# 構造化出力と制約付きデコーディング

> LLMにJSONを要求する。ほとんどの場合JSONが返ってくる。本番環境では「ほとんど」が問題だ。制約付きデコーディングは、サンプリング前にlogitsを編集することで「ほとんど」を「常に」に変える。


## 問題設定

分類器がLLMに「{positive, negative, neutral}のいずれかを返せ」とプロンプトを送る。モデルが返すのは「感情はpositiveです — このレビューは顧客が明示的に述べているように圧倒的に好意的で...」。パーサーがクラッシュする。分類器のF1は0.0になる。

自由形式の生成はコントラクトではない。提案に過ぎない。本番システムにはコントラクトが必要だ。

2026年には3つの層が存在する。

1. **プロンプティング。** 丁寧にお願いする。「JSONオブジェクトのみを返せ」。フロンティアモデルでは約80%機能し、小さなモデルではそれ以下。
2. **ネイティブ構造化出力API。** OpenAIの`response_format`、Anthropicのツール使用、GeminiのJSONモード。サポートされるスキーマでは信頼性が高い。ベンダーにロックインされる。
3. **制約付きデコーディング。** 各生成ステップでlogitsを変更し、モデルが無効なトークンを*生成できない*ようにする。構造的に常に100%有効。任意のローカルモデルで動作する。

このレッスンでは3つすべての直感を構築し、どのような場合にどれを使うかを解説する。

## 概念

![各ステップで無効なトークンをマスクする制約付きデコーディング](../assets/constrained-decoding.svg)

**制約付きデコーディングの仕組み。** 各生成ステップで、LLMは完全な語彙（約100kトークン）に対するlogitベクトルを生成する。*logitプロセッサー*がモデルとサンプラーの間に位置する。ターゲット文法（JSONスキーマ、正規表現、文脈自由文法）における現在位置を考慮して、どのトークンが有効かを計算し、すべての無効なトークンのlogitsを負の無限大に設定する。残りのlogitsのソフトマックスにより確率質量が有効な継続のみに置かれる。

2026年の実装：

- **Outlines。** JSONスキーマまたは正規表現を有限状態機械にコンパイルする。すべてのトークンにO(1)の有効な次のトークンルックアップが得られる。FSMベースのため、再帰的なスキーマには平坦化が必要。
- **XGrammar / llguidance。** 文脈自由文法エンジン。再帰的なJSONスキーマを処理できる。デコーディングオーバーヘッドがほぼゼロ。OpenAIは2025年の構造化出力実装でllguidanceを参照している。
- **vLLM guided decoding。** 組み込みの`guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`（Outlines、XGrammar、またはlm-format-enforcerバックエンド経由）。
- **Instructor。** あらゆるLLMに対するPydanticベースのラッパー。検証失敗時にリトライ。クロスプロバイダー対応だが、logitsは変更しない — リトライと構造化出力対応プロンプトに依存する。

### 反直感的な結果

制約付きデコーディングはしばしば非制約の生成より*速い*。二つの理由がある。第一に、次のトークン検索空間を縮小する。第二に、巧妙な実装では強制されたトークン（`{"name": "`のような足場 — すべてのバイトが確定されている）の生成を完全にスキップする。

### コストを払うピットフォール

フィールドの順序が重要。`answer`を`reasoning`の前に置くと、モデルは考える前に回答にコミットしてしまう。JSONは有効。回答は間違っている。検証では捕捉できない。

```json
// 悪い例
{"answer": "yes", "reasoning": "because ..."}

// 良い例
{"reasoning": "... therefore ...", "answer": "yes"}
```

スキーマのフィールド順序はロジックであり書式ではない。

## 実装する

### ステップ1: スクラッチからの正規表現制約付き生成

スタンドアロンのFSM実装については`code/main.py`を参照。核となるアイデアを30行で：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSMはこれまでに文法のどの部分を満たしたかを追跡する。`valid_tokens(state, tokenizer)`は、受け入れパスを離れることなくFSMを進めることができる語彙トークンを計算する。

### ステップ2: JSONスキーマのためのOutlines

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

検証エラーはゼロ。常に。FSMが無効な出力を到達不能にする。

### ステップ3: プロバイダー非依存なPydantic用のInstructor

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

異なるメカニズム。Instructorはlogitsに触れない。スキーマをプロンプトにフォーマットし、出力をパースし、検証失敗時にリトライする（デフォルト3回）。あらゆるプロバイダーで機能する。リトライはレイテンシとコストを増加させる。クロスプロバイダーの移植性が売り物だ。

### ステップ4: ネイティブベンダーAPI

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

サーバーサイドの制約付きデコーディング。サポートされるスキーマでOutlinesと同等の信頼性。ローカルモデル管理なし。ベンダーにロックインされる。

## ピットフォール

- **再帰的なスキーマ。** OutlinesはIは再帰を固定された深さに平坦化する。ツリー構造の出力（ネストされたコメント、AST）にはXGrammarかllguidance（CFGベース）が必要。
- **巨大な列挙型。** 10,000オプションの列挙型はコンパイルが遅いかタイムアウトする。リトリーバーに切り替える：最初にトップk候補を予測し、それに制約を適用する。
- **文法が厳格すぎる。** `date: "YYYY-MM-DD"`正規表現を強制すると、日付不明のケースでモデルが`"unknown"`を出力できない。モデルは日付を捏造することで補償する。`null`またはセンチネルを許可する。
- **早期コミット。** 上記のフィールド順序のピットフォールを参照。常に推論を先に置く。
- **スキーマなしのベンダーJSONモード。** 純粋なJSONモードは有効なJSONのみを保証し、*あなたのユースケースに対して*有効であることは保証しない。常に完全なスキーマを提供する。

## 使ってみる

2026年のスタック：

| 状況 | 選択 |
|-----------|------|
| OpenAI/Anthropic/Googleモデル、シンプルなスキーマ | ネイティブベンダーの構造化出力 |
| あらゆるプロバイダー、Pydanticワークフロー、リトライを許容 | Instructor |
| ローカルモデル、100%有効性が必要、フラットなスキーマ | Outlines（FSM） |
| ローカルモデル、再帰的なスキーマ | XGrammarかllguidance |
| セルフホストの推論サーバー | vLLM guided decoding |
| リトライが許容できるバッチ処理 | Instructor＋最安モデル |

## 成果物を出す

`outputs/skill-structured-output-picker.md`として保存：

```markdown
---
name: structured-output-picker
description: 構造化出力アプローチ、スキーマ設計、検証計画を選択する。
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

ユースケース（プロバイダー、レイテンシ予算、スキーマの複雑さ、失敗への許容度）を与えられた場合、以下を出力する：

1. メカニズム。ネイティブベンダーの構造化出力、Instructorリトライ、Outlines FSM、またはXGrammar CFG。一文の理由。
2. スキーマ設計。フィールド順序（推論を先に、回答を後に）、「不明」のための nullable フィールド、列挙型対正規表現、必須フィールド。
3. 失敗戦略。最大リトライ数、フォールバックモデル、適切な`null`処理、分布外の拒否。
4. 検証計画。スキーマ準拠率（目標100%）、意味的有効性（LLMジャッジ）、フィールドカバレッジ率、レイテンシp50/p99。

推論フィールドの前に`answer`または`decision`を置く設計を拒否する。スキーマなしの生のJSONモードの使用を拒否する。FSMのみのライブラリの背後にある再帰的なスキーマをフラグする。
```

## 演習

1. **易。** 制約付きデコーディングなしで小さなオープンウェイトモデル（例：Llama-3.2-3B）に`Review(sentiment, confidence, evidence_span)`のプロンプトを送る。100件のレビューで有効なJSONとしてパースできる割合を測定する。
2. **中。** Outlines JSONモードで同じコーパスを試す。準拠率、レイテンシ、意味的精度を比較する。
3. **難。** 電話番号（`\d{3}-\d{3}-\d{4}`）のスクラッチから正規表現制約付きデコーダーを実装する。1000サンプルで無効な出力がゼロであることを確認する。

## キーワード

| 用語 | 一般的な説明 | 実際の意味 |
|------|------------|----------|
| 制約付きデコーディング | 有効な出力を強制 | 各生成ステップで無効なトークンのlogitsをマスクする。 |
| logitプロセッサー | 制約を加えるもの | 関数：`(logits, state) -> masked_logits`。 |
| FSM | 有限状態機械 | コンパイルされた文法表現；O(1)の有効な次のトークンルックアップ。 |
| CFG | 文脈自由文法 | 再帰を処理する文法；FSMより遅いがより表現力が高い。 |
| スキーマフィールド順序 | 重要か？ | はい — 最初のフィールドがコミットする；常に回答の前に推論を置く。 |
| ガイド付きデコーディング | vLLMの呼び方 | 同じ概念が推論サーバーに統合されたもの。 |
| JSONモード | OpenAIの早期バージョン | JSON構文を保証する；スキーマへの適合は保証しない。 |

## 参考資料

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — Outlinesの論文。
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — 高速CFGベースの制約付きデコーディング。
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — 推論サーバーの統合。
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — APIリファレンス＋注意点。
- [Instructor library](https://python.useinstructor.com/) — プロバイダー横断のPydantic＋リトライ。
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — 6つの制約付きデコーディングフレームワークのベンチマーク。
