# LLM評価 — RAGAS、DeepEval、G-Eval

> 完全一致とF1は意味的等価性を見逃す。人間によるレビューはスケールしない。LLM-as-judgeが本番の答えだ — 数値を信頼するための十分なキャリブレーションとともに。


## 問題設定

RAGシステムが回答する：「June 29th, 2007.」
ゴールド参照は：「June 29, 2007.」
完全一致スコアは0。F1スコアは約75%。人間なら100%と採点する。

これを1万件のテストケースで掛け算する。さらに検索器、チャンキング、プロンプト、またはモデルへのすべての変更で掛け算する。意味を理解し、スケールで安価に実行し、回帰について嘘をつかず、正しい失敗モードを浮かび上がらせる評価器が必要だ。

2026年にはこの問題を担う3つのフレームワークがある。

- **RAGAS。** Retrieval-Augmented Generation ASsessment。4つのRAG指標（忠実性、回答関連性、コンテキスト精度、コンテキスト再現率）とNLI + LLM-judgeバックエンド。研究に裏付けられた軽量設計。
- **DeepEval。** LLMのためのpytest。G-Eval、タスク完了、幻覚、バイアス指標。CI/CDネイティブ。
- **G-Eval。** 手法（かつDeepEvalの指標）：思考の連鎖を持つLLM-as-judge、カスタム基準、0〜1スコア。

3つすべてがLLM-as-judgeに依存している。このレッスンではその手法と信頼レイヤーの直感を構築する。

## 概念

![4つの評価次元、LLM-as-judgeアーキテクチャ](../assets/llm-evaluation.svg)

**LLM-as-judge。** 静的な指標を、ルーブリックが与えられた場合に出力をスコアリングするLLMで置き換える。`(クエリ, コンテキスト, 回答)`を与えられた場合、ジャッジLLMにプロンプトを送る：「忠実性を0〜1でスコアリングしてください。」スコアを返す。

なぜ機能するか：LLMはわずかなコストで人間の判断を近似する。GPT-4o-miniは1スコアあたり約$0.003で、1,000サンプルの回帰評価実行が$5未満で可能。

なぜ暗黙的に失敗するか：

1. **ジャッジのバイアス。** ジャッジは長い回答、自分のモデルファミリーからの回答、プロンプトスタイルに一致する回答を好む。
2. **JSONパースの失敗。** 不正なJSON → NaNスコア → 集計から暗黙的に除外される。RAGASユーザーはこの苦痛を知っている。try/except + 明示的な失敗モードでゲートする。
3. **モデルバージョン間のドリフト。** ジャッジのアップグレードですべての指標が変わる。ジャッジモデル + バージョンを固定する。

**RAGの4つ。**

| 指標 | 問い | バックエンド |
|------|------|------------|
| 忠実性 | 回答の各主張は検索されたコンテキストから来ているか？ | NLIベースの含意 |
| 回答関連性 | 回答は質問に対応しているか？ | 回答から仮説的な質問を生成；実際の質問と比較 |
| コンテキスト精度 | 検索されたチャンクのうち、関連したものの割合は？ | LLM-judge |
| コンテキスト再現率 | 検索は必要なすべてのものを返したか？ | ゴールド回答に対するLLM-judge |

**G-Eval。** カスタム基準を定義する：「回答は正しいソースを引用したか？」フレームワークが思考の連鎖評価ステップに自動展開し、その後0〜1をスコアリングする。RAGASがカバーしないドメイン固有の品質次元に適している。

**キャリブレーション。** 人間のラベルとの相関を確認するまで生のジャッジスコアを信頼しないこと。100件の手動ラベル付きサンプルを実行する。ジャッジ対人間をプロットする。スピアマン相関係数を計算する。ρ < 0.7ならジャッジのルーブリックに問題がある。

## 実装する

### ステップ1: NLIによる忠実性（RAGASスタイル）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` は任意の呼び出し可能オブジェクト: プロンプト文字列 -> 生成文字列。
# 例: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

回答をアトミックな主張に分解する。各主張を検索されたコンテキストに対してNLIチェックする。忠実性 = サポートされた割合。

### ステップ2: 回答関連性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: .encode(texts, normalize_embeddings=True) -> ndarray を実装する任意のモデル
# 例: encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

回答が尋ねられた質問とは異なる質問を示唆している場合、関連性が下がる。

### ステップ3: G-Evalカスタム指標

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

評価ステップがルーブリックだ。明示的なステップは暗黙的な「0〜1でスコアリング」プロンプトよりも安定している。

### ステップ4: CIゲート

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

pytestファイルとして出荷する。すべてのPRで実行する。回帰でマージをブロックする。

### ステップ5: スクラッチからのトイ評価

`code/main.py`を参照。stdlibのみの忠実性（回答の主張とコンテキストの重複）と関連性（回答トークンと質問トークンの重複）の近似。本番用ではない。形を示している。

## ピットフォール

- **キャリブレーションなし。** 人間のラベルとの相関が0.3のジャッジはノイズだ。出荷前にキャリブレーション実行を必須とする。
- **自己評価。** 生成と評価に同じLLMを使用するとスコアが10〜20%膨らむ。ジャッジには別のモデルファミリーを使用する。
- **ペア評価での位置バイアス。** ジャッジは最初に提示されたオプションを好む。常に順序をランダム化して両方実行する。
- **生の集計が失敗を隠す。** 平均スコア0.85はしばしば5%の致命的な失敗を隠す。常に底部の分位数を検査する。
- **ゴールデンデータセットの腐敗。** バージョン管理されていない評価セットが時間とともにドリフトすると縦断的比較が壊れる。すべての変更でデータセットにタグを付ける。
- **LLMコスト。** 規模では、ジャッジの呼び出しがコストを支配する。キャリブレーション閾値を満たす最も安いモデルを使用する。GPT-4o-mini、Claude Haiku、Mistral-small。

## 使ってみる

2026年のスタック：

| ユースケース | フレームワーク |
|------------|-------------|
| RAG品質監視 | RAGAS（4指標） |
| CI/CD回帰ゲート | DeepEval + pytest |
| カスタムドメイン基準 | DeepEval内のG-Eval |
| オンラインのライブトラフィック監視 | RAGASの参照フリーモード |
| ヒューマンインザループのスポットチェック | 注釈UIを持つLangSmithまたはPhoenix |
| レッドチーミング/安全評価 | Promptfoo + DeepEval |

典型的なスタック：監視にはRAGAS、CIにはDeepEval、新しい次元にはG-Eval。3つすべてを実行する；それらは有益な意見の相違をする。

## 成果物を出す

`outputs/skill-eval-architect.md`として保存：

```markdown
---
name: eval-architect
description: キャリブレーションされたジャッジとCIゲートを持つLLM評価計画を設計する。
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

ユースケース（RAG/エージェント/生成タスク）を与えられた場合、以下を出力する：

1. 指標。忠実性/関連性/コンテキスト精度/コンテキスト再現率 + 基準を持つカスタムG-Eval指標。
2. ジャッジモデル。名前付きモデル + バージョン、コスト対精度の理由。
3. キャリブレーション。手動ラベル付きセットのサイズ、人間に対するターゲットスピアマンρ > 0.7。
4. データセットのバージョニング。タグ戦略、変更ログ、層化。
5. CIゲート。指標ごとの閾値、回帰ウィンドウロジック、底部分位数アラート。

≥50件の人間ラベル付きサンプルでテストされていないジャッジに依存することを拒否する。自己評価（同じモデルが生成 + 評価）を拒否する。底部10%を浮かび上がらせることなく集計のみの報告を拒否する。並行ベースライン評価なしでジャッジのアップグレードがランディングするパイプラインをフラグする。
```

## 演習

1. **易。** 既知の幻覚がある10件のRAGの例でRAGASを使用する。忠実性指標がそれぞれをキャッチするかを確認する。
2. **中。** 50件のQA回答を正確性について0〜1で手動ラベル付けする。G-Evalでスコアリングする。ジャッジと人間のスピアマンρを測定する。
3. **難。** DeepEvalでpytestのCIゲートを構築する。意図的に検索器を回帰させる。ゲートが失敗することを確認する。最低10%の閾値チェックで底部分位数アラートを追加する。

## キーワード

| 用語 | 一般的な説明 | 実際の意味 |
|------|------------|----------|
| LLM-as-judge | LLMでスコアリング | ジャッジモデルにルーブリックを与えて出力を0〜1でスコアリングするようにプロンプトを送る。 |
| RAGAS | RAG指標ライブラリ | 4つの参照フリーRAG指標を持つオープンソース評価フレームワーク。 |
| 忠実性 | 回答はグラウンドされているか？ | 検索されたコンテキストによって含意された回答の主張の割合。 |
| コンテキスト精度 | 検索されたチャンクは関連していたか？ | 実際に重要だったトップKチャンクの割合。 |
| コンテキスト再現率 | 検索はすべてを見つけたか？ | 検索されたチャンクによってサポートされたゴールド回答の主張の割合。 |
| G-Eval | カスタムLLMジャッジ | ルーブリック + 思考の連鎖評価ステップ + 0〜1スコア。 |
| キャリブレーション | 信頼するが確認する | ジャッジスコアと人間スコアのスピアマン相関。 |

## 参考資料

- [Es et al. (2023). RAGAS: RAGの自動評価](https://arxiv.org/abs/2309.15217) — RAGASの論文。
- [Liu et al. (2023). G-Eval: より良い人間アライメントを持つGPT-4を使用したNLG評価](https://arxiv.org/abs/2303.16634) — G-Evalの論文。
- [DeepEvalドキュメント](https://deepeval.com/docs/metrics-introduction) — オープンな本番スタック。
- [Zheng et al. (2023). MT-BenchとChatbot ArenaによるLLM-as-a-Judgeの評価](https://arxiv.org/abs/2306.05685) — バイアス、キャリブレーション、限界。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — RAGAS、DeepEval、Phoenixを統合する統一フレームワーク。
