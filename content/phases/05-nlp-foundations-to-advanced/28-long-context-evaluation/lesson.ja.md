# 長コンテキスト評価 — NIAH、RULER、LongBench、MRCR

> Gemini 3 Proは1,000万トークンのコンテキストを宣伝している。100万トークンでは、8針のMRCRが26.3%に低下する。宣伝 ≠ 使用可能。長コンテキスト評価は出荷するモデルの実際の容量を教える。


## 問題設定

200ページの契約書がある。モデルは100万トークンのコンテキストを主張する。契約書を貼り付けて尋ねる：「解約条項は何か？」モデルが回答する — しかし解約条項がモデルが実際にアテンドできる範囲を超えた12万トークンの深さにあるため、表紙から回答する。

これが2026年のコンテキスト容量ギャップだ。仕様書は100万または1,000万と言う。現実は使用可能なのはその60〜70%であり、「使用可能」はタスクによる。

- **検索（ヘイスタックの単一針）：** フロンティアモデルでは宣伝された最大値まで完璧に近い。
- **マルチホップ/集計：** ほとんどのモデルで128k以降に急激に劣化する。
- **分散した事実に対する推論：** 最初に失敗するタスク。

長コンテキスト評価はこれらの軸を測定する。このレッスンではベンチマークに名前を付け、それぞれが実際に何を測定するか、そしてドメイン用のカスタム針テストを構築する方法を説明する。

## 概念

![NIAHベースライン、RULERマルチタスク、LongBenchホリスティック](../assets/long-context-eval.svg)

**Needle-in-a-Haystack（NIAH、2023年）。** 長いコンテキストの制御された深さに事実（「マジックワードはパイナップルだ」）を置く。モデルにそれを取得するよう求める。深さ × 長さをスイープする。元の長コンテキストベンチマーク。フロンティアモデルは今これを飽和させる；必要だが十分でないベースライン。

**RULER（Nvidia、2024年）。** 4つのカテゴリにわたる13のタスクタイプ：検索（単一/マルチキー/マルチバリュー）、マルチホップトレース（変数追跡）、集計（一般的な単語頻度）、QA。設定可能なコンテキスト長（4k〜128k以上）。NIAHを飽和させるがマルチホップで失敗するモデルを明らかにする。2024年のリリースでは、32k以上のコンテキストを主張する17モデルのうち半分しか32kで品質を維持しなかった。

**LongBench v2（2024年）。** 503の選択式質問、8k〜200万語のコンテキスト、6つのタスクカテゴリ：単一文書QA、マルチ文書QA、長い文脈内学習、長いダイアログ、コードリポジトリ、長い構造化データ。現実世界の長コンテキスト動作の本番ベンチマーク。

**MRCR（マルチラウンド共参照解析）。** スケールでのマルチターン共参照。8針、24針、100針バリアント。アテンションが劣化する前にモデルが処理できる事実の数を明らかにする。

**NoLiMa。** 「非語彙的針」。針とクエリは文字通りの重複を持たない；検索には意味的推論の一ステップが必要。NIAHより難しい。

**HELMET。** 多くの文書を連結し、任意の一つから質問する。選択的アテンションをテストする。

**BABILong。** 無関係なヘイスタックの中にbAbI推論チェーンを埋め込む。検索だけでなく、ヘイスタック内の推論をテストする。

### 実際に報告すべきもの

- **宣伝されたコンテキストウィンドウ。** 仕様書の数値。
- **有効な検索長。** 一定の閾値（例：90%）でのNIAHパス。
- **有効な推論長。** その閾値でのマルチホップまたは集計パス。
- **劣化曲線。** コンテキスト長に対する精度、タスクタイプ別にプロット。

仕様書のための2つの数字：検索有効と推論有効。通常、推論有効は宣伝されたウィンドウの25〜50%だ。

## 実装する

### ステップ1: ドメイン用カスタムNIAH

`code/main.py`を参照。スケルトン：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # ヘイスタック本体を埋めるのに十分な長さになるまでフィラーを繰り返す。
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

`depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}をスイープする。ヒートマップをプロットする。それがターゲットモデルのNIAHカードだ。

### ステップ2: マルチ針バリアント

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

「3つのマジックワードは何か？」のような質問は3つすべての取得が必要だ。単一針の成功はマルチ針の成功を予測しない。

### ステップ3: マルチホップ変数トレース（RULERスタイル）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

回答には3つの代入をチェーンする必要がある。128kでのフロンティアモデルはここでよく50〜70%の精度に低下する。

### ステップ4: スタックでのLongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

カテゴリ別の精度を報告する。集計スコアは大きなタスクレベルの差異を隠す。

## ピットフォール

- **NIAHのみの評価。** 100万トークンでNIAHに合格することはマルチホップについて何も言わない。常にRULERまたはカスタムマルチホップテストを実行する。
- **均一な深さのサンプリング。** 多くの実装はdepth=0.5のみをテストする。depth=0、0.25、0.5、0.75、1.0をテストする — 「中間で迷子になる」効果は本物だ。
- **フィラーとの語彙の重複。** 針がフィラーとキーワードを共有する場合、検索が自明になる。NoLiMAスタイルの重複しない針を使用する。
- **レイテンシを無視する。** 100万トークンのプロンプトはプリフィルに30〜120秒かかる。精度と並行して最初のトークンまでの時間を測定する。
- **ベンダーの自己報告数値。** OpenAI、Google、Anthropicはすべて自分のスコアを公開する。常に自分のユースケースで独立して再実行する。

## 使ってみる

2026年のスタック：

| 状況 | ベンチマーク |
|------|------------|
| クイックサニティチェック | 3深さ × 3長さのカスタムNIAH |
| 本番向けのモデル選択 | ターゲット長でのRULER（13タスク） |
| 現実世界のQA品質 | LongBench v2の単一文書QAサブセット |
| マルチホップ推論 | BABILongまたはカスタム変数トレース |
| 会話/ダイアログ | ターゲット長でのMRCR 8針 |
| モデルアップグレード回帰 | 固定の社内NIAH + RULERハーネス、すべての新モデルで実行 |

本番のための経験則：意図した長さでNIAH + 1つの推論タスクを持つまでコンテキストウィンドウを信頼しない。

## 成果物を出す

`outputs/skill-long-context-eval.md`として保存：

```markdown
---
name: long-context-eval
description: 特定のモデルとユースケース向けの長コンテキスト評価バッテリーを設計する。
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

ターゲットモデル、ターゲットコンテキスト長、ユースケースを与えられた場合、以下を出力する：

1. テスト。NIAH深さ × 長さグリッド；RULERマルチホップ；カスタムドメインタスク。
2. サンプリング。各長さで深さ0、0.25、0.5、0.75、1.0。
3. 指標。検索パス率；推論パス率；最初のトークンまでの時間；クエリあたりのコスト。
4. カットオフ。有効な検索長（90%パス）と有効な推論長（70%パス）。両方を報告する。
5. 回帰。固定ハーネス、すべてのモデルアップグレードで再実行、デルタを浮かび上がらせる。

モデルカードだけからのコンテキストウィンドウを信頼することを拒否する。マルチホップワークロードのNIAHのみの評価を拒否する。独立した証拠としてベンダーの自己報告の長コンテキストスコアを拒否する。
```

## 演習

1. **易。** 3深さ（0.25、0.5、0.75）× 3長さ（1k、4k、16k）のNIAHを構築する。任意のモデルで実行する。パス率を3×3のヒートマップとしてプロットする。
2. **中。** 3針バリアントを追加する。各長さで3つすべての取得を測定する。同じ長さでの単一針パス率と比較する。
3. **難。** 64kのフィラーに埋め込まれた変数トレースタスク（X1 → X2 → X3、3ホップ）を構築する。3つのフロンティアモデルの精度を測定する。モデルごとの有効な推論長を報告する。

## キーワード

| 用語 | 一般的な説明 | 実際の意味 |
|------|------------|----------|
| NIAH | ヘイスタックの中の針 | フィラーに事実を植え、モデルにそれを取得するよう求める。 |
| RULER | NIAHをパワーアップ | 検索/マルチホップ/集計/QAにわたる13のタスクタイプ。 |
| 有効コンテキスト | 実際の容量 | 精度が閾値以上を保持する長さ。 |
| 中間で迷子になる | 深さバイアス | モデルは長い入力の中間にあるコンテンツにアテンドしにくい。 |
| マルチ針 | 同時に多くの事実 | 複数の針；検索だけでなく、アテンションのジャグリングをテストする。 |
| MRCR | マルチラウンドcoref | 8、24、または100針の共参照；アテンション飽和を明らかにする。 |
| NoLiMa | 非語彙的針 | 針とクエリは文字通りのトークンを共有しない；推論が必要。 |

## 参考資料

- [Kamradt (2023). ヘイスタックの中の針分析](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 元のNIAHリポジトリ。
- [Hsieh et al. (2024). RULER: 長コンテキストLMの実際のコンテキストサイズは何か？](https://arxiv.org/abs/2404.06654) — マルチタスクベンチマーク。
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 現実世界の長コンテキスト評価。
- [Modarressi et al. (2024). NoLiMa: 非語彙的針](https://arxiv.org/abs/2404.06666) — より難しい針。
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — ヘイスタック内の推論。
- [Liu et al. (2024). 中間で迷子になる：言語モデルが長いコンテキストをどのように使うか](https://arxiv.org/abs/2307.03172) — 深さバイアスの論文。
