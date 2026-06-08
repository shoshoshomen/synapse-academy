# 自然言語推論 — テキスト含意

> 「t が h を含意する」とは、t を読んだ人間が h を真であると結論づけることを意味する。NLIは含意/矛盾/中立を予測するタスクだ。表面上は退屈だが、本番環境では重要な役割を担っている。


## 問題設定

要約システムを構築した。要約が生成された。その要約に幻覚が含まれていないことをどうやって確認するか？

チャットボットを構築した。「はい」と回答した。その回答が検索されたパッセージによって支持されているかどうかをどうやって確認するか？

10,000件のニュース記事をトピック別に分類する必要がある。学習ラベルはない。モデルを再利用できるか？

これら3つの問題はすべて自然言語推論に帰着する。NLIは次の問いを立てる：前提`t`と仮説`h`が与えられた場合、`h`は`t`によって含意されるか、矛盾するか、中立（無関係）か？

- **幻覚チェック：** `t` = ソース文書、`h` = 要約の主張。含意でない = 幻覚。
- **グラウンドされたQA：** `t` = 検索されたパッセージ、`h` = 生成された回答。含意でない = 捏造。
- **ゼロショット分類：** `t` = 文書、`h` = 言語化されたラベル（「これはスポーツに関するものだ」）。含意 = 予測されたラベル。

一つのタスク、三つの本番用途。これがすべてのRAG評価フレームワークがNLIモデルを内部に持っている理由だ。

## 概念

![NLI: 3クラス分類、前提対仮説](../assets/nli.svg)

**3つのラベル。**

- **含意（Entailment）。** `t` → `h`。「猫はマットの上で寝ている」は「部屋に猫がいる」を含意する。
- **矛盾（Contradiction）。** `t` → ¬`h`。「猫はマットの上で寝ている」は「猫がいない」と矛盾する。
- **中立（Neutral）。** どちらの推論もない。「猫はマットの上で寝ている」は「猫がお腹をすかせている」に対して中立だ。

**論理的含意ではない。** NLIは*自然*言語推論 — 厳格な論理ではなく、典型的な人間の読者が推論するものだ。「ジョンは犬を散歩させた」はNLIでは「ジョンは犬を持っている」を含意するが、厳格な一階論理では所有を公理化した場合のみ認められる。

**データセット。**

- **SNLI**（2015年）。570kの人手注釈ペア、前提として画像キャプション。ドメインが狭い。
- **MultiNLI**（2017年）。10ジャンルにわたる433kペア。2026年の標準的な学習コーパス。
- **ANLI**（2019年）。敵対的NLI。人間が既存のモデルを壊すよう特別に設計した例を書いた。より難しい。
- **DocNLI、ConTRoL**（2020〜21年）。文書長の前提。マルチホップと長距離推論をテストする。

**アーキテクチャ。** トランスフォーマーエンコーダー（BERT、RoBERTa、DeBERTa）が`[CLS] premise [SEP] hypothesis [SEP]`を読む。`[CLS]`表現が3クラスのソフトマックスに送られる。MNLIで学習し、保留ベンチマークで評価すると、分布内ペアで90%以上の精度が得られる。

**NLIによるゼロショット分類。** 文書と候補ラベルが与えられた場合、各ラベルを仮説に変換する（「このテキストはスポーツに関するものだ」）。各ラベルの含意確率を計算する。最大値を選ぶ。これがHugging Faceの`zero-shot-classification`パイプラインの裏にあるメカニズムだ。

## 実装する

### ステップ1: 事前学習済みNLIモデルを実行する

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # 全ラベルを返す；非推奨のreturn_all_scores=Trueを置き換え

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

本番向けNLIでは、`facebook/bart-large-mnli`と`microsoft/deberta-v3-large-mnli`がオープンデフォルト。DeBERTa-v3がリーダーボードのトップ。

### ステップ2: ゼロショット分類

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

テンプレートはデフォルトで「This example is about {label}.」。`hypothesis_template`でカスタマイズ可能。学習データ不要。微調整不要。そのまま動作する。

### ステップ3: RAG向け忠実性チェック

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

これがRAGAS忠実性の核心部分だ。生成された回答をアトミックな主張に分割する。各主張を検索されたコンテキストに対してチェックする。含意する割合を報告する。

### ステップ4: 手作りNLI分類器（概念的）

stdlibのみのトイ実装については`code/main.py`を参照：語彙の重複と否定検出による前提と仮説の比較。トランスフォーマーモデルとは競合しないが、タスクの形を示している：2つのテキスト入力、3クラスラベル出力、損失 = `{entail, contradict, neutral}`に対するクロスエントロピー。

## ピットフォール

- **仮説のみのショートカット。** モデルはSNLIで仮説のみから約60%のラベルを予測できる。「not」、「nobody」、「never」が矛盾と相関するため。ラベル漏洩を検出するための強いベースライン。
- **語彙重複ヒューリスティック。** 「すべての部分列は含意される」ヒューリスティックはSNLIに合格するがHANS/ANLIには失敗する。敵対的ベンチマークを使用する。
- **文書長での劣化。** 単文NLIモデルは文書長の前提で20以上のF1が低下する。長いコンテキストにはDocNLIで学習されたモデルを使用する。
- **ゼロショットテンプレートの感度。** 「This example is about {label}」対「{label}」対「The topic is {label}」は精度を10ポイント以上変動させることがある。テンプレートを調整する。
- **ドメインの不一致。** MNLIは一般的な英語で学習される。法律、医療、科学のテキストにはドメイン固有のNLIモデルが必要（例：SciNLI、MedNLI）。

## 使ってみる

2026年のスタック：

| ユースケース | モデル |
|---------|-------|
| 汎用NLI | `microsoft/deberta-v3-large-mnli` |
| 高速 / エッジ | `cross-encoder/nli-deberta-v3-base` |
| ゼロショット分類（軽量） | `facebook/bart-large-mnli` |
| 文書レベルNLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多言語 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAGの幻覚検出 | RAGAS / DeepEval内部のNLI層 |

2026年のメタパターン：NLIはテキスト理解のガムテープだ。「AはBをサポートするか？」または「AはBと矛盾するか？」が必要なときは — 別のLLM呼び出しを行う前にNLIに手を伸ばすこと。

## 成果物を出す

`outputs/skill-nli-picker.md`として保存：

```markdown
---
name: nli-picker
description: 分類/忠実性/ゼロショットタスクに向けたNLIモデル、ラベルテンプレート、評価セットアップを選ぶ。
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

ユースケース（忠実性チェック、ゼロショット分類、文書レベル推論）を与えられた場合、以下を出力する：

1. モデル。名前付きNLIチェックポイント。ドメイン、長さ、言語に関連した理由。
2. テンプレート（ゼロショットの場合）。言語化パターン。例。
3. 閾値。判断ルールの含意カットオフ。キャリブレーションに基づいた理由。
4. 評価。保留されたラベル付きセットでの精度、仮説のみのベースライン、敵対的サブセット。

100サンプルのラベル付きサニティチェックなしのゼロショット分類の出荷を拒否する。文書長の前提に対して文レベルのNLIモデルを使用することを拒否する。NLIが幻覚を解決すると主張することをフラグする — それは削減するものであり排除するものではない。
```

## 演習

1. **易。** `facebook/bart-large-mnli`をすべての3クラスをカバーする20の手作りの（前提、仮説、ラベル）トリプルで実行する。精度を測定する。敵対的な「部分列ヒューリスティック」のトラップ（「ケーキを食べなかった」対「ケーキを食べた」）を追加して、それが壊れるかどうかを確認する。
2. **中。** 100件のAGニュースの見出しでゼロショットテンプレート「This text is about {label}」対「The topic is {label}」対「{label}」を比較する。精度の変動を報告する。
3. **難。** RAGの忠実性チェッカーを構築する：アトミッククレームの分解＋クレームごとのNLI。50件のRAGで生成された回答とゴールドコンテキストで評価する。手動ラベルに対する偽陽性率と偽陰性率を測定する。

## キーワード

| 用語 | 一般的な説明 | 実際の意味 |
|------|------------|----------|
| NLI | 自然言語推論 | 前提-仮説の関係の3クラス分類。 |
| RTE | テキスト含意認識 | NLIの古い名前；同じタスク。 |
| 含意 | 「t が h を意味する」 | 典型的な読者が、t を与えられれば h が真であると結論づけること。 |
| 矛盾 | 「t が h を排除する」 | 典型的な読者が、t を与えられれば h が偽であると結論づけること。 |
| 中立 | 「未決定」 | t から h への推論がいずれの方向にもない。 |
| ゼロショット分類 | NLIとして分類器 | ラベルを仮説として言語化し、最大含意を選ぶ。 |
| 忠実性 | 回答はサポートされているか？ | （検索されたコンテキスト、生成された回答）に対するNLI。 |

## 参考資料

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI。
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI。
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — ANLIベンチマーク。
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — 分類器としてのNLI。
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — 2026年のNLIの主力モデル。
