# 多言語NLP

> 一つのモデル、100以上の言語、そのほとんどに学習データがゼロ。言語横断転移学習は2020年代の実用的な奇跡だ。


## 問題設定

英語には数十億の注釈付きサンプルがある。ウルドゥー語には数千しかない。マイティリー語にはほとんどない。グローバルなユーザーを対象とした実用的なNLPシステムは、タスク固有の学習データが存在しない言語のロングテールに対応する必要がある。

多言語モデルはこれを解決する。複数の言語を同時に学習することで共有された表現が生まれ、高リソース言語で学んだスキルを低リソース言語に転移できる。英語の感情分析でモデルを微調整すると、そのままウルドゥー語の感情予測が驚くほどうまくいく。これがゼロショット言語横断転移学習であり、NLPが世界に展開される方法を変えた。

このレッスンではトレードオフ、定番モデル、そして多言語作業に不慣れなチームがつまずく一つの判断（転移のためのソース言語の選択）を解説する。

## 概念

![共有多言語埋め込み空間による言語横断転移](../assets/multilingual.svg)

**共有語彙。** 多言語モデルは、すべてのターゲット言語のテキストで学習されたSentencePieceまたはWordPieceトークナイザーを使用する。語彙は共有される：同じサブワード単位が関連する言語にわたって同じ形態素を表す。英語とイタリア語の`anti-`は同じトークンになる。

**共有表現。** 多くの言語にわたってマスク言語モデリングで事前学習されたトランスフォーマーは、異なる言語で意味的に類似した文が類似した隠れ状態を生成することを学習する。mBERT、XLM-R、NLLBはすべてこれを示している。英語の「cat」の埋め込みはフランス語の「chat」やスペイン語の「gato」の近くに集まり、文レベルの埋め込みも同様である。

**ゼロショット転移。** 一つの言語（通常は英語）の注釈付きデータでモデルを微調整する。推論時には、モデルがサポートする他の言語で実行する。ターゲット言語のラベルは不要。類型的に関連する言語に対しては強い結果が出て、遠い言語に対しては弱くなる。

**フューショット微調整。** ターゲット言語の100〜500個の注釈付きサンプルを追加する。分類タスクで精度が英語ベースラインの95〜98%まで跳ね上がる。これが多言語NLPで最もコスト効果の高い手段だ。

## モデル

| モデル | 年 | 言語カバレッジ | 備考 |
|-------|------|----------|-------|
| mBERT | 2018 | 104言語 | Wikipediaで学習。初の実用的な多言語LM。低リソース言語に弱い。 |
| XLM-R | 2019 | 100言語 | CommonCrawlで学習（Wikipediaよりはるかに大きい）。言語横断ベースラインを設定。Base 270M、Large 550M。 |
| XLM-V | 2023 | 100言語 | XLM-Rと1Mトークン語彙（250kに対して）。低リソース言語に優れる。 |
| mT5 | 2020 | 101言語 | 多言語生成用のT5アーキテクチャ。 |
| NLLB-200 | 2022 | 200言語 | Metaの翻訳モデル；55の低リソース言語を含む。 |
| BLOOM | 2022 | 46言語＋13プログラミング言語 | 多言語で学習されたオープン176B LLM。 |
| Aya-23 | 2024 | 23言語 | CohereのマルチリンガルLLM。アラビア語、ヒンディー語、スワヒリ語に強い。 |

ユースケースで選ぶ。分類はXLM-R-baseを標準的なデフォルトとして使う。生成タスクは翻訳かオープン生成かによってmT5かNLLBを選ぶ。LLMスタイルの作業はAya-23か明示的な多言語プロンプティングでのClaudeと組み合わせる。

## ソース言語の決定（2026年の研究）

ほとんどのチームはデフォルトで英語を微調整ソースとして使用する。最近の研究（2026年）はこれが多くの場合に誤りであることを示している。

言語の類似性はコーパスのサイズよりも転移品質を予測する。スラブ系ターゲットにはドイツ語かロシア語が英語を上回ることが多い。インド系ターゲットにはヒンディー語が英語を上回ることが多い。**qWALS**類似度指標（2026年、世界言語構造アトラスの特徴に基づく）はこれを数値化する。**LANGRANK**（Lin et al.、ACL 2019）は、言語的類似性、コーパスサイズ、遺伝的関連性の組み合わせから候補ソース言語をランク付けする別の、より古い手法だ。

実践的なルール：ターゲット言語に類型的に近い高リソース言語があれば、まずその言語で微調整を試みてから、英語での微調整と比較する。

## 実装する

### ステップ1: ゼロショット言語横断分類

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一つのモデル、三つの言語、同じAPI。NLIデータで学習されたXLM-Rはentailmentトリックによる分類にうまく転移する。

### ステップ2: 多言語埋め込み空間

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻訳は埋め込み空間で近くに位置する。異なる英語の文はより遠くに位置する。これにより言語横断検索、クラスタリング、類似性が機能する。

### ステップ3: フューショット微調整戦略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

100〜500のターゲット言語サンプルの場合、`num_train_epochs=5`と`learning_rate=2e-5`が安全なデフォルト値。高い学習率は多言語アライメントを崩壊させ、英語のみのモデルになってしまう。

## 実際に機能する評価

- **保留セットでの言語ごとの精度。** 集計ではなく。集計はロングテールを隠す。
- **単言語ベースラインとのベンチマーク比較。** 十分なデータがある言語では、ゼロから学習した単言語モデルが多言語モデルを上回ることがある。テストすること。
- **エンティティレベルのテスト。** ターゲット言語での名前付きエンティティ。多言語モデルはラテン文字から遠いスクリプトのトークン化が弱いことが多い。
- **言語横断一貫性。** 二つの言語で同じ意味が同じ予測を生成するべき。ギャップを測定する。

## 使ってみる

2026年のスタック：

| タスク | 推奨 |
|-----|-------------|
| 分類、100言語 | XLM-R-base（約270M）微調整版 |
| ゼロショットテキスト分類 | `joeddav/xlm-roberta-large-xnli` |
| 多言語センテンス埋め込み | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 翻訳、200言語 | `facebook/nllb-200-distilled-600M`（レッスン11参照） |
| 生成型多言語 | Claude、GPT-4、Aya-23、mT5-XXL |
| 低リソース言語NLP | XLM-Vか関連する高リソース言語でのドメイン固有微調整 |

パフォーマンスが重要な場合は必ずターゲット言語での微調整を予算に含めること。ゼロショットは出発点であって最終回答ではない。

### トークン化コスト（低リソース言語で起きること）

多言語モデルはすべての言語間で一つのトークナイザーを共有する。その語彙は英語、フランス語、スペイン語、中国語、ドイツ語が支配するコーパスで学習されている。主要セット外の言語では、三つのコストが静かに複合する：

- **フェルティリティコスト。** 低リソース言語のテキストは英語よりも単語あたり多くのトークンに分割される。ヒンディー語の文は同等の英語の文の3〜5倍のトークンが必要になることがある。この3〜5倍はコンテキストウィンドウ、学習効率、レイテンシを圧迫する。
- **変形回復コスト。** タイプミス、発音区別符号の変形、Unicode正規化の不一致、大文字小文字の変形はすべて埋め込み空間での冷スタートの無関係なシーケンスになる。ネイティブスピーカーには自明な正書法の対応関係をモデルが学習できない。
- **容量スピルオーバーコスト。** コスト1と2がコンテキスト位置、レイヤー深度、埋め込み次元を消費する。同じモデルで高リソース言語が受ける実際の推論に残るものは体系的に少なくなる。

実際の症状：ヒンディー語でモデルを普通に学習でき、損失曲線が正常に見え、評価パープレキシティが妥当に見えるのに、本番出力が微妙におかしい。形態論が文の途中で崩壊する。稀な活用形が回復不能のまま残る。**壊れたトークナイザーからはデータスケーリングで抜け出せない。**

緩和策：ターゲット言語に良いカバレッジを持つトークナイザーを選ぶ（XLM-Vの1Mトークン語彙が直接的な解決策）；学習前に保留されたターゲットテキストでトークン化フェルティリティを確認する；本当に稀なスクリプトにはバイトレベルフォールバックを使用する（SentencePiece `byte_fallback=True`、GPT-2スタイルのバイトレベルBPE）ことで、OOVがなくなる。

## 成果物を出す

`outputs/skill-multilingual-picker.md`として保存：

```markdown
---
name: multilingual-picker
description: 多言語NLPタスクのソース言語、ターゲットモデル、評価計画を選ぶ。
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

要件（ターゲット言語、タスクの種類、言語ごとの利用可能なラベル付きデータ）を与えられた場合、以下を出力する：

1. 微調整のソース言語。デフォルトは英語；ターゲット言語に類型的に近い高リソース言語がある場合はLANGRANKかqWALSを確認する。
2. ベースモデル。XLM-R（分類）、mT5（生成）、NLLB（翻訳）、Aya-23（生成型LLM）。
3. フューショット予算。利用可能であれば100〜500のターゲット言語サンプルから始める。ラベリングが実行不可能な場合のみゼロショット。
4. 評価計画。言語ごとの精度（集計ではなく）、言語横断一貫性、非ラテンスクリプトでのエンティティレベルF1。

集計指標なしでは多言語モデルを出荷しないこと — 集計指標はロングテールの失敗を隠す。バイトフォールバックが必要なモデルなしにスクリプトのカバレッジが低い言語（アムハラ語、ティグリニャ語、多くのアフリカ系言語）をフラグする（SentencePieceのbyte_fallback=True、またはバイトレベルトークナイザー（GPT-2など））。
```

## 演習

1. **易。** 英語、フランス語、ヒンディー語、アラビア語で各10文のゼロショット分類パイプラインを実行する。各言語の精度を報告する。フランス語は強く、ヒンディー語は普通、アラビア語はまちまちになるはずである。
2. **中。** `paraphrase-multilingual-MiniLM-L12-v2`を使って小さな混合言語コーパスで言語横断リトリーバーを構築する。英語でクエリを行い、任意の言語でドキュメントを検索する。recall@5を測定する。
3. **難。** ヒンディー語分類タスクに対して英語ソースとヒンディー語ソースの微調整を比較する。両方の体制で500のターゲット言語サンプルでフューショット微調整を使用する。どちらのソースがより良いヒンディー語精度を出すか、その差はいくらかを報告する。これはLANGRANKの論文の主張をミニ版で再現するものだ。

## キーワード

| 用語 | 一般的な説明 | 実際の意味 |
|------|------------|----------|
| 多言語モデル | 一つのモデル、多くの言語 | 言語間で語彙とパラメータを共有する。 |
| 言語横断転移 | ある言語で学習して別の言語で実行 | ソースで微調整し、ターゲット言語ラベルなしでターゲットで評価する。 |
| ゼロショット | ターゲット言語ラベルなし | ターゲット言語での微調整なしに転移する。 |
| フューショット | 少量のターゲットラベル | 微調整に使用される100〜500のターゲット言語サンプル。 |
| mBERT | 初の多言語LM | Wikipediaで事前学習された104言語のBERT。 |
| XLM-R | 標準的な言語横断ベースライン | CommonCrawlで事前学習された100言語のRoBERTa。 |
| NLLB | Metaの200言語MT | No Language Left Behind。55の低リソース言語を含む。 |

## 参考資料

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-Rの論文。
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — 言語横断転移研究の系譜を始めた分析論文。
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200の論文。
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — CohereのマルチリンガルLLM Aya。
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — qWALS / LANGRANK ソース言語の論文。
