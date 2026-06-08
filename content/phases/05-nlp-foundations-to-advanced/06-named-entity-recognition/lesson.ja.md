# 固有表現認識

> 名前を抽出する。曖昧な境界、ネストエンティティ、ドメイン専門用語に対処するまで簡単そうに見える。


## 問題の背景

「Apple sued Google over its iPhone search deal in the US.」エンティティが5つある：Apple（ORG）、Google（ORG）、iPhone（PRODUCT）、search deal（もしかしたら）、US（GPE）。良いNERシステムは正しいタイプですべてを抽出する。悪いものはiPhoneを見逃し、フルーツのAppleと会社のAppleを混同し、「US」をPERSONとラベル付けする。

NERはあらゆる構造化抽出パイプラインの下で動く主力だ。履歴書解析、コンプライアンスログスキャン、医療記録の匿名化、検索クエリ理解、チャットボット応答のグラウンディング、法律契約抽出。目に見えないが、常に依存している。

このレッスンでは古典的な経路（ルールベース、HMM、CRF）から現代のもの（BiLSTM-CRF、次にTransformer）へと歩む。各ステップは前のものの特定の限界を解決する。そのパターンがレッスンだ。

## 概念

**BIOタグ付け**（またはBILOU）は、エンティティ抽出をシーケンスラベリング問題に変換する。各トークンに`B-TYPE`（エンティティの始まり）、`I-TYPE`（エンティティの内部）、または`O`（どのエンティティの外も）でラベルを付ける。

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

複数トークンのエンティティは連鎖する：`New B-GPE`、`York I-GPE`、`City I-GPE`。BIOを理解するモデルは任意のスパンを抽出できる。

アーキテクチャの進化：

- **ルールベース。** 正規表現+ガゼッター（地名辞典）ルックアップ。既知のエンティティに対しては高い適合率、新しいものにはゼロのカバレッジ。
- **HMM。** 隠れマルコフモデル。タグが与えられたトークンの出力確率、タグ間の遷移確率。Viterbiデコード。ラベル付きデータで訓練。
- **CRF。** 条件付き確率場。HMMに似ているが識別的なので、任意の特徴（単語の形、大文字化、隣接単語）を組み合わせられる。2026年においても低リソース展開での古典的な本番主力。
- **BiLSTM-CRF。** 手作り特徴の代わりにニューラル特徴。LSTMが文章を両方向に読み、上にCRFレイヤーで一貫したタグシーケンスを強制する。
- **Transformerベース。** トークン分類ヘッドでBERTをファインチューニング。最高の精度。最も多くの計算。

## 実装する

### ステップ1: BIOタグ付けヘルパー

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### ステップ2: 手作り特徴

古典的（非ニューラル）NERでは、特徴が勝負だ。有用なもの：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")`は`xXxxxx`を返す。`word_shape("USA-2024")`は`XXX-dddd`を返す。大文字化パターンは固有名詞に対して高いシグナルだ。

### ステップ3: シンプルなルールベース+辞書ベースライン

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

本番のガゼッターにはWikipediaとDBpediaからスクレイピングした数百万のエントリがある。カバレッジは良好だ。曖昧さの解消（フルーツのAppleと会社のApple）はひどい。だから統計モデルが勝った。

### ステップ4: CRFステップ（スケッチ、完全な実装ではない）

50行でゼロからCRFを実装しても、確率論の基礎なしには理解しにくい。代わりに`sklearn-crfsuite`を使う：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1`と`c2`はL1とL2正則化だ。`all_possible_transitions=True`はモデルに非合法シーケンス（例：`O`の後の`I-ORG`）が起こりにくいことを学習させ、BIOの一貫性を制約として書かずにCRFが強制する仕組みだ。

### ステップ5: BiLSTM-CRFが追加するもの

特徴が学習される。入力: トークン埋め込み（GloVeまたはFastText）。LSTMが左から右、右から左に読む。連結された隠れ状態がCRF出力レイヤーを通る。CRFはまだタグシーケンスの一貫性を強制するが、LSTMは手作り特徴を学習済みのものに置き換える。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

CRFレイヤーには`torchcrf.CRF`を使う（pip install pytorch-crf）。手作りCRFに対するゲインは測定可能だが、数万のラベル付き文がない限り期待するほど大きくない。

## 使ってみる

spaCyはすぐに使える本番グレードのNERを提供する。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

`iPhone`が`PRODUCT`ではなく`ORG`とラベル付けされているのに注意 — spaCyの小モデルは製品エンティティのカバレッジが弱い。大きいモデル（`en_core_web_lg`）の方が優れている。Transformerモデル（`en_core_web_trf`）はさらに優れている。

BERTベースのNERにはHugging Face：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"`は連続するB-X、I-Xトークンをスパンに結合する。これがなければトークンレベルのラベルが得られ、自分で結合しなければならない。

### LLMベースNER（2026年のオプション）

ゼロショットおよびフューショットのLLM NERは、多くのドメインでファインチューニングされたモデルと競合するようになり、ラベル付きデータが少ない場合は劇的に優れている。

- **ゼロショットプロンプティング。** LLMにエンティティタイプのリストと例スキーマを与える。JSON出力を求める。すぐに動作するが、新規ドメインでの精度は中程度。
- **ZeroTuneBioスタイルのプロンプティング。** タスクを候補抽出→意味説明→判断→再確認に分解する。マルチステージプロンプト（ワンショットではない）は生物医学NERで精度を大幅に向上させる。同じパターンが法律、金融、科学ドメインで機能する。
- **RAGによる動的プロンプティング。** 推論呼び出しごとに小さな注釈付きシードセットから最も類似したラベル付き例を取得し、動的にフューショットプロンプトを構築する。2026年のベンチマークでは、静的プロンプティングに比べてGPT-4の生物医学NER F1が11〜12%向上する。
- **エンティティタイプごとの分解。** 長い文書では、すべてのエンティティタイプを一度に抽出する単一の呼び出しは、長さが増すにつれて再現率が下がる。エンティティタイプごとに1回の抽出パスを実行する。推論コストは高いが、精度は大幅に向上する。これが臨床メモや法律契約の標準パターンだ。

2026年時点での本番推奨: 訓練データを収集する前にLLMゼロショットベースラインから始める。F1が十分に良ければファインチューニングが不要なことも多い。

### 古典的NERがまだ勝つ場面

LLMが利用可能な場合でも、古典的NERは次の場合に勝つ：

- レイテンシ予算が50ms未満。
- 数千のラベル付き例があり、98%以上のF1が必要。
- ドメインに安定したオントロジーがあり、事前訓練済みCRFまたはBiLSTMが転用できる。
- 規制上の制約により、オンプレミスの非生成モデルが必要。

### 失敗する場所

- **ドメインシフト。** CoNLLで訓練されたNERは法律契約でガゼッターよりもパフォーマンスが低い。ドメインでファインチューニングする。
- **ネストエンティティ。** 「Bank of America Tower」は同時にORGとFACILITYだ。標準BIOは重複スパンを表現できない。ネストNER（マルチパスまたはスパンベースのモデル）が必要だ。
- **長いエンティティ。** 「United States Federal Deposit Insurance Corporation。」トークンレベルのモデルはこれを分割することがある。`aggregation_strategy`を使うか後処理する。
- **スパースタイプ。** DRUG_BRAND、ADVERSE_EVENT、DOSEのような医療NERラベル。汎用モデルは把握できない。ScispaCyとBioBERTがそこでの出発点だ。

## 成果物を出す

`outputs/skill-ner-picker.md`として保存する：

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## 演習

1. **易しい。** `bio_to_spans`（`spans_to_bio`の逆）を実装し、10文でラウンドトリップの一貫性を確認する。
2. **普通。** 上記のsklearn-crfsuite CRFをCoNLL-2003英語NERデータセットで訓練する。`seqeval`を使ってエンティティごとのF1を報告する。典型的な結果は約84 F1。
3. **難しい。** `distilbert-base-cased`をドメイン固有NERデータセット（医療、法律、または金融）でファインチューニングする。spaCy小モデルと比較する。データリーケージチェックを文書化し、驚いたことを報告する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------------------|
| NER | 名前を抽出する | トークンスパンにタイプ（PERSON、ORG、GPE、DATE、...）でラベルを付ける。 |
| BIO | タグ付けスキーム | `B-X`が始まり、`I-X`が続き、`O`が外。 |
| BILOU | より良いBIO | 明確な境界のために`L-X`（最後）、`U-X`（単位）を追加。 |
| CRF | 構造化分類器 | 出力だけでなくラベル間の遷移をモデル化する。有効なシーケンスを強制する。 |
| ネストNER | 重複するエンティティ | あるスパンがそのサブスパンとは異なるエンティティだ。BIOはこれを表現できない。 |
| エンティティレベルF1 | 適切なNER指標 | 予測されたスパンは真のスパンと正確に一致しなければならない。トークンレベルF1は精度を過大評価する。 |

## 参考資料

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF論文。正典。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — 標準となったトークン分類パターンを導入。
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — `Doc.ents`と`Span`のすべての属性の実用的なリファレンス。
- [seqeval](https://github.com/chakki-works/seqeval) — 正しい指標ライブラリ。常に使うこと。
