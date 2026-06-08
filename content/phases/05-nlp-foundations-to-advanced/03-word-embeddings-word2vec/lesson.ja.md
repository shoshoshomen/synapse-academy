# 単語埋め込み — Word2Vecをゼロから

> 単語はその仲間によって知られる。その考えに基づいて浅いネットワークを訓練すると、幾何学が自然と生まれる。


## 問題の背景

TF-IDFは`dog`と`puppy`が異なる単語であることを知っている。しかし、ほぼ同じ意味であることは知らない。`dog`で訓練した分類器は`puppy`に関するレビューに汎化できない。類義語リストを作ることでこれをカバーできるが、稀な用語、ドメインの専門用語、そして予期しないすべての言語では機能しない。

`dog`と`puppy`が空間内で近くに配置される表現が欲しい。`king - man + woman`が`queen`の近くに落ちるような表現が。`dog`で訓練したモデルが`puppy`にシグナルの一部を自動的に転送できるような表現が。

Word2Vecがその空間を与えてくれた。2層のニューラルネットワーク、1兆トークンの訓練、2013年に発表。アーキテクチャはほとんど恥ずかしいほどシンプルだ。結果はNLPを10年間変えた。

## 概念

**分布仮説**（Firth、1957）：「単語はその仲間によって知られる。」2つの単語が似た文脈に現れるなら、おそらく似た意味を持つ。

Word2Vecはその考えを利用した2つのバリアントで提供される。

- **Skip-gram。** 中心語が与えられたら、周辺語を予測する。ウィンドウサイズ2で`cat -> (the, sat, on)`。
- **CBOW（continuous bag of words）。** 周辺語が与えられたら、中心語を予測する。`(the, sat, on) -> cat`。

Skip-gramは訓練が遅いが稀な単語の処理に優れている。デフォルトになった。

ネットワークには非線形性のない1つの隠れ層がある。入力は語彙に対するone-hotベクトル。出力は語彙に対するsoftmax。訓練後、出力層を捨てる。隠れ層の重みが埋め込みだ。

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          これが埋め込み
```

トリック: 10万語のsoftmaxは計算コストが高すぎる。Word2Vecは**ネガティブサンプリング**を使って、バイナリ分類タスクに変換する。「このコンテキスト単語はこの中心語の近くに現れたか、現れなかったか」を予測する。全語彙でsoftmaxを計算する代わりに、訓練ペアごとにいくつかのネガティブ（共起しない）単語をサンプリングする。

## 実装する

### ステップ1: コーパスから訓練ペアを生成

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

ウィンドウ内の（中心語、コンテキスト語）ペアがすべてポジティブな訓練例になる。

### ステップ2: 埋め込みテーブル

2つの行列。`W`は中心語埋め込みテーブル（保持するもの）。`W'`はコンテキスト語テーブル（よく捨てられるが、`W`と平均される場合もある）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

小さなランダム初期化。語彙サイズ1万、次元数100が現実的。教育用には50語彙×16次元で幾何学を確認するのに十分だ。

### ステップ3: ネガティブサンプリング目標

各ポジティブペア`(center, context)`について、語彙から`k`個のランダムな単語をネガティブとしてサンプリングする。内積`W[center] · W'[context]`がポジティブでは高く、ネガティブでは低くなるようにモデルを訓練する。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

魔法の式: ポジティブペアへのロジスティックロス（sigmoidを1に近づける）＋ネガティブペアへのロジスティックロス（sigmoidを0に近づける）。両方のテーブルに勾配が流れる。完全な導出はオリジナルの論文にある。紙とペンで一度追うと頭に残る。

### ステップ4: おもちゃのコーパスで訓練

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

大規模なコーパスで十分なエポック数を経た後、似たコンテキストを共有する単語は似た中心埋め込みを持つ。おもちゃのコーパスではその効果がかすかに見える。数十億のトークンでは劇的に見える。

### ステップ5: アナロジートリック

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

事前訓練済みの300次元Google Newsベクトルでは：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。モデルが王族とは何かを知っているからではない。ベクトル`(king - man)`が「王室」のようなものを捉えており、`woman`に加えると王室女性の領域近くに落ちるからだ。

## 使ってみる

Word2Vecをゼロから書くのは教育だ。本番NLPは`gensim`を使う。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

実際の業務では、Word2Vecを自分で訓練することはほとんどない。事前訓練済みベクトルをダウンロードする。

- **GloVe** — Stanfordの共起行列因子分解アプローチ。50次元、100次元、200次元、300次元のチェックポイント。優れた汎用カバレッジ。レッスン04でGloVeを扱う。
- **FastText** — FacebookのWord2Vec拡張でcharacter n-gramを埋め込む。サブワードを組み合わせることで未知語も処理できる。レッスン04。
- **Google News事前訓練済みWord2Vec** — 300次元、300万語の語彙、2013年に公開。今でも毎日ダウンロードされている。

### 2026年でもWord2Vecが勝つ場面

- 軽量ドメイン固有検索。医療論文要旨でラップトップ1時間で訓練し、一般モデルが捉えられない専門ベクトルを得る。
- アナロジースタイルの特徴量エンジニアリング。`gender_vector = mean(man - woman pairs)`。他の単語からそれを引いてジェンダーニュートラルな軸を得る。公正性研究でまだ使われている。
- 解釈可能性。100次元はPCAやt-SNEでプロットしてクラスタが形成されるのを実際に見るのに十分小さい。
- GPUなしでオンデバイス推論が必要な場所。Word2Vecのルックアップは単一行の取得だ。

### Word2Vecが失敗する場面

多義語の壁。`bank`は1つのベクトルを持つ。「river bank」と「financial bank」がそれを共有する。`table`（スプレッドシートvs家具）も共有する。下流の分類器はベクトルから意味を区別できない。

文脈的埋め込み（ELMo、BERT、以降のすべてのTransformer）は、周囲のコンテキストに基づいて単語の出現ごとに異なるベクトルを生成することでこれを解決した。Word2VecからBERTへのジャンプは静的から文脈的への飛躍だ。フェーズ7でTransformerの部分を扱う。

未知語問題がもう一つの失敗だ。Word2Vecが訓練データに`Zoomer-approved`を見ていなければ、そのベクトルを生成できない。FastTextはサブワード合成でこれを修正する（レッスン04）。

## 成果物を出す

`outputs/skill-embedding-probe.md`として保存する：

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## 演習

1. **易しい。** 猫と犬に関する小さなコーパス（20文）で訓練ループを実行する。200エポック後、`nearest(vocab, W, W[vocab["cat"]])`がトップ3に`dog`を返すことを確認する。そうでなければ、エポック数を増やすか語彙を増やす。
2. **普通。** 頻出単語のサブサンプリングを追加する。頻度が`10^-5`を超える単語は、頻度に比例した確率で訓練ペアから除外される。稀な単語の類似性への影響を測定する。
3. **難しい。** 20 Newsgroupsコーパスでモデルを訓練する。`he - she`と`doctor - nurse`の2つのバイアス軸を計算する。職業語を両軸に投影する。どの職業が最大のバイアスギャップを持つかを報告する。これは公正性研究者が使う種類のプローブだ。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------------------|
| 単語埋め込み | 単語をベクトルとして | コンテキストから学習した密で低次元（通常100〜300）の表現。 |
| Skip-gram | Word2Vecのトリック | 中心語からコンテキスト語を予測する。CBOWより遅いが稀な単語に優れる。 |
| ネガティブサンプリング | 訓練のショートカット | 全語彙のsoftmaxを`k`個のランダムな単語に対するバイナリ分類に置き換える。 |
| 静的埋め込み | 単語ごとに1つのベクトル | コンテキストに関係なく同じベクトル。多義語で失敗する。 |
| 文脈的埋め込み | コンテキスト依存ベクトル | 周囲の単語に基づいて出現ごとに異なるベクトル。Transformerが生成するもの。 |
| OOV | 未知語 | 訓練中に見なかった単語。Word2Vecはこれらのベクトルを生成できない。 |

## 参考資料

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — ネガティブサンプリングの論文。短くて読みやすい。
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — オリジナル論文の数学が難しければ、最も明快な勾配の導出。
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — 実際に機能する本番訓練設定。
