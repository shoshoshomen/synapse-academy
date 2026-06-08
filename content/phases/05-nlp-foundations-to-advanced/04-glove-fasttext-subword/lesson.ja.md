# GloVe、FastText、サブワード埋め込み

> Word2Vecは単語ごとに1つの埋め込みを訓練した。GloVeは共起行列を因数分解した。FastTextはパーツを埋め込んだ。BPEはTransformerへの橋渡しをした。


## 問題の背景

Word2Vecは2つの未解決の問いを残した。

第一に、Word2Vecの反復的なSkip-gram更新ではなく、共起行列を直接因数分解する（LSA、HAL）並行研究の流れがあった。Word2Vecの反復的なアプローチは根本的に優れているのか、それとも2つの方法が数値を扱う方法の違いによるアーティファクトに過ぎないのか？**GloVe**がこれに答えた: 注意深く設計された損失関数による行列因数分解はWord2Vecと同等かそれ以上の性能を発揮し、訓練コストも低い。

第二に、どちらの方法も未知語に対するアプローチがなかった。`Zoomer-approved`、`dogecoin`、先週作られた固有名詞、稀な語根の活用形すべてが問題だ。**FastText**はcharacter n-gramを埋め込むことでこれを修正した: 単語は形態素を含むその部分の総和なので、未知語でも適切なベクトルが得られる。

第三に、Transformerが登場すると問いはさらに変わった。単語レベルの語彙は約100万語が限界だが、実際の言語はそれより開かれている。**Byte-pair encoding（BPE）**とその派生手法は、あらゆるものをカバーする頻出サブワード単位の語彙を学習することでこれを解決した。現代のすべてのLLMのすべてのトークナイザーはサブワードトークナイザーだ。

このレッスンでは3つを順に説明し、それぞれをいつ使うべきかを解説する。

## 概念

**GloVe（Global Vectors）。** 単語-単語共起行列`X`（`X[i][j]`は単語`j`が単語`i`のコンテキストに現れる回数）を構築する。`v_i · v_j + b_i + b_j ≈ log(X[i][j])`となるようにベクトルを訓練する。頻出ペアが支配しないように損失を重み付けする。完了。

**FastText。** 単語はそのcharacter n-gramと単語自体の総和だ。`where`は`<wh, whe, her, ere, re>, <where>`になる。単語ベクトルはそれらの成分ベクトルの総和だ。Word2Vecと同様に訓練する。利点: 未知語（`whereupon`）は既知のn-gramから組み合わせられる。

**BPE（Byte-Pair Encoding）。** 個々のバイト（または文字）の語彙から始まる。コーパス内の隣接するすべてのペアを数える。最も頻繁なペアを新しいトークンに統合する。`k`回繰り返す。結果: 頻出列（`ing`、`tion`、`the`）が単一のトークンで、稀な単語は馴染みのある部分に分割された、`k + 256`トークンの語彙。どの文章も何かにトークン化される。

## 実装する

### GloVe: 共起行列を因数分解

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

名前を付けておくべき2つの動く部品。重み関数`f(x) = (x/x_max)^alpha`は非常に頻繁なペア（`(the, and)`のような）が損失を支配しないように重みを下げる。最終的な埋め込みは`W`（中心）と`W_tilde`（コンテキスト）テーブルの和だ。両方を合計するのは、どちらか一方を使うよりも優れている傾向にある公表済みのトリックだ。

### FastText: サブワードを考慮した埋め込み

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

各単語はそのn-gramのセット（通常3〜6文字）で表現される。単語埋め込みはそのn-gram埋め込みの総和だ。Skip-gram訓練では、Word2Vecが単一ベクトルを使う場所にこれを接続する。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

未知語でも、そのn-gramの一部が知られていればベクトルが得られる。`whereupon`は`where`と`<wh`、`her`、`ere`、`<where`を共有するので、2つは近くに落ちる。

### BPE: 学習されたサブワード語彙

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

最初のイテレーションで最も頻出する隣接ペアを統合する。十分なイテレーション後、頻出部分文字列（`low`、`est`、`tion`）が単一トークンになり、稀な単語はきれいに分割される。

実際のGPT/BERT/T5トークナイザーは3万〜10万の統合を学習する。結果: どんなテキストも既知IDの有限長シーケンスにトークン化され、OOVは決して起きない。

## 使ってみる

実際には、これらを自分で訓練することはほとんどない。事前訓練済みチェックポイントを読み込む。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

Transformer時代のBPEスタイルのサブワードトークン化では：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ`プレフィックスは単語の境界を示す（GPT-2の慣例）。現代のすべてのトークナイザーはBPEのバリアント、WordPiece（BERT）、またはSentencePiece（T5、LLaMA）だ。

### どれを選ぶか

| 状況 | 選択肢 |
|-----------|------|
| 事前訓練済みの汎用単語ベクトル、OOV耐性不要 | GloVe 300次元 |
| 事前訓練済みの汎用単語ベクトル、スペルミス/新語/形態的に豊かな言語を処理する必要 | FastText |
| Transformerに入力するもの（訓練または推論） | モデルに同梱されたトークナイザーを使う。絶対に交換しない。 |
| 独自の言語モデルをゼロから訓練する | まずコーパスでBPEまたはSentencePieceトークナイザーを訓練する |
| 線形モデルを使った本番テキスト分類 | まだTF-IDF。レッスン02。 |

## 成果物を出す

`outputs/skill-embeddings-picker.md`として保存する：

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## 演習

1. **易しい。** `char_ngrams("playing")`と`char_ngrams("played")`を実行する。2つのn-gramセットのJaccard重複を計算する。`pla`、`lay`、`play`などの共有部分が相当あるはずで、これがFastTextが形態的バリアント間で転用できる理由だ。
2. **普通。** `learn_bpe`を拡張して語彙の成長を追跡する。統合回数の関数として、コーパス文字あたりのトークン数をプロットする。最初は急速な圧縮が見られ、1トークン約2〜3文字前後で漸近するはずだ。
3. **難しい。** シェイクスピアの全作品で1000統合のBPEを訓練する。一般的な単語と稀な固有名詞のトークン化を比較する。統合前後の単語あたりの平均トークン数を測定する。驚いたことを報告する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------------------|
| 共起行列 | 単語-単語頻度テーブル | `X[i][j]` = 単語`j`が単語`i`周辺のウィンドウに現れる頻度。 |
| サブワード | 単語の一部 | Character n-gram（FastText）または学習トークン（BPE/WordPiece/SentencePiece）。 |
| BPE | Byte-pair encoding | 語彙が目標サイズに達するまで最頻出の隣接ペアを反復的に統合する。 |
| OOV | 未知語 | モデルが見たことのない単語。Word2Vec/GloVeは失敗する。FastTextとBPEは処理できる。 |
| バイトレベルBPE | 生バイトに対するBPE | GPT-2の方式。語彙は256バイトから始まるので、OOVは決して発生しない。 |

## 参考資料

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe論文、7ページ、今でも最高の損失の導出。
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText。
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPEを現代NLPに導入した論文。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — BPE、WordPiece、SentencePieceが実際にどう違うか。
