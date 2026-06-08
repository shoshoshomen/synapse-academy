# Bag of Words、TF-IDF、テキスト表現

> まず数えよ、後で考えよ。TF-IDFは2026年においても明確なタスクで埋め込みを上回る。


## 問題の背景

モデルには数値が必要だ。手元にあるのは文字列だ。

すべてのNLPパイプラインは同じ問いに答えなければならない。可変長のトークンストリームを、分類器が消費できる固定サイズのベクトルにどう変換するか。フィールドが最初に辿り着いた答えは、機能する最も単純なものだった。単語を数えて、ベクトルを作れ。

そのベクトルは、どんな埋め込みモデルよりも多くの本番NLPを支えてきた。スパムフィルタ、トピック分類器、ログ異常検出、検索ランキング（BM25以前）、感情分析の第一波、NLP学術ベンチマークの最初の10年。2026年の実務者も、狭い分類タスクでは今でもこれを最初に使う。高速で、解釈可能で、単語の存在が重要なタスクでは4億パラメータの埋め込みモデルと区別がつかないことが多い。

このレッスンでは、Bag of WordsとTF-IDFをゼロから構築する。それからscikit-learnが同じことを3行でやる方法を示す。そして、埋め込みに切り替えるべき失敗パターンを名指しする。

## 概念

**Bag of Words (BoW)** は順序を捨てる。各文書について、語彙中の各単語が何回登場するかを数える。ベクトルの長さは語彙サイズだ。位置`i`は単語`i`の出現回数だ。

**TF-IDF** はBoWを再重み付けする。すべての文書に現れる単語は情報を持たないので、スコアを下げる。コーパス全体では稀だが特定の文書に頻繁に現れる単語はシグナルなので、スコアを上げる。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

`TF`は文書内の単語頻度、`df`は文書頻度（その単語を含む文書数）、`N`は文書総数。`log`は頻出単語の重みを制限する。

重要な特性: どちらも解釈可能な軸を持つスパースベクトルを生成する。訓練済み分類器の重みを見れば、どの単語が文書を各クラスに向けているかを読み取れる。768次元のBERT埋め込みではこれができない。

## 実装する

### ステップ1: 語彙の構築

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

入力: トークン化された文書のリスト（単語レベルのトークナイザーなら何でもよい。このレッスンの`code/main.py`では簡略化された小文字バリアントを使用）。出力: `{単語: インデックス}`辞書。挿入順序が安定しているので、単語インデックス0は最初の文書で最初に見た単語だ。慣例は様々で、scikit-learnはアルファベット順に並べる。

### ステップ2: Bag of Words

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行は文書、列は語彙インデックス。エントリ`[i][j]`は「文書`i`に単語`j`が何回現れるか」だ。文書1には`cat`が2回あるのは、実際にそうだからだ。文書0には`ran`が0回なのは、含まれていないからだ。

### ステップ3: 単語頻度と文書頻度

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

名前を付けておくべきスムージングトリック2つ。`(n+1)/(d+1)`は`log(x/0)`を避ける。末尾の`+1`は、すべての文書に現れる単語のIDFが0ではなく1になることを保証し、scikit-learnのデフォルトに合わせる。生の`log(N/df)`を使う実装もある。どちらも機能するが、スムージング版の方が扱いやすい。

### ステップ4: TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

3つの文書、5語の語彙（`the`、`cat`、`sat`、`dog`、`ran`）。`the`は3つ全部に現れるのでIDFが低い。`dog`は1つにしか現れないのでIDFが高い。ベクトルはスパースで（ほとんどのエントリは小さい）、識別力のある単語が際立つ。

### ステップ5: L2正規化

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

正規化なしでは、長い文書の方が大きなベクトルを持ち、類似スコアを支配する。L2正規化はすべての文書を単位超球上に配置する。行間のコサイン類似度は単なる内積になる。

## 使ってみる

scikit-learnが本番版を提供している。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer`はトークン化、語彙構築、BoWを1回の呼び出しで行う。`TfidfVectorizer`はIDF重み付けとL2正規化を追加する。どちらもスパース行列を返す。10万文書の場合、密なバージョンはメモリに収まらない。分類器が密なものを要求するまでスパースのままにする。

結果を大きく変えるパラメータ:

| パラメータ | 効果 |
|-----|--------|
| `ngram_range=(1, 2)` | バイグラムを含める。通常は分類を向上させる。 |
| `min_df=2` | 2つ未満の文書に現れる単語を除外する。ノイズの多いデータで語彙を削減する。 |
| `max_df=0.95` | 95%以上の文書に現れる単語を除外する。ハードコードされたリストなしでストップワード除去を近似する。 |
| `stop_words="english"` | scikit-learnの組み込みストップワードリスト。タスク依存 — 感情分析では否定語を除外すべきでない。 |
| `sublinear_tf=True` | 生の`tf`の代わりに`1 + log(tf)`を使う。1つの文書で単語が何度も繰り返す場合に有効。 |

### TF-IDFがまだ勝つ場面（2026年時点）

- スパム検出、トピックラベリング、ログ異常フラグ。単語の存在が重要で、意味的なニュアンスは不要。
- 少ないデータの場合（ラベル付き例が数百件）。TF-IDF+ロジスティック回帰には事前訓練コストがない。
- レイテンシが重要な場所。TF-IDF+線形モデルはマイクロ秒単位で応答する。Transformerで文書を埋め込むには10〜100msかかる。
- 予測を説明しなければならないシステム。分類器の係数を調べれば、どの単語が理由かを読み取れる。

### TF-IDFが失敗する場面

意味的盲目性の失敗。この2つの文書を考えてみよう：

- 「The movie was not good at all.」
- 「The movie was excellent.」

一方は否定的なレビュー、もう一方は肯定的だ。TF-IDFの重複は正確に`{the, movie, was}`だ。Bag of Wordsの分類器は、`not`が近くにある`good`がラベルを反転させることを暗記しなければならない。十分なデータがあれば学習できるが、構文を理解するモデルほど優雅にはできない。

もう一つの失敗: 推論時の未知語。IMDbレビューで訓練されたBoWモデルは、`Zoomer-approved`というトークンが訓練に現れなかった場合、どうすればいいかわからない。サブワード埋め込み（レッスン04）はこれを処理する。TF-IDFはできない。

### ハイブリッド: TF-IDF重み付き埋め込み

中程度のデータ分類のための2026年の実用的なデフォルト: 単語埋め込みに対するアテンションとしてTF-IDF重みを使う。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

埋め込みから意味的能力を得つつ、TF-IDFから稀な単語の強調を得る。分類器はプールされたベクトルで訓練する。これは、約5万件のラベル付き例以下での感情、トピック、意図分類で、どちら単独よりも優れている。

## 成果物を出す

`outputs/prompt-vectorization-picker.md`として保存する：

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## 演習

1. **易しい。** L2正規化されたTF-IDF出力に対して`cosine_similarity(doc_vec_a, doc_vec_b)`を実装する。同一文書のスコアが1.0、語彙が重ならない文書のスコアが0.0になることを確認する。
2. **普通。** `bag_of_words`にn-gramサポートを追加する。パラメータ`n`でn-gramの出現回数を生成する。`n=2`の`["the", "cat", "sat"]`が`["the cat", "cat sat"]`のバイグラム出現回数を生成することをテストする。
3. **難しい。** GloVe 100dベクトルを使って上記のTF-IDF重み付き埋め込みハイブリッドを構築する（一度ダウンロードしてキャッシュ）。20 Newsgroupsデータセットで、純粋なTF-IDFと純粋な平均プール埋め込みに対して分類精度を比較する。どちらがどこで勝つかを報告する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------------------|
| BoW | 単語頻度ベクトル | 1つの文書内の語彙単語の出現回数。順序を捨てる。 |
| TF | 単語頻度 | 文書内の単語の出現回数。文書長で正規化する場合もある。 |
| DF | 文書頻度 | その単語を少なくとも1回含む文書の数。 |
| IDF | 逆文書頻度 | `log(N / df)`をスムージング。どこにでも現れる単語の重みを下げる。 |
| スパースベクトル | ほとんど0 | 語彙は通常1万〜10万語で、ほとんどが特定の文書には存在しない。 |
| コサイン類似度 | ベクトルの角度 | L2正規化ベクトルの内積。1は同一、0は直交。 |

## 参考資料

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 標準APIリファレンス、すべてのパラメータに関する注記付き。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — TF-IDFを10年間のデフォルトにした論文。
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 古い手法がいつ、なぜ勝つかについての2026年の見解。
