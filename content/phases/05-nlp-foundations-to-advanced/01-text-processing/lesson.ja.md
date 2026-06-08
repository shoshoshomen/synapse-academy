# テキスト処理 — トークン化、ステミング、レンマ化

> 言語は連続している。モデルは離散的だ。前処理はその橋渡しである。


## 問題の背景

モデルは「The cats were running.」を読めない。読むのは整数値だ。

すべてのNLPシステムは同じ3つの問いから始まる。単語はどこで始まるのか。単語の語根は何か。「run」「running」「ran」を同じものとして扱うべき場合と、別々のものとして扱うべき場合をどう判断するか。

トークン化を誤ると、モデルはゴミデータから学習することになる。トークナイザーが`don't`を1トークンとして扱い、`do n't`を2トークンとして扱うなら、訓練データの分布が分裂する。ステマーが`organization`と`organ`を同じ語幹に落とすなら、トピックモデリングが機能しなくなる。レンマタイザーが品詞のコンテキストを必要とするのに渡さなければ、動詞が名詞として扱われる。

このレッスンでは3つの前処理ステップをゼロから構築し、NLTKとspaCyが同じ処理をどう行うかを示す。トレードオフを理解するためだ。

## 概念

3つの処理。それぞれに役割と失敗パターンがある。

**トークン化** は文字列をトークンに分割する。「トークン」という言葉が意図的に曖昧なのは、適切な粒度がタスクによって異なるからだ。古典的なNLPでは単語レベル。Transformerではサブワード。空白のない言語では文字レベル。

**ステミング** はルールに従って接尾辞を切り落とす。高速で積極的だが粗雑だ。`running -> run`。`organization -> organ`。後者が失敗パターンだ。

**レンマ化** は文法的知識を使って単語を辞書形に戻す。遅いが正確で、ルックアップテーブルか形態素解析器が必要だ。`ran -> run`（「ran」が「run」の過去形だと知る必要がある）。`better -> good`（比較形を知る必要がある）。

経験則として、速度が重要でノイズを許容できる場合はステミング（検索インデックス、大まかな分類）。意味が重要な場合はレンマ化（質問応答、セマンティック検索、ユーザーが読むもの）。

## 実装する

### ステップ1: 正規表現ワードトークナイザー

英数字以外の文字で分割しながら句読点を独立したトークンとして保持する、最もシンプルで実用的なトークナイザー。完璧でも最終版でもないが、1行で動く。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

優先順位順の3つのパターン。内側のアポストロフィを持つ可能性のある単語（`don't`、`it's`）。純粋な数値。独立したトークンとして扱う空白以外の非英数字（句読点）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

注意すべき失敗パターン。`3pm`が`['3', 'pm']`に分割される。文字の連続と数字の連続を交互に認識するためだ。ほとんどのタスクには十分だ。URL、メールアドレス、ハッシュタグはすべて壊れる。本番環境では、汎用パターンの前にそれらのパターンを追加する。

### ステップ2: Porterステマー（ステップ1aのみ）

完全なPorterアルゴリズムには5フェーズのルールがある。ステップ1aだけで英語の最も頻出する接尾辞を処理でき、パターンを学べる。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

ルールを上から読む。`ies -> i`のルールにより`ponies -> poni`となる（`pony`ではない）。実際のPorterにはステップ1bがあり、これを修正する。ルールは競合する。先のルールが優先する。順序は個々のルールよりも重要だ。

### ステップ3: ルックアップベースのレンマタイザー

レンマ化には形態論が必要だ。小さなレンマテーブルとフォールバックを使う教育用の実装は実現可能だ。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最後のケースが重要な学習ポイントだ。`watched`はテーブルになく、フォールバックは`ing`しか処理しない。実際のレンマ化では`ed`、不規則動詞、比較級形容詞、音変化を伴う複数形（`children -> child`）を処理する。だから本番システムはWordNet、spaCyの形態素解析器、または完全な形態素解析器を使う。

### ステップ4: それらをパイプラインにつなぐ

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

足りないのは品詞タガーだ。フェーズ5·07（品詞タグ付け）で構築する。今は全トークンをデフォルトで`NOUN`とし、その制限を認識する。

## 使ってみる

NLTKとspaCyが本番用のバージョンを提供している。数行で使える。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize`は縮約形、Unicode、正規表現が見落とすエッジケースを処理する。`PorterStemmer`は5フェーズすべてを実行する。`WordNetLemmatizer`はNLTKのPenn TreebankスキームからWordNetの略称セットに品詞タグを変換する必要がある。上の変換コードがほとんどのチュートリアルで省略される部分だ。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCyは`nlp(text)`の裏でパイプライン全体を隠蔽する。トークン化、品詞タグ付け、レンマ化がすべて実行される。大規模処理ではNLTKより高速で、すぐに使える精度が高い。トレードオフは、個々のコンポーネントを簡単に入れ替えられないことだ。

### どちらを選ぶか

| 状況 | 選択肢 |
|-----------|------|
| 教育目的、研究、コンポーネントの入れ替え | NLTK |
| 本番環境、多言語、速度重視 | spaCy |
| Transformerパイプライン（どうせモデルのトークナイザーでトークン化する） | `tokenizers` / `transformers` を使い、古典的前処理はスキップ |

### 誰も警告しない2つの失敗パターン

ほとんどのチュートリアルはアルゴリズムを教えて終わる。実際の前処理パイプラインで痛い目を見る2つのことがあるが、ほとんど取り上げられない。

**再現性のドリフト。** NLTKとspaCyはバージョン間でトークン化とレンマタイザーの挙動が変わる。spaCy 2.xで`['do', "n't"]`を生成したものが、3.xでは`["don't"]`を生成するかもしれない。モデルはある分布で訓練された。推論は別の分布で実行される。精度は静かに低下し、誰も理由がわからない。`requirements.txt`でライブラリバージョンを固定する。20のサンプル文の期待されるトークン化を固定する前処理のリグレッションテストを書く。すべてのアップグレード時に実行する。

**訓練/推論のミスマッチ。** アグレッシブな前処理（小文字化、ストップワード除去、ステミング）で訓練し、生のユーザー入力で推論し、パフォーマンスが崩壊するのを見る。これは本番NLPで最も一般的な失敗パターンだ。訓練中に前処理するなら、推論中も同一の関数を実行しなければならない。前処理をノートブックのセルとしてではなく、モデルパッケージ内の関数としてデプロイする。

## 成果物を出す

前処理戦略を選ぶのに3冊の教科書を読まなくて済むよう、エンジニアを助ける再利用可能なプロンプト。

`outputs/prompt-preprocessing-advisor.md`として保存する：

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## 演習

1. **易しい。** `tokenize`を拡張してURLを単一トークンとして保持するようにする。テスト: `tokenize("Visit https://example.com today.")`はURLトークンを1つ生成すべきだ。
2. **普通。** Porterのステップ1bを実装する。単語に母音が含まれ、`ed`または`ing`で終わる場合、それを除去する。重子音ルールを処理する（`hopping -> hop`、`hopp`ではない）。
3. **難しい。** WordNetをルックアップテーブルとして使い、WordNetに項目がない場合はPorterステマーにフォールバックするレンマタイザーを構築する。タグ付きコーパスで、純粋なWordNetと純粋なPorterに対する精度を測定する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------------------|
| トークン | 単語 | モデルが消費する単位。単語、サブワード、文字、バイトのいずれかになりうる。 |
| 語幹 | 単語の語根 | ルールベースの接尾辞切り落としの結果。必ずしも実在する単語ではない。 |
| レンマ | 辞書形 | 辞書で調べる形。正確に計算するには文法的コンテキストが必要。 |
| 品詞タグ | 品詞 | NOUN、VERB、ADJのようなカテゴリ。正確にレンマ化するために必要。 |
| 形態論 | 単語の形のルール | 時制、数、格に基づいて単語がどのように変形するか。レンマ化はこれに依存する。 |

## 参考資料

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — オリジナル論文、5ページ、今でも最も明快な説明。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 実際のパイプラインがどのように配線されているか。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — まだ考えていないトークン化のエッジケース。
