# 感情分析

> NLPの典型的なタスク。古典的なテキスト分類について知るべきことのほとんどがここに現れる。


## 問題の背景

「The food was not great.」ポジティブかネガティブか？

感情分析は単純そうに見える。レビュアーが何かを好きかどうかを言った。文章にラベルを付けよ。これが典型的なNLPタスクになった理由は、簡単そうに見えるケースほど難しいケースを隠しているからだ。否定は意味を反転させる。皮肉はそれをひっくり返す。「Not bad at all」は2つの否定語を持つにもかかわらずポジティブだ。絵文字は周囲のテキストより多くのシグナルを持つ。ドメイン語彙が重要だ（音楽レビューでの`tight`とファッションレビューでの`tight`）。

感情分析は古典的なNLPの実験室だ。各ナイーブなベースラインが特定の失敗パターンを持つ理由を理解すれば、各より豊かなモデルがなぜ発明されたかを理解できる。このレッスンでは、ナイーブベイズのベースラインをゼロから構築し、ロジスティック回帰を追加し、本番感情分析をコンプライアンスグレードの問題にする罠を名指しする。

## 概念

古典的な感情分析は2ステップのレシピだ。

1. **表現。** テキストを特徴ベクトルに変換する。BoW、TF-IDF、またはn-gram。
2. **分類。** ラベル付き例で線形モデル（ナイーブベイズ、ロジスティック回帰、SVM）を適合させる。

ナイーブベイズは機能する最も単純なモデルだ。ラベルが与えられた場合、すべての特徴が独立していると仮定する。カウントから`P(word | positive)`と`P(word | negative)`を推定する。推論時に確率を掛け合わせる。「ナイーブ」な独立仮定は明らかに間違っているのに、結果は驚くほど強力だ。理由は: スパースなテキスト特徴と中程度のデータがあれば、分類器は各単語がどちらに傾くかを「どれだけ」というよりも気にするからだ。

ロジスティック回帰は独立仮定を修正する。特徴ごとに重みを学習し、負の重みも含む。`not good`というバイグラム特徴は負の重みを得る。ナイーブベイズはラベル付けしたことのないバイグラムではそれができない。

## 実装する

### ステップ1: 実際のミニデータセット

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

意図的に小さくしている。実際の業務では数万の例を使う（IMDb、SST-2、Yelp polarity）。数学は同じだ。

### ステップ2: 多項ナイーブベイズをゼロから

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

加算スムージング（alpha=1.0）はLaplaceスムージングだ。これがなければ、クラスに現れたことのない単語の確率がゼロになり、対数が爆発する。`alpha=0.01`は実際によく使われる。`alpha=1.0`は教育用のデフォルトだ。

### ステップ3: ロジスティック回帰をゼロから

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2正則化がここで重要だ。テキスト特徴はスパース。L2なしではモデルが訓練例を暗記する。`0.01`から始めてチューニングする。

### ステップ4: 否定の処理（失敗パターン）

「not good」と「not bad」を考えてみよう。BoW分類器は`{not, good}`と`{not, bad}`を見て、訓練でどちらがより多く現れたかから学習する。バイグラム分類器は`not_good`と`not_bad`を別の特徴として見て、それらを区別して学習する。それで十分なことが多い。

バイグラムがない場合に機能する粗削りな修正: **否定スコープ化**。否定語に続くトークンを次の句読点まで`NOT_`プレフィックスを付ける。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

`good`と`NOT_good`が別の特徴になった。分類器は逆に重み付けできる。3行の前処理で、感情分析ベンチマークで測定可能な精度の向上が得られる。

### ステップ5: 重要な評価指標

クラスが不均衡な場合、正確度だけでは誤解を招く。実際の感情コーパスは通常70〜80%がポジティブまたはネガティブで、多数派定数分類器は80%の正確度を得るが役に立たない。以下のすべてを報告する：

- **クラスごとの適合率と再現率。** クラスごとに1ペア。クラスバランスを尊重する単一の数値を得るためにマクロ平均を取る。
- **マクロF1（不均衡データの主要指標）。** クラスごとのF1スコアの平均、等しく重み付け。クラスが不均衡な場合は正確度の代わりにこれを使う。
- **重み付きF1（代替案）。** マクロと同じだがクラス頻度で重み付け。不均衡自体がビジネス的意味を持つ場合はマクロF1と一緒に報告する。
- **混同行列。** 生のカウント。スカラー指標を信頼する前に必ず確認する。どのクラスペアをモデルが混同するかを明らかにする。
- **クラスごとのエラーサンプル。** クラスごとに5つの誤った予測を引き出す。読む。実際のエラーを読むことに代わるものはない。

深刻に不均衡なデータ（95対5以上の比率）では、正確度の代わりに**AUROC**と**AUPRC**を報告する。AUPRCは少数クラスにより敏感で、通常気にするのは少数クラスだ（スパム、詐欺、稀な感情）。

**避けるべき一般的なバグ。** 不均衡データでマイクロF1の代わりにマクロF1を報告すると、多数クラスに支配された高い数値が得られる。マクロF1は少数クラスのパフォーマンスを強制的に見せる。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 使ってみる

scikit-learnは6行で正しく処理する。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

注目すべき3点。`stop_words=None`は否定語を保持する。`ngram_range=(1, 2)`はバイグラムを追加するので`not_good`が特徴になる。`sublinear_tf=True`は繰り返し単語を抑制する。この3つのフラグがSST-2で75%から85%の精度ベースラインの差を生む。

### Transformerを使うべき場面

- 皮肉検出。古典的なモデルはここで失敗する。確実に。
- 文書の中盤で感情が変わる長いレビュー。
- アスペクトベースの感情分析。「カメラは良かったがバッテリーはひどかった。」アスペクトに感情を帰属させる必要がある。TransformerまたはStructured Outputモデルのみ。
- 低リソースの非英語言語。多言語BERTが無料でゼロショットベースラインを提供する。

上記が必要な場合は、フェーズ7（Transformer詳細）に進む。そうでなければ、TF-IDF+バイグラム+否定処理に対するナイーブベイズまたはロジスティック回帰が2026年の本番ベースラインだ。

### 再現性の罠（再び）

感情モデルの再訓練は日常的だ。再評価は日常的ではない。論文で報告される正確度の数値は特定のスプリット、特定の前処理、特定のトークナイザーを使用している。同一のパイプラインを使わずに新しいモデルをベースラインと比較すると、誤解を招くデルタが得られる。常に論文の数値ではなく自分のパイプラインでベースラインを再生成する。

## 成果物を出す

`outputs/prompt-sentiment-baseline.md`として保存する：

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## 演習

1. **易しい。** `apply_negation`をscikit-learnパイプラインの前処理ステップとして追加し、小さな感情データセットでF1の変化を測定する。
2. **普通。** クラス重み付きロジスティック回帰を実装する（scikit-learnに`class_weight="balanced"`を渡すか、勾配を自分で導出する）。合成した90対10のクラス不均衡での効果を測定する。
3. **難しい。** 感情モデルの残差で皮肉検出器を構築する。実験設定を文書化する。精度が偶然以下の場合（2クラス皮肉の偶然レベルは約50%で、最初の試みの多くはそこに落ちる）読者に警告する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------------------|
| 極性 | ポジティブまたはネガティブ | バイナリラベル。ニュートラルや細粒度（5段階）に拡張されることもある。 |
| アスペクトベース感情分析 | アスペクトごとの極性 | テキストで言及された特定のエンティティや属性に感情を帰属させる。 |
| 否定スコープ化 | 近隣トークンを反転させる | 「not」の後のトークンに句読点まで`NOT_`プレフィックスを付ける。 |
| Laplaceスムージング | カウントに1を加える | ナイーブベイズでゼロ確率特徴を防ぐ。 |
| L2正則化 | 重みを縮小する | 損失に`lambda * sum(w^2)`を加える。スパーステキスト特徴に必須。 |

## 参考資料

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — 基礎となるサーベイ。長いが、最初の4セクションで古典的なものをすべてカバー。
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — バイグラム+ナイーブベイズが短いテキストで難しく対抗できることを示した論文。
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — `CountVectorizer`、`TfidfVectorizer`、チューニングするすべてのパラメータのリファレンス。
