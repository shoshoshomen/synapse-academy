# トピックモデリング — LDAとBERTopic

> LDA: 文書はトピックの混合、トピックは単語の分布。BERTopic: 文書は埋め込み空間でクラスタリングされ、クラスタがトピック。ゴールは同じ、分解の仕方が違う。


## 問題設定

顧客サポートのチケットが1万件、ニュース記事が5万件、ツイートが20万件あるとする。ラベル付きカテゴリはない。カテゴリがいくつあるかすらわからない。そのコレクションが何について書かれているかを知る必要がある。

トピックモデリングは教師なしでその問いに答える。コーパスを与えると、まとまりのあるトピックの小さなセットと、各文書についてトピックへの分布が返ってくる。

2つのアルゴリズムファミリーが主流だ。LDA（2003年）は各文書を潜在的なトピックの混合として、各トピックを単語の分布として扱う。推論はベイズ的だ。混合メンバーシップのトピック割り当てと説明可能な単語レベルの確率分布が必要なプロダクションでは今でも現役だ。

BERTopic（2020年）はBERTで文書をエンコードし、UMAPで次元削減し、HDBSCANでクラスタリングし、クラスベースのTF-IDFでトピック語を抽出する。短いテキスト、ソーシャルメディア、単語の重複より意味的な類似性が重要な場合に勝る。1つの文書に1つのトピックしか割り当てられないという制限は、長文コンテンツには不向きだ。

このレッスンでは両方の直感を養い、与えられたコーパスに対してどちらを選ぶかを解説する。

## 概念

![LDA mixture model vs BERTopic clustering](../assets/topic-modeling.svg)

**LDAの生成ストーリー。** 各トピックは単語の分布だ。各文書はトピックの混合だ。文書中の単語を生成するには、文書の混合からトピックをサンプリングし、そのトピックの分布から単語をサンプリングする。推論はこれを逆転させる: 観測された単語から文書ごとのトピック分布とトピックごとの単語分布を推論する。崩壊ギブスサンプリングまたは変分ベイズが計算を担う。

LDAの主要な出力：

- `doc_topic`: 行列 `(n_docs, n_topics)`、各行の合計は1（文書のトピック混合）。
- `topic_word`: 行列 `(n_topics, vocab_size)`、各行の合計は1（トピックの単語分布）。

**BERTopicパイプライン。**

1. センテンストランスフォーマー（例: `all-MiniLM-L6-v2`）で各文書をエンコードする。384次元ベクトル。
2. UMAPで次元を約5次元に削減する。BERT埋め込みはクラスタリングには高次元すぎる。
3. HDBSCANでクラスタリングする。密度ベースで、可変サイズのクラスタと「外れ値」ラベルを生成する。
4. 各クラスタで、クラスタの文書にクラスベースのTF-IDFを計算して上位単語を抽出する。

出力は文書ごとに1つのトピック（加えて-1の外れ値ラベル）。オプションで、HDBSCANの確率ベクトルによるソフトメンバーシップもある。

## 実装する

### ステップ1：scikit-learnでのLDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

注意点: ストップワードを除去し、min_dfとmax_dfでレアな語と頻出すぎる語をフィルタリングし、LDAは生のカウントを期待するためCountVectorizer（TfidfVectorizerではない）を使う。

### ステップ2：BERTopic（プロダクション向け）

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

`Topic != -1` のフィルタはBERTopicの外れ値バケット（HDBSCANがクラスタリングできなかった文書）を除外する。`min_topic_size`はHDBSCANの最小クラスタサイズを制御する。ライブラリのデフォルトは10で、このレッスンのスケールに合わせて15に明示的に設定している。文書数が1万を超えるコーパスでは50や100に増やすこと。

### ステップ3：評価

どちらの方法もトピック語を出力する。問題はそれらの語がまとまりがあるかどうかだ。

- **トピックコヒーレンス（c_v）。** スライディングウィンドウコンテキスト内の上位単語ペアのNPMI（正規化点相互情報量）を組み合わせ、スコアをトピックベクトルに集約し、コサイン類似度でベクトルを比較する。高いほど良い。`gensim.models.CoherenceModel`を`coherence="c_v"`で使う。
- **トピック多様性。** 全トピックの上位単語にわたるユニークな語の割合。高いほど良い（トピックが重複しない）。
- **定性的な検査。** 各トピックの上位単語を読む。それらは実際の何かを指しているか？人間の判断が最後の砦だ。

## どちらを選ぶか

| 状況 | 選択 |
|------|------|
| 短いテキスト（ツイート、レビュー、見出し） | BERTopic |
| トピックが混在する長文書 | LDA |
| GPU不要/計算リソースが限られる | LDAまたはNMF |
| 文書レベルのマルチトピック分布が必要 | LDA |
| トピックラベル付けのためのLLM統合 | BERTopic（直接サポートあり） |
| リソースが制約されたエッジデプロイ | LDA |
| 最大限の意味的コヒーレンス | BERTopic |

最大の実践的な考慮事項は文書の長さだ。BERT埋め込みはトランケートされる。LDAのカウントはどんな長さでも機能する。埋め込みモデルのコンテキストより長い文書には、チャンク分割+集約するかLDAを使う。

## 使ってみる

2026年のスタック：

- **BERTopic。** 短いテキストとセマンティクスが重要な場合のデフォルト。
- **`gensim.models.LdaModel`。** プロダクション向けのクラシックなLDA。成熟していて実績がある。
- **`sklearn.decomposition.LatentDirichletAllocation`。** 実験向けの簡単なLDA。
- **NMF。** 非負値行列分解。LDAへの高速な代替。短いテキストで同等の品質。
- **Top2Vec。** BERTopicと似た設計。コミュニティは小さいが一部のベンチマークで良い。
- **FASTopic。** 新しく、非常に大きなコーパスではBERTopicより高速。
- **LLMベースのラベリング。** 任意のクラスタリングを行い、モデルに各クラスタに名前を付けるようプロンプトする。

## 成果物を出す

`outputs/skill-topic-picker.md` として保存する：

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## 演習

1. **（簡単）** 20 Newsgroupsデータセットで5トピックのLDAをフィッティングする。トピックごとに上位10語を出力する。各トピックに手動でラベルを付ける。アルゴリズムは実際のカテゴリを見つけたか？
2. **（中程度）** 同じ20 Newsgroupsのサブセットでも BERTopicをフィッティングする。見つかったトピック数、上位語、定性的なコヒーレンスをLDAと比較する。どちらが実際のカテゴリをよりはっきりと浮かび上がらせるか？
3. **（難しい）** コーパスのLDAとBERTopic両方でc_vコヒーレンスを計算する。各手法を5、10、20、50トピックで実行する。コヒーレンス対トピック数をプロットする。どちらの手法がトピック数にわたってより安定しているかを報告する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|----------------|----------------------|
| トピック | コーパスが扱っているもの | 単語の確率分布（LDA）または類似文書のクラスタ（BERTopic） |
| 混合メンバーシップ | 文書が複数のトピックを持つ | LDAは各文書に全トピックにわたる分布を割り当てる |
| UMAP | 次元削減 | ローカル構造を保持するマニフォールド学習。BERTopicで使われる |
| HDBSCAN | 密度クラスタリング | 可変サイズのクラスタを見つける。外れ値には「ノイズ」ラベル（-1）を生成する |
| c_vコヒーレンス | トピック品質メトリクス | スライディングウィンドウ内の上位トピック語の平均点相互情報量 |

## 参考資料

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA論文
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — BERTopic論文
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — c_vなどを導入した論文
- [BERTopicドキュメント](https://maartengr.github.io/BERTopic/) — プロダクション向けリファレンス。充実したサンプルあり
