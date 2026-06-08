# 情報検索と検索

> BM25は正確だが脆弱だ。高密度は広い網を投げるがキーワードを見逃す。ハイブリッドが2026年のデフォルトだ。それ以外はチューニングだ。


## 問題の背景

ユーザーが「お金を得るために嘘をついたらどうなるか」と入力して、実際にそれをカバーする法規：「IPC第420条」を見つけることを期待する。キーワード検索は完全に見逃す（共有語彙なし）。セマンティック検索は埋め込みが法律テキストで訓練されていなければ見逃す。実際の検索は両方を扱わなければならない。

IRはすべてのRAGシステム、すべての検索バー、すべてのドキュメントサイトのファジールックアップの下にあるパイプラインだ。2026年に本番で機能するアーキテクチャは単一の方法ではない。相補的な方法のチェーンで、それぞれが前のものの失敗を捕捉する。

このレッスンでは各ピースを構築して、それぞれが捕捉する失敗を示す。

## 概念

4つのレイヤー。必要なものを選ぶ。

1. **スパース検索（BM25）。** 高速で完全一致に正確、セマンティクスは苦手。転置インデックスに対して実行する。数百万文書に対してクエリあたり10ms未満。法規参照、製品コード、エラーメッセージ、固有表現を正確に得る。
2. **高密度検索。** クエリとドキュメントをベクトルにエンコードする。最近傍探索。言い換えと意味的類似度を捉える。1文字異なる完全なキーワードマッチを見逃す。FAISSまたはベクターDBでクエリあたり50〜200ms。
3. **フュージョン。** スパースと高密度のランク付きリストをマージする。Reciprocal Rank Fusion（RRF）は生スコア（異なるスケールに存在する）を無視してランク位置のみを使うため簡単なデフォルトだ。重み付きフュージョンはドメインで一方のシグナルが支配的な場合のオプションだ。
4. **クロスエンコーダーリランク。** フュージョンから上位30個を取る。クロスエンコーダー（クエリとドキュメントを一緒に、各ペアをスコアリング）を実行する。上位5個を保持する。クロスエンコーダーはバイエンコーダーよりペアあたり遅いが、はるかに正確だ。上位30個のみで実行することで償却する。

3方向検索（BM25 + 高密度 + SPLADEのような学習済みスパース）は2026年のベンチマークで2方向を上回るが、学習済みスパースインデックスのインフラが必要だ。ほとんどのチームにとって、2方向プラスクロスエンコーダーリランクがスイートスポットだ。

## 実装する

### ステップ1: BM25をゼロから

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

知っておく価値のある2つのパラメーター。`k1=1.5`は単語頻度の飽和を制御する。高いほど単語の繰り返しに多くの重みがある。`b=0.75`は長さの正規化を制御する。0はドキュメント長を無視し、1は完全に正規化する。デフォルトはオリジナルの論文からのRobertsonの推奨であり、ほとんどチューニングする必要はない。

### ステップ2: バイエンコーダーによる高密度検索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

ドット積がコサインと等しくなるようにL2正規化する。`all-MiniLM-L6-v2`は384次元で速く、ほとんどの英語検索に十分強い。多言語作業には`paraphrase-multilingual-MiniLM-L12-v2`を使う。最高の精度には`bge-large-en-v1.5`または`e5-large-v2`。

### ステップ3: Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60`定数はオリジナルのRRF論文から来ている。高い`k`はランク差の寄与を平坦化する。低い`k`はトップランクが支配するようにする。60が公開されているデフォルトでほとんどチューニングする必要はない。

### ステップ4: ハイブリッド検索+リランク

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

3つのステージを組み合わせる。BM25は語彙的なマッチを見つける。高密度はセマンティックなマッチを見つける。RRFはスコアキャリブレーションなしに2つのランキングをマージする。クロスエンコーダーはバイエンコーダーが見逃した細かい粒度の関連性を捉えるために上位30個をクエリ-ドキュメントペアを一緒に使って再スコアリングする。上位5個を保持する。

### ステップ5: 評価

| 指標 | 意味 |
|------|------|
| Recall@k | 正しいドキュメントが存在するクエリのうち、上位kに含まれる割合。 |
| MRR（Mean Reciprocal Rank） | 最初の関連ドキュメントの1/rankの平均。 |
| nDCG@k | 単なる二値関連/非関連ではなく、関連性の段階を考慮する。 |

RAGには特に、検索器の**Recall@k**が最も重要な数字だ。正しいパッセージが取得されたセットにない場合、読み取り器は答えられない。

デバッグのヒント：失敗したクエリについて、スパースと高密度のランキングを比較する。一方が正しいドキュメントを見つけてもう一方が見つけない場合、語彙のミスマッチ（修正：欠けている半分を追加）またはセマンティックの曖昧さ（修正：より良い埋め込みまたはリランカー）がある。

## 使ってみる

2026年のスタック：

| スケール | スタック |
|---------|---------|
| 1k〜100k文書 | インメモリBM25 + `all-MiniLM-L6-v2`埋め込み + RRF。別個のDBなし。 |
| 100k〜10M文書 | 高密度にFAISSまたはpgvector + BM25にElasticsearch/OpenSearch。並列実行。 |
| 10M以上 | ハイブリッドサポートを持つQdrant/Weaviate/Vespa/Milvus。上位30にクロスエンコーダーリランク。 |
| 最高品質フロンティア | 3方向（BM25 + 高密度 + SPLADE）+ ColBERTレイトインタラクションリランキング |

何を選んでも評価に予算を確保する。エンドツーエンドのRAG精度をベンチマークする前に検索再現率をベンチマークする。読み取り器は検索器が見逃したものを修正できない。

### 2026年本番RAGから得られた教訓

- **RAG失敗の80%はモデルではなく取り込みとチャンキングに起因する。** チームはLLMを交換してプロンプトを調整するのに何週間も費やすが、検索は静かに3つのクエリごとに間違ったコンテキストを返している。まずチャンキングを修正する。
- **チャンキング戦略はチャンクサイズより重要だ。** 固定サイズの分割はテーブル、コード、ネストされたヘッダーを壊す。センテンス認識がデフォルト。技術ドキュメントと製品マニュアルにはセマンティックまたはLLMベースのチャンキングが効果を発揮する。
- **親ドキュメントパターン。** 精度のために小さい「子」チャンクを取得する。同じ親セクションから複数の子が現れたら親ブロックに切り替えてコンテキストを保持する。これはリトレーニングなしで一貫して回答品質を向上させる。
- **k_rerank=3が通常最適だ。** それを超えた追加のチャンクはトークンコストと生成レイテンシを追加するが回答品質は向上しない。k=8がまだk=3より良ければ、リランカーが不十分だ。
- **HyDE/クエリ拡張。** クエリから仮想の回答を生成してそれを埋め込み、検索する。短い質問と長いドキュメントの間のフレーズのギャップを埋める。訓練なしの無料の精度向上。
- **コンテキスト予算は8Kトークン未満。** その制限でのコンスタントなヒットはリランカーのしきい値が緩すぎることを意味する。
- **すべてをバージョン管理する。** プロンプト、チャンキングルール、埋め込みモデル、リランカー。どんなドリフトも静かに回答品質を壊す。事実性、コンテキスト精度、未回答質問率のCIゲートでユーザーが見る前に回帰をブロックする。
- **3方向検索（BM25 + 高密度 + SPLADEのような学習済みスパース）は2026年のベンチマークで2方向を上回る**。特に固有名詞とセマンティクスを混合するクエリに対して。インフラがSPLADEインデックスをサポートするときに出荷する。

適切な検索設計は2026年の業界測定によると幻覚を70〜90%削減する。ほとんどのRAGパフォーマンス向上はモデルのファインチューニングではなく、より良い検索から来る。

## 成果物を出す

`outputs/skill-retrieval-picker.md`として保存する：

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## 演習

1. **易しい。** 上記の`hybrid_search`を500文書のコーパスで実装する。20個のクエリをテストする。BM25のみ、高密度のみ、ハイブリッドの間でrecall at 5を比較する。
2. **普通。** MRR計算を追加する。既知の正しいドキュメントを持つ各テストクエリについて、BM25、高密度、ハイブリッドのランキングで正しいドキュメントのランクを見つける。それぞれのMRRを報告する。
3. **難しい。** MultipleNegativesRankingLoss（Sentence Transformers）を使ってドメインで高密度エンコーダーをファインチューニングする。500個のクエリ-ドキュメントペアから訓練セットを構築する。ファインチューニング前後の再現率を比較する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------------------|
| BM25 | キーワード検索 | Okapi BM25。単語頻度、IDF、長さでドキュメントをスコアリングする。 |
| 高密度検索 | ベクトル検索 | クエリとドキュメントをベクトルにエンコードして最近傍を見つける。 |
| バイエンコーダー | 埋め込みモデル | クエリとドキュメントを独立してエンコードする。クエリ時に高速。 |
| クロスエンコーダー | リランカーモデル | クエリとドキュメントを一緒にエンコードする。遅いが正確。 |
| RRF | ランクフュージョン | `1/(k + rank)`を合計して2つのランキングを結合する。 |
| Recall@k | 検索指標 | 上位kに関連ドキュメントが含まれるクエリの割合。 |

## 参考資料

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — 決定的なBM25の扱い。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR、標準的なバイエンコーダー。
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 高密度とのギャップを埋める学習済みスパース検索器。
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF論文。
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — レイトインタラクション検索。
