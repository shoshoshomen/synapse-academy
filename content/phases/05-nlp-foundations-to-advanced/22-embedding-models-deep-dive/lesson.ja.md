# 埋め込みモデル — 2026年の詳細ガイド

> Word2Vecは単語ごとにベクトルを与えた。現代の埋め込みモデルはパッセージごとに、言語横断で、スパース・密・マルチベクトルのビューを持ち、インデックスに合わせたサイズのベクトルを与える。間違った選択をするとRAGは間違ったものを検索する。


## 問題設定

RAGシステムが40%の確率で間違ったパッセージを検索する。原因はほとんどの場合、ベクトルデータベースでもプロンプトでもない。埋め込みモデルだ。

2026年における埋め込みの選択は5つの軸にわたる：

1. **密対スパース対マルチベクトル。** パッセージごとに一つのベクトル、トークンごとに一つ、またはスパースな重み付きバッグ・オブ・ワーズ。
2. **言語カバレッジ。** 単言語の英語モデルは英語のみのタスクで依然として勝る。多言語モデルはコーパスが混在する場合に勝る。
3. **コンテキスト長。** 512トークン対8,192対32,768 — そして実際の有効容量はしばしば宣告上限の60〜70%だ。
4. **次元予算。** フル精度での3,072フロート = ベクトルあたり12 KB。1億ベクトルではストレージが月1,300ドルになる。Matryoshkaトランケーションによりこれを4倍削減できる。
5. **オープン対ホスト型。** オープンウェイトはスタックとデータを制御できる。ホスト型は制御と引き換えに常に最新版を得る。

このレッスンではトレードオフを解説するので、前四半期に人気だったものではなく証拠に基づいて選べるようになる。

## 概念

![密・スパース・マルチベクトル埋め込み](../assets/embedding-modes.svg)

**密埋め込み。** パッセージごとに一つのベクトル（通常384〜3,072次元）。コサイン類似度がパッセージを意味的な近さでランク付けする。OpenAIの`text-embedding-3-large`、BGE-M3の密モード、Voyage-3。デフォルトの選択。

**スパース埋め込み。** SPLADEスタイル。トランスフォーマーがすべての語彙トークンの重みを予測し、そのほとんどをゼロにする。結果は|vocab|サイズのスパースベクトル。学習された用語重み付きの語彙マッチング（BM25のような）を捉える。キーワード重視のクエリに強い。

**マルチベクトル（遅延インタラクション）。** ColBERTv2、Jina-ColBERT。トークンごとに一つのベクトル。MaxSimでスコアリング：各クエリトークンに対して最も類似したドキュメントトークンを見つけ、スコアを合計する。ストレージとスコアリングのコストが高いが、長いクエリとドメイン固有のコーパスで勝る。

**BGE-M3: 三つ同時に。** 単一モデルが密、スパース、マルチベクトル表現を同時に出力する。それぞれを独立してクエリできる；スコアは重み付き和で統合される。柔軟性が必要な場合の2026年のデフォルト。

**Matryoshka表現学習。** ベクトルの最初のN次元が有用なスタンドアロン埋め込みを形成するよう学習される。1,536次元のベクトルを256次元に切り捨てると、精度約1%の損失で6倍のストレージ節約が得られる。OpenAIのtext-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5以降でサポートされる。

### MTEBリーダーボードは一部の話しか伝えない

Massive Text Embedding Benchmark — 立ち上げ時に8タスクタイプで56タスク（2022年）、MTEB v2では100以上のタスクに拡張。2026年初頭、Gemini Embedding 2が検索でトップ（67.71 MTEB-R）。Cohere embed-v4が一般部門をリード（65.2 MTEB）。BGE-M3がオープンウェイト多言語でリード（63.0）。リーダーボードは必要条件だが十分条件ではない — 常に自分のドメインでベンチマークを取ること。

### 3層パターン

| ユースケース | パターン |
|----------|---------|
| 高速な第一パス | 密バイエンコーダー（BGE-M3、text-3-small） |
| 再現率向上 | スパース（SPLADE、BGE-M3スパース）＋RRFフュージョン |
| トップ50の精度 | マルチベクトル（ColBERTv2）またはクロスエンコーダーリランカー |

ほとんどの本番スタックは3つすべてを使用する。

## 実装する

### ステップ1: ベースライン — Sentence-BERTによる密埋め込み

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True`によりドット積がコサイン類似度と等しくなる。常に設定すること。

### ステップ2: Matryoshkaトランケーション

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

トランケーション後に再正規化する。Nomic v1.5、OpenAI text-3、Voyage-4はこれが無損失になるよう学習されている。Matryoshkaでないモデル（オリジナルのSentence-BERT）はトランケーション時に急激に劣化する。

### ステップ3: BGE-M3の多機能性

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

3つのインデックス、一度の推論呼び出し。スコアの統合：

```python
dense_score = ... # dense_vecsに対するコサイン
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

自分のドメインで重みを調整する。

### ステップ4: カスタムタスクでのMTEB評価

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

候補モデルを*代表的な*サブセットで実行する。リーダーボードのランクだけを信頼しない — 自分のドメインが重要だ。

### ステップ5: スクラッチからのコサイン計算

`code/main.py`を参照。平均化ハッシュトリック埋め込み（stdlibのみ）。トランスフォーマー埋め込みとは競合しないが、形を示している：トークン化 → ベクトル → 正規化 → ドット積。

## ピットフォール

- **クエリとドキュメントに同じモデル。** 一部のモデル（Voyage、Jina-ColBERT）はクエリとドキュメントで異なるパスを通る非対称エンコーディングを使用する。常にモデルカードを確認する。
- **プレフィックスの欠落。** `bge-*`モデルはクエリに「Represent this sentence for searching relevant passages: 」を前置する必要がある。忘れると再現率が3〜5ポイント低下する。
- **Matryoshkaの過度のトランケーション。** 1,536 → 256は通常安全。1,536 → 64は安全でない。評価セットで検証する。
- **コンテキストの切り捨て。** ほとんどのモデルは最大長を超える入力を静かに切り捨てる。長い文書にはチャンキングが必要（レッスン23参照）。
- **レイテンシのテールを無視する。** MTEBスコアはp99レイテンシを隠す。600Mモデルは335Mモデルより2ポイント高くなる可能性があるが、クエリあたり3倍のコストがかかる。

## 使ってみる

2026年のスタック：

| 状況 | 選択 |
|-----------|------|
| 英語のみ、高速、API | `text-embedding-3-large`か`voyage-3-large` |
| オープンウェイト、英語 | `BAAI/bge-large-en-v1.5` |
| オープンウェイト、多言語 | `BAAI/bge-m3`か`Qwen3-Embedding-8B` |
| 長いコンテキスト（32k以上） | Voyage-3-large、Cohere embed-v4、Qwen3-Embedding-8B |
| CPUのみのデプロイ | Nomic Embed v2（137Mパラメータ、MoE） |
| ストレージ制約 | Matryoshkaトランケーション＋int8量子化 |
| キーワード重視のクエリ | SPLADEスパースを追加し、密とRRFフュージョン |

2026年パターン：BGE-M3かtext-3-largeから始め、MTEBで自分のドメインを評価し、ドメイン固有モデルが3ポイント以上勝てばスワップする。

## 成果物を出す

`outputs/skill-embedding-picker.md`として保存：

```markdown
---
name: embedding-picker
description: 特定のコーパスとデプロイに合わせた埋め込みモデル、次元、検索モードを選ぶ。
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

コーパス（サイズ、言語、ドメイン、平均長さ）、デプロイターゲット（クラウド / エッジ / オンプレミス）、レイテンシ予算、ストレージ予算を与えられた場合、以下を出力する：

1. モデル。名前付きチェックポイントまたはAPI。一文の理由。
2. 次元。フル / Matryoshkaトランケーション / int8量子化。ストレージ予算に関連した理由。
3. モード。密 / スパース / マルチベクトル / ハイブリッド。理由。
4. モデルカードが必要とするクエリプレフィックス / テンプレート。
5. 評価計画。ドメインに関連するMTEBタスク＋nDCG@10を持つ保留ドメイン評価。

ドメイン検証なしにMatryoshkaを64次元未満に切り捨てる推奨を拒否する。10k未満のパッセージのコーパスにColBERTv2を推奨することを拒否する（オーバーヘッドが正当化されない）。512トークンのウィンドウを持つモデルにルーティングされる長文書コーパス（8k以上のトークン）をフラグする。
```

## 演習

1. **易。** 100文を`bge-small-en-v1.5`でフル次元（384）、その後Matryoshka 128でエンコードする。10クエリでのMRR低下を測定する。
2. **中。** 自分のドメインの500パッセージでBGE-M3の密、スパース、colbertを比較する。recall@10でどれが勝つか？RRFフュージョンは最良の単一モードを上回るか？
3. **難。** 上位2つのドメインタスクで3つの候補モデルにMTEBを実行する。MTEBスコア、100クエリバッチのp99レイテンシ、1Mクエリあたりのコストを報告する。パレート最適なものを選ぶ。

## キーワード

| 用語 | 一般的な説明 | 実際の意味 |
|------|------------|----------|
| 密埋め込み | ベクトル | テキストあたりの固定サイズのベクトル。コサイン類似度でランキング。 |
| スパース埋め込み | 学習されたBM25 | 語彙トークンあたりの重み；ほとんどがゼロ；エンドツーエンドで学習。 |
| マルチベクトル | ColBERTスタイル | トークンごとのベクトル；MaxSimスコアリング；インデックスが大きく再現率が良い。 |
| Matryoshka | ロシア人形のトリック | 最初のN次元が単独で有効な小さな埋め込みを形成する。 |
| MTEB | ベンチマーク | Massive Text Embedding Benchmark — 立ち上げ時56タスク、v2では100以上。 |
| BEIR | 検索ベンチマーク | 18のゼロショット検索タスク；クロスドメインの堅牢性でよく引用される。 |
| 非対称エンコーディング | クエリ ≠ ドキュメントパス | モデルがクエリとドキュメントに異なる射影を使用する。 |

## 参考資料

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — バイエンコーダーの論文。
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — リーダーボードの論文。
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — 統合三モードモデル。
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — 次元ラダー学習目標。
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — 本番での遅延インタラクション。
- [Hugging Face上のMTEBリーダーボード](https://huggingface.co/spaces/mteb/leaderboard) — ライブランキング。
