# キャップストーン 04 — マルチモーダル文書QA（ビジョンファースト：PDF、表、グラフ）

> 2026年の文書QAのフロンティアは、OCR→テキストからビジョンファーストの遅延インタラクションへと移行した。ColPali、ColQwen2.5、ColQwen3-omniは各PDFページを画像として扱い、マルチベクターの遅延インタラクションで埋め込み、クエリがパッチに直接アテンドできるようにする。金融の10-K、科学論文、手書きメモではこのパターンがOCRファーストを大きく上回る。10kページのパイプラインをエンドツーエンドで構築し、OCR→テキストとの比較を公開せよ。

**演習するフェーズ:** P4 · P5 · P7 · P11 · P12 · P17

## 問題

企業は、OCRパイプラインが台無しにするPDFを大量に抱えている：回転した表のあるスキャンされた10-K、方程式が密な科学論文、画像としてのみ意味をなすグラフ、手書きの注記。これらをテキストファーストとして扱うと信号の半分を失う。2026年の答えは、生のページ画像に対する遅延インタラクションマルチベクター検索だ。ColPali（Illuin Tech）がそれを導入した；ColQwen2.5-v0.2とColQwen3-omniが精度を向上させた。ViDoRe v3では、ビジョンファーストの検索がOCR→テキストを大きなマージンで上回り、グラフ・表・手書きでそのギャップが広がる。

トレードオフはストレージとレイテンシーだ。ColQwen埋め込みはページあたり約2048のパッチベクターであり、単一の1024次元ベクターではない。生のストレージが膨張する。DocPruner（2026年）は測定可能な精度損失なしに50%の剪定を実現する。10kページをインデックスし、ViDoRe v3 nDCG@5を測定し、2秒以下で回答を提供し、OCR→テキストのベースラインと直接比較する。

## コンセプト

遅延インタラクションとは、すべてのクエリトークンがすべてのパッチトークンに対してスコアリングし、クエリトークンごとの最大スコアが合計されることを意味する。単一のプールされたベクターを必要とせずに細かい粒度のマッチングが得られる。マルチベクターインデックス（Vespa、Qdrantマルチベクター、またはAstraDB）はパッチごとの埋め込みを保存し、検索時にMaxSimを実行する。

回答者は視覚言語モデルで、クエリとトップkの取得済みページを画像として受け取り、エビデンス領域（バウンディングボックスまたはページ参照）付きで回答を書く。Qwen3-VL-30B、Gemini 2.5 Pro、InternVL3が2026年のフロンティアの選択肢だ。方程式と科学表記については、OCRフォールバック（Nougat、dots.ocr）がオプションのテキストチャンネルとして組み込まれる。

評価は2次元のマトリックスだ。1軸：コンテンツタイプ（プレーンテキスト段落、密な表、棒/折れ線グラフ、手書きメモ、方程式）。もう1軸：検索アプローチ（ビジョンファースト遅延インタラクション対OCR→テキスト対ハイブリッド）。各セルはnDCG@5と回答精度を得る。レポートが成果物だ。

## アーキテクチャ

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## スタック

- ページレンダリング: 180 DPIでPyMuPDF（fitz）、縦向き正規化
- 遅延インタラクションモデル: ColQwen2.5-v0.2またはColQwen3-omni（Hugging FaceのvidoreチームによるもΔ）
- インデックス: マルチベクターフィールドを持つVespa、またはQdrantマルチベクター、またはMaxSimを持つAstraDB
- 剪定: DocPruner 2026ポリシー（高分散パッチを保持、0.5%以下の精度損失で50%圧縮）
- OCRフォールバック（方程式/密な表）: dots.ocrまたはNougat
- VLM回答者: セルフホストQwen3-VL-30BまたはホストGemini 2.5 Pro；InternVL3をフォールバックとして
- 評価: ViDoRe v3ベンチマーク、マルチページ推論用M3DocVQA
- ビューアUI: エビデンス領域のcanvasオーバーレイを持つNext.js 15

## 実装する

1. **取り込み。** 10-K、科学論文、スキャン文書にわたる10kページのPDFコーパスを処理する。各ページを1536x2048 PNGにレンダリングする。`{doc_id, page_num, image_path}`を永続化する。

2. **埋め込み。** 各ページ画像にColQwen2.5-v0.2を実行する。出力形状は次元128の約2048パッチ埋め込み。DocPrunerを適用して最も信号の高い半分を保持する。Vespaのマルチベクターフィールドまたはqdrantのマルチベクターに書き込む。

3. **クエリ。** 各受信クエリについて、クエリタワー（トークンレベルの埋め込み）で埋め込む。インデックスに対してMaxSimを実行する：すべてのクエリトークンについて、ページパッチ埋め込みに対する最大ドット積を取り、合計する。上位kページを返す。

4. **合成。** クエリとトップ5のページ画像を使ってQwen3-VL-30Bを呼び出す。プロンプト：「提供されたページのみを使って回答すること。各主張を（doc_id、ページ）で引用し、領域（図、表、段落）を名前で挙げること。」

5. **エビデンス領域。** 引用された領域を抽出するために回答を後処理する。VLMがバウンディングボックスを出力する場合（Qwen3-VLは出力する）、ビューアでオーバーレイとしてレンダリングする。

6. **OCRフォールバック。** 方程式が密なページ（画像の分散に関するヒューリスティック）については、NougatまたはDots.ocrを実行し、OCRテキストを画像と一緒に追加チャンネルとして渡す。

7. **評価。** ViDoRe v3（検索nDCG@5）とM3DocVQA（マルチページQA精度）を実行する。同じシンセサイザーを使って同じコーパスにOCR→テキストパイプラインも実行する。コンテンツタイプ×アプローチのマトリックスを作成する。

8. **UI。** まずStreamlitプロトタイプ；次にページごとのエビデンス領域オーバーレイを持つNext.js 15本番ビューア。

## 使ってみる

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## 成果物を出す

`outputs/skill-doc-qa.md`には成果物が記述される：ViDoRe v3でのOCR→テキストのベースラインと比較して評価された、特定のコーパスに調整されたビジョンファーストのマルチモーダル文書QAシステム。

| 配点 | 基準 | 測定方法 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA精度 | OCR→テキストのベースラインと公開リーダーボードとのベンチマーク数値比較 |
| 20 | エビデンス領域のグラウンディング | 引用された領域が実際に回答スパンを含む割合 |
| 20 | ストレージとレイテンシーエンジニアリング | DocPruner圧縮率、インデックスp95、回答p95 |
| 20 | マルチページ推論 | 手動ラベル付き100問マルチページセットでの精度 |
| 15 | ソース検査UX | ビューアの明確さ、オーバーレイの忠実度、並列比較ツール |
| **100** | | |

## 演習

1. 同じコーパスでColQwen2.5-v0.2対ColQwen3-omniを測定する。一方が正解して他方が不正解のページはどれか？コンテンツクラスタグをインデックスに追加してタイプ別にルーティングする。

2. 積極的に埋め込みを剪定する（75%、90%）。圧縮の崖を見つける：ViDoRe nDCG@5がOCRのベースラインを下回る点。

3. ハイブリッドを構築する：OCR→テキストとColQWenを並列に実行し、RRFで融合し、クロスエンコーダーでリランクする。ハイブリッドはどちらかを単独で上回るか？どこで最も役立つか？

4. Qwen3-VL-30Bを小さいVLM（Qwen2.5-VL-7B）に交換する。精度対コストの曲線を測定する。

5. 手書きメモのサポートを追加する。手書きコーパスをレンダリングし、ColQwenで埋め込み、検索を測定する。手書きOCRパイプラインと比較する。

## キーワード

| 用語 | 一般的な呼び方 | 実際の意味 |
|------|-----------------|------------------------|
| 遅延インタラクション | 「ColPaliスタイルの検索」 | クエリトークンがページパッチに対して独立してスコアリング；MaxSimが集約 |
| マルチベクター | 「パッチごとの埋め込み」 | 各文書は1つのプールされたベクターではなく多くのベクターを持つ |
| MaxSim | 「遅延インタラクションスコアリング」 | すべてのクエリトークンについて、文書ベクターに対する最大類似度を取り、合計 |
| DocPruner | 「パッチ圧縮」 | 精度損失を無視できるほど保ちながら50%のパッチを保持する2026年の剪定 |
| ViDoRe v3 | 「文書検索ベンチマーク」 | 視覚文書検索を測定するための2026年の標準 |
| エビデンス領域 | 「引用されたバウンディングボックス」 | 回答スパンを局所化するソースページ上のbbox |
| OCRフォールバック | 「方程式チャンネル」 | 方程式または表が多いページで視覚と並行して使われるテキストパイプライン |

## 参考資料

- [ColPali（Illuin Tech）リポジトリ](https://github.com/illuin-tech/colpali) — リファレンス遅延インタラクション文書検索
- [ColPali論文（arXiv:2407.01449）](https://arxiv.org/abs/2407.01449) — 基礎的手法論文
- [Hugging Face上のColQwenファミリー](https://huggingface.co/vidore) — 本番対応チェックポイント
- [M3DocRAG（Adobe）](https://arxiv.org/abs/2411.04952) — マルチページマルチモーダルRAGのベースライン
- [Vespaマルチベクターチュートリアル](https://docs.vespa.ai/en/colpali.html) — リファレンスサービングスタック
- [Qdrantマルチベクターサポート](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — 代替インデックス
- [AstraDBマルチベクター](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — 代替マネージドインデックス
- [Nougat OCR](https://github.com/facebookresearch/nougat) — 方程式対応OCRフォールバック
