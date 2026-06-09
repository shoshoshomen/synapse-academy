# 推論プラットフォームの経済学 — Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 2026年の推論市場はもはやGPU時間のレンタルではない。カスタムシリコン（Groq、Cerebras、SambaNova）、GPUプラットフォーム（Baseten、Together、Fireworks、Modal）、APIファーストマーケットプレイス（Replicate、DeepInfra）の3つに分岐する。Fireworksは2026年5月1日にGPUを1ドル/時値上げし、1日10兆トークン以上という数字が示す40億ドル評価額はボリューム駆動モデルが機能することを証明している。Basetenは2026年1月に50億ドルで3億ドルのシリーズEを完了した。競争ポジショニングのルールはシンプルだ：Fireworksはレイテンシを最適化し、TogetherはカタログBreadthを最適化し、Basetenはエンタープライズの洗練度を最適化し、ModalはPythonネイティブDXを最適化し、Replicateはマルチモーダルのリーチを最適化し、Anyscaleは分散Pythonを最適化する。このレッスンでは、創業者に手渡せるマトリクスを提供する。


## 学習目標

- 3つの市場セグメント（カスタムシリコン、GPUプラットフォーム、APIファースト）を挙げ、各ベンダーをセグメントにマッピングする。
- 「トークン単価」のAPIプライシングモデルが、ハードウェアのコストカーブではなく配信エンジンのコストカーブに収束する理由を説明する。
- 少なくとも3つのベンダーで有効なリクエスト単価を計算し、分単価（Baseten、Modal）がトークン単価に勝る場面を説明する。
- 特定のワークロード（サーバーレスバースト、安定した高スループット、ファインチューニングバリアント、マルチモーダル）に適切なデフォルトプラットフォームを特定する。

## 問題

マネージドハイパースケーラープラットフォームを評価した。より狭く速いプロバイダーが必要だと判断した — レイテンシにFireworks、幅にTogether、カスタムモデルのファインチューニングにBaseten。今6つの現実的な選択肢があるが、価格ページが揃わない。Fireworksはドル/Mトークンで表示、Basetenはドル/分、Modalはドル/秒、Replicateはドル/予測。ワークロードをモデル化しないと比較できない。

さらに悪いことに、各価格ページの背後にあるビジネスモデルが異なる。Fireworksは共有GPU上で自社のカスタムエンジン（FireAttention）を実行し、トークン単価は利用率カーブを反映する。BasetenはTruss + 専用GPUを提供し、分単価は専有性を反映する。Modalは真のPythonサーバーレス — 秒単位課金でコールドスタートが1秒未満。同じ出力（LLMレスポンス）、3つの異なるコスト関数だ。

このレッスンでは6つをモデル化し、それぞれが勝る場面を示す。

## コンセプト

### 3つのセグメント

**カスタムシリコン** — Groq（LPU）、Cerebras（WSE）、SambaNova（RDU）。同一モデルのGPUベースクラスターと比較して通常5〜10倍速いデコード。トークン単価は高い（2025年後半のGroqはLlama-70Bで約0.99ドル/M）が、レイテンシセンシティブなユースケースでは比類ない。Groqはボイスエージェントとリアルタイム翻訳の本番選択肢だ。

**GPUプラットフォーム** — Baseten、Together、Fireworks、Modal、Anyscale。NVIDIA（2026年のH100、H200、B200）またはAMDで動作。「生のGPUレンタル」（RunPod、Lambda）と「ハイパースケーラーマネージドサービス」（Bedrock）の中間にある経済層。

**APIファーストマーケットプレイス** — Replicate、DeepInfra、OpenRouter、Fal。幅広いカタログ、予測単価または秒単価、最初の呼び出しまでの時間を重視する。

### Fireworks — レイテンシ最適化GPUプラットフォーム

- FireAttentionエンジン（カスタム）。同等設定でvLLMと比較して4倍低レイテンシとマーケティング。
- 非インタラクティブワークロード向けのバッチティア（サーバーレス料金の約50%）。
- ファインチューニングされたモデルはベースモデルと同じ料金で提供 — LoRAにプレミアムを請求するプロバイダーとの実質的な差別化。
- 2026年中盤：2026年5月1日よりオンデマンドGPUレンタルを1ドル/時値上げ。スケール時のボリューム価格は交渉可能。
- 財務シグナル：40億ドル評価額、1日10兆トークン以上の処理。

### Together — 幅最適化

- 上流公開から数日以内のオープンソースリリースを含む200以上のモデル。
- 同等のLLMモデルでReplicateより50〜70%安価 — 「AIネイティブクラウド」のポジショニングはボリュームとカタログ。
- 1つのAPIで推論 + ファインチューニング + トレーニング。

### Baseten — エンタープライズ洗練度最適化

- Trussフレームワーク：1つのマニフェストに依存関係、シークレット、配信設定を含むモデルパッケージング。
- T4からB200までのGPUレンジ。合理的なコールドスタート軽減付きの分単価課金。
- SOC 2 Type II、HIPAA対応。一般的なフィンテックとヘルスケアの選択。
- 50億ドル評価額、2026年1月シリーズE（CapitalG、IVP、NVIDIAから3億ドル）。

### Modal — Pythonネイティブ最適化

- 純粋なPythonによるInfrastructure as Code。`@modal.function(gpu="A100")`でデコレートして1コマンドでデプロイ。
- 秒単価課金。事前ウォームアップで2〜4sのコールドスタート。小モデルは1秒未満。
- 2025年のシリーズBで11億ドル評価額（8700万ドル）。独立調査で最高の開発者体験スコア。

### Replicate — マルチモーダルの幅

- 予測単価。画像、動画、音声モデルのデフォルトプラットフォーム。
- 統合エコシステム（Zapier、Vercel、CMSプラグイン）。
- LLMトークン単価では競争力が低いが、マルチモーダルの多様性で勝つ。

### Anyscale — Rayネイティブ

- Rayの上に構築。RayTurboはAnyscaleの独自推論エンジン（vLLMと競合）。
- 推論ステップが大きなグラフの1ノードである分散Pythonワークロードに最適。
- マネージドRayクラスター。Ray AIRとRay Serveとの緊密な統合。

### トークン単価 vs 分単価 — それぞれが勝る場面

トークン単価は、ワークロードがレイテンシ非感応でバースト的な場合に意味がある — 使った分だけ払う。分単価は、利用率が高く予測可能な場合に意味がある — 専用GPUを飽和させれば勝る。

おおまかなルール：専用GPU持続利用率が約30%を超えるワークロードでは、分単価（Baseten、Modal）がトークン単価（Fireworks、Together）に勝ち始める。それ以下では、アイドル時間を払わないのでトークン単価が勝る。

### カスタムエンジンが真のモート

上記すべてのプラットフォームがvLLMとSGLangの上でカスタムエンジンを主張している。FireAttention、RayTurbo、Basetenの推論スタック。カスタムエンジンの主張はマーケティングの影 — 正直な言い方では、vLLM + SGLangが本番オープンソース推論の約80%を占め、プラットフォーム層での差別化要因はDX、アトリビューション、SLAだ。

### 覚えておくべき数値

- Fireworks GPUレンタル：2026年5月1日より1ドル/時値上げ
- Fireworksの主張：同等設定でvLLMより4倍低レイテンシ
- Together：LLMでReplicateより50〜70%安価
- Baseten評価額：50億ドル（2026年1月シリーズE、3億ドル調達）
- Modal評価額：11億ドル（2025年シリーズB）
- 持続利用率約30%を超えると分単価がトークン単価に勝る

## 使ってみる

`code/main.py`は6つのベンダーを価格モデル横断で合成ワークロードと比較する。1日あたりのドルと有効なドル/Mトークンを報告する。トークン単価と分単価の損益分岐点を見つけるために実行する。

## 成果物を出す

このレッスンでは`outputs/skill-inference-platform-picker.md`を作成する。ワークロードプロファイル、SLA、予算から主要推論プラットフォームを選択し、次点を挙げる。

## 演習

1. `code/main.py`を実行する。1台のH100上の70Bモデルで、Baseten（分単価）がFireworks（トークン単価）に勝つ持続GPU利用率はどこか？損益分岐点を自分で導き、目安と比較する。
2. プロダクトが画像生成、チャット、音声認識を提供する。各モダリティのプラットフォームを選び、それらを統合するゲートウェイパターンを挙げる。
3. Fireworksが主要モデルの価格を1ドル/時値上げした。トラフィックの40%がバッチティア（50%オフ）に移動した場合のブレンドコスト影響をモデル化する。
4. 規制された顧客がSOC 2 Type II + HIPAA + 専用GPUを要求する。どの3つのプラットフォームが有力候補で、FinOpsで勝つのはどれか？
5. Fireworksサーバーレス、Togetherオンデマンド、Baseten専用、Replicate APIで、Llama 3.1 70Bの1,000予測あたりのコストを比較する。1日10予測で最も安いのはどれか？10,000予測では？

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|----------------|------------|
| カスタムシリコン | 「GPU以外のチップ」 | Groq LPU、Cerebras WSE、SambaNova RDU — デコードに最適化 |
| FireAttention | 「Fireworksエンジン」 | カスタムアテンションカーネル。vLLMより4倍低レイテンシとマーケティング |
| Truss | 「Basetenのフォーマット」 | モデルパッケージングマニフェスト。依存関係 + シークレット + 配信設定 |
| トークン単価 | 「API価格」 | 消費トークン数で課金。アイドル時間は払わない |
| 分単価 | 「専用価格」 | GPU時計時間で課金。高利用率で有利 |
| 予測単価 | 「Replicate価格」 | モデル呼び出しごとに課金。画像/動画で一般的 |
| RayTurbo | 「Anyscaleエンジン」 | Ray上の独自推論。RayクラスターでvLLMと競合 |
| バッチティア | 「50%オフ」 | 割引料金の非インタラクティブキュー。Fireworks、OpenAIで一般的 |
| LoRA基本料金 | 「Fireworks LoRA」 | LoRA配信リクエストをベースモデルの料金で課金（差別化要因） |

## 参考資料

- [Fireworks Pricing](https://fireworks.ai/pricing) — トークン単価、バッチティア、GPUレンタル
- [Baseten Pricing](https://www.baseten.co/pricing/) — 分単価、コミットメントキャパシティ、エンタープライズティア
- [Modal Pricing](https://modal.com/pricing) — 秒単価GPUレートと無料ティア
- [Together AI Pricing](https://www.together.ai/pricing) — モデルカタログとトークン単価
- [Anyscale Pricing](https://www.anyscale.com/pricing) — RayTurboとマネージドRay価格
- [Northflank — Fireworks AI Alternatives](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — 比較評価
- [Infrabase — AI Inference API Providers 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — ベンダーランドスケープ
