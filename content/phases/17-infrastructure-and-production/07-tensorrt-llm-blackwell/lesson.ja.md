# FP8とNVFP4を使ったBlacwellでのTensorRT-LLM

> TensorRT-LLMはNVIDIA専用だが、Blackwellで勝る。Dynamoオーケストレーション付きのGB200 NVL72では、SemiAnalysis InferenceXが2026年Q1〜Q2に120Bモデルでトークン百万あたり0.012ドルを計測し、H100 + vLLMの0.09ドル/Mに対して7倍の経済的差を示す。スタックは3つの浮動小数点制度が重なっている：FP8はKVキャッシュとアテンションカーネルに重要なまま、必要なダイナミックレンジを持つため。NVFP4（4ビットマイクロスケーリング）はウェイトとアクティベーションを扱う。マルチトークン予測（MTP）と分散プリフィル/デコードがさらに2〜3倍を上乗せする。デイ0モデルサポートはポストトレーニング変換なしにFP4ウェイトを直接ロードする。2026年のエンジニアリングチームへの落とし穴：TRT-LLMはクローズドのNVIDIAスタックなので採用するとスループットのためにポータビリティを犠牲にする。コミットする前にモデルとハードウェアのミックスで数学を実行すること。


## 学習目標

- ウェイトがNVFP4にある場合でもFP8がKVキャッシュとアテンションに重要なままである理由を説明する。
- BF16、FP8、NVFPフロンティアモデルのHBMフットプリントを計算し、節約がどこから来るかを推論する。
- TRT-LLMが利用するBlackwell固有の機能（デイ0 FP4、MTP、分散配信、all-to-allプリミティブ）を挙げる。
- TRT-LLMのNVIDIAロックがHopper上vLLMとの7倍コスト差に値するかを判断する。

## 問題

2026年の推論経済のフロンティアは「1ドルあたりどれだけのトークンか」だ。答えは4つの積み重なった選択に依存する：ハードウェア世代（HopperのH100/H200 vs BlackwellのB200/GB200）、精度（BF16 → FP8 → NVFP4）、配信エンジン（vLLM vs SGLang vs TRT-LLM）、オーケストレーション（プレーン vs 分散 vs Dynamo）。

Hopper + vLLMでは、120BのMoEが約0.09ドル/Mトークンで動く。Blackwell + TRT-LLM + Dynamoでは、同じモデルが約0.012ドル — 7倍安い。差の一部はハードウェア（BlackwellはHopperに対してGPU1台あたりLLMスループットが11〜15倍）。一部はスタック：FP4ウェイト、MTPドラフト、分散プリフィル/デコード、MoEエキスパート通信のためのNVLink 5 all-to-allだ。

NVIDIAのスタックの外でこれを再現できない。それがトレードオフ — ポータビリティと経済性の交換。どのスタック選択がギャップのどの割合を与えるかを理解することがこのレッスンの要点だ。

## コンセプト

### FP8がKVキャッシュのフロアである理由

2026年でよくある誤り：NVFPがどこにでも適用されると仮定すること。適用されない。KVキャッシュはアテンションのキーと値を格納し、それらはワイドなダイナミックレンジにまたがる。KVをFP4に量子化すると壊滅的な精度損失 — 分布のテールが落ち、アテンションスコアが崩壊する。FP8の指数ビットがKVキャッシュに必要なレンジを与える。

NVFP4（2025〜2026）はウェイトとアクティベーションに適用する。マイクロスケーリング：ウェイトの各ブロックには独自のスケールファクターがあり、小さなブロックがテンソルスケール損失なしに異なるダイナミックレンジにまたがれる。アクティベーションについては、アクティベーションがレイヤー内で小レンジであるためFP4が成り立つ。

典型的なBlackwell設定：

- ウェイト：NVFP4（4ビットマイクロスケーリング）
- アクティベーション：NVFP4
- KVキャッシュ：FP8
- アテンションアキュムレーター：FP32（softmax安定性）

### TRT-LLMが使うBlackwell固有のプリミティブ

- **デイ0 FP4ウェイト**：モデルプロバイダーがFP4ウェイトを直接出荷。TRT-LLMはポストトレーニング変換なしにロード。FP4にはAWQ/GPTQステップが不要。
- **マルチトークン予測（MTP）**：EAGLE（フェーズ17·05）と同じアイデアだがTRT-LLMビルドに統合。
- **分散配信**：別のGPUプール上でのプリフィルとデコード、NVLinkまたはInfiniBand経由でKVキャッシュを転送。Dynamoと同じアイデア（フェーズ17·20）。
- **All-to-all通信プリミティブ**：NVLink 5がMoEエキスパート通信レイテンシをHopperの3倍削減。TRT-LLMのMoEカーネルはこれに最適化されている。
- **NVFP4 + MXFP8マイクロスケーリング**：Blackwellテンソルコアでハードウェアアクセラレートされたスケールファクター処理。

### 覚えておくべき数値

- TRT-LLM経由のHGX B200で0.02ドル/M（GPT-OSS-120B）
- Dynamo（TRT-LLMをオーケストレート）経由のGB200 NVL72で0.012ドル/M
- H100 + vLLM ≈ 0.09ドル/M（同等ワークロード）
- 3ヶ月でのTRT-LLMアップデートによるスループット2.8倍ゲイン（2026年）
- GPU1台あたりのLLMスループット11〜15倍、Blackwell vs Hopper
- MLPerf Inference v6.0（2026年4月）：Blackwellがすべてのサブミットされたタスクを支配

### FP4が品質に実際にかかるコスト

NVFPは積極的だ。推論重いワークロード（思考連鎖、数学、長コンテキストでのコード生成）ではFP4ウェイトが目に見えて劣化する。ブロックごとのキャリブレーションが軽減するが排除しない。推論モデルを出荷するチームは、FP8ウェイト + FP4アクティベーションを妥協として使うか、FP8全体でH200に留まることが多い。

ルール：NVFPウェイト変換にコミットする前に、評価セットでタスク品質を必ず検証する。

### これがNVIDIAロックの決断である理由

TRT-LLMはC++ + CUDA + クローズドソースのカーネルだ。モデルは特定のGPU SKU用にコンパイルする必要がある。AMD、Intel、ARMはなし。インフラ戦略がマルチベンダーの場合、TRT-LLMは配信ティア（混合ハードウェア上のvLLMでは引き続き配信できる）には選択肢にならない。NVIDIA専用なら、7倍の差がロックの代価を払う。

### 2026年の実践レシピ

年間1億ドル以上の推論請求書なら、Hopper + vLLMでの実行は7〜10倍を無駄にしている。コスト支配的なワークロードをBlackwell + TRT-LLM + Dynamoに移行する。モデルイテレーション速度のためにH100 + vLLMで実験ティアを維持する。本番投入前にNVFP変換された各モデルの品質を検証する。

### 分散ボーナス

TRT-LLMの分散配信（別のプリフィルとデコードプール）はフェーズ17·20で詳しく扱う。Blackwellでは倍数が積み重なる：FP4ウェイト × MTPスピードアップ × 分散配置 × キャッシュ対応ルーティング。7倍の数値はこの完全なスタックを前提とする。

## 使ってみる

`code/main.py`は3つのスタックにわたってHBMフットプリント、デコードスループット（メモリバウンド制度）、ドル/Mトークンを計算する：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。実行して複合効果と各変更がギャップのどの割合に寄与するかを確認する。

## 成果物を出す

このレッスンでは`outputs/skill-trtllm-blackwell-advisor.md`を作成する。ワークロード、モデルサイズ、年間トークン量から、Blackwell + TRT-LLMスタックがNVIDIAロックに値するかを決定する。

## 演習

1. `code/main.py`を実行する。アクティブパラメーター30%の120B MoEで、H100 BF16、H100 FP8、B200 NVFP4/FP8のメモリ帯域幅制限デコードスループットを計算する。最大のジャンプはどこか？
2. 顧客がH100 + vLLMに年間200万ドルを費やしている。7倍の経済的差を考慮して、12ヶ月でTRT-LLMへの移行を償却するために購入が必要なBlacwellGPUの損益分岐数を計算する。
3. NVFP4ウェイト変換後にMATHで精度が3ポイント下がるのを見る。2つの回復パスを挙げる：品質ファーストのもの（FP8ウェイトを維持）とコストファーストのもの（ドメイン内データでキャリブレート）。
4. MLPerf v6.0の推論結果を読む。BlackwellのHopperに対する差が最も小さいタスクはどれで、なぜか？
5. NVFPウェイト + FP8 KVキャッシュ（128kコンテキスト）の405BモデルのHBM必要量を計算する。単一のGB200 NVL72ノードに収まるか？

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|----------------|------------|
| FP8 | 「8ビット浮動小数点」 | 8ビット浮動小数点。ダイナミックレンジのためKVキャッシュとアテンションに使用 |
| NVFP4 | 「4ビットマイクロ」 | NVIDIAの4ビットマイクロスケーリングFP形式。Blackwellでウェイトとアクティベーションに |
| MXFP8 | 「MXエイト」 | マイクロスケーリングFP8バリアント。BlackwellテンソルコアでハードウェアアクセラレートΑ |
| デイ0 FP4 | 「FP4ウェイト出荷」 | モデルプロバイダーがすでにFP4のウェイトをリリース。ポストトレーニング変換ステップなし |
| MTP | 「マルチトークン予測」 | TRT-LLMの統合された投機的デコードドラフト（フェーズ17·05） |
| 分散配信 | 「プリフィル/デコード分割」 | 別のGPUプール上のプリフィルとデコード。NVLink/IB経由でKV転送 |
| All-to-all | 「MoEエキスパート通信」 | トークンをエキスパートGPUにルーティングする通信パターン。NVLink 5が3倍削減 |
| InferenceX | 「SemiAnalysis推論ベンチ」 | 2026年の業界で認められたトークン単価ベンチマーク |

## 参考資料

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026年4月MLPerf結果
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 all-to-allとMoEカーネル
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) — 公式エンジンドキュメント
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — TRT-LLM上の分散オーケストレーション
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — Blackwell数値を公開するベンチマークスイート
