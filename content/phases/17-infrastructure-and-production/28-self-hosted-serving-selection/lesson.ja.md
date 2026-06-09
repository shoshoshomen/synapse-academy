# セルフホスティング推論エンジン選定 — llama.cpp、Ollama、TGI、vLLM、SGLang

> 2026年のセルフホスティング推論では4つのエンジンが主流だ。ハードウェア、スケール、エコシステムに基づいて選択する。**llama.cpp**はCPUで最速——最も広いモデルサポート、量子化とスレッドの完全な制御。**Ollama**は開発ラップトップでの1コマンドインストール、llama.cppより約15〜30%遅い（Go＋CGo＋HTTPシリアライゼーション）、本番相当の負荷では3倍のスループット差がある。**TGIは2025年12月11日にメンテナンスモードに入った**——バグ修正のみ、vLLMよりも生のスループットが約10%遅いが歴史的には最高のオブザーバビリティとHFエコシステム統合。そのメンテナンスステータスにより新規プロジェクトでは長期的なリスクがある——新規プロジェクトにはSGLangかvLLMの方が安全なデフォルト。**vLLM**は汎用的な本番デフォルト——v0.15.1（2026年2月）でPyTorch 2.10、RTX Blackwell SM120、H200最適化を追加。**SGLang**はエージェンティックなマルチターン/プレフィックスヘビーのスペシャリスト——本番で40万以上のGPU（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。ハードウェア制約：CPUのみ→llama.cppのみ。AMD/非NVIDIA→vLLMのみ（TRT-LLMはNVIDIAロック）。2026年のパイプラインパターン：開発=Ollama、ステージング=llama.cpp、本番=vLLMまたはSGLang。同じGGUF/HFウェイトを通じて使用。


## 学習目標

- ハードウェア（CPU/AMD/NVIDIA Hopper/Blackwell）、スケール（1ユーザー/100/10,000）、ワークロード（一般チャット/エージェント/長コンテキスト）に応じてエンジンを選ぶ。
- 2026年のTGIメンテナンスモードのステータス（2025年12月11日）と新規プロジェクトでvLLMまたはSGLangが推奨される理由を述べる。
- 同じGGUFまたはHFウェイトを使い続ける開発/ステージング/本番パイプラインを説明する。
- 「CPUのみ」がllama.cppを強制し、「AMD」がTRT-LLMを除外する理由を説明する。

## 問題の背景

チームが新しいセルフホスティングLLMプロジェクトを始める。あるエンジニアはOllamaと言い、別のエンジニアはvLLMと言い、3人目のエンジニアは「TGIはすぐ使えるんじゃないの？」と言う。3人とも文脈によっては正しい。3人とも全ての文脈で正しいわけではない。

2026年では選択ツリーが重要だ：ハードウェア優先、スケール次、ワークロード3番目。そして2025年の特定のイベント——TGIが12月11日にメンテナンスモードに入ったこと——が新規プロジェクトのデフォルトを変える。

## コンセプト

### 5つのエンジン

| エンジン | 最適な用途 | 備考 |
|---------|---------|------|
| **llama.cpp** | CPU/エッジ/最小依存/最も広いモデルサポート | CPUで最速、完全な制御 |
| **Ollama** | 開発ラップトップ、1ユーザー、1コマンドインストール | llama.cppより15〜30%遅い；本番スループット差3倍 |
| **TGI** | HFエコシステム、規制産業 | **2025年12月11日にメンテナンスモード** |
| **vLLM** | 汎用本番、100ユーザー以上 | 広い本番デフォルト；v0.15.1 2026年2月 |
| **SGLang** | エージェンティックなマルチターン、プレフィックスヘビーワークロード | 本番で40万以上のGPU |

### ハードウェア優先の決定

**CPUのみ** → llama.cpp。Ollamaも動くが遅い。CPUでは他のエンジンは競争力がない。

**AMD GPU** → vLLM（AMD ROCmサポート）。SGLangも動く。TRT-LLMはNVIDIAロックなので除外。

**NVIDIA Hopper（H100/H200）** → vLLM、SGLang、TRT-LLMのいずれか。3つともトップレベル。

**NVIDIA Blackwell（B200/GB200）** → TRT-LLMがスループットリーダー（フェーズ17・07）。vLLMとSGLangが後に続く。

**Apple Silicon（Mシリーズ）** → llama.cpp（Metal）。OllamaはこれをラップしてPOV。

### スケール次の決定

**1ユーザー/ローカル開発** → Ollama。1コマンド、数秒で最初のトークン。

**10〜100ユーザー/小規模チーム** → vLLMシングルGPU。

**100〜1万ユーザー/本番** → vLLM本番スタック（フェーズ17・18）またはSGLang。

**1万以上のユーザー/エンタープライズ** → vLLM本番スタック＋分離（フェーズ17・17）＋LMCache（フェーズ17・18）。

### ワークロード3番目の決定

**一般チャット/Q&A** → 広いデフォルトでvLLMが勝る。

**エージェンティックなマルチターン（ツール、プランニング、メモリ）** → SGLangのRadixAttention（フェーズ17・06）が圧倒。

**プレフィックスの大量再利用があるRAG** → SGLang。

**コード生成** → vLLMで十分；SGLangはキャッシュで若干上。

**長コンテキスト（128K以上）** → vLLM＋チャンクプリフィル；SGLang＋ティアードKV。

### TGIのメンテナンストラップ

Hugging Face TGIは2025年12月11日にメンテナンスモードに入った——以降はバグ修正のみ。歴史的には：トップレベルのオブザーバビリティ、最高クラスのHFエコシステム統合（モデルカード、安全ツール）、生のスループットでvLLMよりわずかに遅い。

2026年の新規プロジェクト：デフォルトはTGI以外。既存のTGIデプロイメントは継続できるが最終的には移行すべき。SGLangとvLLMがより安全なデフォルト。

### パイプラインパターン

開発（Ollama）→ステージング（llama.cpp）→本番（vLLM）。同じGGUFまたはHFウェイトを通じて使用。エンジニアはラップトップで素早くイテレーション；ステージングは本番の量子化をミラーリング；本番がサービング対象。

### Ollamaの注意点

Ollamaは開発に最適だ。共有本番環境には向かない：Go HTTPシリアライゼーションがオーバーヘッドを追加し、同時実行管理がvLLMより単純で、OpenTelemetryサポートが遅れている。Ollamaが輝く場所——1ユーザー、1コマンド——で使い、共有環境ではvLLMに切り替える。

### セルフホストと管理型は別の決定

フェーズ17・01（管理型ハイパースケーラー）、・02（推論プラットフォーム）が管理型をカバーしている。このレッスンはセルフホストを決めた後を前提とする。セルフホストの理由：データ居住性、カスタムファインチューン、スケールでの総保有コスト、ホスト型で利用できないドメインモデル。

### 覚えておくべき数値

- TGIのメンテナンスモード：2025年12月11日。
- vLLM v0.15.1：2026年2月；PyTorch 2.10；Blackwell SM120サポート。
- SGLangの本番フットプリント：40万以上のGPU。
- llama.cppとのOllamaスループット差：15〜30%遅い；本番負荷では3倍。

## 使ってみる

`code/main.py`は決定ツリーウォーカー：ハードウェア＋スケール＋ワークロードを与えるとエンジンを選んでその理由を説明する。

## 成果物を出す

このレッスンでは`outputs/skill-engine-picker.md`を生成する。制約を考慮してエンジンを選び、移行計画を書く。

## 演習

1. `code/main.py`を自分のハードウェア/スケール/ワークロードで実行する。出力は直感と合っているか？
2. インフラが12台のH100と8台のMI300X AMDだ。どのエンジンか？TRT-LLMがなぜ選外なのか？
3. チームが「慣れているから」という理由で2026年もTGIを使いたがっている。移行の理由を主張する。
4. Ollama開発からvLLM本番へ：量子化、設定、オブザーバビリティで何が変わるか？
5. P99プレフィックス長8K、テナント間の再利用が多いRAG製品。エンジンを選んでフェーズ17・11＋18とスタックする。

## キーワード

| 用語 | よく言われる表現 | 実際の意味 |
|------|----------------|------------|
| llama.cpp | 「CPU版」 | 最も広いモデルサポート、CPUで最速 |
| Ollama | 「ラップトップ版」 | 1コマンドインストール、開発グレードのスループット |
| TGI | 「HFのサービング」 | 2025年12月からメンテナンスモード |
| vLLM | 「デフォルト」 | 2026年の広い本番ベースライン |
| SGLang | 「エージェンティック版」 | プレフィックスヘビー、RadixAttention |
| TRT-LLM | 「NVIDIAロック」 | Blackwellスループットリーダー、NVIDIAのみ |
| GGUF | 「llama.cppフォーマット」 | バンドルされたKクォントバリアント |
| Production-stack | 「vLLM K8s」 | フェーズ17・18リファレンスデプロイメント |
| Pipeline pattern | 「開発→ステージ→本番」 | 同じウェイトでOllama→llama.cpp→vLLM |

## 参考資料

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — リリースノート。
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)
