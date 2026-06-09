# エッジ推論 — Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson

> エッジの中核的な制約はコンピュートではなくメモリ帯域幅だ。モバイルDRAMは50〜90 GB/s；データセンターHBM3は2〜3 TB/sをクリア — 30〜50倍の差。デコードはメモリバウンドなのでこの差は決定的だ。2026年のランドスケープは4つに分かれる。Apple M4/A18 Neural Engineは38 TOPSを達成し統合メモリを持つ（CPU↔NPUコピーなし）。Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagonは45 TOPSを達成する。WebGPU + WebLLMはM3 MaxでLlama 3.1 8B（Q4）を約41 tok/sで実行する（ネイティブの概ね70〜80%）；GitHubスター17.6k、OpenAI互換API、モバイルカバレッジ約70〜75%。NVIDIA Jetson Orin Nano Super（8GB）はLlama 3.2 3B / Phi-3に適合；AGX OrinはvLLM経由でgpt-oss-20bを約40 tok/sで実行；Jetson T4000（JetPack 7.1）はAGX Orinの2倍。TensorRT Edge-LLMはEAGLE-3、NVFP4、chunked prefillをサポート — CES 2026でBosch、ThunderSoft、MediaTekが発表。


## 学習目標

- モバイルLLM推論がメモリ帯域幅バウンドでコンピュートが二次的な理由を説明する。
- 4つのエッジターゲット（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson）を列挙し各々をユースケースとマッチングする。
- 2026年のWebGPUカバレッジギャップ（Firefox Androidが追いつき中）とSafari iOS 26のリリースを挙げる。
- ターゲットごとに量子化フォーマットを選ぶ（ANEはCore ML INT4+FP16、HexagonはQNN INT8/INT4、ブラウザはWebGPU Q4、Jetson ThorはNVFP4）。

## 問題

顧客がオンデバイスチャットボットを求めている：音声ファースト、プライバシーデフォルト、オフライン動作。MacBook Pro M3 MaxでLlama 3.1 8B Q4は約55 tok/sで動作 — 問題なし。iPhone 16 Proで同じモデルは3 tok/s — 問題あり。Snapdragon 8 Gen 3搭載ミッドレンジAndroidで7 tok/s。Chrome Android v121+経由のWebGPUでブラウザから4〜8 tok/s（デバイスによる）。

スループットの差はポーティング問題ではない。帯域幅の差×量子化フォーマット×NPUがユーザースペースからアクセス可能かどうかの掛け算だ。2026年のエッジ推論は4つの異なる問題に4つの異なる解決策がある。

## コンセプト

### 帯域幅が真の上限

デコードはすべてのトークンに対してウェイトのフルセットを読み込む。Q4の7Bモデルは3.5 GB。50 GB/sで3.5 GBを読むのに70ms — 理論上の上限は約14 tok/s。90 GB/s（ハイエンドモバイルDRAM）では上限は約25 tok/sに移動する。この数値以下ではどれだけコンピュートを追加しても助けにならない。

データセンターのHBM3（3 TB/s）では同じ3.5 GBを1.2msでクリア — 上限は830 tok/s。同じモデル、同じウェイト。異なるメモリサブシステム。

### Apple Neural Engine（M4 / A18）

- 最大38 TOPS。統合メモリ（CPUとANEが同じプールを共有）— コピーオーバーヘッドなし。
- Core ML + `.mlmodel`コンパイル済みモデル経由、またはMetal Performance Shaders（MPS）経由のPyTorchでアクセス。
- Llama.cpp MetalバックエンドはANEではなくMPSを使用；ネイティブANEにはCore ML変換が必要。
- 2026年のiOSアプリの最良の実用的パス：INT4ウェイト+FP16アクティベーションのCore ML。

### Qualcomm Hexagon（Snapdragon X Elite / 8 Gen 4）

- 最大45 TOPS。SoC内でCPUとGPUと統合されているが別のメモリドメインを持つ。
- QNN（Qualcomm Neural Network）SDKとAI HubがPyTorch/ONNXからの変換を提供。
- チャットテンプレート、Llama 3.2、Phi-3はAI Hubでファーストクラスのアーティファクトとして提供される。

### Intel / AMD NPU（Lunar Lake、Ryzen AI 300）

- 40〜50 TOPS。ソフトウェアはApple/Qualcommに遅れ；OpenVINOは改善中だがニッチ。
- Windows ARMコパイロットアプリに最適；AMD/Intelデスクトップでのローカルファーストに対してネイティブ。

### WebGPU + WebLLM

- WebGPUコンピュートシェーダー経由でブラウザ内でモデルを実行；インストール不要。
- M3 MaxでLlama 3.1 8B Q4が約41 tok/s — 同じバックエンド経由でネイティブの概ね70〜80%。
- WebLLMは17.6kのGitHubスター；OpenAI互換JS API；Apache 2.0。
- 2026年のカバレッジ：Chrome Android v121+、Safari iOS 26 GA、Firefox Androidはまだ追いつき中。全体的にモバイルカバレッジ約70〜75%。

### NVIDIA Jetsonファミリー

- Orin Nano Super（8GB）：Llama 3.2 3B、Phi-3が良好なtok/sで適合する。
- AGX Orin：vLLM経由でgpt-oss-20bを約40 tok/sで実行。
- Thor / T4000（JetPack 7.1）：AGX Orinの2倍のパフォーマンス、EAGLE-3とNVFP4をサポート。
- TensorRT Edge-LLM（2026）はEAGLE-3投機的デコード、NVFP4ウェイト、chunked prefillをサポート — データセンターの最適化をエッジに移植。

### ターゲットごとの量子化選択

| ターゲット | フォーマット | 備考 |
|---------|--------|-------|
| Apple ANE | INT4ウェイト + FP16アクティベーション | Core ML変換パス |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hubコンバーター |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | `mlc_llm convert_weight`+コンパイル済み`.wasm`を使用；GGUFはサポートされない |
| Jetson Orin Nano | Q4 GGUFまたはTRT-LLM INT4 | メモリバウンド |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLMパス |

### エッジでの長コンテキストの罠

Llama 3.1の128Kコンテキストはデータセンター向けの機能だ。8GB RAMのスマートフォンでは、4GBのモデル+32Kトークン用の2GB KVキャッシュ+OSオーバーヘッド=OOMになる。エッジデプロイメントでは積極的なKV量子化（Q4 KV）を受け入れない限り、コンテキストを4K〜8Kに維持する。

### 音声がキラーアプリ

音声エージェントはレイテンシに敏感だ（最初のトークン < 500ms）。ローカル推論はネットワークレイテンシを完全に排除する。音声認識（Whisper Turboバリアントはエッジで動作）と組み合わせると、エッジ推論は本番品質の音声ループになる。

### 覚えておくべき数値

- Apple M4 / A18 ANE：38 TOPS。
- Qualcomm Hexagon SD X Elite：45 TOPS。
- WebLLM M3 Max：Llama 3.1 8B Q4で約41 tok/s。
- AGX Orin：vLLM経由でgpt-oss-20bを約40 tok/s。
- データセンター対エッジの帯域幅差：30〜50倍。
- WebGPUモバイルカバレッジ：約70〜75%（Firefox Androidが遅れ）。

## 使ってみる

`code/main.py`はエッジターゲット全体で帯域幅バウンド数学から理論上のデコードスループット上限を計算する。観測されたベンチマークと比較し、帯域幅がコンピュートでなくボトルネックになっている箇所を強調する。

## 成果物を出す

このレッスンは`outputs/skill-edge-target-picker.md`を作成する。プラットフォーム（iOS/Android/ブラウザ/Jetson）、モデル、レイテンシ/メモリ予算を与えると量子化フォーマットと変換パイプラインを選択する。

## 演習

1. `code/main.py`を実行する。Snapdragon 8 Gen 3（約77 GB/s帯域幅）のQ4の7Bモデルのデコード上限を計算する。観測された6〜8 tok/sと比較する — ランタイムは効率的か？
2. AndroidでのWebGPUにはChrome v121+が必要だ。古いブラウザ用のフォールバックを設計する — 同じOpenAI互換APIを介したサーバーサイド。
3. iOSアプリは4Kコンテキストストリーミングが必要だ。iPhone 16で4GBのアクティブメモリ以下に収まるモデル/フォーマットの組み合わせはどれか？
4. Jetson AGX OrinはGPT-oss-20bを40 tok/sで実行する。Jetson Nanoは3Bしか適合しない。製品が両方をターゲットとする場合、推論スタックをどのように統一するか？
5. 「WebLLMは2026年に本番対応か」を論じる。カバレッジ、パフォーマンス、Firefox Androidギャップを引用する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|----------------|------------|
| ANE | 「Apple neural engine」 | Mシリーズ・Aシリーズのオンデバイスシリコン；統合メモリ |
| Hexagon | 「Qualcomm NPU」 | Snapdragon NPU；アクセスにはQNN SDKを使用 |
| WebGPU | 「ブラウザGPU」 | W3C標準化ブラウザGPU API；Chrome/Safari 2026 |
| WebLLM | 「ブラウザLLMランタイム」 | MLC-LLMプロジェクト；Apache 2.0；OpenAI互換JS |
| Jetson | 「NVIDIAエッジ」 | Orin Nano / AGX / Thor / T4000ファミリー |
| TRT Edge-LLM | 「エッジTensorRT」 | 2026年エッジポートのTensorRT-LLM；EAGLE-3+NVFP4 |
| 統合メモリ | 「共有プール」 | CPUとNPUが同じRAMを参照；コピーオーバーヘッドなし |
| 帯域幅バウンド | 「メモリ制限」 | ウェイトを読む秒あたりバイト数にゲートされたデコード |
| Core ML | 「Apple変換」 | ANEネイティブモデル用のAppleフレームワーク |
| QNN | 「Qualcommスタック」 | Qualcomm Neural Network SDK |

## 参考資料

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — ランドスケープとベンチマーク
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026年エッジポート発表
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — 設計とベンチマーク
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANEネイティブ変換
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — Hexagon用事前変換モデル
