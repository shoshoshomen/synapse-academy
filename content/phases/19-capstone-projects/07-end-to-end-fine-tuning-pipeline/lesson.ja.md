# キャップストーン 07 — エンドツーエンドファインチューニングパイプライン（データからSFT、DPOを経てサービングまで）

> 自分自身のデータでトレーニングされた8Bモデル、自分自身の好みでDPOアラインメントされ、量子化され、投機的デコードされ、測定可能な$/1Mトークンでサービングされる。2026年のオープンスタックは、Axolotl v0.8、TRL 0.15、反復用のUnsloth、量子化用のGPTQ/AWQ/GGUF、サービング用のEAGLE-3付きvLLM 0.7だ。このキャップストーンはパイプライン全体を再現可能に実行すること — YAMLを入力として、提供済みエンドポイントを出力として — そして2026年のモデル開放性フレームワーク（MOF）の下でモデルカードを公開することだ。

**演習するフェーズ:** P2 · P3 · P7 · P10 · P11 · P17 · P18

## 問題

2026年のすべての真剣なAIチームはファインチューニングパイプラインを手元に持っている。フロンティアベースモデルを出荷するからではなく、下流の適応（ドメインSFT、ラベル付きの好みに対するDPO、投機的デコード用の蒸留ドラフト、EAGLE-3を使ったサービング）が測定可能な成果が生まれる場所だからだ。Axolotl v0.8はマルチGPU SFTの設定を処理する。TRL 0.15はDPOとGRPOを処理する。Unslothは高速な単一GPU反復を提供する。vLLM 0.7とEAGLE-3は品質損失なしにデコードスループットを2-3倍向上させる。ツールは機能する；技術はYAML・データの清潔さ・評価の規律にある。

8Bベース（Llama 3.3、Qwen3、またはGemma 3）をタスク固有のデータでSFTしてDPOにかけ、サービングのために量子化し、lm-evaluation-harness・RewardBench-2・MT-Bench-v2・MMLU-Proに対するゲインを測定する。2026年のMOFの下でモデルカードを作成する。要点は再現可能性 — 1つのコマンドでパイプライン全体をエンドツーエンドで再実行できる。

## コンセプト

パイプラインには5つのステージがある。**データ**：重複排除（MinHash / Datatrove）、品質フィルター（Nemotron-CCスタイルの分類器）、PII除去、公開ベンチマーク汚染に対するスプリット健全性チェック。**SFT**：Axolotl YAML、8xH100でZeRO-3、コサインスケジュール、パックされたシーケンス、2-3エポック。**DPOまたはGRPO**：TRL設定、1エポック、人間またはモデルがラベル付けした好みのペア、ベータのチューニング。**量子化**：デプロイの柔軟性のためのGPTQ + AWQ + GGUF。**サービング**：EAGLE-3の投機的ヘッドを持つvLLM 0.7（またはSpecForge付きSGLang）、K8sデプロイメント、キュー待ちでのHPA。

アブレーションが成果物だ：3つのタスク固有のベンチマークでSFTのみ対SFT+DPO対SFT+GRPOの比較。サービングメトリクス：バッチ1/8/32でのトークン/秒、EAGLE-3の受理率、$/1Mトークン。安全評価：Llama Guard 4の合格率。モデルカード：バイアス評価、再現可能性シード、データライセンス。

## アーキテクチャ

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## スタック

- データ: 重複排除用Datatrove、品質用Nemotron-CC分類器、PII用Presidio
- ベース: Llama 3.3 8B、Qwen3 14B、またはGemma 3 12B
- SFT: ZeRO-3・Flash Attention 3・パックされたシーケンスを持つAxolotl v0.8
- 好みチューニング: DPOまたはGRPO用TRL 0.15；単一GPU反復用Unsloth
- 量子化: GPTQ（Marlin）、AWQ、llama.cpp経由のGGUF
- サービング: EAGLE-3投機的デコード付きvLLM 0.7（またはSpecForge付きSGLang 0.4）
- 評価: lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- 安全評価: Llama Guard 4、ShieldGemma-2
- インフラ: Kubernetes + NVIDIAデバイスプラグイン、キュー待ちメトリクスでのHPA
- 可観測性: トレーニング用W&B、推論用Langfuse

## 実装する

1. **データパイプライン。** 生コーパスにDatatroveの重複排除を実行する。Nemotron-CCスタイルの品質分類器を適用する。PresidioがPIIを除去する。明示的なシードでtrain/valスプリットを書く。

2. **汚染チェック。** すべてのバリデーションスプリットについて、MMLU-Pro・MT-Bench-v2・RewardBench-2のテストセットに対してMinHashを計算する。重複があれば拒否する。

3. **Axolotl SFT。** ZeRO-3・FA3・シーケンスパッキングを持つYAML。8xH100で2-3エポック。W&Bにログ。

4. **TRL DPO / GRPO。** SFTチェックポイントを取り、好みのペアで1エポックのDPOを実行する（または数学/コードの検証可能な報酬でGRPO）。ベータをスウィープする。

5. **量子化。** 3つのクォントを生成する：GPTQ-INT4-Marlin、AWQ-INT4、llama.cpp用GGUF-Q4_K_M。サイズと名目スループットを記録する。

6. **投機的デコードでサービング。** Red Hat SpeculatorsでトレーニングされたEAGLE-3ドラフトヘッドを持つvLLM 0.7設定。バッチ1/8/32での受理率とテールレイテンシーを測定する。同じ評価で$/1Mトークン対Anthropic/OpenAIをレポートする。

7. **評価マトリックス。** ベース・SFTのみ・SFT+DPO・SFT+GRPOに対してlm-eval-harness・RewardBench-2・MT-Bench-v2・MMLU-Proを実行する。テーブルを作成する。

8. **安全評価。** 開発セットでのLlama Guard 4の合格率。ShieldGemma-2の出力フィルター。

9. **モデルカード。** MOF 2026テンプレート：データ、トレーニング、評価、安全性、ライセンス、YAMLとコミットSHAを持つ再現可能性セクション。

## 使ってみる

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## 成果物を出す

`outputs/skill-finetuning-pipeline.md`は成果物を説明する。単一コマンドがデータをSFTからDPOから量子化からサービングから評価まで実行し、モデルカードと提供済みエンドポイントを出力する。

| 配点 | 基準 | 測定方法 |
|:-:|---|---|
| 25 | ベースに対する評価デルタ | ターゲットタスクでの測定ゲイン（MMLU-Pro、MT-Bench-v2、タスク固有） |
| 20 | パイプラインの再現可能性 | 同一シードでエンドツーエンドを再実行する1コマンド |
| 20 | データの清潔さ | 重複排除率、PII除去カバレッジ、汚染チェックの緑色 |
| 20 | サービング効率 | bs=1/8/32でのトークン/秒、EAGLE-3受理率、$/1Mトークン |
| 15 | モデルカード + 安全評価 | 2026 MOFの完全性 + Llama Guard 4の合格率 |
| **100** | | |

## 演習

1. 同じタスク固有のベンチマークでSFTのみ対SFT+DPO対SFT+GRPOを実行する。どの好みメソッドが勝つかとどのくらい差があるかをレポートする。

2. Llama 3.3 8BをQwen3 14Bに交換する。同じ品質での$/1Mトークンを測定する。

3. ドメインデータとジェネリックなShareGPTでのEAGLE-3の受理率を測定する。デルタと、それがレイテンシー予算に何を意味するかをレポートする。

4. トレーニングデータに1%の汚染を注入（MMLU-Proの答えをトレーニングデータに漏洩させる）して評価を再実行する。MMLU-Proの精度が非現実的に上昇するのを確認する。これを捉える汚染チェックCIゲートを構築する。

5. フルファインチューニングの代わりにLoRA SFTを追加する。メモリが10倍少ない場合の品質ギャップを測定する。

## キーワード

| 用語 | 一般的な呼び方 | 実際の意味 |
|------|-----------------|------------------------|
| Axolotl | 「SFTトレーナー」 | SFT・DPO・蒸留のための統一YAMLドリブントレーナー |
| TRL | 「好みチューナー」 | LLMでのDPO・GRPO・PPO用のHugging Faceライブラリ |
| GRPO | 「グループ相対ポリシー最適化」 | 検証可能な報酬を使ったDeepSeek R1のRLレシピ |
| EAGLE-3 | 「投機的デコードのドラフト」 | ターゲットモデルの隠れ状態でトレーニングされたドラフトヘッド；vLLMがターゲットモデルで検証 |
| MOF | 「モデル開放性フレームワーク」 | データ・コード・ライセンスでのモデルリリースのグレーディングの2026年標準 |
| 汚染チェック | 「スプリット健全性」 | テストセットのトレーニングへの漏洩のMinHashベースの検出 |
| 受理率 | 「EAGLE / MTPメトリクス」 | ターゲットモデルが受け入れるドラフトトークンの割合 |

## 参考資料

- [Axolotlドキュメント](https://axolotl-ai-cloud.github.io/axolotl/) — リファレンスSFT / DPOトレーナー
- [TRLドキュメント](https://huggingface.co/docs/trl) — DPOとGRPOのリファレンス実装
- [Unsloth](https://github.com/unslothai/unsloth) — 単一GPU反復のリファレンス
- [DeepSeek R1論文（arXiv:2501.12948）](https://arxiv.org/abs/2501.12948) — GRPOの方法論
- [vLLM + EAGLE-3ドキュメント](https://docs.vllm.ai) — リファレンスサービングスタック
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 代替投機的デコードトレーナー
- [モデル開放性フレームワーク 2026](https://isocpp.org/) — オープンリリースグレーディング標準
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — カノニカル評価ランナー
