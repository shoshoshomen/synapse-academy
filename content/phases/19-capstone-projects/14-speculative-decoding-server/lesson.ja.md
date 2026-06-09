# キャップストーン 14 — 投機的デコーディング推論サーバー

> vLLM 0.7のEAGLE-3が実際のトラフィックで2.5-3倍のスループットを達成する。P-EAGLE（AWS 2026）が並行投機をさらに推し進めた。SGLangのSpecForgeがスケールでドラフトヘッドを学習させた。Red HatのSpeculatorsハブが一般的なオープンモデル向けのアライメントされたドラフトを公開した。TensorRT-LLMがNVIDIA上で投機的デコーディングをファーストクラスにした。2026年の本番サービングスタックはEAGLE系ドラフト・FP8またはINT4量子化・キュー待ちでのHPAを持つvLLMまたはSGLangだ。このキャップストーンは2つのオープンモデルをベースラインの2.5倍以上のスループットで提供し、完全なテールレイテンシレポートを作成することだ。

**演習するフェーズ:** P3 · P7 · P10 · P17

## 問題

投機的デコーディングは2026年にコモディティになった。EAGLE-3のドラフトヘッドはターゲットモデルの隠れ状態で学習し、N個のトークンを先読みして予測する；ターゲットモデルは1回のパスで検証する。60-80%のアクセプタンスレートはエンドツーエンドで2-3倍のスループットに相当する。vLLM 0.7がこれをネイティブに統合する。SGLang + SpecForgeが学習パイプラインを提供する。Red Hat's SpeculatorsがLlama 3.3 70B・Qwen3-Coder-30B MoE・GPT-OSS-120Bのアライメントされたドラフトを公開する。

技術はモデルではなくサービング操作にある。アクセプタンスレートはトラフィック分布（ShareGPT対コード対ドメインデータ）によってドリフトする。リジェクション時のテールレイテンシは投機なしより悪い — 安定状態のtokens/secだけでなく、複数のバッチサイズでのp99を報告しなければならない。Anthropic/OpenAI APIと比較した1Mトークンあたりのコストが信頼性のレバーだ。

## コンセプト

投機的デコーディングには2つのレイヤーがある。**ドラフト**モデル（EAGLE-3ヘッド・ngram・またはより小さなターゲットアライメントモデル）がステップごとにk個の候補トークンを提案する。**ターゲット**モデルが1回のパスですべてのkを検証し；受理されたプレフィックスがグリーディパスを置き換える。アクセプタンスレートはドラフト-ターゲットのアライメントと入力分布に依存する。

EAGLE-3はほとんどのトラフィックでngramドラフトを上回る。P-EAGLEはより深いドラフトツリーのために並行投機を実行する。トレードオフ：リジェクション時のP99レイテンシは検証パスが大きいため高くなる。サービング設定はこれを表面化するためにバッチサイズ区分されたレイテンシを報告しなければならない。

デプロイはKubernetesだ。vLLM 0.7がGPUごとまたはテンソル並行シャードごとに1つのレプリカを実行する。HPAはCPUではなくキュー待ちでオートスケールする。FP8（Marlin）とINT4（AWQ）量子化がGPUメモリをH100/H200のエンベロープ内に収める。エンドツーエンドのレポートはスループット・アクセプタンスレート・バッチ1/8/32でのp50/p99・$/1Mトークンだ。

## アーキテクチャ

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## スタック

- サービング: vLLM 0.7またはSGLang 0.4
- 投機的手法: EAGLE-3ドラフトヘッド・P-EAGLE並行投機・ngramフォールバック
- ドラフト学習: SpecForge（SGLang）またはRed Hat Speculators
- ターゲットモデル: Llama 3.3 70B・Qwen3-Coder-30B MoE・GPT-OSS-120B
- 量子化: FP8（Marlin）・INT4 AWQ
- デプロイ: Kubernetes + NVIDIAデバイスプラグイン；キュー待ちメトリクスでのHPA
- 評価: ShareGPT・MT-Bench-v2・GSM8K・HumanEvalでドメイン分散アクセプタンス計測
- リファレンス: ベンダーベースラインとしてのTensorRT-LLM投機的デコーディング

## 実装する

1. **ターゲットモデル準備。** Llama 3.3 70Bを選ぶ。MarlinでFP8に量子化する。vLLM 0.7を1xH100（または2x tensor-parallel）でデプロイする。

2. **ドラフトソース。** Red Hat SpeculatorsからアライメントされたEAGLE-3ドラフトヘッドを取得する（またはSpecForgeで学習させる）。vLLMの投機的デコーディング設定にロードする。

3. **ベースライン数値。** 投機前：バッチ1/8/32でのtokens/s・p50/p99レイテンシ・GPU利用率。公開する。

4. **EAGLE-3を有効化する。** 設定を切り替え；同じベンチマークを再実行する。スピードアップ・アクセプタンスレート・p99テールレイテンシのデルタを報告する。

5. **P-EAGLE。** 並行投機を有効化し；より深いドラフトツリー対シリアルEAGLE-3を計測する。P-EAGLEが有効対無効な変曲点を報告する。

6. **ドメイントラフィック。** ShareGPT対HumanEval対ドメイン固有のトラフィックを同じサーバーで実行する。分布ごとのアクセプタンスレートを計測する。ドラフトがドリフトする時を特定する。

7. **2番目のターゲットモデル。** 同じパイプラインをQwen3-Coder-30B MoEで実行する。ドラフトはより難しい（MoEルーティングノイズ）。報告する。

8. **K8s HPA。** `queue_wait_ms`を追跡するHPAを持つK8sでデプロイする。負荷が3倍になった時のスケールアウトを実証する。

9. **コスト比較。** 同じ評価でAnthropic Claude Sonnet 4.7とOpenAI GPT-5.4対の$/1Mトークンを計算する。公開する。

## 使ってみる

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## 成果物を出す

`outputs/skill-inference-server.md`が成果物を説明する。投機的デコーディングを持つ計測されたサービングスタック・完全なベンチマークレポート・K8sデプロイメント。

| 配点 | 基準 | 測定方法 |
|:-:|---|---|
| 25 | ベースライン対の計測されたスピードアップ | 2つのモデルで同等品質における2.5倍以上のスループット |
| 20 | 現実的なトラフィックでのアクセプタンスレート | 分布ごとのアクセプタンスレートレポート |
| 20 | P99テールレイテンシの規律 | 投機ありなしでバッチ1/8/32でのp99 |
| 20 | 運用 | K8sデプロイ・キュー待ちでのHPA・スムーズなロールアウト |
| 15 | レポートと方法論 | 何が変わりなぜかの明確な説明 |
| **100** | | |

## 演習

1. ドラフトがターゲットより1バージョン遅れている場合のアクセプタンスレート低下を計測する（例：Llama 3.3 -> 3.4ドリフト）。監視アラートを構築する。

2. ngramフォールバックを実装する：EAGLE-3のアクセプタンスがしきい値以下に落ちた場合、ngramドラフトに切り替える。信頼性の改善を報告する。

3. 制御されたMoE実験を実行する：ルーティングノイズを注入したQwen3-Coder-30B対なし。ドラフトアクセプタンスの感度を計測する。

4. H200（141 GB）に拡張する。得られたモデルサイズあたりのレプリカヘッドルームと量子化なしのLlama 3.3 70Bが提供できるかを報告する。

5. 同じH100ハードウェアでTensorRT-LLM投機的デコーディングをベンチマークする。vLLMに対して勝る場所を報告する。

## キーワード

| 用語 | 一般的な呼び方 | 実際の意味 |
|------|-----------------|------------------------|
| ドラフトモデル | 「スペキュレーター」 | ターゲットが検証するN個のトークンを提案する小さなモデル |
| EAGLE-3 | 「2026年ドラフトアーキテクチャ」 | ターゲットの隠れ状態で学習したドラフトヘッド；アクセプタンス約75% |
| P-EAGLE | 「並行投機」 | 1回のターゲットパスで検証されるドラフトブランチのツリー |
| アクセプタンスレート | 「ヒット率」 | リサンプリングなしで受理されたドラフトトークンの割合 |
| 量子化 | 「FP8 / INT4」 | GPUメモリにより多くのモデルを収めるための低精度重み |
| キュー待ち | 「HPAメトリクス」 | 推論開始前にリクエストがペンディングキューで待つ時間 |
| Speculatorsハブ | 「アライメントされたドラフト」 | 一般的なオープンモデル向けEAGLEドラフトのRed Hat Neural Magicハブ |

## 参考資料

- [vLLM EAGLEおよびP-EAGLEドキュメント](https://docs.vllm.ai) — リファレンスサービングスタック
- [P-EAGLE（AWS 2026）](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — 並行投機的デコーディングの論文+統合
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — ドラフトヘッド学習パイプライン
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — アライメントされたドラフトハブ
- [TensorRT-LLM投機的デコーディング](https://nvidia.github.io/TensorRT-LLM/) — ベンダー代替
- [Fireworks.aiサービングアーキテクチャ](https://fireworks.ai/blog) — 商用リファレンス
- [EAGLE-3論文（arXiv:2503.01840）](https://arxiv.org/abs/2503.01840) — 手法論文
- [vLLMリポジトリ](https://github.com/vllm-project/vllm) — コードとベンチマーク
