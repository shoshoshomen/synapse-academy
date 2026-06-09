# Kubernetes上のGPUオートスケーリング — Karpenter、KAI Scheduler、ギャングスケジューリング

> 3つの層があり、1つではない。Karpenterはノードを動的にプロビジョニングする（1分未満で、Cluster Autoscalerより40%速い）。KAI Schedulerはギャングスケジューリング、トポロジー認識、階層型キューを処理する — 7ノードが1枚不足のGPUを待ちながら無駄に消費する「8枚中7枚の部分割り当てトラップ」を防ぐ。アプリケーションレベルのオートスケーラー（NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler）はCPU/DCGMデューティサイクルではなく、推論固有のシグナル — キュー深さ、KVキャッシュ利用率 — でスケールする。古典的なHPAの罠は`DCGM_FI_DEV_GPU_UTIL`がデューティサイクル計測であること：100%は10リクエストかもしれないし100かもしれない。vLLMはKVキャッシュメモリを事前割り当てするため、メモリでスケールダウンがトリガーされることはない。このレッスンでは3つの層を組み合わせ、実行中のGPUジョブを終了させるKarpenterのデフォルト`WhenEmptyOrUnderutilized`ポリシーを回避する方法を教える。


## 学習目標

- 3つのオートスケーリング層（ノードプロビジョニング、ギャングスケジューリング、アプリケーションレベル）を図示し、各層で使用するツールを挙げる。
- vLLMに対して`DCGM_FI_DEV_GPU_UTIL`が誤ったHPAシグナルである理由を説明し、2つの代替（キュー深さ、KVキャッシュ利用率）を挙げる。
- ギャングスケジューリングとKAI Schedulerが防ぐ部分割り当て障害モード（8枚中7枚のGPUがアイドル）を説明する。
- 実行中のGPUジョブを終了させるKarpenterのコンソリデーションポリシー（`WhenEmptyOrUnderutilized`）を挙げ、2026年の安全な代替を述べる。

## 問題

チームがKubernetes上でLLM配信サービスをデプロイした。HPAのシグナルに`DCGM_FI_DEV_GPU_UTIL`を設定した。業務時間中にサービスが100%利用率で固まる。HPAはスケールアップしない — すでに満杯だと思っている。手動でレプリカを追加するとTTFTが下がる。HPAはまだスケールしない。シグナルが嘘をついている。

別途、ノードにはCluster Autoscalerを使っている。深夜2時に100万トークンのプロンプトが届く。クラスターはノードのプロビジョニングに3分かかり、リクエストがタイムアウトする。

さらに別途、8枚のGPUを2ノードにわたって要求する70Bモデルをデプロイする。クラスターには7枚のGPUと3ノードに分散した1枚がある。Cluster Autoscalerが不足の1枚GPUのためにノードをプロビジョニングする。7ノードが4分間、最後のGPUが立ち上がるのを待ちながら費用を消費する。

3つの層、3つの異なる障害モード。2026年のGPU対応オートスケーリングは「HPAをオンにする」ことではない。ノードプロビジョニング、ギャングスケジューリング、アプリケーションシグナルオートスケーリングを組み合わせることだ。

## コンセプト

### 第1層 — ノードプロビジョニング（Karpenter）

Karpenterはペンディングポッドを監視し、約45〜60秒でノードをプロビジョニングする（GPUノードではCluster Autoscalerが通常90〜120秒かかる）。`NodePool`制約に基づいてインスタンスタイプを動的に選択する — ポッドが8枚のH100を必要とし、クラスターに一致するノードがなければ、既存グループをスケールする代わりに直接プロビジョニングする。

**コンソリデーションの罠**：KarpenterのデフォルトのコンソリデーションポリシーはGPUプールには危険だ `consolidationPolicy: WhenEmptyOrUnderutilized`。実行中のGPUノードを終了させ、ポッドをより安価で適切なサイズのインスタンスに移行しようとする。推論ワークロードでは実行中のリクエストを退去させ、新しいノード上で70Bモデルをリロードすることを意味する。損失は数分のキャパシティとリクエスト失敗だ。

GPUプール向けの安全な設定：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

これによりKarpenterは1時間後に真に空のノードをコンソリデートできるが、実行中のジョブは退去させない。

### 第2層 — ギャングスケジューリング（KAI Scheduler）

KAI Scheduler（プロジェクト「Karp」後に改名）はデフォルトのkube-schedulerが処理しないものを扱う：

**ギャングスケジューリング** — オールオアナッシングでスケジュール。8枚のGPUを要求する分散推論ポッドは、8枚すべてが一緒に起動するか、どれも起動しないかのどちらかだ。これがないと部分割り当てトラップが発生する：7枚中8枚のポッドが起動し、無期限に待ち、費用を消費する。

**トポロジー認識** — どのGPUがNVLinkを共有し、どれが同じラックにあり、どれの間にInfiniBandがあるかを把握する。それに応じてポッドを配置する。DeepSeek-V3 67Bテンソル並列ワークロードは1つのNVLinkドメインに留まらなければならない。KAI Schedulerはそれを尊重する。

**階層型キュー** — 複数のチームが優先度とクォータを持ちながら同じGPUプールを競い合う。チームAの本番ピンチがチームBのトレーニングジョブにプリエンプトされるのは、優先度ルールが許可する場合のみだ。

KAIはkube-schedulerの隣にセカンダリスケジューラーとしてデプロイされる。ワークロードにアノテーションを付けて使用する。RayとvLLMの本番スタックは両方とも統合されている。

### 第3層 — アプリケーションレベルのシグナル

**HPAの罠**：`DCGM_FI_DEV_GPU_UTIL`はデューティサイクルメトリクス — 各サンプリング間隔でGPUが作業していたかどうかを測定する。100%の利用率は10の同時リクエストか100を意味するかもしれない。GPUはどちらの場合も忙しかった。デューティサイクルでのスケーリングは盲目的なスケーリングだ。

さらに悪いことに、vLLMと類似エンジンはKVキャッシュメモリを事前割り当てする（`--gpu-memory-utilization`まで）。メモリ使用量は1リクエストでも90%近くに留まる。メモリベースのHPAはスケールダウンしない。

**2026年の代替シグナル**：

- キュー深さ（プリフィルを待っているリクエスト数）
- KVキャッシュ利用率（アクティブなシーケンスに割り当てられているブロックの割合）
- レプリカごとのP99 TTFT（SLAシグナル）
- グッドプット（秒あたりすべてのSLOを満たしているリクエスト数）

NVIDIA Dynamo PlannerとllmードWorkload Variant Autoscalerがこれらのシグナルを消費してレプリカをスケールする。これらはLLM配信のHPAを完全に置き換える。

### いつ何を使うか

| スケール判断 | ツール |
|-------------|-------|
| ノードの追加/削除 | Karpenter |
| マルチGPUジョブのスケジュール | KAI Scheduler |
| レプリカの追加/削除 | Dynamo Planner / llm-d WVA（またはキュー深さでのカスタムHPA） |
| GPUタイプの選択 | Karpenter NodePool |
| 低優先度をプリエンプト | KAI Schedulerキュー |

### 分散プリフィル/デコードがすべてを複雑にする

分散プリフィル/デコードを実行する場合（フェーズ17·17）、異なるスケーリングトリガーを持つ2つのポッドクラスがある：プリフィルポッドはキュー深さでスケールし、デコードポッドはKVキャッシュ圧力でスケールする。llm-dはこれらをロールごとのHPAを持つ独立した`Services`として公開する。両方の前に単一のHPAを置こうとしないこと。

### コールドスタートもここで重要

コールドスタート軽減（フェーズ17·10）は、ノードプロビジョニング時間がユーザーに見える場面だ。Karpenterの45〜60秒のウォームアップ + 20GBモデルロード + エンジン初期化は、ゼロからのリクエストが2〜5分かかることを意味する。SLOクリティカルなパスには温かいプール（`min_workers=1`）を維持するか、アプリケーション層でModalスタイルのチェックポイントを使用する。

### 覚えておくべき数値

- Karpenterのノードプロビジョニング：約45〜60s vs Cluster Autoscalerの約90〜120s（GPUノード）
- KAI Schedulerが部分割り当ての無駄を防ぐ — 8枚中7枚のトラップ
- HPAシグナルとしての`DCGM_FI_DEV_GPU_UTIL`：壊れている。キュー深さかKV利用率を使う
- Karpenterの`WhenEmptyOrUnderutilized`：実行中のGPUジョブを終了させる。推論には`WhenEmpty + consolidateAfter: 1h`を使う

## 使ってみる

`code/main.py`はバースト的なGPUワークロードで3層オートスケーラーをシミュレートする。ナイーブなHPA（デューティサイクル）、キュー深さHPA、KAIギャングスケジューリングを比較する。未処理リクエスト数、アイドルGPU分、複合スコアを報告する。

## 成果物を出す

このレッスンでは`outputs/skill-gpu-autoscaler-plan.md`を作成する。クラスタートポロジー、ワークロードシェイプ、SLOに基づいて3層オートスケーリングプランを設計する。

## 演習

1. `code/main.py`を実行する。バースト的なワークロードで、ナイーブなデューティサイクルHPAがキュー深さHPAが捕捉するリクエストをいくつドロップするか？差はどこから来るか？
2. H100 SXM5上でLlama 3.3 70B FP8を配信するクラスターのKarpenter NodePoolを設計する。`capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`、GPU以外のワークロードをノードから遠ざけるテイントを指定する。
3. 「GPUは空いているがポッドがスケジュールされない」というデプロイメントがPendingで止まっているとチームが報告する。診断する — これはKarpenter、kube-scheduler、KAI Schedulerのどれか？どのメトリクスが確認するか？
4. 分散プリフィルポッドをオートスケールするシグナルと、デコードポッドに別のシグナルを選ぶ。両方を正当化する。
5. `WhenEmptyOrUnderutilized`コンソリデーショントラップのコストを、P99 TTFT > 10sのリクエストドロップイベントが1日平均60回ある24時間365日の本番サービスで計算する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|----------------|------------|
| Karpenter | 「ノードプロビジョナー」 | Kubernetesノードオートスケーラー。1分未満のプロビジョニング |
| Cluster Autoscaler | 「古いスケーラー」 | Kubernetesノードオートスケーラーの前身。低速でグループベース |
| KAI Scheduler | 「GPUスケジューラー」 | ギャング + トポロジー + キューのセカンダリスケジューラー |
| ギャングスケジューリング | 「オールオアナッシング」 | Nポッドをアトミックにスケジュール、またはすべて延期 |
| トポロジー認識 | 「ラック認識」 | NVLink/IB/ラック配置に基づいてポッドを配置 |
| `DCGM_FI_DEV_GPU_UTIL` | 「GPU利用率」 | デューティサイクルメトリクス。LLMのスケーリングシグナルではない |
| キュー深さ | 「待機リクエスト」 | プリフィルバウンドスケーリングの正しいHPAシグナル |
| KVキャッシュ利用率 | 「メモリ圧力」 | デコードバウンドスケーリングの正しいHPAシグナル |
| コンソリデーション | 「Karpenterコンソリデーション」 | より安価なインスタンスタイプへのノード終了 |
| `WhenEmpty + 1h` | 「安全なコンソリデーション」 | 実行中のGPUジョブを退去させないポリシー |

## 参考資料

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — 設計ドキュメントと設定例
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) — コンソリデーションポリシーのセマンティクスとGPU安全デフォルト
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Plannerスケーリングシグナル
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray統合パターン
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — マネージドKubernetes固有のガイダンス
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Workload Variant Autoscaler設計
