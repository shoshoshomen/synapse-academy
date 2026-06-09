# キャップストーン 06 — Kubernetes向けDevOpsトラブルシューティングエージェント

> AWSのDevOps AgentがGAになり、Resolve AIはK8sプレイブックを公開し、NeuBirdはセマンティックモニタリングをデモし、MetoroはAI SREをサービスごとのSLOに結び付けた。本番環境の形は決まっている：アラートWebhookが発火し、エージェントはテレメトリーを読み取り、K8sオブジェクトのグラフを歩き、根本原因の仮説をランク付けし、承認ボタン付きのSlackのブリーフを投稿する。デフォルトでリードオンリー。すべての修復は人間によってゲートされる。このキャップストーンはそのエージェントであり、20の合成インシデントで評価され、3つの共有ケースでAWSのエージェントと比較される。

**演習するフェーズ:** P11 · P13 · P14 · P15 · P17 · P18

## 問題

2025-2026年のSREの物語は「AIエージェントがインシデントをトリアージし、人間が修復を承認する」となった。AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOpsはすべてこの形を本番環境で出荷している。エージェントはPrometheusメトリクス・Lokiログ・Tempoトレース・kube-state-metrics・K8sオブジェクトの知識グラフを読む。5分以内にテレメトリーの引用を持つランク付けされた根本原因仮説を生成する。Slack経由の明示的な人間の承認なしに破壊的なコマンドを実行することはない。

難しい作業のほとんどはスコーピングと安全性であり、推論ではない。エージェントはリードオンリーデフォルトのRBACサーフェス・強化されたMCPツールサーバー・検討されたコマンド対実行されたコマンドの監査ログを必要とする。深度を超えたときにいつエスカレートするかを知る必要がある。OOMキルカスケードが$5kのエージェント請求書を生成しないよう十分安価に実行する必要もある。

## コンセプト

エージェントは知識グラフ上で動作する。ノードはK8sオブジェクト（Pod、Deployment、Service、Node、HPA、PVC）とテレメトリーソース（Prometheusシリーズ、Lokiストリーム、Tempoトレース）だ。エッジは所有権（Pod -> ReplicaSet -> Deployment）、スケジューリング（Pod -> Node）、観察（Pod -> Prometheusシリーズ）をエンコードする。グラフはkube-state-metricsの同期によって新鮮に保たれ、すべてのアラートで再サンプリングされる。

アラートが発火すると、エージェントは影響を受けたオブジェクトから根本原因を特定する。エッジを歩き、関連するテレメトリースライス（過去15分）を取得し、仮説を草案する。仮説はエビデンスでランク付けされる：いくつのテレメトリー引用がそれをサポートするか、どれだけ最近か、どれだけ具体的か。上位3つの仮説がグラフパスの視覚化と修復アクションの承認ボタン付きでSlackに送られる。

修復はゲートされる。許可されるデフォルトアクションはリードオンリーだ。破壊的なアクション（スケールダウン、ロールバック、Podの削除）はSlackの承認が必要；ArgoDBのロールバックフックはエージェントが保持しない認証トークンを必要とする。監査ログはエージェントが「検討した」すべてのコマンドを記録する — 実行されたものだけでなく — レビュープロセスがニアミスを捉える。

## アーキテクチャ

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## スタック

- 可観測性ソース: Prometheus、Loki、Tempo、kube-state-metrics
- 知識グラフ: K8sオブジェクト + テレメトリーエッジのNeo4j（マネージド）またはkuzu（組み込み）
- エージェント: ツールごとの許可リストを持つLangGraph、デフォルトでリードオンリー
- ツールトランスポート: StreamableHTTP経由のFastMCP；承認ゲートの後ろにある破壊的ツール用の別サーバー
- モデル: 根本原因推論用Claude Sonnet 4.7、ログサマリー用Gemini 2.5 Flash
- 修復: ArgoDBロールバックWebhook、PagerDutyエスカレート、Slack承認カード
- 監査: 追記専用の構造化ログ（検討、実行、承認、結果）
- デプロイ: 独自の狭いRBACロールを持つK8sデプロイメント；別のNamespace

## 実装する

1. **グラフ取り込み。** kube-state-metricsをNeo4j/kuzuに30秒ごとに同期する。ノード：Pod、Deployment、Node、Service、PVC、HPA。エッジ：OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS、SCALES。テレメトリーオーバーレイエッジ：OBSERVED_BY（PodはPrometheusシリーズによって観察される）。

2. **アラートレシーバー。** PagerDutyまたはAlertmanagerのWebhookを受け付けるFastAPIエンドポイント。影響を受けたオブジェクトとSLO違反を抽出する。

3. **リードオンリーツールサーフェス。** kubectl、Prometheusクエリ、Loki logql、Tempo tracequlをFastMCP経由でラップする。すべてのツールは狭いRBACの動詞を持つ（「get」、「list」、「describe」）。デフォルトサーバーには「delete」、「exec」、「scale」はない。

4. **根本原因エージェント。** 3つのノードを持つLangGraph：`sample`は過去15分のテレメトリースライスを取得し、`walk`は隣接オブジェクトのグラフをクエリし、`hypothesize`はテレメトリー引用付きのランク付けされた根本原因候補を草案する。

5. **エビデンススコアリング。** 各仮説のスコア = 最新性 * 具体性 * グラフパス長逆数 * 引用数。上位3を返す。

6. **Slackブリーフ。** 仮説・グラフパスの視覚化（サーバー側でレンダリングされたサブグラフ画像）・最大1つの修復アクションの承認ボタンを持つアタッチメントを投稿する。

7. **修復ゲート。** 破壊的ツール（スケールダウン、ロールバック、削除）は承認トークンの後ろにある第2のMCPサーバーに存在する。エージェントはSlackカードが人間によって承認された後にのみそれらを呼び出せる。

8. **監査ログ。** 追記専用JSONL：すべての候補コマンドについて、検討されたかどうか、実行されたかどうか、誰が承認したかをログに記録する。毎日S3に出荷する。

9. **合成インシデントスイート。** 20のシナリオを構築する：OOMキルカスケード、DNSフラップ、HPAスラッシング、PVC満杯、うるさいネイバー、障害のあるサイドカー、不良ConfigMapのロールアウト、証明書のローテーション、イメージプルのバックオフなど。根本原因の精度と仮説への時間でエージェントをスコアリングする。

## 使ってみる

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## 成果物を出す

`outputs/skill-devops-agent.md`が成果物。K8sクラスターとアラートソースを与えると、エージェントはランク付けされた根本原因仮説とSlackゲートされた修復フローを生成する。

| 配点 | 基準 | 測定方法 |
|:-:|---|---|
| 25 | シナリオスイートでのRCA精度 | 20の合成インシデントで80%以上の正しい根本原因 |
| 20 | 安全性 | 監査ログでSlack承認なしに破壊的アクションガードが発火しない |
| 20 | 仮説への時間 | アラートからSlackブリーフまでのp50が5分以下 |
| 20 | 説明可能性 | すべての仮説にグラフパスとテレメトリー引用がある |
| 15 | 統合の完全性 | PagerDuty、Slack、ArgoCD、Prometheusのエンドツーエンド動作 |
| **100** | | |

## 演習

1. AWSのDevOps Agentがデモされている同じ3つのインシデントでエージェントを実行する。並列比較を公開する。エージェントが発散する箇所をレポートする。

2. エージェントが承認なしに破壊的だったかもしれない*検討した*コマンドにフラグを立てる「ニアミス」監査を追加する。1週間でのニアミス率を測定する。

3. 仮説モデルをClaude Sonnet 4.7からセルフホストのLlama 3.3 70Bに交換する。RCA精度のデルタとインシデントあたりのコストを測定する。

4. 因果フィルターを構築する：相関するテレメトリースパイクを真の根本原因から区別する。20シナリオのラベルに対して小さな分類器をトレーニングする。

5. ロールバックのドライランを追加する：同じマニフェストを持つステージングクラスターに対するArgoDBロールバック。Slack承認ボタンの前にライブクラスターでロールバックプランを検証する。

## キーワード

| 用語 | 一般的な呼び方 | 実際の意味 |
|------|-----------------|------------------------|
| K8s知識グラフ | 「クラスターグラフ」 | ノード = K8sオブジェクト + テレメトリーシリーズ；エッジ = 所有権、スケジューリング、観察 |
| リードオンリーデフォルト | 「スコープドRBAC」 | エージェントのサービスアカウントはget/list/describeの動詞のみ；破壊的動詞は承認の後ろの別サーバーにある |
| 監査ログ | 「検討対実行」 | すべての候補コマンドの追記専用記録、実行されたかどうか、誰が承認したか |
| 仮説ランキング | 「エビデンススコア」 | 最新性 × 具体性 × グラフパス長逆数 × 引用数 |
| Slack承認カード | 「HITL（Human-In-The-Loop）ゲート」 | 修復ボタンを持つインタラクティブなSlackメッセージ；人間がクリックするまでエージェントは続行できない |
| テレメトリー引用 | 「エビデンスポインター」 | 主張をサポートするPrometheusクエリ・Lokiセレクター・TempoトレースのURL |
| MTTR | 「解決までの時間」 | アラート発火からSLO回復までのウォールクロック |

## 参考資料

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 2026年のカノニカルリファレンス
- [Resolve AI K8sトラブルシューティング](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — 競合他社リファレンス
- [NeuBirdセマンティックモニタリング](https://www.neubird.ai) — セマンティックグラフアプローチ
- [Metoro AI SRE](https://metoro.io) — SLOファーストの本番フレーミング
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — クラスターステートソース
- [LangGraph](https://langchain-ai.github.io/langgraph/) — リファレンスエージェントオーケストレーター
- [FastMCP](https://github.com/jlowin/fastmcp) — PythonのMCPサーバーフレームワーク
- [ArgoDBロールバック](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — ゲートされた修復ターゲット
