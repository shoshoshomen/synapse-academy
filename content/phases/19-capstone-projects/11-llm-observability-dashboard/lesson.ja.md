# キャップストーン 11 — LLM可観測性・評価ダッシュボード

> LangfuseはオープンコアになりArize Phoenixが2026年GenAI semconvマッピングを公開した。HeliconeとBraintrustはどちらもユーザーごとのコスト帰属に注力した。TraceloopのOpenLLMetryがデファクトのSDKインストルメンテーションになった。本番環境の形はClickHouseでトレース・Postgresでメタデータ・Next.jsでUI・そしてサンプリングされたトレースに対して実行される小さな評価ジョブ群（DeepEval・RAGAS・LLM-judge）だ。セルフホストで1つ構築し、少なくとも4つのSDKファミリーからインジェストし、注入されたリグレッションを5分以内に検知することを実証する。

**演習するフェーズ:** P11 · P13 · P17 · P18

## 問題

2026年に本番トラフィックを流しているすべてのAIチームは、モデルのそばに可観測性プレーンを置いている。コスト帰属。ハルシネーション検出。ドリフト監視。ジェイルブレークシグナル。SLOダッシュボード。PIIリークアラート。オープンソースのリファレンス（Langfuse・Phoenix・OpenLLMetry）はインジェストスキーマとしてOpenTelemetry GenAIセマンティック規約に収束した。今ではOpenAI・Anthropic・Google・LangChain・LlamaIndex・vLLMを1つのSDKでインストルメントし、互換性のあるスパンを送り出せる。

少なくとも4つのSDKファミリーからインジェストし、サンプリングされたトレースに対して小さな評価ジョブセットを実行し、ドリフトを検出してアラートを出す、セルフホストのダッシュボードを構築する。計測基準：意図的に注入されたリグレッション（PIIを生成し始めるプロンプト）をダッシュボードが5分以内に検知してアラートを発火させること。

## コンセプト

インジェストはOTLP HTTPだ。SDKはGenAI-semconvスパンを生成する：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.response.id`、`llm.prompts`、`llm.completions`。スパンはカラム指向分析のためにClickHouseに保存され、メタデータ（ユーザー・セッション・アプリ）はPostgresに保存される。

評価はサンプリングされたトレースに対してバッチジョブとして実行される。DeepEvalがfaithfulness・toxicity・answer relevanceをスコアリングする。RAGASはトレースがリトリーバルコンテキストを持つ場合にリトリーバルメトリクスをスコアリングする。カスタムLLM-judgeがドメイン固有のチェック（PIIリーク・ポリシー外のレスポンス）を実行する。評価実行結果は元のLLMコールスパンにリンクされた評価スパンとして同じClickHouseに書き戻される。

ドリフト検出は時間経過に伴う埋め込み空間の分布を監視する（プロンプト埋め込みのPSIまたはKL発散）および評価スコアのトレンド。アラートはPrometheus Alertmanagerを経由してSlack/PagerDutyへ送られる。UIはRechartsを使ったNext.js 15だ。

## アーキテクチャ

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## スタック

- インジェスト: OpenTelemetry SDKs + GenAI semantic conventions; OTLP HTTPトランスポート
- コレクター: テールサンプリングプロセッサ付きOpenTelemetry Collector（コスト管理のため）
- ストレージ: スパン用ClickHouse・メタデータ用Postgres・生イベントアーカイブ用S3
- 評価: DeepEval・RAGAS 0.2・Arize Phoenix evaluatorパック・カスタムLLM-judge
- ドリフト: プールされたプロンプト埋め込み（sentence-transformers）のPSI/KLを週次算出
- アラート: Prometheus Alertmanager -> Slack / PagerDuty
- UI: Next.js 15 App Router + Recharts + server actions
- 標準対応SDK: OpenAI・Anthropic・Google GenAI・LangChain・LlamaIndex・vLLM

## 実装する

1. **コレクター設定。** OTLP HTTPレシーバー・エラートレースを100%保持して成功を10%保持するテールサンプラー・ClickHouseとS3へのエクスポーターを持つOpenTelemetry Collectorを設定する。

2. **ClickHouseスキーマ。** GenAI semconvを反映したカラムを持つ`spans`テーブル：`gen_ai_system`・`gen_ai_request_model`・`input_tokens`・`output_tokens`・`latency_ms`・`prompt_hash`・`trace_id`・`parent_span_id`、および長いペイロード用JSONバッグ。user_idとapp_idにセカンダリインデックスを追加する。

3. **SDKカバレッジテスト。** 各SDK（OpenAI・Anthropic・Google・LangChain・LlamaIndex・vLLM）をOpenLLMetry自動インストルメントで使う小さなクライアントアプリを書く。それぞれが正規のGenAIスパンを生成してClickHouseに保存されることを確認する。

4. **評価ジョブ。** スケジューラーが直近15分間のサンプリングされたトレースを読み、DeepEval faithfulness・toxicity・answer relevanceを実行する。出力は親トレースにリンクされた評価スパンだ。

5. **カスタムLLM-judge。** PIIリークjudge：レスポンスが与えられると、ガードLLMを呼び出してPIIリークの可能性をスコアリングする。高スコアのレスポンスはトリアージキューに入る。

6. **ドリフト検出。** 週次ジョブが今週のプールされたプロンプト埋め込みと直近4週間のベースラインの間のPSIを計算する。PSIがしきい値を超えたらアラートを出す。

7. **ダッシュボード。** Next.js 15のページ：概要（スパン/秒・ユーザーあたりコスト・p95レイテンシ）・トレース（検索+ウォーターフォール）・評価（faithfulnessトレンド・toxicity）・ドリフト（時系列PSI）・アラート。

8. **アラートチェーン。** Prometheusエクスポーターが評価スコア集計とレイテンシパーセンタイルを読み取り；AlertmanagerがSlackへの警告とPagerDutyへの重大アラートをルーティングする。

9. **リグレッションプローブ。** バグを注入する：評価対象のチャットボットが1%の確率で偽のSSNを漏洩し始める。MTTRを計測する：バグのデプロイからSlackアラートまで。

## 使ってみる

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## 成果物を出す

`outputs/skill-llm-observability.md`が成果物。LLMアプリケーションが与えられると、ダッシュボードはそのトレースをインジェストし、評価を実行し、ドリフトにアラートを出し、Next.jsでユーザーあたりのコスト内訳を表示する。

| 配点 | 基準 | 測定方法 |
|:-:|---|---|
| 25 | トレーススキーマカバレッジ | 正規のGenAIスパンを生成するSDKファミリーの数（目標: 6+） |
| 20 | 評価の正確さ | 手動ラベル付きセットに対するDeepEval/RAGASスコア |
| 20 | ダッシュボードUX | 注入されたリグレッションのMTTR（5分未満目標） |
| 20 | コスト/スケール | バックログなしで1kスパン/秒の持続的インジェスト |
| 15 | アラート+ドリフト検出 | Prometheus/Alertmanagerチェーンのエンドツーエンド検証 |
| **100** | | |

## 演習

1. HaystackフレームワークのカスタムインストルメンテーションをGenAI semconvに追加する。正規のスパンがClickHouseに`gen_ai.*`属性付きで届くことを確認する。

2. 同じトレースでDeepEvalをPhoenix evaluatorsに置き換える。2つの評価エンジン間のスコアドリフトを計測する。

3. ドリフト検出器を改善する：PSIをグローバルではなくapp-idごとに計算する。app別ドリフトの推移を表示する。

4. 「ユーザーインパクト」ページを追加する：スパークライン付きのユーザーあたりコストとユーザーあたり失敗率。

5. toxicity > 0.5のトレースを100%保持し、残りを10%層別サンプリングするテールサンプリングポリシーを構築する。導入されるサンプリングバイアスを計測する。

## キーワード

| 用語 | 一般的な呼び方 | 実際の意味 |
|------|-----------------|------------------------|
| GenAI semconv | 「OTel LLMアトリビュート」 | LLMスパンアトリビュートのための2025年OpenTelemetry仕様（system・model・tokens） |
| テールサンプリング | 「ポストトレースサンプル」 | トレース完了後にコレクターが保持か破棄かを決定する（エラーをピーク可能） |
| PSI | 「人口安定性指数」 | 2つの分布を比較するドリフトメトリクス；> 0.2は通常意味のあるドリフトを示す |
| LLM-judge | 「評価としてのモデル」 | ルーブリック（faithfulness・toxicity・PII）で別のLLMの出力をスコアリングするLLM |
| テールサンプリングポリシー | 「保持ルール」 | どのトレースを保持してどれを破棄するかを決定するルール；エラー+サンプリングレート |
| 評価スパン | 「リンクされた評価トレース」 | 元のLLMコールスパンにリンクされた評価スコアを持つ子スパン |
| ユーザーあたりコスト | 「ユニットエコノミクス」 | ウィンドウ内のuser_idに帰属するドルコスト；主要なプロダクトメトリクス |

## 参考資料

- [Langfuse](https://github.com/langfuse/langfuse) — リファレンスオープンコア可観測性プラットフォーム
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — ドリフトサポートが強力な代替リファレンス
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — 自動インストルメンテーションSDKファミリー
- [OpenTelemetry GenAIセマンティック規約](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — インジェストスキーマ
- [Helicone](https://www.helicone.ai) — 代替ホスト型可観測性
- [Braintrust](https://www.braintrust.dev) — 代替評価ファーストプラットフォーム
- [ClickHouseドキュメント](https://clickhouse.com/docs) — カラム指向スパンストア
- [DeepEval](https://github.com/confident-ai/deepeval) — evaluatorライブラリ
