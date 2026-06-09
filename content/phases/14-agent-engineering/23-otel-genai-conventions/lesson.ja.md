# OpenTelemetry GenAI セマンティック規約

> OpenTelemetry の GenAI SIG（2024年4月立ち上げ）はエージェントテレメトリの標準スキーマを定義します。スパン名、属性、コンテンツキャプチャルールはベンダー間で統一されているため、エージェントトレースは Datadog、Grafana、Jaeger、Honeycomb で同じ意味を持ちます。


## 学習目標

- GenAI スパンカテゴリを説明できる: モデル/クライアント、エージェント、ツール。
- `invoke_agent` の CLIENT スパンと INTERNAL スパンの違いと、それぞれが適用される場面を説明できる。
- トップレベルの GenAI 属性を一覧できる: プロバイダー名、リクエストモデル、データソース ID。
- コンテンツキャプチャコントラクトを説明できる: オプトイン、`OTEL_SEMCONV_STABILITY_OPT_IN`、外部参照の推奨。

## 問題設定

すべてのベンダーが独自のスパン名を考案します。運用チームはフレームワークごとのダッシュボードを構築することになります。OpenTelemetry の GenAI SIG はエコシステム全体がターゲットとする1つの標準を定義することでこれを解決します。

## コンセプト

### スパンカテゴリ

1. **モデル/クライアントスパン.** 生の LLM 呼び出しをカバーします。プロバイダー SDK（Anthropic、OpenAI、Bedrock）とフレームワークのモデルアダプターによって発行されます。
2. **エージェントスパン.** `create_agent`（エージェントが構築されたとき）と `invoke_agent`（実行されたとき）。
3. **ツールスパン.** ツール呼び出しごとに1つ。エージェントスパンに親子関係で接続されます。

### エージェントスパンの命名

- スパン名: 名前付きの場合は `invoke_agent {gen_ai.agent.name}`。フォールバックは `invoke_agent`。
- スパンカインド:
  - **CLIENT** — リモートエージェントサービス（OpenAI Assistants API、Bedrock Agents）の場合。
  - **INTERNAL** — インプロセスのエージェントフレームワーク（LangChain、CrewAI、ローカル ReAct）の場合。

### 主要属性

- `gen_ai.provider.name` — `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model` — モデル ID。
- `gen_ai.response.model` — 解決されたモデル（ルーティングにより要求と異なる場合あり）。
- `gen_ai.agent.name` — エージェント識別子。
- `gen_ai.operation.name` — `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id` — RAG 用: どのコーパスまたはストアが参照されたか。

Anthropic、Azure AI Inference、AWS Bedrock、OpenAI に対してテクノロジー固有の規約が存在します。

### コンテンツキャプチャ

デフォルトルール: インストルメンテーションはデフォルトで入力/出力をキャプチャしてはいけません（SHOULD NOT）。キャプチャは以下を介してオプトインです:

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推奨される本番パターン: コンテンツを外部（S3、ログストア）に保存し、スパンに参照（ポインター ID、散文ではなく）を記録します。これはレッスン27のコンテンツポイズニング防御がオブザーバビリティに組み込まれたものです。

### 安定性

ほとんどの規約は 2026年3月時点では実験的です。安定したプレビューにオプトインするには:

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37 以降は GenAI 属性を LLM Observability スキーマにネイティブにマッピングします。他のバックエンド（Grafana、Honeycomb、Jaeger）は生の属性をサポートします。

### このパターンが失敗するケース

- **スパン内で完全なプロンプトをキャプチャする.** 運用チームが読めるトレースに PII、シークレット、顧客データが入ります。外部に保存してください。
- **`gen_ai.provider.name` がない.** マルチプロバイダーのダッシュボードが帰属なしに壊れます。
- **親リンクのないスパン.** 孤立したツールスパン。常にコンテキストを伝播してください。
- **安定性オプトインの未設定.** バックエンドのアップグレード時に属性の名称が変更される可能性があります。

## 実装する

`code/main.py` は GenAI 規約に合わせた標準ライブラリのスパンエミッターを実装しています:

- GenAI 属性スキーマを持つ `Span`。
- `start_span`、ネストされたコンテキストを持つ `Tracer`。
- スクリプト化されたエージェント実行が発行するもの: `create_agent`、`invoke_agent`（INTERNAL）、ツールごとのスパン、LLM 呼び出しの `chat` スパン。
- プロンプトを外部に保存してスパンに ID を記録するコンテンツキャプチャモード。

実行:

```
python3 code/main.py
```

出力: 必要なすべての GenAI 属性を持つスパンツリーと、オプトインのコンテンツ参照を示す「外部ストア」。

## 使ってみる

- **Datadog LLM Observability**（v1.37 以降）は属性をネイティブにマッピングします。
- **Langfuse / Phoenix / Opik**（レッスン24）— エコシステムを自動インストルメント。
- **Jaeger / Honeycomb / Grafana Tempo** — 生の OTel トレース。GenAI 属性からダッシュボードを構築。
- **セルフホスト** — GenAI プロセッサーを持つ OTel コレクターを実行。

## 成果物を出す

`outputs/skill-otel-genai.md` はコンテンツキャプチャのデフォルトと外部参照ストレージを持つ既存エージェントに OTel GenAI スパンを組み込みます。

## 演習

1. レッスン01の ReAct ループを `invoke_agent`（INTERNAL）+ ツールごとのスパンでインストルメントします。Jaeger インスタンスに送信します。
2. 「参照のみ」モードでコンテンツキャプチャを追加します: プロンプトを SQLite に、スパン属性には行 ID のみ。
3. `gen_ai.data_source.id` の仕様を読みます。レッスン09の Mem0 検索に組み込みます。
4. `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` を設定し、コレクターで属性の名称が変更されないことを確認します。
5. ダッシュボードを構築します: 「どのツールエラーがどのモデルと相関しているか」を GenAI 属性だけから。

## キーワード

| 用語 | 一般的な言い方 | 実際の意味 |
|------|----------------|------------|
| GenAI SIG | "OpenTelemetry GenAI グループ" | スキーマを定義する OTel ワーキンググループ |
| invoke_agent | "エージェントスパン" | エージェント実行を表すスパンの名前 |
| CLIENT スパン | "リモート呼び出し" | リモートエージェントサービスへの呼び出しのスパン |
| INTERNAL スパン | "インプロセス" | インプロセスのエージェント実行のスパン |
| gen_ai.provider.name | "プロバイダー" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG ソース" | 検索がヒットしたコーパス/ストア |
| コンテンツキャプチャ | "プロンプトログ" | メッセージのオプトインキャプチャ。本番では外部保存 |
| 安定性オプトイン | "プレビューモード" | 実験的な規約を固定する環境変数 |

## 参考資料

- [OpenTelemetry GenAI セマンティック規約](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 仕様
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — デフォルトで GenAI スパンを発行
- [AutoGen v0.4（Microsoft Research）](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — OTel スパン組み込み済み
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C トレースコンテキスト伝播
