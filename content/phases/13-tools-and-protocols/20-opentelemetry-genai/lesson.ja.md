# OpenTelemetry GenAI — ツール呼び出しのエンドツーエンドトレーシング

> エージェントが5つのツール、3つのMCPサーバー、2つのサブエージェントを呼び出す。すべてにわたる1つのトレースが必要だ。OpenTelemetry GenAIセマンティック規約（v1.37以降の安定属性）は2026年の標準であり、Datadog、Langfuse、Arize Phoenix、OpenLLMetry、AgentOpsがネイティブサポートしている。このレッスンでは必須属性を定義し、スパン階層（エージェント → LLM → ツール）を説明し、任意のOTelエクスポーターに接続できるstdlibスパンエミッターを構築する。


## 学習目標

- LLMスパンとツール実行スパンの必須OTel GenAI属性を名前で説明できる。
- エージェントループ、LLM呼び出し、ツール呼び出し、MCPクライアントディスパッチをカバーするトレース階層を構築できる。
- デフォルトで編集すべき内容と、オプトインでキャプチャすべき内容を決定できる。
- ツールコードを書き直すことなくスパンをローカルコレクター（Jaeger、Langfuse）に出力できる。

## 問題背景

2026年2月のデバッグ：ユーザーが「エージェントが時々30秒かかることがあるが、他の時は3秒だ」と報告する。トレースなし。ログはLLM呼び出しを示しているが、ツールのディスパッチ、MCPサーバーのラウンドトリップ、サブエージェントは示していない。推測するしかない。最終的に判明：1つのMCPサーバーがコールドスタート時に時々ハングしていた。

エンドツーエンドのトレーシングなしにはこれを見つけられない。OTel GenAIがそれを解決する。

この規約は2025〜2026年にOpenTelemetryのセマンティック規約グループの下で定められた。Datadog、Langfuse、Phoenix、OpenLLMetry、AgentOpsがすべて同じスパンを解析できるよう安定した属性名を定義している。一度インストルメントするとどのバックエンドにも出荷できる。

## コンセプト

### スパン階層

```
agent.invoke_agent  (トップ、INTERNALスパン)
 ├── llm.chat       (CLIENTスパン)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENTスパン)
 ├── llm.chat       (CLIENTスパン)
 └── subagent.invoke (INTERNAL)
```

すべてが1つのトレースIDの下にネストされる。スパンIDが親子関係をリンクする。

### 必須属性

2025〜2026年のセマンティック規約：

- `gen_ai.operation.name` — `"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name` — `"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model` — リクエストされたモデル文字列（例：`"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` — 実際に提供されたモデル。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` — 相関のためのプロバイダーレスポンスID。

ツールスパン：

- `gen_ai.tool.name` — ツール識別子。
- `gen_ai.tool.call.id` — 特定の呼び出しID。
- `gen_ai.tool.description` — ツールの説明（オプション）。

エージェントスパン：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### スパンの種類

- `SpanKind.CLIENT`：プロセス境界を越える呼び出し（LLMプロバイダー、MCPサーバー）。
- `SpanKind.INTERNAL`：エージェント自身のループステップとツール実行。

### オプトインのコンテンツキャプチャ

デフォルトでは、スパンはメトリクスとタイミングを含み、プロンプトやコンプリーションは含まない。大きなペイロードとPIIはデフォルトでオフだ。`OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` と特定のコンテンツキャプチャの環境変数を設定するとコンテンツが含まれる。本番環境で有効にする前に慎重に確認する。

### スパンのイベント

トークンレベルのイベントをスパンイベントとして追加できる：

- `gen_ai.content.prompt` — 入力メッセージ。
- `gen_ai.content.completion` — 出力メッセージ。
- `gen_ai.content.tool_call` — 記録されたツール呼び出し。

イベントはスパン内で時系列に並び、詳細なリプレイができる。

### エクスポーター

OTelスパンは以下にエクスポートできる：

- **Jaeger / Tempo。** OSS、オンプレミス。
- **Langfuse。** LLMオブザーバビリティ特化；トークン使用量を可視化。
- **Arize Phoenix。** 評価とトレーシングの統合。
- **Datadog。** 商用；`gen_ai.*` 属性をネイティブに解析。
- **Honeycomb。** 列指向；クエリしやすい。

すべてがOTLPというワイヤーフォーマットで話す。コードは気にしない。

### MCP越しのプロパゲーション

MCPクライアントがサーバーを呼び出す際、リクエストにW3C traceparentヘッダーを注入する。Streamable HTTPは標準ヘッダーをサポートしている。StdioはネイティブにHTTPヘッダーを運ばない；仕様の2026年ロードマップではJSON-RPCの呼び出しに `_meta.traceparent` フィールドを追加することを議論している。

それがリリースされるまで：すべてのリクエストの `_meta` にtraceparentを手動で含める。サーバーがトレースIDをログに記録する。

### メトリクス

スパンに加えて、GenAIセマンティック規約はメトリクスを定義している：

- `gen_ai.client.token.usage` — ヒストグラム。
- `gen_ai.client.operation.duration` — ヒストグラム。
- `gen_ai.tool.execution.duration` — ヒストグラム。

呼び出しごとの詳細が不要なダッシュボードにはこれらを使用する。

### AgentOpsレイヤー

AgentOps（2024年設立）はGenAIオブザーバビリティに特化している。一般的なフレームワーク（LangGraph、Pydantic AI、CrewAI）をラップしてOTelスパンを自動的に出力する。使用しているスタックがサポートされているフレームワークを使っているなら便利；そうでなければ手動インストルメンテーションを使う。

## 使ってみる

`code/main.py` は、LLMを呼び出し、2つのツールをディスパッチし、1回のMCPラウンドトリップを行うエージェントのOTel形状のスパンをstdout（OTLP-JSONライク形式）に出力する。実際のエクスポーターなし — レッスンはスパン形状と属性セットに集中している。OTLPと互換性のあるビューワーに出力を貼り付けるか、そのまま読むだけでよい。

確認すべきポイント：

- トレースIDはすべてのスパンで共有される。
- 親子リンクは `parentSpanId` でエンコードされる。
- 必須の `gen_ai.*` 属性が入力されている。
- コンテンツキャプチャはデフォルトでオフ；1つのシナリオが環境変数でそれを有効にする。

## 成果物を出す

このレッスンでは `outputs/skill-otel-genai-instrumentation.md` を生成する。エージェントのコードベースが与えられると、このスキルはインストルメンテーション計画を生成する：どこにスパンを追加するか、どの属性を入力するか、どのエクスポーターをターゲットにするか。

## 演習

1. `code/main.py` を実行する。スパンを数え、どれがCLIENTでどれがINTERNALかを特定する。

2. コンテンツキャプチャを有効にし（環境変数）、`gen_ai.content.prompt` と `gen_ai.content.completion` イベントが表示されることを確認する。PIIに対する意味を確認する。

3. ツール実行メトリクス `gen_ai.tool.execution.duration` を追加し、呼び出しごとにヒストグラムサンプルとして出力する。

4. 親エージェントスパンからMCPリクエストの `_meta.traceparent` フィールドにtraceparentをプロパゲートする。MCPサーバーが同じトレースIDを見ることを確認する。

5. OTel GenAIセマンティック規約の仕様を読む。このレッスンのコードが出力していないセマンティック規約にリストされている属性を1つ特定する。それを追加する。

## キーワード

| 用語 | よく言われる表現 | 実際の意味 |
|------|----------------|------------------------|
| OTel | "OpenTelemetry" | トレース、メトリクス、ログのオープン標準 |
| GenAIセマンティック規約 | "GenAIの属性規約" | LLM / ツール / エージェントスパンの安定した属性名 |
| `gen_ai.*` | "属性ネームスペース" | すべてのGenAI属性がこのプレフィックスを共有 |
| スパン | "タイムド操作" | 開始、終了、属性を持つ作業の単位 |
| トレース | "クロススパンの祖先関係" | トレースIDを共有するスパンのツリー |
| SpanKind | "CLIENT / SERVER / INTERNAL" | スパンの方向についてのヒント |
| OTLP | "OpenTelemetryラインプロトコル" | エクスポーター用のワイヤーフォーマット |
| オプトインコンテンツ | "プロンプト / コンプリーションキャプチャ" | デフォルトでオフ；環境変数で有効化 |
| traceparent | "W3Cヘッダー" | サービスをまたいでトレースコンテキストをプロパゲート |
| エクスポーター | "バックエンド固有のシッパー" | Jaeger / Datadog等にスパンを送信するコンポーネント |

## 参考資料

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAIスパン、メトリクス、イベントの正規規約
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLMとツール実行スパンの属性リスト
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — エージェントレベルの `invoke_agent` スパン
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHubホストの情報源
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 本番統合のウォークスルー
