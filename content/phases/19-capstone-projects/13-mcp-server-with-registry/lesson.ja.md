# キャップストーン 13 — レジストリとガバナンスを持つMCPサーバー

> Model Context Protocolは2026年に未来から現在のデフォルトツール使用仕様になった。Anthropic・OpenAI・Google・そしてすべての主要IDEがMCPクライアントを搭載する。PinterestがMCPサーバーの内部エコシステムを公開した。AAIFレジストリが`.well-known`でのケイパビリティメタデータを形式化した。AWS ECSがリファレンスのステートレスデプロイメントを公開した。BlockのGooseエージェントが同じプロトコルをホスト型アシスタントに組み込んだ。2026年の本番環境の形は：StreamableHTTPトランスポート・OAuth 2.1スコープ・OPAポリシーゲーティング・そしてプラットフォームチームがサーバーを発見・検証・有効化できるレジストリだ。これをエンドツーエンドで構築する。

**演習するフェーズ:** P11 · P13 · P14 · P17 · P18

## 問題

MCPがツール使用の共通語になった。Claude Code・Cursor 3・Amp・OpenCode・Gemini CLI・そしてすべてのマネージドエージェントがMCPサーバーを消費する。本番環境の課題はサーバーのオーサリングではなく（FastMCPが簡単にする）、企業要件でのスケールデプロイだ：テナントごとのOAuthスコープ・破壊的ツールへのOPAポリシー・StreamableHTTPステートレススケーリング・発見のためのレジストリ・ツール呼び出しごとの監査ログ。PinterestのMCPエコシステムとAAIFレジストリ仕様が2026年の基準を設定した。

Postgresの読み取り専用・S3リスティング・Jira・Linear・Datadogなど10個の内部ツールを公開するMCPサーバー・プラットフォーム発見のためのレジストリUI・そして破壊的ツールへのヒューマンアプルーバルゲートを構築する。ロードテストがStreamableHTTPの水平スケーリングを実証する。監査証跡が企業のセキュリティレビューを満たす。

## コンセプト

MCP 2026リビジョンはデフォルトトランスポートとしてStreamableHTTPを義務付ける。以前のstdio-and-SSEの形とは異なり、StreamableHTTPはデフォルトでステートレスだ：単一のHTTPエンドポイントがJSON-RPCリクエストを受け付け・レスポンスをストリームし・通知のための長期接続をサポートする。ステートレスはロードバランサー背後での水平スケーリングを意味する。

認証はツールごとのスコープを持つOAuth 2.1だ。トークンは`jira:read`・`s3:list`・`postgres:query:readonly`のようなスコープを持つ。MCPサーバーはセッション開始時だけでなくツール呼び出し時にスコープをチェックする。高リスクツールについては、スコープが直近N分以内に`approved:by:human`に昇格されていない呼び出しをサーバーが拒否する。この昇格はSlackのレビューカードから来る。

レジストリは別のサービスだ。すべてのMCPサーバーはツールマニフェスト・トランスポートURL・認証要件を持つ`.well-known/mcp-capabilities`ドキュメントを公開する。レジストリがポーリング・検証・インデックスを行う。プラットフォームチームはレジストリUIを使って利用可能なツール・必要なスコープ・所有チームを確認する。

## アーキテクチャ

```
MCP client (Claude Code, Cursor 3, ...)
          |
          v
StreamableHTTP over HTTPS (JSON-RPC + streaming)
          |
          v
MCP server (FastMCP) behind load balancer
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 listing  Jira       Linear     Datadog
(read-only) (paged)     (read)     (read)     (query)
          |
   +------+-------------+
   v                    v
 OPA policy gate   destructive tool MCP (separate server)
                        |
                        v
                   human approval via Slack
                        |
                        v
                   audit log (append-only, per-tenant)

  registry service
     |
     v  GET /.well-known/mcp-capabilities from each server
     v
     UI: search / validate / enable-disable / ownership
```

## スタック

- サーバーフレームワーク: FastMCP（Python）または`@modelcontextprotocol/sdk`（TypeScript）
- トランスポート: StreamableHTTP over HTTPS（ステートレス）
- 認証: SPIFFE/SPIRE経由のワークロードアイデンティティを持つOAuth 2.1
- ポリシー: ツールごとのOPA/Regoルール；リクエストごとのポリシー決定サービス
- レジストリ: セルフホスト型、`.well-known/mcp-capabilities`マニフェストを消費
- ヒューマンアプルーバル: 破壊的ツール用のSlackインタラクティブメッセージ
- デプロイ: AWS ECS FargateまたはFly.io、テナントごとに1サーバーまたはテナントスコープ付き共有
- 監査: 呼び出しごとの系統情報を持つテナントごとのJSONLバケット

## 実装する

1. **ツールサーフェス。** 10個の内部ツールを公開する：Postgres読み取り専用クエリ・S3オブジェクトリスト・Jira検索/フェッチ・Linear検索/フェッチ・Datadogメトリクスクエリ・PagerDutyオンコール参照・GitHub読み取り専用・Notion検索・Slack検索・Salesforce読み取り。各ツールは型付きスキーマとスコープラベルを持つ。

2. **FastMCPサーバー。** ツールをマウントする。StreamableHTTPトランスポートを設定する。OAuthトークンイントロスペクションとスコープ適用のためのミドルウェアを追加する。

3. **OPAポリシー。** ツールごとのRegoポリシー：呼び出しを許可するスコープ・適用するPIIリダクション・適用するペイロードサイズ上限。ツール呼び出しごとに呼ばれる決定サービス。

4. **レジストリサービス。** 登録されたサーバーから`.well-known/mcp-capabilities`をポーリングし・JSONスキーマで検証し・リスト/検索/検証/有効化-無効化UIを公開する、別のGoまたはTSサービス。

5. **ケイパビリティマニフェスト。** 各サーバーが`.well-known/mcp-capabilities`を公開する：ツールリスト・認証要件・トランスポートURL・オーナーチーム・SLO。

6. **破壊的ツールの分離。** 状態を変更するツール（Jira作成・Linear作成・Postgresへの書き込み）は2番目のMCPサーバーに置き、より厳しい認証フローを持つ：15分以内にSlackカードで昇格した`approved:by:human`スコープをトークンが必要とする。

7. **監査ログ。** テナントごとのアペンドオンリーJSONL：`{timestamp, user, tool, args_redacted, response_redacted, outcome}`。書き込み前にPresidioでPIIリダクション。

8. **ロードテスト。** StreamableHTTPで100の同時クライアント。2番目のレプリカを追加してセッションスティッキネスなしでロードバランサーが再分配する様子を示し、水平スケーリングを実証する。

9. **適合性テスト。** 両方のサーバーに対して公式MCP適合性スイートを実行する。必須セクションをすべて通過する。

## 使ってみる

```
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }
```

## 成果物を出す

`outputs/skill-mcp-server.md`が成果物を説明する。OAuth 2.1スコープとOPAゲーティングを持つ内部ツール用のプロダクショングレードMCPサーバー+レジストリ+監査レイヤー。

| 配点 | 基準 | 測定方法 |
|:-:|---|---|
| 25 | 仕様適合性 | StreamableHTTP+ケイパビリティマニフェストがMCP適合性テストを通過 |
| 20 | セキュリティ | スコープ適用・すべてのツールへのOPAカバレッジ・シークレット衛生 |
| 20 | 可観測性 | PIIリダクション付きのツール呼び出しごとの監査ログ |
| 20 | スケール | 100クライアントロードテストの水平スケーリング実証 |
| 15 | レジストリUX | 発見/検証/有効化-無効化ワークフロー |
| **100** | | |

## 演習

1. 新しいツール（Confluence検索）を追加する。コアサーバーに触れずにレジストリ検証フローを通じて出荷する。

2. `email`・`ssn`・`phone`という名前のカラムを含むPostgresクエリ結果をリダクトするOPAポリシーを書く。プローブクエリで検証する。

3. StreamableHTTPとstdioのローカルレイテンシをベンチマークする。呼び出しごとのp50/p95を報告する。

4. テナントごとのクォータを実装する：ツールごと・テナントごとに1分間あたりN回の最大呼び出し。2番目のOPAルールで適用する。

5. [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance)からMCP適合性スイートを実行し、すべての失敗を修正する。

## キーワード

| 用語 | 一般的な呼び方 | 実際の意味 |
|------|-----------------|------------------------|
| StreamableHTTP | 「2026 MCPトランスポート」 | ステートレスHTTP+ストリーミング；ネットワークサーバーのSSE+stdioを置き換える |
| ケイパビリティマニフェスト | 「Well-knownドキュメント」 | ツールリスト・認証・トランスポートURLを持つ`.well-known/mcp-capabilities` |
| OPA / Rego | 「ポリシーエンジン」 | ツール呼び出しを外部ルールで認可するOpen Policy Agent |
| スコープ昇格 | 「ヒューマンアプルーバル」 | Slackアプルーバル経由で付与される短期スコープ、破壊的ツールに必要 |
| レジストリ | 「ツール発見」 | ケイパビリティマニフェストからMCPサーバーをインデックスするサービス |
| ワークロードアイデンティティ | 「SPIFFE / SPIRE」 | OAuthトークン発行のための暗号学的サービスアイデンティティ |
| 適合性スイート | 「仕様テスト」 | StreamableHTTP+ツールマニフェスト正確性のための公式MCPテストバッテリー |

## 参考資料

- [Model Context Protocol 2026ロードマップ](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP・ケイパビリティメタデータ・レジストリ
- [AAIF MCPレジストリ仕様](https://github.com/modelcontextprotocol/registry) — 2026年レジストリ仕様
- [AWS ECSリファレンスデプロイメント](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) — リファレンス本番デプロイメント
- [Pinterest内部MCPエコシステム](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) — リファレンス内部デプロイメント
- [Block `goose` MCPの使用](https://block.github.io/goose/) — リファレンスエージェント消費パターン
- [FastMCP](https://github.com/jlowin/fastmcp) — PythonサーバーフレームワークJ
- [Open Policy Agent](https://www.openpolicyagent.org/) — ポリシーエンジンリファレンス
- [SPIFFE / SPIRE](https://spiffe.io) — ワークロードアイデンティティリファレンス
