# MCP基礎 — プリミティブ、ライフサイクル、JSON-RPCベース

> MCP以前のあらゆる統合は個別実装でした。Model Context Protocolは2024年11月にAnthropicが初めて提供し、現在はLinux FoundationのAgentic AI Foundationが管理しています。これを使えば、任意のクライアントが任意のサーバーと通信できるよう、発見と呼び出しを標準化します。2025-11-25仕様では6つのプリミティブ（サーバー側3つ、クライアント側3つ）、3フェーズのライフサイクル、JSON-RPC 2.0ワイヤーフォーマットが定義されています。これらを理解すれば、MCPチャプターの残りは読み進めるだけです。


## 学習目標

- MCPの6つすべてのプリミティブ（サーバー側：tools、resources、prompts；クライアント側：roots、sampling、elicitation）を挙げ、それぞれのユースケースを述べられる。
- 3フェーズのライフサイクル（initialize、operation、shutdown）を説明し、各フェーズでどちらがどのメッセージを送るかを述べられる。
- JSON-RPC 2.0のリクエスト、レスポンス、通知エンベロープをパースして出力できる。
- `initialize`でのケイパビリティネゴシエーションとは何か、それなしに何が壊れるかを説明できる。

## 問題

MCP以前は、ツールを使う各エージェントが独自のプロトコルを持っていました。CursorはMCP形式だが互換性のないツールシステムを持っていました。Claude Desktopは別のものを提供していました。VS CodeのCopilot拡張機能はさらに別のものを持っていました。「Postgresクエリ」ツールを構築したチームは、それぞれ異なるホストのAPIに対して同じツールを3回書きました。再利用にはコードのコピーが必要でした。

その結果は、個別統合のカンブリア爆発とエコシステム速度の上限でした。

MCPはワイヤーフォーマットを標準化することでこれを解決します。単一のMCPサーバーはすべてのMCPクライアントで動作します：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf、2026年4月時点で300以上のクライアント。月間SDKダウンロード1億1,000万件。10,000以上のパブリックサーバー。Linux Foundationは2025年12月に新しいAgentic AI Foundationの下で管理権を引き受けました。

このフェーズで使用する仕様のリビジョンは**2025-11-25**です。非同期Tasks（SEP-1686）、URLモードelicitation（SEP-1036）、ツール付きsampling（SEP-1577）、インクリメンタルスコープ同意（SEP-835）、OAuth 2.1リソースインジケータセマンティクスが追加されています。Phase 13 · 09〜16でそれらの拡張を説明します。このレッスンはベースで止まります。

## 概念

### 3つのサーバープリミティブ

1. **Tools。** 呼び出し可能なアクション。Phase 13 · 01と同じ4ステップループです。
2. **Resources。** 公開されたデータ。URIでアドレス可能な読み取り専用コンテンツ：`file:///path`、`db://query/...`、カスタムスキーム。
3. **Prompts。** 再利用可能なテンプレート。ホストUIのスラッシュコマンド；サーバーがテンプレートを提供し、クライアントが引数を埋めます。

### 3つのクライアントプリミティブ

4. **Roots。** サーバーが触ることを許可されたURIのセット。クライアントが宣言し、サーバーが尊重します。
5. **Sampling。** サーバーがクライアントのモデルに補完を実行するよう要求します。サーバーサイドAPIキーなしでサーバーホストのエージェントループを実現します。
6. **Elicitation。** サーバーが実行中にクライアントのユーザーに構造化入力を要求します。フォームまたはURL（SEP-1036）。

MCPのすべてのケイパビリティはこれら6つのうちの1つに属します。Phase 13 · 10〜14でそれぞれを詳しく説明します。

### ワイヤーフォーマット：JSON-RPC 2.0

すべてのメッセージはこれらのフィールドを持つJSONオブジェクトです：

- リクエスト：`{jsonrpc: "2.0", id, method, params}`。
- レスポンス：`{jsonrpc: "2.0", id, result | error}`。
- 通知：`{jsonrpc: "2.0", method, params}` — `id`なし、レスポンス不要。

ベース仕様には約15のメソッドがあり、プリミティブごとにグループ化されています。重要なもの：

- `initialize` / `initialized`（ハンドシェイク）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（サーバーからクライアントへ）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 3フェーズのライフサイクル

**フェーズ1：initialize。**

クライアントは自身の`capabilities`と`clientInfo`を含む`initialize`を送ります。サーバーは自身の`capabilities`、`serverInfo`、対応する仕様バージョンで応答します。クライアントはレスポンスを消化したら`notifications/initialized`を送ります。ここからは交渉されたケイパビリティに従って、どちらの側もリクエストを送ることができます。

**フェーズ2：operation。**

双方向。クライアントは`tools/list`で発見し、次に`tools/call`で呼び出します。サーバーは`sampling`ケイパビリティを宣言していれば`sampling/createMessage`を送ることができます。サーバーはツールセットが変更されると`notifications/tools/list_changed`を送ることができます。クライアントはユーザーがルートスコープを変更すると`notifications/roots/list_changed`を送ることができます。

**フェーズ3：shutdown。**

どちらの側もトランスポートを閉じます。MCPには構造化されたシャットダウンメソッドはありません；トランスポート（stdioまたはStreamable HTTP、Phase 13 · 09）が接続終了シグナルを運びます。

### ケイパビリティネゴシエーション

`initialize`ハンドシェイクの`capabilities`が契約です。サーバーからの例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

サーバーは`tools/list_changed`通知を出力し、`resources/subscribe`をサポートすることを宣言します。クライアントは自身のものを宣言することで同意します：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

クライアントが`sampling`を宣言しない場合、サーバーは`sampling/createMessage`を呼び出してはなりません。対称的に：サーバーが`resources.subscribe`を宣言しない場合、クライアントはサブスクライブしてはなりません。

これがエコシステムのドリフトを防ぎます。samplingをサポートしないクライアントは依然として有効なMCPクライアントです；`sampling`を呼び出さないサーバーは依然として有効なMCPサーバーです。ただし、その機能を互いに使わないだけです。

### 構造化コンテンツとエラー形式

`tools/call`は型付きブロックの`content`配列を返します：`text`、`image`、`resource`。Phase 13 · 14はそのリストにMCP Apps（`ui://`インタラクティブUI）を追加します。

エラーはJSON-RPCエラーコードを使います。仕様で定義された追加：`-32002`「リソースが見つかりません」、`-32603`「内部エラー」、`error.data`としてのMCP固有エラーデータ。

### クライアントケイパビリティ対ツール呼び出し詳細

よくある混乱：`capabilities.tools`はクライアントがtool-list-changed通知をサポートするかどうかです。クライアントが特定のツールを実際に呼び出すかどうかは、ケイパビリティフラグではなくそのモデルによるランタイムの選択です。ケイパビリティフラグは仕様レベルの契約です。モデルの選択はそれとは直交します。

### なぜJSON-RPCでRESTではないのか？

JSON-RPC 2.0（2010）は軽量な双方向プロトコルです。RESTはクライアント起点です。MCPはサーバー起点のメッセージ（sampling、通知）が必要だったため、対称的なリクエスト/レスポンス形式を持つJSON-RPCがstdioとWebSocket/Streamable HTTPに自然に合いました。

## 使ってみる

`code/main.py`には最小限のJSON-RPC 2.0パーサとエミッタが付属しており、`initialize` → `tools/list` → `tools/call` → `shutdown`シーケンスを手動で実行してすべてのメッセージを出力します。実際のトランスポートはありません；メッセージの形式だけです。各エンベロープを確認するために「参考資料」にリンクされた仕様と比較してください。

確認すべき点：

- `initialize`は両方向でケイパビリティを宣言します；レスポンスには`serverInfo`と`protocolVersion: "2025-11-25"`があります。
- `tools/list`は`tools`配列を返し、各エントリには`name`、`description`、`inputSchema`があります。
- `tools/call`は`params.name`と`params.arguments`を使います。
- レスポンスの`content`は`{type, text}`ブロックの配列です。

## 成果物を出す

このレッスンは`outputs/skill-mcp-handshake-tracer.md`を生成します。MCPクライアントとサーバーのやり取りのpcapスタイルのトランスクリプトを受け取り、各メッセージにプリミティブ、ライフサイクルフェーズ、依存するケイパビリティをアノテートします。

## 演習

1. `code/main.py`を実行します。ケイパビリティネゴシエーションが発生する行を特定し、サーバーが`tools.listChanged`を宣言しなかった場合に何が変わるかを説明します。

2. パーサを拡張して`notifications/progress`を処理します。メッセージ形式：`{method: "notifications/progress", params: {progressToken, progress, total}}`。長時間実行中の`tools/call`の途中で出力し、クライアントハンドラがプログレスバーを表示することを確認します。

3. MCP 2025-11-25仕様を最初から最後まで読みます — ドキュメント全体は約80ページです。ほとんどのサーバーが必要としない1つのケイパビリティフラグを特定します。ヒント：リソースサブスクリプションに関連します。

4. 仮想の「cronジョブ」機能はどのプリミティブに属するかを紙上でスケッチします。（ヒント：サーバーがクライアントにスケジュールされた時間に呼び出すことを望んでいます。今日の6つのプリミティブはどれも合いません。）MCPの2026年ロードマップにはこのためのドラフトSEPがあります。

5. GitHubのオープンなMCPサーバーから1つのセッションログをパースします。リクエスト対レスポンス対通知メッセージをカウントします。ライフサイクル対オペレーションのトラフィックの割合を計算します。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|----------------|------------------------|
| MCP | 「Model Context Protocol」 | モデルとツールの発見と呼び出しのためのオープンプロトコル |
| サーバープリミティブ | 「サーバーが公開するもの」 | tools（アクション）、resources（データ）、prompts（テンプレート） |
| クライアントプリミティブ | 「クライアントがサーバーに許可するもの」 | roots（スコープ）、sampling（LLMコールバック）、elicitation（ユーザー入力） |
| JSON-RPC 2.0 | 「ワイヤーフォーマット」 | 対称的なリクエスト/レスポンス/通知エンベロープ |
| `initialize`ハンドシェイク | 「ケイパビリティネゴシエーション」 | 最初のメッセージペア；サーバーとクライアントがサポートする機能を宣言 |
| `tools/list` | 「発見」 | クライアントがサーバーに現在のツールセットを尋ねる |
| `tools/call` | 「呼び出し」 | クライアントがサーバーに引数付きでツールを実行するよう要求する |
| `notifications/*_changed` | 「変更イベント」 | サーバーがプリミティブリストが変更されたことをクライアントに伝える |
| コンテンツブロック | 「型付き結果」 | ツール結果の`{type: "text" \| "image" \| "resource" \| "ui_resource"}` |
| SEP | 「仕様進化提案」 | 名前付きドラフト提案（例：非同期Tasksの SEP-1686） |

## 参考資料

- [Model Context Protocol — 仕様2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 標準仕様ドキュメント
- [Model Context Protocol — アーキテクチャの概念](https://modelcontextprotocol.io/docs/concepts/architecture) — 6プリミティブのメンタルモデル
- [Anthropic — Model Context Protocolの紹介](https://www.anthropic.com/news/model-context-protocol) — 2024年11月の発表記事
- [MCPブログ — MCP初周年](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 1周年の振り返りと2025-11-25仕様の変更点
- [WorkOS — MCP 2025-11-25仕様の更新](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686、1036、1577、835、1724のサマリー
