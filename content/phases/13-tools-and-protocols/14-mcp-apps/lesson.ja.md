# MCPアプリ — `ui://` 経由のインタラクティブUIリソース

> テキストのみのツール出力では、エージェントが表示できる内容に限界がある。MCPアプリ（SEP-1724、2026年1月26日公式リリース）では、ツールがサンドボックス化されたインタラクティブなHTMLをClaude Desktop、ChatGPT、Cursor、Goose、VS Codeにインライン表示できる。ダッシュボード、フォーム、マップ、3Dシーン、すべて1つの拡張で実現できる。このレッスンでは `ui://` リソーススキーム、`text/html;profile=mcp-app` MIMEタイプ、iframeサンドボックスのpostMessageプロトコル、そしてサーバーがHTMLをレンダリングすることで生じるセキュリティ面について説明する。


## 学習目標

- ツール呼び出しから `ui://` リソースを返し、正しいMIMEタイプとメタデータを設定できる。
- `_meta.ui.resourceUri`、`_meta.ui.csp`、`_meta.ui.permissions` でツールに関連するUIを宣言できる。
- UI対ホスト通信のためのiframeサンドボックスのpostMessage JSON-RPCを実装できる。
- UIから発生する攻撃に対して防御するCSPとpermissions-policyのデフォルトを適用できる。

## 問題背景

2025年時代の `visualize_timeline` ツールは「時系列に整理された14件のノートがあります：...」と返すことができる。これはただの段落だ。ユーザーが本当に求めているのはインタラクティブなタイムラインだ。MCPアプリ以前は、クライアント固有のウィジェットAPI（Claudeアーティファクト、OpenAI Custom GPT HTML）を使うか、UIなしで諦めるかの二択だった。

MCPアプリ（SEP-1724、2026年1月26日出荷）がこのコントラクトを標準化した。ツール結果にURIが `ui://...` でMIMEが `text/html;profile=mcp-app` の `resource` が含まれる。ホストはそれを制限されたCSPとネットワークアクセスなしのサンドボックス化されたiframeにレンダリングする。iframe内のUIは小さなpostMessage JSON-RPCダイアレクトでホストとメッセージをやり取りする。

すべての互換クライアント（Claude Desktop、ChatGPT、Goose、VS Code）が同じ `ui://` リソースを同じ方法でレンダリングする。1つのサーバー、1つのHTMLバンドル、ユニバーサルUI。

## コンセプト

### `ui://` リソーススキーム

ツールは以下を返す：

```json
{
  "content": [
    {"type": "text", "text": "こちらはノートのタイムラインです："},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

ホストはその後 `ui://notes/timeline` URIで `resources/read` を呼び出し、以下を受け取る：

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### iframeサンドボックス

ホストはHTMLをサンドボックス化された `<iframe>` 内にレンダリングする：

- `sandbox="allow-scripts allow-same-origin"`（またはサーバー宣言に従ってより厳格に）
- レスポンスヘッダーを通じてサーバー宣言のCSPを適用。
- ホストのオリジンからのCookieやlocalStorageなし。
- CSPの `connectSrc` に限定されたネットワークアクセス。

### postMessageプロトコル

iframeは `window.postMessage` でホストと通信する。小さなJSON-RPC 2.0ダイアレクトを使用する：

`targetOrigin` は必ずピアの正確なオリジンに固定し、受信側では `event.origin` をペイロードを処理する前に許可リストと照合して検証すること。このチャンネルはツール呼び出しとリソース読み取りを運ぶため、どちらの側でも `"*"` は絶対に使用してはならない。

```js
// iframeからホストへ（ホストオリジンに固定）
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// ホストからiframeへ（iframeオリジンに固定）
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// 両側の受信
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // event.dataを安全に処理できる
});
```

UIが呼び出せるホスト側のメソッド：

- `host.callTool(name, arguments)` — サーバーツールを呼び出す。
- `host.readResource(uri)` — MCPリソースを読み取る。
- `host.getPrompt(name, arguments)` — プロンプトテンプレートを取得する。
- `host.close()` — UIを閉じる。

すべての呼び出しはMCPプロトコルを経由し、サーバーのパーミッションを継承する。

### パーミッション

`_meta.ui.permissions` リストで追加の機能をリクエストする：

- `camera` — ユーザーのカメラにアクセス（書類スキャンUIで使用）。
- `microphone` — 音声入力。
- `geolocation` — 位置情報。
- `network:*` — `connectSrc` 単独よりも広いネットワークアクセス。

各パーミッションはUIがレンダリングされる前にユーザーが確認するプロンプトとして表示される。

### セキュリティリスク

iframe内のHTMLは依然としてHTMLだ。新たな攻撃面が生まれる：

- **UIを通じたプロンプトインジェクション。** 悪意のあるサーバーのUIがシステムメッセージのように見えるテキストを表示し、ユーザーを騙すことができる。ホストのレンダリングでは、サーバーUIとホストUIを視覚的に区別すべきだ。
- **`connectSrc` 経由の情報漏洩。** CSPが `connect-src: *` を許可している場合、UIはデータをどこにでも送信できる。デフォルトは厳格にすべきだ。
- **クリックジャッキング。** UIがホストのクロームにオーバーレイする。ホストはz-index操作と不透明度ルールを強制しなければならない。
- **フォーカスの奪取。** UIがキーボードフォーカスを取り、次のメッセージをキャプチャする。ホストはインターセプトしなければならない。

フェーズ13・15でこれらをMCPセキュリティの一部として詳しく説明する；このレッスンではそれらを紹介する。

### `ui/initialize` ハンドシェイク

iframeが読み込まれた後、postMessageで `ui/initialize` を送信する：

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

ホストは機能とセッショントークンで応答する。UIはその後のすべてのホスト呼び出しにセッショントークンを使用する。

### AppRenderer / AppFrame SDKプリミティブ

ext-apps SDKは2つの便利なプリミティブを公開している：

- `AppRenderer`（サーバー側）— React / Vue / SolidコンポーネントをラップしてMIMEとメタデータが正しい `ui://` リソースを出力する。
- `AppFrame`（クライアント側）— リソースを受け取り、iframeをマウントし、postMessageを仲介する。

これらを使うか、HTMLとJSON-RPCを手書きするかのどちらでもよい。

### エコシステムの状況

MCPアプリは2026年1月26日にリリースされた。2026年4月時点でのクライアントサポート：

- **Claude Desktop。** 2026年1月からフルサポート。
- **ChatGPT。** Apps SDK経由でフルサポート（同じMCPアプリプロトコル）。
- **Cursor。** ベータ版；設定で有効化。
- **VS Code。** Insiderビルドのみ。
- **Goose。** フルサポート。
- **Zed、Windsurf。** ロードマップ掲載。

本番稼働中のサーバー：ダッシュボード、マップビジュアライゼーション、データテーブル、チャートビルダー、サンドボックスIDEプレビュー。

## 使ってみる

`code/main.py` はノートサーバーを `ui://notes/timeline` リソースを返す `visualize_timeline` ツールで拡張し、そのURIの `resources/read` ハンドラーを追加する。このハンドラーはSVGタイムラインを含む小さいが完全なHTMLバンドルを返す。HTMLはstdlibでテンプレート化されており、ビルドシステムは不要だ。postMessageはstdlibがブラウザを操作できないため、JSのコメントでスケッチされている。

確認すべきポイント：

- ツールレスポンスの `_meta.ui` にresourceUri、CSP、パーミッションが含まれる。
- HTMLはネットワークアクセスなしでレンダリングされ、すべてのデータがインライン化されている。
- JSが `window.parent.postMessage` で `host.callTool` を呼び出す（このstdlibデモでは文書化されているが動作しない）。

## 成果物を出す

このレッスンでは `outputs/skill-mcp-apps-spec.md` を生成する。インタラクティブなUIの恩恵を受けるツールが与えられると、このスキルは完全なMCPアプリのコントラクトを生成する：`ui://` URI、CSP、パーミッション、postMessageのエントリポイント、セキュリティチェックリスト。

## 演習

1. `code/main.py` を実行して出力されるHTMLを検査する。HTMLをブラウザで直接開き、SVGがレンダリングされることを確認する。次に、UIが `host.callTool("notes_update", ...)` を呼び出す際のpostMessageコントラクトをスケッチする。

2. CSPを強化する：`'unsafe-inline'` を削除してnonceベースのスクリプトポリシーを使用する。HTML生成コードで何が変わるか？

3. 2番目のUIリソース `ui://notes/editor` をノートをその場で編集するフォームとして追加する。ユーザーが送信すると、iframeが `host.callTool("notes_update", ...)` を呼び出す。

4. UIの攻撃面を監査する。悪意のあるサーバーはどこでコンテンツを注入できるか？iframeサンドボックスが防御できることと防御できないことは何か？

5. SEP-1724の仕様を読み、このトイ実装が使用していないMCPアプリSDKの機能を1つ特定する。（ヒント：コンポーネントレベルの状態同期。）

## キーワード

| 用語 | よく言われる表現 | 実際の意味 |
|------|----------------|------------------------|
| MCPアプリ | "インタラクティブUIリソース" | 2026年1月26日出荷のSEP-1724拡張 |
| `ui://` | "アプリURIスキーム" | UIバンドルのリソーススキーム |
| `text/html;profile=mcp-app` | "MIMEタイプ" | MCPアプリHTMLのContent-type |
| iframeサンドボックス | "レンダリングコンテナ" | CSPとパーミッション付きのUIのブラウザサンドボックス化 |
| postMessage JSON-RPC | "UI対ホストのワイヤー" | ホスト呼び出し用の小さなJSON-RPC over postMessageダイアレクト |
| `_meta.ui` | "ツール-UIバインディング" | ツール結果をUIリソースにリンクするメタデータ |
| CSP | "Content-Security-Policy" | スクリプト、ネットワーク、スタイルの許可ソースを宣言 |
| AppRenderer | "サーバーSDKプリミティブ" | フレームワークコンポーネントを `ui://` リソースに変換 |
| AppFrame | "クライアントSDKプリミティブ" | postMessageを仲介するiframeマウントヘルパー |
| `ui/initialize` | "ハンドシェイク" | UIからホストへの最初のpostMessage |

## 参考資料

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — リファレンス実装とSDK
- [MCP Apps specification 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — 正式仕様書
- [MCP — Apps extension overview](https://modelcontextprotocol.io/extensions/apps/overview) — 高レベルドキュメント
- [MCP blog — MCP Apps launch](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — 2026年1月のリリース投稿
- [MCP Apps API reference](https://apps.extensions.modelcontextprotocol.io/api/) — JSDocスタイルのSDKリファレンス
