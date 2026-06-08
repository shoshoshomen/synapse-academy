# MCPサーバーの構築 — Python + TypeScript SDK

> ほとんどのMCPチュートリアルはstdioのhello-worldだけを示します。本物のサーバーはtools、resources、promptsを公開し、ケイパビリティネゴシエーションを処理し、構造化されたエラーを出力し、SDKをまたいで同じように機能します。このレッスンではノートサーバーをエンドツーエンドで構築します：標準ライブラリのstdioトランスポート、JSON-RPCディスパッチ、3つのサーバープリミティブ、そしてPython SDKのFastMCPまたはTypeScript SDKに切り替える際にそのまま使えるピュア関数スタイル。


## 学習目標

- `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list`、`prompts/get`メソッドを実装できる。
- stdinからJSON-RPCメッセージを読み取り、stdoutにレスポンスを書き込むディスパッチループを作成できる。
- JSON-RPC 2.0仕様とMCPの追加コードに従って構造化されたエラーレスポンスを出力できる。
- ツールロジックを書き直さずに標準ライブラリ実装をFastMCP（Python SDK）またはTypeScript SDKに移行できる。

## 問題

リモートトランスポート（Phase 13 · 09）や認証レイヤー（Phase 13 · 16）を使う前に、クリーンなローカルサーバーが必要です。ローカルとはstdioを意味します：サーバーはクライアントによって子プロセスとして起動され、メッセージはstdin/stdoutを経由して改行区切りで流れます。

2025-11-25仕様ではstdioメッセージが明示的な`\n`セパレータを持つJSONオブジェクトとしてエンコードされることが規定されています。ここにSSEはありません；SSEは古いリモートモードであり2026年半ばに削除されています（AtlassianのRovo MCPサーバーは2026年6月30日に廃止；Keboola は2026年4月1日；残りのほとんどのエンタープライズサーバーは2026年末までに）。stdioの場合、1行に1つのJSONオブジェクトがワイヤーフォーマットの全てです。

ノートサーバーは3つすべてのサーバープリミティブを使うため良い形状です。Toolsは変更（`notes_create`）を行います。Resourcesはデータ（`notes://{id}`）を公開します。Promptsはテンプレート（`review_note`）を提供します。このレッスンの形状はあらゆるドメインに汎用化できます。

## 概念

### ディスパッチループ

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

3つのルール：

- JSON-RPCエンベロープ以外のものをstdoutに出力しない。デバッグログはstderrに送る。
- すべてのリクエストは同じ`id`を持つレスポンスとマッチしなければならない。
- 通知にはレスポンスしてはならない。

### `initialize`の実装

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

サポートするものだけを宣言します。クライアントはケイパビリティセットに依存して機能をゲートします。

### `tools/list`と`tools/call`の実装

`tools/list`は各エントリに`name`、`description`、`inputSchema`を持つ`{tools: [...]}`を返します。`tools/call`は`{name, arguments}`を受け取り、`{content: [blocks], isError: bool}`を返します。

コンテンツブロックは型付きです。最も一般的なもの：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

ツールエラーは2つの形があります。プロトコルレベルのエラー（未知のメソッド、不正なパラメータ）はJSON-RPCエラーです。ツールレベルのエラー（有効な呼び出しだがツールが失敗）は`{content: [...], isError: true}`として返されます。これによりモデルはコンテキストで失敗を確認できます。

### リソースの実装

リソースは設計上読み取り専用です。`resources/list`はマニフェストを返し；`resources/read`はコンテンツを返します。URIは`file://...`、`http://...`、または`notes://`のようなカスタムスキームが使えます。

データをツールではなくリソースとして公開する場合：

- モデルはそれを「呼び出す」のではなく；クライアントがユーザーのリクエストでそれをコンテキストに注入できます。
- サブスクリプションによりサーバーはリソースが変更されたときに更新をプッシュできます（Phase 13 · 10）。
- Phase 13 · 14はこれを`ui://`でインタラクティブリソースに拡張します。

### プロンプトの実装

プロンプトは名前付き引数を持つテンプレートです。ホストはそれらをスラッシュコマンドとして表示します。`review_note`プロンプトは`note_id`引数を受け取り、クライアントがモデルに渡す複数メッセージのプロンプトテンプレートを生成します。

### stdioトランスポートの細かい点

- 改行区切りJSON。長さプレフィクスのフレーミングなし。
- バッファリングしない。書き込みのたびに`sys.stdout.flush()`。
- クライアントがライフタイムを制御します。stdinが閉じたら（EOF）クリーンに終了します。
- SIGPIPEを無視して終了しない；ログに記録して終了します。

### アノテーション

各ツールは安全特性を説明する`annotations`を持てます：

- `readOnlyHint: true` — 純粋な読み取り、再試行可能。
- `destructiveHint: true` — 元に戻せない副作用；クライアントは確認すべき。
- `idempotentHint: true` — 同じ入力が同じ出力を生成する。
- `openWorldHint: true` — 外部システムと連携する。

クライアントはこれらを使ってUX（確認ダイアログ、ステータスインジケータ）とルーティング（Phase 13 · 17）を決定します。

### 移行パス

`code/main.py`の標準ライブラリサーバーは約180行です。FastMCP（Python）はデコレータスタイルで同じロジックを表現します：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDKは同等の形状を持ちます。概念（ケイパビリティ、ディスパッチ、コンテンツブロック）は同じなので、準備ができたらドロップインで移行できます。

## 使ってみる

`code/main.py`はstdioを使った完全なノートMCPサーバーで、標準ライブラリのみです。`initialize`、3つのツールの`tools/list`と`tools/call`（`notes_list`、`notes_search`、`notes_create`）、各ノートの`resources/list`と`resources/read`、`review_note`プロンプトを処理します。JSON-RPCメッセージをパイプで渡して実行できます：

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

確認すべき点：

- ディスパッチャはメソッド名をキーとする`dict[str, Callable]`です。
- すべてのツールエグゼキュータは裸の文字列ではなくコンテンツブロックのリストを返します。
- エグゼキュータが例外を発生させると`isError: true`が設定されます。

## 成果物を出す

このレッスンは`outputs/skill-mcp-server-scaffolder.md`を生成します。ドメイン（ノート、チケット、ファイル、データベース）を受け取り、適切なtools / resources / promptsの分割とSDK移行パスを持つMCPサーバーの雛形を生成します。

## 演習

1. `code/main.py`を実行し、手作りのJSON-RPCメッセージで操作します。`notes_create`を実行し、次に`resources/read`で新しいノートを取得します。

2. `annotations: {destructiveHint: true}`を持つ`notes_delete`ツールを追加します。クライアントが確認ダイアログを表示することを確認します（実際のホストが必要；Claude Desktopが動作します）。

3. ノートが変更されるたびにサーバーが`notifications/resources/updated`をプッシュするよう`resources/subscribe`を実装します。キープアライブタスクを追加します。

4. サーバーをFastMCPに移植します。Pythonファイルは80行未満に縮小すべきです。ワイヤーの動作は同一でなければなりません；同じJSON-RPCテストハーネスで確認します。

5. 仕様の`server/tools`セクションを読み、このレッスンのサーバーで実装されていないツール定義の1つのフィールドを特定します。（ヒント：いくつかあります；1つを選んで追加してください。）

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|----------------|------------------------|
| MCPサーバー | 「ツールを公開するもの」 | stdioまたはHTTPでMCP JSON-RPCを話すプロセス |
| stdioトランスポート | 「子プロセスモデル」 | サーバーはクライアントによって起動され；stdin/stdoutで通信する |
| ディスパッチャ | 「メソッドルーター」 | JSON-RPCメソッド名からハンドラ関数へのマップ |
| コンテンツブロック | 「ツール結果チャンク」 | ツールレスポンスの`content`配列の型付き要素 |
| `isError` | 「ツールレベルの失敗」 | ツールが失敗したことを通知；JSON-RPCエラーと区別する |
| アノテーション | 「安全ヒント」 | readOnly / destructive / idempotent / openWorldフラグ |
| FastMCP | 「Python SDK」 | MCPプロトコルの上にあるデコレータベースの高レベルフレームワーク |
| リソースURI | 「アドレス可能なデータ」 | リソースを識別する`file://`、`db://`、またはカスタムスキーム |
| プロンプトテンプレート | 「スラッシュコマンドの概要」 | ホストUIの引数スロットを持つサーバー提供テンプレート |
| ケイパビリティ宣言 | 「機能トグル」 | `initialize`で宣言されるプリミティブごとのフラグ |

## 参考資料

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — Pythonリファレンス実装
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — 並行TS実装
- [FastMCP — サーバーフレームワーク](https://gofastmcp.com/) — MCPサーバー用デコレータスタイルPython API
- [MCP — クイックスタートサーバーガイド](https://modelcontextprotocol.io/quickstart/server) — いずれかのSDKを使ったエンドツーエンドチュートリアル
- [MCP — サーバーtools仕様](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — tools/*メッセージの完全リファレンス
