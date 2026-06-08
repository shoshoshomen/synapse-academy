# 非同期タスク（SEP-1686）— 即時呼び出し・後取得による長時間処理

> 実際のエージェント処理はCIの実行、深層リサーチの統合、バッチエクスポートなど数分から数時間かかることがある。同期ツール呼び出しでは接続が切断されたり、タイムアウトが発生したり、UIがフリーズしたりする。2025年11月25日にマージされたSEP-1686はTasksプリミティブを追加した。任意のリクエストをタスクとして拡張でき、結果を後から取得したり状態通知でストリーミングしたりできる。ドリフトリスクの注意：TasksはH1 2026までは実験的扱いであり、SDKの表面はまだ仕様に合わせて設計中である。


## 学習目標

- ツールを同期からタスク拡張へ格上げすべきタイミングを判断できる（サーバー側で30秒以上かかる処理）。
- タスクのライフサイクルを把握する：`working` → `input_required` → `completed` / `failed` / `cancelled`。
- クラッシュしても進行中の処理が失われないよう、タスク状態を永続化できる。
- `tasks/status` のポーリングと `tasks/result` の取得を正しく実装できる。

## 問題背景

`generate_report` ツールが数分かかる抽出パイプラインを実行するとする。同期モデルでの選択肢は以下の通りだ。

1. 3分間接続を保持し続ける。リモートトランスポートは切断し、クライアントはタイムアウトし、UIはフリーズする。
2. プレースホルダーを即返却し、クライアントにカスタムエンドポイントをポーリングさせる。MCPの統一性が損なわれる。
3. Fire-and-forget（結果なし）。

どれも良くない。SEP-1686は4番目の選択肢を追加する：タスク拡張だ。任意のリクエスト（通常は `tools/call`）をタスクとしてタグ付けできる。サーバーはタスクIDを即座に返し、クライアントは `tasks/status` をポーリングして完了時に `tasks/result` を取得する。サーバー側の状態は再起動後も保持される。

## コンセプト

### タスク拡張

`params._meta.task.required: true`（または `optional: true`、サーバーが判断）を設定することでリクエストをタスクに昇格できる。サーバーは即座に以下を返す：

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` は状態保持に対するサーバーの保証時間で、ttl経過後にタスク結果は破棄される。

### ツール単位のオプトイン

ツールアノテーションでタスクサポートを宣言できる：

- `taskSupport: "forbidden"` — このツールは常に同期実行。高速なツールに適している。
- `taskSupport: "optional"` — クライアントがタスク拡張をリクエストしてもよい。
- `taskSupport: "required"` — クライアントはタスク拡張を必ず使用しなければならない。

`generate_report` ツールは `required`、`notes_search` ツールは `forbidden` になる。

### 状態

```
working  -> input_required -> working  (elicitation経由のループ)
working  -> completed
working  -> failed
working  -> cancelled
```

ステートマシンは追記専用：`completed`、`failed`、または `cancelled` になると、タスクは終端状態となる。

### メソッド

- `tasks/status {taskId}` — 現在の状態と進捗ヒントを返す。
- `tasks/result {taskId}` — ブロックするか、未完了の場合は404を返す。
- `tasks/cancel {taskId}` — 冪等性あり；終端状態では無視される。
- `tasks/list` — オプション；アクティブなタスクと最近完了したタスクを列挙する。

### 状態変更のストリーミング

サーバーがサポートする場合、クライアントは状態通知をサブスクライブできる：

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

ポーリングよりもストリーミングの方がUXが良い。ポーリングは最小限のサーフェスとして常にサポートされている。

### 耐久性のある状態

仕様では、タスクサポートを宣言するサーバーが状態を永続化することを要求している。クラッシュしてもttl内の完了結果が失われてはならない。ストアはSQLiteからRedisまたはファイルシステムまで様々だ。Lesson 13のハーネスはファイルシステムを使用している。

### キャンセルのセマンティクス

`tasks/cancel` は冪等性がある。タスクが実行中の場合、サーバーは停止を試みる（エグゼキューターの協調キャンセルを確認する）。すでに終端状態の場合、リクエストはno-opとなる。

### クラッシュリカバリ

サーバープロセスが再起動する際：

1. すべての永続化されたタスク状態を読み込む。
2. プロセスが終了した `working` 状態のタスクを `CRASH_RECOVERY` エラーで `failed` としてマークする。
3. `completed` / `failed` / `cancelled` はttlまで保持する。

### 非同期タスクとサンプリング

タスク自体が `sampling/createMessage` を呼び出せる。長期実行リサーチタスクはこのように機能する：サーバーのタスクスレッドが必要に応じてクライアントのモデルをサンプリングし、クライアントのUIはタスクを `working` として定期的な進捗更新と共に表示する。

### なぜ実験的なのか

SEP-1686は2025年11月25日にリリースされたが、より広いロードマップには3つのオープン問題が指摘されている：耐久性のあるサブスクリプションプリミティブ、サブタスク（親子タスク関係）、および結果のTTL標準化だ。仕様は2026年を通じて進化すると予想される。本番コードでは、一般的なケースのみTasksを安定したものとして扱い、サブタスクの将来のSDK変更に備えてガードすべきだ。

## 使ってみる

`code/main.py` は耐久性のあるタスクストア（ファイルシステムバックエンド）と、バックグラウンドスレッドで動作する `generate_report` ツールを実装している。クライアントはツールを呼び出してタスクIDを即座に受け取り、ワーカーが進捗を更新する間 `tasks/status` をポーリングし、完了時に `tasks/result` を取得する。キャンセルと、ワーカースレッドを強制終了してから状態を再読み込みするクラッシュリカバリのシミュレーションも動作する。

確認すべきポイント：

- タスク状態のJSONが `/tmp/lesson-13-tasks/<id>.json` に永続化される。
- ワーカースレッドが `progress` フィールドを更新し、ポーリングで進捗が進む様子が確認できる。
- クライアント側からのキャンセルがイベントを設定し、ワーカーが確認して早期終了する。
- 「クラッシュ」時の状態再読み込みで、進行中のタスクが `CRASH_RECOVERY` で `failed` としてマークされる。

## 成果物を出す

このレッスンでは `outputs/skill-task-store-designer.md` を生成する。長期実行ツール（リサーチ、ビルド、エクスポート）が与えられると、このスキルはタスクストア（状態の形状、ttl、耐久性）を設計し、適切なtaskSupportフラグを選択し、進捗通知のスケッチを作成する。

## 演習

1. `code/main.py` を実行する。`generate_report` タスクを開始し、ステータスをポーリングして、結果を取得する。

2. 実行途中に `tasks/cancel` を呼び出す。ワーカーがそれを尊重し、状態が `cancelled` になることを確認する。

3. クラッシュリカバリをシミュレートする：ワーカースレッドを強制終了し、ローダーを再起動して、`CRASH_RECOVERY` の失敗モードを観察する。

4. ストアをSQLiteに拡張する。耐久性の利点は同じだが、クエリのオプションが増える（セッションXのすべてのタスクをリスト表示するなど）。

5. MCPの2026年ロードマップ記事を読む。次の1年間でSDK APIの設計に最も影響を与えそうなTasksに関連するオープン問題を1つ特定する。

## キーワード

| 用語 | よく言われる表現 | 実際の意味 |
|------|----------------|------------------------|
| Task | "長期実行ツール呼び出し" | 非同期実行のための `_meta.task` で拡張されたリクエスト |
| SEP-1686 | "Tasksの仕様" | 2025年11月25日にTasksを追加したSpec Evolution Proposal |
| `_meta.task` | "タスクエンベロープ" | id、state、ttlを含むリクエストごとのメタデータ |
| taskSupport | "ツールフラグ" | ツールごとの `forbidden` / `optional` / `required` |
| `tasks/status` | "ポーリングメソッド" | 現在の状態とオプションの進捗ヒントを取得 |
| `tasks/result` | "結果の取得" | 完了したペイロードを返すか、未完了なら404 |
| `tasks/cancel` | "停止" | 冪等性のあるキャンセルリクエスト |
| ttl | "保持期間" | サーバーがタスク状態を保持すると約束するミリ秒数 |
| `notifications/tasks/updated` | "状態プッシュ" | サーバーが開始する状態変更イベント |
| Durable store | "クラッシュセーフな状態" | ファイルシステム / SQLite / Redisの永続化層 |

## 参考資料

- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — 起源となる提案と完全な議論
- [WorkOS — MCP async tasks for AI agent workflows](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) — 根拠付きの設計ウォークスルー
- [DeepWiki — MCP task system and async operations](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) — メカニズムとステートマシン
- [FastMCP — Tasks](https://gofastmcp.com/servers/tasks) — SDKレベルのタスク実装パターン
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — サブタスクを含む2026年の優先事項とオープン問題
