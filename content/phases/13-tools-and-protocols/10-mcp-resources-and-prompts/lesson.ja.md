# MCPリソースとプロンプト — ツールを超えたコンテキスト公開

> ToolsはMCPの注目の90%を占めます。他の2つのサーバープリミティブは異なる問題を解決します。Resourcesは読み取り用のデータを公開します；Promptsはスラッシュコマンドとして再利用可能なテンプレートを公開します。多くのサーバーはToolsで読み取りをラップする代わりにResourcesを使うべきであり、クライアントプロンプトにワークフローをハードコードする代わりにPromptsを使うべきです。このレッスンでは判断ルールを示し、`resources/*`と`prompts/*`メッセージを説明します。


## 学習目標

- 特定のドメインに対してケイパビリティをtool、resource、promptのどれとして公開するかを決定できる。
- `resources/list`、`resources/read`、`resources/subscribe`を実装し、`notifications/resources/updated`を処理できる。
- 引数テンプレートを持つ`prompts/list`と`prompts/get`を実装できる。
- ホストがpromptをスラッシュコマンドとして表示する場合と自動注入コンテキストとして表示する場合を認識できる。

## 問題

ノートアプリの素朴なMCPサーバーはすべてをtoolとして公開します：`notes_read`、`notes_list`、`notes_search`。これはすべてのデータアクセスをモデル駆動のtool呼び出しにラップします。結果として：

- モデルはコンテキストから恩恵を受けるかもしれないすべてのクエリで`notes_read`を呼び出すかどうかを決定しなければなりません。
- 読み取り専用コンテンツはホストのサイドパネルにサブスクライブしたりストリームしたりできません。
- クライアントUI（Claude Desktopのリソース添付パネル、Cursorの「ファイルを含める」ピッカー）はデータを表示できません。

正しい分割：データはresourceとして公開し、変更または計算されたアクションはtoolとして公開し、再利用可能なマルチステップワークフローはpromptとして公開します。各プリミティブにはそのUXアフォーダンスとアクセスパターンがあります。

## 概念

### tools対resources対prompts — 判断ルール

| ケイパビリティ | プリミティブ |
|------------|-----------|
| ユーザーがデータを検索、フィルタ、変換したい | tool |
| ユーザーがホストにこのデータをコンテキストとして含めてほしい | resource |
| ユーザーが再実行できるテンプレートワークフローを望んでいる | prompt |

ガイドライン：モデルが関連するすべてのクエリでそれを呼び出すことから恩恵を受けるならtoolです。ユーザーがそれを会話に添付することから恩恵を受けるならresourceです。全体のマルチステップワークフローがユーザーが再利用したい単位ならpromptです。

### リソース

`resources/list`は`{resources: [{uri, name, mimeType, description?}]}`を返します。`resources/read`は`{uri}`を受け取り`{contents: [{uri, mimeType, text | blob}]}`を返します。

URIはアドレス可能なものであれば何でも：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（カスタムスキーム）
- `memory://session-2026-04-22/recent`（サーバー固有）

`contents[]`はテキストとバイナリの両方をサポートします。バイナリはbase64エンコードされた文字列の`blob`プラス`mimeType`を使います。

### リソースサブスクリプション

ケイパビリティに`{resources: {subscribe: true}}`を宣言します。クライアントは`resources/subscribe {uri}`を呼び出します。サーバーはリソースが変更されると`notifications/resources/updated {uri}`を送ります。クライアントは再読み取りします。

ユースケース：リソースがディスク上のファイルのノートサーバー；ファイルウォッチャーが更新通知をトリガー；Claude Desktopはホスト外で編集されたときにコンテキストにファイルを再取得します。

### リソーステンプレート（2025-11-25の追加）

`resourceTemplates`はパラメータ化されたURIパターンを公開できます：`notes://{id}`に`id`を補完ターゲットとして。クライアントはリソースピッカーでIDをオートコンプリートできます。

### プロンプト

`prompts/list`は`{prompts: [{name, description, arguments?}]}`を返します。`prompts/get`は`{name, arguments}`を受け取り`{description, messages: [{role, content}]}`を返します。

プロンプトはホストがモデルに渡すメッセージのリストにレンダリングするテンプレートです。例えば、`code_review`プロンプトは`file_path`引数を受け取り、クライアントがモデルに渡す3メッセージシーケンスを返します：システムメッセージ、ファイル本体を含むユーザーメッセージ、推論テンプレートを含むアシスタントキックオフ。

### ホストとプロンプト

Claude Desktop、VS Code、CursorはチャットUIのスラッシュコマンドとしてプロンプトを公開します。ユーザーは`/code_review`と入力してフォームから引数を選びます。サーバーのプロンプトは「ユーザーのショートカット」と「モデルに送られる完全なプロンプト」の間の契約です。

まだすべてのクライアントがプロンプトをサポートしているわけではありません — ケイパビリティネゴシエーションを確認してください。prompts ケイパビリティを宣言したサーバーでもpromptサポートのないクライアントはスラッシュコマンドを表示しません。

### 「リスト変更」通知

ResourcesとPromptsの両方はセットが変更されると`notifications/list_changed`を出力します。20個の新しいノートをインポートしたばかりのノートサーバーは`notifications/resources/list_changed`を出力します；クライアントは追加を取得するために`resources/list`を再呼び出しします。

### コンテンツタイプの慣例

テキストの場合：`mimeType: "text/plain"`、`text/markdown`、`application/json`。
バイナリの場合：`image/png`、`application/pdf`、プラス`blob`フィールド。
MCP Apps（レッスン14）の場合：`ui://`URIの`text/html;profile=mcp-app`。

### 動的リソース

リソースURIは静的ファイルに対応する必要はありません。`notes://recent`は毎回の読み取りで最新の5つのノートを返すことができます。`db://query/users/active`はパラメータ化されたクエリを実行できます。サーバーはコンテンツを動的に計算することができます。

ルール：クライアントがURIでキャッシュできる場合、URIは安定していなければなりません。計算が一回限りの場合、URIにはタイムスタンプまたはノンスが含まれていて、クライアントキャッシュが古くなるのを防ぐべきです。

### サブスクリプション対ポーリング

サブスクリプション対応クライアントは`notifications/resources/updated`経由でサーバープッシュを受け取ります。サブスクリプション前またはサポートしないホストはリソースを再読み取りすることでポーリングします。どちらも仕様準拠です。サーバーのケイパビリティ宣言がどちらをサポートするかをクライアントに伝えます。

サブスクリプションのコスト：サーバー上のセッションごとの状態（誰が何をサブスクライブしているか）。サブスクライブセットは境界を設けてください；切断されたクライアントはタイムアウトすべきです。

### Prompts対システムプロンプト

MCPのPromptsはシステムプロンプトではありません。ホストのシステムプロンプト（独自の操作指示）とMCPプロンプト（ユーザーが呼び出すサーバー提供のテンプレート）は並行して存在します。適切に動作するクライアントは決してサーバープロンプトが独自のシステムプロンプトを上書きすることを許可しません；それらをレイヤーとして重ねます。

## 使ってみる

`code/main.py`はLesson 07のノートサーバーを以下で拡張します：

- `resources/subscribe`サポートを持つノートごとのリソース（`notes://note-1`など）。
- 3メッセージテンプレートにレンダリングする`review_note`プロンプト。
- ノートが変更されると`notifications/resources/updated`を出力するファイルウォッチャーシミュレーション。
- 常に最新の5つのノートを返す`notes://recent`動的リソース。

デモを実行して完全なフローを確認します。

## 成果物を出す

このレッスンは`outputs/skill-primitive-splitter.md`を生成します。提案されたMCPサーバーを受け取り、各ケイパビリティをtool / resource / promptとして根拠付きで分類します。

## 演習

1. `code/main.py`を実行します。初期のリソースリストを観察し、ノートの編集をトリガーして`notifications/resources/updated`イベントが発火することを確認します。

2. `resources/list_changed`エミッターを追加します：新しいノートが作成されると、クライアントが再発見できるように通知を送ります。

3. GitHubのMCPサーバーのための3つのプロンプトを設計します：`summarize_pr`、`triage_issue`、`release_notes`。引数スキーマをそれぞれに。プロンプト本体はさらに編集なしに実行可能であるべきです。

4. Lesson 07サーバー内の既存のtoolを取り、それがtoolのままであるべきか、resource + toolのペアに分割すべきかを分類します。1文で理由を述べます。

5. 仕様の`server/resources`と`server/prompts`セクションを読みます。`resources/read`のめったに使われないが仕様サポートされている1つのフィールドを特定します。ヒント：リソースコンテンツの`_meta`を見てください。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|----------------|------------------------|
| Resource | 「公開されたデータ」 | ホストが読み取れるURIアドレス可能なコンテンツ |
| リソースURI | 「データへのポインタ」 | スキームプレフィクス付き識別子（`file://`、`notes://`など） |
| `resources/subscribe` | 「変更を監視する」 | 特定のURIのサーバープッシュ更新のクライアントオプトイン |
| `notifications/resources/updated` | 「リソースが変更された」 | サブスクライブされたリソースに新しいコンテンツがあるというクライアントへのシグナル |
| リソーステンプレート | 「パラメータ化されたURI」 | ホストピッカーの補完ヒント付きのURIパターン |
| Prompt | 「スラッシュコマンドテンプレート」 | 引数スロットを持つ名前付きマルチメッセージテンプレート |
| プロンプト引数 | 「テンプレート入力」 | ホストがレンダリング前に収集する型付きパラメータ |
| `prompts/get` | 「テンプレートのレンダリング」 | サーバーが埋め込まれたメッセージリストを返す |
| コンテンツブロック | 「型付きチャンク」 | `{type: text \| image \| resource \| ui_resource}` |
| スラッシュコマンドUX | 「ユーザーショートカット」 | ホストが`/`で始まるコマンドとしてプロンプトを表示 |

## 参考資料

- [MCP — 概念：Resources](https://modelcontextprotocol.io/docs/concepts/resources) — リソースURI、サブスクリプション、テンプレート
- [MCP — 概念：Prompts](https://modelcontextprotocol.io/docs/concepts/prompts) — プロンプトテンプレートとスラッシュコマンド統合
- [MCP — サーバーresources仕様2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — resources/*メッセージの完全リファレンス
- [MCP — サーバーprompts仕様2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — prompts/*メッセージの完全リファレンス
- [MCP — プロトコル情報サイト：resources](https://modelcontextprotocol.info/docs/concepts/resources/) — 公式ドキュメントを拡張するコミュニティガイド
