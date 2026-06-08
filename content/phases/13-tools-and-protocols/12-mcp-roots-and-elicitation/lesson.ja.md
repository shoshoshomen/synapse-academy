# RootsとElicitation — スコーピングとフライト中のユーザー入力

> ハードコードされたパスはユーザーが別のプロジェクトを開いた瞬間に壊れます。事前に埋められたtool引数はユーザーが不十分な指定をすると壊れます。RootsはサーバーをユーザーコントロールのURIセットにスコープします；Elicitationは構造化されたユーザー入力をフォームやURL経由でフォームからtool呼び出しの途中に求めるために一時停止します。2つのクライアントプリミティブ、一般的なMCP障害モードへの2つの修正。SEP-1036（URLモードelicitation、2025-11-25）は2026年上半期を通じて実験的 — 依存する前にSDKバージョンを確認してください。


## 学習目標

- `roots`を宣言し、`notifications/roots/list_changed`に応答できる。
- サーバーのファイル操作を宣言されたrootセット内のURIに制限できる。
- `elicitation/create`を使ってtool呼び出しの途中で確認や構造化入力をユーザーに求めることができる。
- フォームモードとURLモードのelicitationのどちらかを選択できる（後者は実験的；ドリフトリスクを記載）。

## 問題

本番でノートMCPサーバーが遭遇する2つの具体的な失敗：

**壊れたパスの前提。** サーバーは`~/notes`に対して書かれています。ノートが`~/Documents/Notes`に入っている別のマシンのユーザーは、サイレントに失敗する（ファイルが見つからない）または最悪の場合、間違った場所に書き込むtool呼び出しを受け取ります。

**ユーザーが知っているはずの引数の欠落。** ユーザーは「古いTPS reportノートを削除して」と頼みます。モデルは`notes_delete(title: "TPS report")`を呼び出しますが、2023年、2024年、2025年から3つの一致するノートがあります。toolは推測できません。「曖昧」で失敗するのは面倒；3つすべてで実行するのは壊滅的です。

Rootsは最初の問題を解決します：クライアントは`initialize`でサーバーが触れることのできるURIセットを宣言します。Elicitationは2番目を解決します：サーバーはtool呼び出しを一時停止して`elicitation/create`を送り、ユーザーにどれを選ぶかを尋ねます。

## 概念

### Roots

クライアントは`initialize`でrootリストを宣言します：

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

サーバーはその後`roots/list`を呼び出せます：

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

サーバーはrootsを境界として扱わなければなりません：rootセット外のファイルの読み取りまたは書き込みは拒否されます。これはクライアントによって強制されません（サーバーはユーザーが信頼したコードです）が、仕様準拠のサーバーはこれを尊重します。

ユーザーがrootを追加または削除すると、クライアントは`notifications/roots/list_changed`を送ります。サーバーは`roots/list`を再呼び出しし、その境界を更新します。

### なぜrootsがクライアントプリミティブなのか

Rootsはクライアントによって宣言されます。ユーザーの同意モデルを表しているためです。ユーザーはClaude Desktopに「このノートサーバーにこれら2つのディレクトリへのアクセスを与える」と伝えました。サーバーはそのスコープを拡大できません。

### Elicitation：デフォルトのフォームモード

`elicitation/create`はフォームスキーマと自然言語のプロンプトを受け取ります：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

クライアントはフォームをレンダリングし、ユーザーの回答を収集し、返します：

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

3つのアクション：`accept`（ユーザーが入力した）、`decline`（ユーザーが閉じた）、`cancel`（ユーザーがtool呼び出し全体を中断した）。

フォームスキーマはフラットです — ネストされたオブジェクトはv1ではサポートされていません。SDKは通常、単一レイヤーより複雑なものを拒否します。

### Elicitation：URLモード（SEP-1036、実験的）

2025-11-25の新機能。スキーマの代わりに、サーバーはURLを送ります：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

クライアントはブラウザでURLを開き、完了を待ち、ユーザーが戻ってくると返ります。Oauthフロー、支払い認証、フォームでは不十分なドキュメント署名に役立ちます。

ドリフトリスク注意：SEP-1036のレスポンス形式はまだ安定化中です；一部のSDKはコールバックURLを返し、他は完了トークンを返します。URLモードを本番で使う前にSDKのリリースノートを読んでください。

### Elicitationが正しいツールである場合

- 破壊的なアクション前のユーザー確認（destructiveヒント + elicitation）。
- 曖昧さの解消（N件の一致から1つを選ぶ）。
- 初回セットアップ（APIキー、ディレクトリ、設定）。
- OAuthスタイルのフロー（URLモード）。

### Elicitationが間違っている場合

- モデルが散文で尋ねることができたtoolの必須引数を埋める。通常の再プロンプトを使い、elicitationダイアログではない。
- 高頻度の呼び出し。Elicitationは会話を中断します；ループ内で発火させないでください。
- サーバーが事後に検証できるもの。検証し、エラーを返し、モデルにテキストでユーザーに尋ねさせます。

### 人間参加型のブリッジ

ElicitationとSamplingを組み合わせることでMCPの「人間参加型」モデルが実現します。サーバーのエージェントループはユーザー入力（elicitation）またはモデルの推論（sampling）のどちらかのために一時停止できます。Phase 13 · 11はsamplingを、このレッスンはelicitationをカバーしました。それらを組み合わせることでループの完全な制御が可能になります。

## 使ってみる

`code/main.py`はノートサーバーを以下で拡張します：

- サーバーがroot-list-changed通知後に再クエリする`roots/list`レスポンス。
- 複数のノートが一致する場合に`elicitation/create`を使って曖昧さを解消する`notes_delete`tool。
- シミュレートされた初回設定ページを開くURLモードelicitationを使う`notes_setup`tool。
- 宣言されたrootsの外のURIに対する操作を拒否する境界チェック。

デモは3つのシナリオを実行します：ハッピーパス（1件の一致）、曖昧さ解消（3件の一致、elicitationが発火）、root外書き込み（拒否）。

## 成果物を出す

このレッスンは`outputs/skill-elicitation-form-designer.md`を生成します。ユーザー確認または曖昧さ解消が必要かもしれないtoolを受け取り、elicitationフォームスキーマとメッセージテンプレートを設計します。

## 演習

1. `code/main.py`を実行します。曖昧さ解消パスをトリガーし、シミュレートされたユーザーの回答がtoolに戻ってルーティングされることを確認します。

2. 毎回（destructiveヒント付き）elicitation確認を必要とする新しいtool `notes_archive`を追加します。UXを確認します：これはモデルがテキストで再尋ねするのと比べてどうですか？

3. 初回OAuthフローのためのURLモードelicitationを実装します。ドリフトリスクに注意してSDKバージョンガードを追加します。

4. `roots/list`処理を拡張します：通知が届いたとき、サーバーはスコープ外になったかもしれない開いているファイルハンドルをアトミックに再読み取りして再スキャンすべきです。

5. GitHubのSEP-1036イシュー議論スレッドを読みます。サーバーがURLモードコールバックをどのように処理すべきかに影響する1つの未解決の質問を特定します。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|----------------|------------------------|
| Root | 「同意境界」 | クライアントがサーバーに触れることを許可したURI |
| `roots/list` | 「サーバーがスコープを尋ねる」 | クライアントが現在のrootセットを返す |
| `notifications/roots/list_changed` | 「ユーザーがスコープを変更した」 | クライアントがrootセットが変更されたことを通知 |
| Elicitation | 「呼び出し中にユーザーに尋ねる」 | 構造化されたユーザー入力のためのサーバー起点リクエスト |
| `elicitation/create` | 「メソッド」 | elicitationリクエストのJSON-RPCメソッド |
| フォームモード | 「スキーマ駆動フォーム」 | クライアントUIでフォームとしてレンダリングされるフラットJSON Schema |
| URLモード | 「ブラウザリダイレクト」 | SEP-1036 実験的；URLを開いて待機 |
| `accept` / `decline` / `cancel` | 「ユーザーの応答結果」 | サーバーが処理する3つの分岐 |
| 曖昧さ解消 | 「1つを選ぶ」 | toolがN個の候補を持つ場合の一般的なelicitationユースケース |
| フラットフォーム | 「トップレベルプロパティのみ」 | Elicitationスキーマはネストできない |

## 参考資料

- [MCP — クライアントroots仕様](https://modelcontextprotocol.io/specification/draft/client/roots) — roots標準リファレンス
- [MCP — クライアントelicitation仕様](https://modelcontextprotocol.io/specification/draft/client/elicitation) — elicitation標準リファレンス
- [Cisco — MCPのelicitation、構造化コンテンツ、OAuth拡張の新機能](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — 2025-11-25の追加ウォークスルー
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — URLモードelicitation提案（実験的、ドリフトリスク）
- [The New Stack — MCPのelicitationはどのようにAIツールに人間参加型をもたらすか](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — UXウォークスルー
