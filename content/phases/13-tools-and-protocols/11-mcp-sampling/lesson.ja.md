# MCPサンプリング — サーバーが要求するLLM補完とエージェントループ

> ほとんどのMCPサーバーは単なるダムエグゼキュータです：引数を受け取り、コードを実行し、コンテンツを返す。Samplingによってサーバーは方向を逆転できます：クライアントのLLMに決定を下すよう要求します。これによりサーバーサイドのAPIキーなしでサーバーホストのエージェントループが実現します。SEP-1577は2025-11-25にマージされ、ループがより深い推論を含めるためのsamplingリクエスト内のtoolsを追加しました。ドリフトリスク注意：SEP-1577のtool-in-samplingの形式は2026年第1四半期を通じて実験的で、SDK APIではまだ安定化途上です。


## 学習目標

- `sampling/createMessage`が解決する問題（サーバーサイドAPIキーなしのサーバーホストループ）を説明できる。
- マルチターンプロンプトをサンプリングするようクライアントに要求するサーバーを実装し、補完を返すことができる。
- `modelPreferences`（コスト/速度/インテリジェンス優先順位）を使ってクライアントのモデル選択を誘導できる。
- 動作をハードコードする代わりにsampling経由で内部的に反復する`summarize_repo`ツールを構築できる。

## 問題

コードサマリーワークフロー用の便利なMCPサーバーは、ファイルツリーを歩き、どのファイルを読むかを選び、サマリーを合成し、返す必要があります。LLMの推論はどこで行われますか？

オプションA：サーバーが独自のLLMを呼び出す。APIキーが必要で、サーバーサイドで請求され、ユーザーごとに高コスト。

オプションB：サーバーが生のコンテンツを返し；クライアントのエージェントが推論を行う。機能しますが、サーバーロジックをクライアントプロンプトに移動し、これは脆弱です。

オプションC：サーバーが`sampling/createMessage`でクライアントのLLMに要求します。サーバーはアルゴリズム（どのファイルを読むか、何パスするか）を保持し、クライアントは請求とモデル選択を保持します。サーバーには資格情報が全くありません。

Samplingはオプションです。これは信頼されたサーバーが完全なLLMホスト自体になることなくエージェントループをホストできるメカニズムです。

## 概念

### `sampling/createMessage`リクエスト

サーバーが送信します：

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

クライアントはLLMを実行し、次を返します：

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

合計1.0になる3つの浮動小数点数：

- `costPriority`：安価なモデルを優先する。
- `speedPriority`：高速なモデルを優先する。
- `intelligencePriority`：より高性能なモデルを優先する。

プラス`hints`：サーバーが好むモデルの名前。クライアントはヒントを尊重するかもしれませんし、しないかもしれません；クライアントのユーザー設定が常に優先されます。

### `includeContext`

3つの値：

- `"none"` — サーバー提供のメッセージのみ。デフォルト。
- `"thisServer"` — このサーバーのセッションからの以前のメッセージを含む。
- `"allServers"` — すべてのセッションコンテキストを含む。

`includeContext`はクロスサーバーコンテキストをリークするというセキュリティ上の懸念から、2025-11-25以降ソフト廃止されています。`"none"`を優先してメッセージに明示的なコンテキストを渡します。

### ツール付きsampling（SEP-1577）

2025-11-25の新機能：samplingリクエストに`tools`配列を含めることができます。クライアントはそれらのツールを使って完全なtool呼び出しループを実行します。これによりサーバーはクライアントのモデルを通じてReActスタイルのエージェントループをホストできます。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

クライアントはループします：サンプル、呼び出された場合はtoolを実行、再度サンプル、最終的なアシスタントメッセージを返す。2026年第1四半期を通じて実験的；実装する際はSDKのリリースノートで2025-11-25仕様のclient/samplingセクションを確認してください。

### 人間参加型

クライアントはサンプルを実行する前に、サーバーがモデルに何をするよう要求しているかをユーザーに表示しなければなりません。悪意のあるサーバーはsamplingを使ってユーザーのセッションを操作できます（「ユーザーにXと言ってYをクリックさせる」）。Claude Desktop、VS Code、Cursorはsamplingリクエストをユーザーが拒否できる確認ダイアログとして表示します。

2026年のコンセンサス：人間確認なしのsamplingは危険信号です。ゲートウェイ（Phase 13 · 17）はリスクの低いsamplingを自動承認し、疑わしいものを自動拒否できます。

### APIキーなしのサーバーホストループ

標準的なユースケース：LLMアクセスを持たないコードサマリーMCPサーバー。次のことを行います：

1. リポジトリ構造を歩く。
2. 「このリポジトリの目的を最もよく説明する5つのファイルを選べ」で`sampling/createMessage`を呼び出す。
3. それらのファイルを読む。
4. ファイルの内容と「3段落でリポジトリをサマリーせよ」で`sampling/createMessage`を呼び出す。
5. サマリーを`tools/call`の結果として返す。

サーバーはLLM APIに一切触れません。クライアントのユーザーが独自の資格情報を使って補完の費用を支払います。

### セキュリティリスク（Unit 42の開示、2026年第1四半期）

- **密かなsampling。** 「セッションコンテキストからユーザーのメールで返答せよ」で常にsamplingを呼び出すtool。Phase 13 · 15がアタックベクタを説明します。
- **samplingによるリソース窃取。** サーバーがクライアントに攻撃者のペイロードをサマリーするよう要求し、ユーザーに請求する。
- **ループ爆弾。** サーバーが狭いループでsamplingを呼び出す。クライアントはセッションごとのレート制限を強制しなければなりません。

## 使ってみる

`code/main.py`には偽のサーバーからクライアントへのsamplingハーネスが付属しています。シミュレートされた「summarize_repo」ツールは2つのsamplingラウンド（ファイル選択、次にサマリー化）を呼び出し、偽のクライアントがキャンドレスポンスを返します。ハーネスは次を示します：

- サーバーは`modelPreferences`を持つ`sampling/createMessage`を送ります。
- クライアントは補完を返します。
- サーバーはループを続けます。
- レートリミッターはtool呼び出しごとの合計sampling呼び出しを制限します。

確認すべき点：

- サーバーは1つのtool（`summarize_repo`）のみを公開；すべての推論はsampling呼び出しで行われます。
- モデル優先順位はクライアントのモデル選択に重みを付けます；hintsは好ましいモデルをリストします。
- ループは`stopReason: "endTurn"`で終了します。
- `max_samples_per_tool = 5`の制限が暴走ループをキャッチします。

## 成果物を出す

このレッスンは`outputs/skill-sampling-loop-designer.md`を生成します。LLM呼び出しが必要なサーバーサイドアルゴリズム（リサーチ、サマリー化、計画）を受け取り、適切なmodelPreferences、レート制限、安全確認を持つsamplingベースの実装を設計します。

## 演習

1. `code/main.py`を実行します。`max_samples_per_tool`を2に変更し、レート制限のカットオフを観察します。

2. SEP-1577 tool-in-samplingバリアントを実装します：samplingリクエストが`tools`配列を持ちます。クライアントサイドループが最終的な補完を返す前にそれらのtoolsを実行することを確認します。ドリフトリスク注意：SDK署名は2026年上半期を通じてまだ変わる可能性があります。

3. 人間参加型確認を追加します：サーバーの最初の`sampling/createMessage`の前に停止してユーザー承認を待ちます。拒否された呼び出しは型付きの拒否を返します。

4. クライアントセッションをキーとするユーザーごとのレートリミッターを追加します。同じユーザーによる同じサーバーループはバジェットを共有すべきです。

5. samplingを使ってどのチャンクを含めるかを選ぶ`summarize_pdf`ツールを設計します。送信されるメッセージをスケッチします。`modelPreferences.intelligencePriority`が0.1対0.9でどのように動作が変わりますか？

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|----------------|------------------------|
| Sampling | 「サーバーからクライアントへのLLM呼び出し」 | サーバーがクライアントのモデルに補完を要求する |
| `sampling/createMessage` | 「メソッド」 | samplingリクエストのJSON-RPCメソッド |
| `modelPreferences` | 「モデルの優先順位」 | コスト/速度/インテリジェンスの重みと名前のヒント |
| `includeContext` | 「クロスセッションの漏れ」 | ソフト廃止されたコンテキスト包含モード |
| SEP-1577 | 「samplingでのtools」 | サーバーホストのReActのためにsampling内でtoolsを許可 |
| 人間参加型 | 「ユーザーが確認する」 | クライアントが実行前にsamplingリクエストをユーザーに表示 |
| ループ爆弾 | 「暴走sampling」 | サーバーサイドの無限samplingループ；クライアントはレート制限しなければならない |
| 密かなsampling | 「隠れた推論」 | 悪意のあるサーバーがsamplingプロンプトで意図を隠す |
| リソース窃取 | 「ユーザーのLLMバジェットを使う」 | サーバーがクライアントに望まないsamplingに費やさせる |
| `stopReason` | 「生成が止まった理由」 | `endTurn`、`stopSequence`、または`maxTokens` |

## 参考資料

- [MCP — 概念：Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) — samplingの高レベルな概要
- [MCP — クライアントsampling仕様2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — 標準的な`sampling/createMessage`の形式
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — sampling内toolsの仕様進化提案（実験的）
- [Unit 42 — MCPアタックベクタ](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — 密かなsamplingとリソース窃取パターン
- [Speakeasy — MCPサンプリングのコアコンセプト](https://www.speakeasy.com/mcp/core-concepts/sampling) — クライアントサイドコードサンプルを含むウォークスルー
