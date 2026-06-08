# Function Calling 詳解 — OpenAI、Anthropic、Gemini

> 3つのフロンティアプロバイダは2024年に同じツール呼び出しループに収束しましたが、その後は細部で分岐しました。OpenAIは`tools`と`tool_calls`を使います。Anthropicは`tool_use`と`tool_result`ブロックを使います。Geminiは`functionDeclarations`と一意IDによる相関を使います。このレッスンでは3つを並べて比較することで、あるプロバイダで動作するコードが移植時に壊れないようにします。


## 学習目標

- OpenAI、Anthropic、GeminiのFunction callingペイロードの3つの形式上の違い（宣言、呼び出し、結果）を述べられる。
- 1つのツール宣言を3つのプロバイダ形式すべてに変換し、ストリクトモードの制約がどこで異なるかを予測できる。
- 各プロバイダで`tool_choice`を使ってツール呼び出しを強制、禁止、または自動選択できる。
- プロバイダごとのハード制限（ツール数、スキーマ深さ、引数長）と、制限違反時のエラー特徴を把握している。

## 問題

Function callingリクエストの形式はプロバイダによって異なります。2026年の本番スタックからの具体的な3例を示します：

**OpenAI Chat Completions / Responses API。** `tools: [{type: "function", function: {name, description, parameters, strict}}]`を渡します。モデルのレスポンスには`choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`が含まれ、`arguments`はパースが必要なJSON文字列です。ストリクトモード（`strict: true`）は制約付きデコーディングによるスキーマ準拠を強制します。

**Anthropic Messages API。** `tools: [{name, description, input_schema}]`を渡します。レスポンスは`content: [{type: "text"}, {type: "tool_use", id, name, input}]`として返ってきます。`input`はすでにパース済みです（文字列ではなくオブジェクト）。新しい`user`メッセージに`{type: "tool_result", tool_use_id, content}`ブロックを含めて返信します。

**Google Gemini API。** `tools: [{functionDeclarations: [{name, description, parameters}]}]`（`functionDeclarations`以下にネスト）を渡します。レスポンスは`candidates[0].content.parts: [{functionCall: {name, args, id}}]`として届きます。Gemini 3以降では並列呼び出しの相関のために`id`が一意です。`{functionResponse: {name, id, response}}`で返信します。

同じループです。フィールド名、ネスト、文字列対オブジェクトの慣例、相関メカニズムが異なります。OpenAIで天気エージェントを書いたチームはAnthropicへの移植に2日、さらにGeminiへの移植に1日かかります — 単なる配管作業のために。

このレッスンでは3つの形式を1つの標準ツール宣言に統一し、エッジでルーティングするトランスレータを構築します。Phase 13 · 17では同じパターンをLLMゲートウェイに汎用化します。

## 概念

### 共通構造

すべてのプロバイダが必要とする5つのもの：

1. **ツールリスト。** ツールごとの名前、説明、入力スキーマ。
2. **ツール選択。** 特定のツールを強制、ツールを禁止、またはモデルに決めさせる。
3. **呼び出しの出力。** ツールと引数を指定する構造化出力。
4. **呼び出しID。** 応答を正しい呼び出しに対応付ける（並列の場合に重要）。
5. **結果の注入。** 結果を呼び出しに紐付けるメッセージまたはブロック。

### フィールドごとの形式の違い

| 観点 | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| 宣言エンベロープ | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| スキーマフィールド | `parameters` | `input_schema` | `parameters` |
| レスポンスコンテナ | アシスタントメッセージの`tool_calls[]` | `content[]`の`tool_use`タイプ | `parts[]`の`functionCall`タイプ |
| 引数の型 | 文字列化JSON | パース済みオブジェクト | パース済みオブジェクト |
| IDフォーマット | `call_...`（OpenAI生成） | `toolu_...`（Anthropic） | UUID（Gemini 3以降） |
| 結果ブロック | ロール`tool`、`tool_call_id` | `tool_result`と`tool_use_id`を含む`user` | 一致する`id`を持つ`functionResponse` |
| ツールを強制 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| ツールを禁止 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| ストリクトスキーマ | `strict: true` | スキーマはスキーマ（常に強制） | リクエストレベルの`responseSchema` |

### 実際に引っかかる制限

- **OpenAI。** リクエストごとに128ツール。スキーマ深さ5。引数文字列8192バイト以下。ストリクトモードでは`$ref`なし、重複なしの`oneOf`/`anyOf`/`allOf`なし、全プロパティが`required`に列挙されていること。
- **Anthropic。** リクエストごとに64ツール。スキーマ深さは事実上無制限だが実用上限は10。ストリクトモードフラグなし、スキーマは契約でありモデルは準拠する傾向がある。
- **Gemini。** リクエストごとに64関数。スキーマタイプはOpenAPI 3.0サブセット（JSON Schema 2020-12とわずかに異なる）。Gemini 3から並列呼び出しの一意ID。

### `tool_choice`の動作

3つのモード（全プロバイダがサポート、名前が異なる）：

- **Auto。** モデルがツールまたはテキストを選ぶ。デフォルト。
- **Required / Any。** モデルは少なくとも1つのツールを呼び出す必要がある。
- **None。** モデルはツールを呼び出してはならない。

さらに各プロバイダ固有のモード：

- **OpenAI。** 名前で特定のツールを強制。
- **Anthropic。** 名前で特定のツールを強制；`disable_parallel_tool_use`フラグで単一対複数を分離。
- **Gemini。** `mode: "VALIDATED"`はモデルの意図に関わらず全レスポンスをスキーマバリデータにルーティング。

### 並列呼び出し

OpenAIの`parallel_tool_calls: true`（デフォルト）は1つのアシスタントメッセージで複数の呼び出しを出力します。すべて実行して`tool_call_id`ごとに1エントリを含むバッチ化されたtoolロールメッセージで返信します。Anthropicは歴史的に単一呼び出しでした；`disable_parallel_tool_use: false`（Claude 3.5以降デフォルト）で複数が有効になります。Gemini 2は並列呼び出しを許可していましたが安定したIDがありませんでした；Gemini 3でUUIDが追加され、順序どおりでない応答もクリーンに相関付けられます。

### ストリーミング

3プロバイダすべてがストリーミングツール呼び出しをサポートします。ワイヤーフォーマットが異なります：

- **OpenAI。** `tool_calls[i].function.arguments`のデルタチャンクが段階的に届きます。チャンクは`index`（呼び出しリスト内の位置）を持ちます。インデックスごとに蓄積し、最初に表れた`id`を読み取り、`finish_reason: "tool_calls"`でJSONをパースします。
- **Anthropic。** ブロック開始/ブロックデルタ/ブロック停止イベント。`input_json_delta`チャンクが部分的な引数を運びます。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3以降）は複数の並列呼び出しがインターリーブできるよう`functionCallId`付きでチャンクを出力します。

Phase 13 · 03では並列+ストリーミングの再組み立てを詳しく説明します。このレッスンは宣言と単一呼び出しの形式に焦点を当てます。

### エラーと修復

無効引数エラーもそれぞれ異なります。

- **OpenAI（非ストリクト）。** モデルは`arguments: "{bad json}"`を返し、JSONパースが失敗し、エラーメッセージを注入して再呼び出しします。
- **OpenAI（ストリクト）。** デコーディング中に検証が行われ、無効JSONは不可能ですが`refusal`が現れることがあります。
- **Anthropic。** `input`に予期しないフィールドが含まれることがあります；スキーマは推奨です。サーバーサイドで検証してください。
- **Gemini。** OpenAPI 3.0の癖：オブジェクトフィールドの`enum`は無視されます；自分で検証してください。

### トランスレータパターン

コード内の標準ツール宣言はこのようになります（形式はあなたが選びます）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

3つの小さな関数がこれを3つのプロバイダ形式に変換します。`code/main.py`のハーネスはまさにこれを行い、各プロバイダのレスポンス形式を通じて偽のツール呼び出しをラウンドトリップします。ネットワーク不要 — このレッスンはHTTPではなく形式を教えます。

本番チームはこのトランスレータを`AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）、または`BaseTool`（LlamaIndex）にラップします。Phase 13 · 17では3つのいずれかの前にOpenAI形式のAPIを公開するゲートウェイを提供します。

## 使ってみる

`code/main.py`は1つの標準`Tool`データクラスとOpenAI、Anthropic、Gemini宣言JSONを出力する3つのトランスレータを定義します。次に各形式の手作りプロバイダレスポンスを同じ標準呼び出しオブジェクトにパースし、意味的に同一であることを示します。実行して3つの宣言を並べて比較してください。

確認すべき点：

- 3つの宣言ブロックはエンベロープとフィールド名のみが異なります。
- 3つのレスポンスブロックは呼び出しの場所が異なります（トップレベルの`tool_calls`、`content[]`ブロック、`parts[]`エントリ）。
- 1つの`canonical_call()`関数が3つのレスポンス形式すべてから`{id, name, args}`を抽出します。

## 成果物を出す

このレッスンは`outputs/skill-provider-portability-audit.md`を生成します。あるプロバイダに対するFunction calling統合を受け取り、依存するプロバイダ制限、変更が必要なフィールド、各プロバイダへの移植で壊れる箇所を含む移植性監査を生成します。

## 演習

1. `code/main.py`を実行し、3つのプロバイダ宣言JSONがすべて同じ基盤となる`Tool`オブジェクトをシリアライズすることを確認します。標準ツールにenumパラメータを追加して、Geminiトランスレータのみが OpenAPIの癖を処理する必要があることを確認します。

2. 各プロバイダの`ListToolsResponse`パーサを追加して、モデルが`list_tools`または探索呼び出し後に返すツールリストを抽出します。OpenAIにはネイティブにそれがありません；この非対称性に注意してください。

3. `tool_choice`変換を実装します：標準の`ToolChoice(mode="force", tool_name="x")`を3つのプロバイダ形式すべてにマップします。次に`mode="any"`と`mode="none"`をマップします。レッスンの差分表と照合してください。

4. 3つのプロバイダのうち1つを選び、そのFunction callingガイドを最初から最後まで読みます。他の2つがサポートしていないスキーマ仕様の1つのフィールドを見つけます。候補：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. テストベクタを作成します：宣言されたスキーマに違反する引数を持つツール呼び出し。各プロバイダのバリデータ（Lesson 01の標準ライブラリのものがプロキシとして機能します）を通じて実行し、どのエラーが発生するかを記録します。厳格さの点でどのプロバイダを本番で使うかを文書化します。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|----------------|------------------------|
| Function calling | 「ツール使用」 | 構造化ツール呼び出し出力のためのプロバイダレベルAPI |
| ツール宣言 | 「ツール仕様」 | 名前＋説明＋JSONスキーマ入力ペイロード |
| `tool_choice` | 「強制/禁止」 | 自動/必須/なし/特定名前モード |
| ストリクトモード | 「スキーマ強制」 | デコーディングをスキーマに一致させるよう制約するOpenAIフラグ |
| `tool_use`ブロック | 「Anthropicの呼び出し形式」 | id、名前、inputを含むインラインコンテンツブロック |
| `functionCall`パート | 「Geminiの呼び出し形式」 | 名前、args、idを含む`parts[]`エントリ |
| 文字列化引数 | 「文字列化JSON」 | OpenAIはargsをオブジェクトではなくJSON文字列として返す |
| 並列ツール呼び出し | 「1ターンでのファンアウト」 | 1つのアシスタントメッセージ内の複数ツール呼び出し |
| Refusal | 「モデルの拒否」 | 呼び出しの代わりにストリクトモード専用の拒否ブロック |
| OpenAPI 3.0サブセット | 「Geminiスキーマの癖」 | GeminiはわずかなJSON Schema方言の違いを持つ |

## 参考資料

- [OpenAI — Function callingガイド](https://platform.openai.com/docs/guides/function-calling) — ストリクトモードと並列呼び出しを含む標準リファレンス
- [Anthropic — ツール使用の概要](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use`と`tool_result`ブロックセマンティクス
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — 並列呼び出し、一意ID、OpenAPIサブセット
- [Vertex AI — Function callingリファレンス](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — GeminiのエンタープライズサーフェAI
- [OpenAI — 構造化出力](https://platform.openai.com/docs/guides/structured-outputs) — ストリクトモードスキーマ強制の詳細
