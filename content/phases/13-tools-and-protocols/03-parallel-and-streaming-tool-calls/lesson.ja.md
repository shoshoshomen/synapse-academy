# 並列ツール呼び出しとストリーミング

> 独立した3つの天気ルックアップを逐次実行すれば3回のラウンドトリップです。並列で実行すれば合計時間は最も遅い1つの呼び出しと同じになります。現在すべてのフロンティアプロバイダは1ターンで複数のツール呼び出しを出力します。この恩恵は現実的ですが、配管は微妙です。このレッスンでは並列ファンアウトとストリーミング引数の再組み立ての両方を、ID相関のトラップに重点を置いて説明します。


## 学習目標

- `parallel_tool_calls: true`が存在する理由と、無効にすべき場合を説明できる。
- 並列ファンアウト中にストリーミングされた引数チャンクを正しいツール呼び出しIDに対応付けられる。
- 部分的な`arguments`文字列を早期パースせずに完全なJSONに再組み立てできる。
- 逐次対並列のレイテンシを示す3都市の天気ベンチマークを実行できる。

## 問題

並列呼び出しなしで、「ベンガルール、東京、チューリッヒの天気は？」に回答するエージェントはこうなります：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

LLMのラウンドトリップが3回で、各々がエグゼキュータのレイテンシも払います。理想的な壁時計時間の約4倍です。

並列呼び出しで：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

LLMのラウンドトリップが1回。エグゼキュータ時間は3つの合計ではなく最大値です。OpenAI、Anthropic、Geminiの本番ベンチマークでは、ファンアウトワークロードで60〜70%の壁時計時間削減が示されています。

代償は相関の複雑さです。3つの呼び出しが順序どおりに完了しない場合、結果には対応する`tool_call_id`が含まれていて、モデルが照合できる必要があります。結果がストリーミングされる場合は、実行前に部分的な引数フラグメントを完全なJSONに組み立てる必要があります。Gemini 3で一意IDが追加されたのは、同じツールへの2つの並列呼び出しが区別できなかったという実際の問題を解決するためでした。

## 概念

### 並列を有効にする

- **OpenAI。** `parallel_tool_calls: true`がデフォルトでオン。`false`に設定して逐次を強制。
- **Anthropic。** `disable_parallel_tool_use: false`（Claude 3.5以降デフォルト）で並列。`true`に設定して逐次。
- **Gemini。** 常に並列対応；`tool_config.function_calling_config.mode = "AUTO"`でモデルに決めさせる。

ツールが順序依存関係を持つ場合（`create_file`の後に`write_file`）、一方の出力が他方の入力を形成する場合、またはレート制限がファンアウトを処理できない場合は並列を無効にします。

### ID相関

モデルが出力するすべての呼び出しにはIDがあります。ホストが返すすべての結果には同じIDが含まれている必要があります。これがなければ結果が曖昧になります。

- **OpenAI。** 各ツールロールメッセージの`tool_call_id`。
- **Anthropic。** 各`tool_result`ブロックの`tool_use_id`。
- **Gemini。** 各`functionResponse`の`id`（Gemini 3以降；Gemini 2は名前でマッチしていたため同名の並列呼び出しで問題が生じました）。

### 呼び出しを並列実行する

ホストは各呼び出しのエグゼキュータを独自のスレッド、コルーチン、またはリモートワーカーで実行します。最も単純なハーネスはスレッドプールを使います；本番では`asyncio.gather`または構造化並行性を使います。完了順は予測不可能 — IDが識別子です。

よくあるバグ：完了順ではなく呼び出しリスト順で結果を返す。モデルは`tool_call_id`だけを気にするため通常は機能しますが、結果が欠落または重複する場合は順序どおりでない送信のデバッグが難しくなります。明示的なIDを使って完了順に返すことをお勧めします。

### ストリーミングツール呼び出し

モデルがストリーミングする場合、`arguments`は断片的に届きます。3つの並列呼び出しの3つの別々のストリームがワイヤーでインターリーブします。IDごとに1つのアキュムレータが必要です。

プロバイダごとの形式：

- **OpenAI。** 各チャンクは`choices[0].delta.tool_calls[i].function.arguments`（部分文字列）。チャンクは`index`（呼び出しリスト内の位置）を持ちます。インデックスごとに蓄積し、最初に現れた時点で`id`を読み取り、`finish_reason = "tool_calls"`でJSONをパースします。
- **Anthropic。** ストリームイベントは`message_start`、次に`tool_use`タイプの各ブロックに1つの`content_block_start`（id、名前、空のinputを含む）。`content_block_delta`イベントは`input_json_delta`チャンクを運びます。`content_block_stop`が各ブロックを閉じます。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3以降）は複数の並列呼び出しがクリーンにインターリーブできるよう`functionCallId`付きでチャンクを出力します。Gemini 3以前は、ストリーミングは1回に1つの完全な呼び出しを返していました。

### 部分的JSONと早期パーストラップ

`arguments`が完全になるまでパースできません。`{"city": "Beng`のような部分的なJSONは有効ではなく例外を発生させます。正しいゲートはプロバイダの呼び出し終了シグナルです：OpenAIの`finish_reason = "tool_calls"`、Anthropicの`content_block_stop`、またはGeminiのストリーム終了イベント。その後に`json.loads`を試みます。より堅牢なアプローチは、構造が完成するにつれてイベントを生成するインクリメンタルJSONパーサを使うことです；OpenAIのストリーミングガイドは「考え中」インジケータを表示するUXのためにこれを推奨しています。括弧カウントは完全性テストとして信頼できません（引用符内や転義コンテンツ内の括弧が誤検知を引き起こします）、インフォーマルなデバッグヒューリスティックとしてのみ使うべきです。

### 順序どおりでない完了

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

ホストの返信には引き続きIDを引用する必要があります：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

OpenAIまたはAnthropicでは返信の順序は正確さに影響しません。GeminiはIDが一致する限りどの順序でも受け入れます。

### ベンチマーク：逐次対並列

`code/main.py`のハーネスは400、600、800msのレイテンシを持つ3つのエグゼキュータをシミュレートします。逐次では合計1800ms。並列ではmax(400, 600, 800) = 800ms。差分は一定で比例しないため、ツール数が増えると節約が増えます。

現実世界の注意：並列呼び出しはダウンストリームAPIにストレスをかけます。レート制限されたサービスへの10方向ファンアウトは失敗します。Phase 13 · 17ではゲートウェイレベルのバックプレッシャーを説明します。

### ストリーミングファンアウトの壁時計

モデル自体がストリームする場合、全呼び出しが確定するのを待つ代わりに、1つの呼び出しの引数が完成した直後に実行を開始できます。これはOpenAIがドキュメントに記載している最適化ですが、全SDKが公開しているわけではありません。このレッスンのハーネスはそれを実行します：シミュレートされたストリームが完全な引数オブジェクトを生成した直後、ホストはその呼び出しを開始します。

## 使ってみる

`code/main.py`には2つの部分があります。最初は`concurrent.futures.ThreadPoolExecutor`を使って3つのシミュレートされた天気呼び出しを逐次と並列で実行し、壁時計時間を出力します。後半は偽のストリーミングレスポンスを再生します — 3つの並列呼び出しの`arguments`チャンクが1つのストリームにインターリーブされたもの — を`StreamAccumulator`でIDごとに再組み立てします。LLMもネットワークも不要、再組み立てロジックだけです。

確認すべき点：

- 逐次タイマーは1.8秒。並列タイマーは同じ偽のレイテンシで0.8秒。
- アキュムレータは順序どおりに届かないチャンクをIDごとにバッファリングし、各呼び出しのJSONが完全になってからパースします。
- エグゼキュータは全ストリームの終了後ではなく、IDの引数が確定した直後に開始します。

## 成果物を出す

このレッスンは`outputs/skill-parallel-call-safety-check.md`を生成します。ツールレジストリを受け取り、どのツールが並列化安全か、どれが順序依存関係を持つか、どれがダウンストリームレート制限を超過するかを監査し、ツールごとの`parallel_safe`フラグを持つ修正済みレジストリを返します。

## 演習

1. `code/main.py`を実行し、シミュレートされたレイテンシを変えてみます。並列対逐次の比率がおよそ`max/sum`であることを確認します（実際の実行はスレッドスケジューリング、シリアライズ、ハーネスオーバーヘッドのため理想とわずかに異なります）。どのレイテンシ分布で並列化が意味をなさなくなりますか？

2. アキュムレータを拡張して「呼び出しがストリーム中にキャンセルされた」ケースを処理します：バッファを削除して`cancelled`イベントを発生させます。どのプロバイダがこのケースを明示的にドキュメント化していますか？Anthropicの`content_block_stop`セマンティクスとOpenAIの`finish_reason: "length"`動作を確認してください。

3. スレッドプールを`asyncio.gather`に置き換えます。両方をベンチマークします。エグゼキュータが実際のI/Oを行う場合のみ、コンテキストスイッチコストが低いためasyncの方がわずかに有利なはずです。

4. 並列化すべきでない2つのツールを選びます（例：`create_file`の後に`write_file`）。レジストリに`ordering_dependency`グラフを追加し、そのグラフに基づいて並列ファンアウトをゲートします。これは依存関係を考慮したスケジューリングの最小機構であり、将来のエージェントエンジニアリングフェーズでフォーマル化されます。

5. OpenAIの並列Function callingセクションとAnthropicの`disable_parallel_tool_use`ドキュメントを読みます。Anthropicが並列化を無効にすることを推奨する1つの現実世界のツールタイプを特定します。（ヒント：同じリソースへの結果的な変更。）

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|----------------|------------------------|
| 並列ツール呼び出し | 「1ターンでのファンアウト」 | モデルが1つのアシスタントメッセージで複数のツール呼び出しを出力 |
| `parallel_tool_calls` | 「OpenAIのフラグ」 | 複数呼び出しの出力を有効/無効にする |
| `disable_parallel_tool_use` | 「Anthropicの逆フラグ」 | オプトアウトフラグ；デフォルトは並列有効 |
| ツール呼び出しID | 「相関ハンドル」 | 結果メッセージが繰り返さなければならない呼び出しごとの識別子 |
| アキュムレータ | 「ストリームバッファ」 | 部分的な`arguments`チャンクのIDごとの文字列バッファ |
| 順序どおりでない完了 | 「最速が最初」 | 並列呼び出しは予測不可能な順序で終了；IDが接着剤 |
| 依存グラフ | 「順序制約」 | 出力が他のツールの入力に供給されるツール；並列化不可 |
| 早期パーストラップ | 「JSON.parseが爆発した」 | 不完全な`arguments`文字列をパースしようとする |
| `streamFunctionCallArguments` | 「Gemini 3の機能」 | 呼び出しごとに一意IDを持つストリーミング引数チャンク |
| 完了順返信 | 「全部待つな」 | IDをキーにして届いた順に結果を返す |

## 参考資料

- [OpenAI — 並列Function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — デフォルト動作とオプトアウトフラグ
- [Anthropic — ツール使用：ツール使用の実装](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use`と結果バッチ処理
- [Google — Gemini function callingの並列セクション](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3からのID相関並列呼び出し
- [OpenAI — ツールを使ったストリーミングレスポンス](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAIストリームのチャンク化引数再組み立て
- [Anthropic — メッセージのストリーミング](https://docs.anthropic.com/en/api/messages-streaming) — `input_json_delta`を持つ`content_block_delta`
