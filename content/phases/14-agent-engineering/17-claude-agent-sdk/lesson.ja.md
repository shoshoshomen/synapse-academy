# Claude Agent SDK: サブエージェントとセッションストア

> Claude Agent SDK は Claude Code ハーネスのライブラリ形態です。組み込みツール、コンテキスト分離のためのサブエージェント、フック、W3C トレース伝播、セッションストアのパリティ。Claude Managed Agents は長時間実行の非同期作業のためのホスト型代替手段です。


## 学習目標

- Anthropic Client SDK（生の API）と Claude Agent SDK（ハーネス形状）の違いを説明できる。
- サブエージェントの2つの目的（並列化とコンテキスト分離）と、いつ使うべきかを説明できる。
- Python SDK のセッションストアインターフェース（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）と `--session-mirror` の役割を説明できる。
- 組み込みツール、分離されたコンテキストを持つサブエージェント生成、ライフサイクルフック、セッションストアを持つ標準ライブラリハーネスを実装できる。

## 問題設定

生の LLM API では1回のラウンドトリップしか得られません。本番エージェントにはツール実行、MCP サーバー、ライフサイクルフック、サブエージェント生成、セッション永続化、トレース伝播が必要です。Claude Agent SDK はこの形状をライブラリとして提供します — Claude Code が使用するのと同じハーネスが、カスタムエージェント向けに公開されています。

## コンセプト

### Client SDK vs Agent SDK

- **Client SDK（`anthropic`）.** 生の Messages API。ループ、ツール、状態は自分で管理します。
- **Agent SDK（`claude-agent-sdk`）.** 組み込みのツール実行、MCP 接続、フック、サブエージェント生成、セッションストア。Claude Code ループのライブラリ版。

### 組み込みツール

SDK は 10 以上のツールをすぐに使える状態で提供します: ファイル読み書き、シェル、grep、glob、Web フェッチ、その他。カスタムツールは標準のツールスキーマインターフェース経由で登録します。

### サブエージェント

Anthropic が文書化した2つの目的:

1. **並列化.** 独立した作業を並行して実行します。「これら 20 個のモジュールそれぞれのテストファイルを見つけて」は 20 個の並列サブエージェントタスクです。
2. **コンテキスト分離.** サブエージェントは独自のコンテキストウィンドウを使用します。結果だけがオーケストレーターに返ります。オーケストレーターの予算は保持されます。

Python SDK の最近の追加: `list_subagents()`、`get_subagent_messages()` でサブエージェントのトランスクリプトを読み込めます。

### セッションストア

TypeScript とのプロトコルパリティ:

- `append(session_id, message)` — ターンを追加します。
- `load(session_id)` — 会話を復元します。
- `list_sessions()` — 列挙します。
- `delete(session_id)` — サブエージェントセッションへのカスケードと共に。
- `list_subkeys(session_id)` — サブエージェントキーを一覧表示します。

`--session-mirror`（CLI フラグ）はデバッグのために、トランスクリプトをストリームしながら外部ファイルにミラーします。

### フック

登録できるライフサイクルフック:

- `PreToolUse`、`PostToolUse` — ツール呼び出しをゲートまたは監査します。
- `SessionStart`、`SessionEnd` — セットアップとティアダウン。
- `UserPromptSubmit` — モデルが見る前にユーザー入力に対して処理します。
- `PreCompact` — コンテキストコンパクションの前に実行します。
- `Stop` — エージェント終了時のクリーンアップ。
- `Notification` — サイドチャネルアラート。

フックは pro-workflow（フェーズ14カリキュラム参照）などのシステムがクロスカッティングな動作を追加する方法です。

### W3C トレースコンテキスト

呼び出し元でアクティブな OTel スパンは W3C トレースコンテキストヘッダー経由で CLI サブプロセスに伝播されます。マルチプロセスのトレース全体がバックエンドで1つのトレースとして表示されます。

### Claude Managed Agents

ホスト型の代替手段（ベータヘッダー `managed-agents-2026-04-01`）。長時間実行の非同期作業、組み込みのプロンプトキャッシング、組み込みのコンパクション。管理されたインフラストラクチャのためにコントロールをトレードします。

### このパターンが失敗するケース

- **サブエージェントの過剰生成.** 100 個の小さなタスクに 100 個のサブエージェントを生成する。オーバーヘッドが支配的になります。代わりにバッチ処理してください。
- **フッククリープ.** すべてのチームがフックを追加する。起動時間が膨らみます。フックを四半期ごとにレビューしてください。
- **セッション肥大化.** セッションが蓄積し、サイズが増大します。`list_sessions` と有効期限ポリシーを使用してください。

## 実装する

`code/main.py` は SDK の形状を標準ライブラリで実装しています:

- 組み込みの `read_file`、`write_file`、`list_dir` を持つ `Tool`、`ToolRegistry`。
- `Subagent` — プライベートコンテキスト、分離された実行、結果が返ります。
- `SessionStore` — append、load、list、delete、list_subkeys。
- `Hooks` — `pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- デモ: メインエージェントが3つのサブエージェントを並列生成し（それぞれ分離）、結果を集約し、セッションを永続化します。

実行:

```
python3 code/main.py
```

トレースはサブエージェントのコンテキスト分離（オーケストレーターのコンテキストサイズが制限内に保持される）、フック実行、セッション永続化を示します。

## 使ってみる

- **Claude Agent SDK** は Claude Code ハーネスの形状が欲しい Claude ファーストのプロダクトに。
- **Claude Managed Agents** はホスト型の長時間実行非同期作業に。
- **OpenAI Agents SDK**（レッスン16）は OpenAI ファーストの対応物として。
- **LangGraph + カスタムツール** はグラフ形状のステートマシンが欲しい場合に。

## 成果物を出す

`outputs/skill-claude-agent-scaffold.md` はサブエージェント、フック、セッションストア、MCP サーバーアタッチメント、W3C トレース伝播を持つ Claude Agent SDK アプリを足場として提供します。

## 演習

1. 20 個のタスクを5つの並列サブエージェントのグループにバッチ処理するサブエージェントスポーナーを追加します。タスクごとに1つ生成した場合とオーケストレーターのコンテキストサイズを比較します。
2. `write_file` 呼び出しをレート制限する `PreToolUse` フックを実装します（セッションごとに1分間に5回）。動作をトレースします。
3. `list_subkeys` を接続してサブエージェントツリーをレンダリングします。深いネストはどのように見えますか？
4. 実際の `claude-agent-sdk` Python パッケージに移植します。ツール登録はどのように変わりますか？
5. Claude Managed Agents のドキュメントを読みます。セルフホストからマネージドに切り替えるのはいつですか？

## キーワード

| 用語 | 一般的な言い方 | 実際の意味 |
|------|----------------|------------|
| Agent SDK | "ライブラリとしての Claude Code" | ハーネス形状: ツール、MCP、フック、サブエージェント、セッションストア |
| Subagent | "子エージェント" | 独自のコンテキスト・予算を持ち、結果がバブルアップする |
| Session store | "会話 DB" | サブエージェントカスケードと共にターンを永続化・ロード・一覧・削除 |
| Hook | "ライフサイクルコールバック" | ツールの前後、セッション、プロンプト送信、コンパクト、停止 |
| W3C トレースコンテキスト | "クロスプロセストレース" | 親スパンが CLI サブプロセスに伝播される |
| Managed Agents | "ホスト型ハーネス" | Anthropic がホストする長時間実行非同期作業 |
| `--session-mirror` | "トランスクリプトミラー" | セッションのターンをストリームしながら外部ファイルに書き込む |
| MCP サーバー | "ツールサーフェス" | エージェントにアタッチされる外部ツール・リソースソース |

## 参考資料

- [Claude Agent SDK 概要](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Code のライブラリ形態
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 本番パターン
- [Claude Managed Agents 概要](https://platform.claude.com/docs/en/managed-agents/overview) — ホスト型代替手段
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 対応物
