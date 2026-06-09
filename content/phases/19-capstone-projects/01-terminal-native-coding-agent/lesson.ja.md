# キャップストーン 01 — ターミナルネイティブコーディングエージェント

> 2026年には、コーディングエージェントの形は完成している。TUIハーネス、ステートフルなプラン、サンドボックス化されたツールサーフェス、計画・実行・観察・回復を行うループ。Claude Code、Cursor 3、OpenCodeは50フィート離れて見れば同じに見える。このキャップストーンでは、それをエンドツーエンドで構築し（CLIで入力、プルリクエストを出力として）、SWE-bench ProでのminiSWEエージェントおよびLive-SWE-agentと比較して測定する。ハードな部分はモデル呼び出しではなく、ツールループ・サンドボックス・50ターンの実行におけるコスト上限であることを学ぶ。

**演習するフェーズ:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18

## 問題

コーディングエージェントは2026年に支配的なAIアプリケーションカテゴリになった。Claude Code（Anthropic）、Cursor 3（Composer 2とAgent Tabs搭載）、Amp（Sourcegraph）、OpenCode（スター数112k）、Factory Droids、Google Julesはすべて同じアーキテクチャのバリエーションを実装している。ターミナルハーネス、許可されたツールサーフェス、サンドボックス、フロンティアモデルを中心に構築されたplan-act-observeループだ。フロンティアは狭い（Live-SWE-agentはOpus 4.5でSWE-bench Verifiedで79.2%に達した）が、エンジニアリングの技術は広い。ほとんどの失敗モードはモデルのミスではない。ツールループの不安定さ、コンテキストの汚染、制御不能なトークンコスト、そして破壊的なファイルシステム操作だ。

外部からこれらのエージェントについて推論することはできない。実際に構築し、ripgrepが8MBのマッチを返したときにターン47でループがクラッシュするのを見て、トランケーション層を再構築しなければならない。これがこのキャップストーンの目的だ。

## コンセプト

ハーネスには4つのサーフェスがある。**Plan** はモデルが各ターンで書き直すTodoWriteスタイルのステートオブジェクトを管理する。**Act** はツール呼び出し（read、edit、run、search、git）をディスパッチする。**Observe** はstdout / stderr / 終了コードをキャプチャし、トランケートして、サマリーをフィードバックする。**Recover** はコンテキストウィンドウを吹き飛ばすことなく、または永久にループすることなく、ツールエラーを処理する。2026年の形は一つのことを追加する：**フック**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop`、`PreCompact` — オペレーターがポリシー、テレメトリー、ガードレールを注入する設定可能な拡張ポイントだ。

サンドボックスはE2BまたはDaytona。各タスクは読み書き可能にマウントされたgitワークツリーを持つフレッシュなdevcontainerで実行される。ハーネスはホストファイルシステムには触れない。ワークツリーは成功または失敗時に破棄される。コスト制御は3つのレイヤーで実施される：ターンごとのトークン上限、セッションごとのドル予算、ハードなターン制限（通常50）。可観測性レイヤーはGenAIセマンティック規約を持つOpenTelemetryスパンで、セルフホストのLangfuseに出力する。

## アーキテクチャ

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## スタック

- ハーネスランタイム: Bun 1.2 + Ink 5（React-in-terminal）
- モデルアクセス: Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5（最も難しいタスク用）を使ったOpenRouter統合API
- ツールトランスポート: Model Context Protocol StreamableHTTP（MCP 2026年改訂版）
- サンドボックス: E2Bサンドボックス（JS SDK）またはDaytonaデブコンテナ
- コード検索: ripgrepサブプロセス、17言語用のtree-sitterパーサー（プリコンパイル済み）
- 分離: タスクごとに`git worktree add`、成功/失敗時にクリーンアップ
- 評価ハーネス: SWE-bench Pro（検証済みサブセット）+ Terminal-Bench 2.0 + 独自30タスクホールドアウト
- 可観測性: `gen_ai.*`セマンティック規約を持つOpenTelemetry SDK → セルフホストLangfuse
- PRの投稿: ターゲットリポジトリへの権限が限定されたGitHub App

## 実装する

1. **TUIとコマンドループ。** BunプロジェクトをInkでスキャフォールドする。`agent run <repo> "<task>"`を受け付ける。分割ビューを表示する：プランペイン（上部）、ツール呼び出しストリーム（中央）、トークン予算（下部）。終了前に`SessionEnd`フックを発火するCtrl-Cキャンセルを追加する。

2. **プランステート。** 型付きのTodoWriteスキーマを定義する（メモ付きのpending / in_progress / doneアイテム）。モデルは各ターンにツール呼び出しとして完全なステートを書き直す — インクリメンタルな変更は許可しない。クラッシュからの再開ができるよう`.agent/state.json`にプランを永続化する。

3. **ツールサーフェス。** 6つのツールを定義する：`read_file`、`edit_file`（差分プレビュー付き）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（タイムアウト付き）、`git`（status / diff / commit / push）。ハーネスがトランスポート非依存になるようMCP StreamableHTTP経由で公開する。すべてのツールはトランケートされた出力を返す（呼び出しごとに4kトークンの上限）。

4. **サンドボックスラッピング。** 各タスクはE2Bサンドボックスを起動する。`git worktree add -b agent/$TASK_ID`でフレッシュなブランチを作成する。すべてのツール呼び出しはサンドボックス内で実行される。ホストファイルシステムには到達できない。

5. **フック。** 2026年の8つすべてのフックタイプを実装する。少なくとも4つのユーザー作成フックを接続する：(a) ワークツリー外での`rm -rf`をブロックする`PreToolUse`破壊コマンドガード、(b) トークン会計の`PostToolUse`、(c) 予算初期化の`SessionStart`、(d) 最終トレースバンドルを書く`Stop`。

6. **評価ループ。** SWE-bench Pro PythonのIssue30件サブセットをクローンする。各Issueに対してハーネスを実行する。mini-swe-agent（最小ベースライン）と比較して、pass@1、タスクあたりのターン数、タスクあたりのコストを測定する。結果を`eval/results.jsonl`に書く。

7. **コスト制御。** ハードカットオフ：50ターン、200kコンテキスト、タスクあたり$5。`PreCompact`フックは150kマークで古いターンを事前ステートブロックに要約し、プランを失わずに新しい観察のためのスペースを確保する。

8. **PRの投稿。** 成功時に最終ステップは`git push` + プランと差分サマリーをボディに持つPRを開くGitHub API呼び出し。

## 使ってみる

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## 成果物を出す

成果物スキルは`outputs/skill-terminal-coding-agent.md`に置く。リポジトリパスとタスクの説明を与えると、完全なplan-act-observeループをサンドボックスで実行し、PR URLとトレースバンドルを返す。このキャップストーンのルーブリック：

| 配点 | 基準 | 測定方法 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 対ベースライン | 30件の一致したPythonタスクでのハーネス対mini-swe-agent |
| 20 | アーキテクチャの明確さ | Plan/act/observe分離、フックサーフェス、ツールスキーマ — Live-SWE-agentレイアウトとのレビュー |
| 20 | 安全性 | サンドボックスエスケープテスト、パーミッションプロンプト、破壊コマンドガードのレッドチーム合格 |
| 20 | 可観測性 | トレースの完全性（ツール呼び出しの100%がスパン化）、ターンごとのトークン会計 |
| 15 | 開発者UX | コールドスタート2秒未満、クラッシュ回復でプランを再開、Ctrl-Cがツールの中途でクリーンにキャンセル |
| **100** | | |

## 演習

1. バッキングモデルをClaude Sonnet 4.7からvLLM上で提供されるQwen3-Coder-30Bに交換する。pass@1とタスクあたりのコストを比較する。オープンモデルが劣る箇所をレポートする。

2. PRの投稿前に差分を読んで修正ループを要求できる`reviewer`サブエージェントを追加する。誤検知レビューがシングルエージェントのベースラインよりSWE-bench pass rateを下げるかどうかを測定する（ヒント：通常そうなる）。

3. サンドボックスのストレステスト：外部URLに`curl`しようとするタスクとワークツリー外に書き込もうとするタスクを作成する。両方がPreToolUseフックによってブロックされることを確認する。試みをログに記録する。

4. 小さいモデル（Haiku 4.5）を使った`PreCompact`要約を実装する。3倍のコンパクションでプランの忠実度がどれほど失われるかを測定する。

5. MCPのStreamableHTTPトランスポートをstdioに交換する。コールドスタートと呼び出しごとのレイテンシーをベンチマークする。ローカルのみの使用で勝者を選ぶ。

## キーワード

| 用語 | 一般的な呼び方 | 実際の意味 |
|------|-----------------|------------------------|
| ハーネス | 「エージェントループ」 | ツールをディスパッチし、プランステートを管理し、予算を実施するモデルを取り囲むコード |
| フック | 「エージェントイベントリスナー」 | ハーネスが8つのライフサイクルイベントのいずれかで実行するユーザー作成スクリプト |
| ワークツリー | 「Gitサンドボックス」 | 別のパスにあるリンクされたgitチェックアウト；メインクローンに触れずに破棄可能 |
| TodoWrite | 「プランステート」 | モデルが各ターンで書き直すpending/in-progress/doneアイテムの型付きリスト |
| StreamableHTTP | 「MCPトランスポート」 | 2026年MCP改訂：双方向ストリーミングを持つ長時間のHTTP接続；SSEを置き換える |
| トークン上限 | 「コンテキスト予算」 | 入力+出力トークンのターンごとまたはセッションごとの上限；コンパクションまたは終了をトリガー |
| pass@1 | 「シングル試行合格率」 | リトライやテストセットの覗き見なしで最初の実行で解決されたSWE-benchタスクの割合 |

## 参考資料

- [Claude Codeドキュメント](https://docs.anthropic.com/en/docs/claude-code) — AnthropicのリファレンスハーネスClaude Code
- [Cursor 3変更ログ](https://cursor.com/changelog) — Agent TabsとComposer 2の製品ノート
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — SWE-benchハーネス比較のための最小ベースライン
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — Opus 4.5でSWE-bench Verified 79.2%
- [OpenCode](https://opencode.ai) — オープンハーネス、スター数112k
- [SWE-benchProリーダーボード](https://www.swebench.com) — このキャップストーンが対象とする評価
- [Model Context Protocol 2026ロードマップ](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、機能メタデータ
- [OpenTelemetry GenAIセマンティック規約](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — ツール呼び出しとトークン使用のスパンスキーマ
