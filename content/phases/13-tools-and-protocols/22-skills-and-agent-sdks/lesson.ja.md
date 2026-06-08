# スキルとエージェントSDK — Anthropicスキル、AGENTS.md、OpenAI Apps SDK

> MCPは「どんなツールが存在するか」を伝える。スキルは「どのようにタスクを行うか」を伝える。2026年のスタックは両方を重ねている。AnthropicのAgent Skills（オープン標準、2025年12月）はプログレッシブディスクロージャー付きのSKILL.mdとして出荷される。OpenAIのApps SDKはMCPにウィジェットメタデータを加えたものだ。AGENTS.md（現在60,000以上のリポジトリに存在）はプロジェクトレベルのエージェントコンテキストとしてリポジトリルートに置かれる。このレッスンでは各コンポーネントが何をカバーするかを定義し、エージェントをまたいで移植できる最小限のSKILL.md + AGENTS.mdバンドルを構築する。


## 学習目標

- 3つの層を区別できる：AGENTS.md（プロジェクトコンテキスト）、SKILL.md（再利用可能なノウハウ）、MCP（ツール）。
- YAMLフロントマターとプログレッシブディスクロージャー付きのSKILL.mdを書ける。
- スキルをエージェントランタイムにファイルシステム形式でロードできる。
- スキルをMCPサーバーとAGENTS.mdと組み合わせて1つのパッケージがClaude Code、Cursor、Codexで動作するようにできる。

## 問題背景

あるエンジニアがリリースノート作成ワークフローを多ステップのプロンプトに蒸留した：「最新のマージされたPRを読む。エリアごとにグループ化する。各PRを要約する。チームのスタイルに従ったchangelogエントリを書く。Slackの下書きに投稿する。」Notionのドキュメントとしてチームに共有した。

今度はこのワークフローをClaude Code、Cursor、Codex CLIから使いたい。各エージェントには異なる命令の読み込み方がある：Claude Codeのスラッシュコマンド、Cursorのルール、Codexの `.codex.md`。エンジニアはワークフローを3回コピーして3つのコピーを管理することになる。

AGENTS.mdとSKILL.mdが一緒にこれを解決する：

- **AGENTS.md** がリポジトリルートに置かれる。互換性のあるすべてのエージェントがセッション開始時にそれを読む。「このプロジェクトはどのように動作するか？慣例は何か？どのコマンドでテストを実行するか？」
- **SKILL.md** は移植可能なバンドルだ：YAMLフロントマター（名前、説明）+ マークダウン本文 + オプションのリソース。スキルをサポートするエージェントが名前でオンデマンドにロードする。
- **MCP**（フェーズ13・06-14）がスキルが呼び出す必要のあるツールを処理する。

3つの層、1つの移植可能なアーティファクト。

## コンセプト

### AGENTS.md（agents.md）

2025年後半に立ち上がり、2026年4月までに60,000以上のリポジトリが採用。リポジトリルートに1つのファイル。形式：

```markdown
# プロジェクト：my-service

## 慣例
- strictモードのTypeScript。
- Python側ではPydanticをモデルに使用。
- テストは `pnpm test` で実行。

## ビルドと実行
- `pnpm dev` でローカル開発サーバー。
- `pnpm build` で本番ビルド。
```

エージェントがセッション開始時にこれを読み、そのプロジェクトに対する動作を調整する。2026年のすべてのコーディングエージェントがAGENTS.mdをサポートしている：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.mdの形式

AnthropicのAgent Skills（2025年12月にオープン標準としてリリース）：

```markdown
---
name: release-notes-writer
description: このプロジェクトのスタイルに従って最新のマージされたPRのchangelogエントリを書く。
---

# リリースノートライター

呼び出されたら、以下のステップを実行する：

1. 最後のタグ以降にマージされたPRをリスト化する。`gh pr list --base main --state merged` を使用する。
2. ラベルでグループ化する：feature、fix、chore、docs。
3. 各グループの各PRに1行書く：`- <タイトル> (#<番号>)`。
4. リリースノートをドラフトし、CHANGELOG.mdにステージングする。

ユーザーが「ship」と言ったら、`git tag vX.Y.Z` と `gh release create` を実行する。

## 注記

- PRのないコミットは含めない。
- パブリックなchangelogから「chore」エントリはスキップする。
```

フロントマターがスキルのアイデンティティを宣言する。本文はスキルがロードされる際にモデルに表示されるプロンプトだ。

### プログレッシブディスクロージャー

スキルはエージェントが必要な時だけ取得するサブリソースを参照できる。例：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.mdは「スタイルルールはstyle-guide.mdを参照」と言う。エージェントはスキルが実際に動作している時だけstyle-guide.mdをプルする。これによりモデルが必要としないかもしれない詳細でプロンプトが膨らむのを避ける。

### ファイルシステムによる発見

エージェントランタイムが既知のディレクトリでSKILL.mdファイルをスキャンする：

- `~/.anthropic/skills/*/SKILL.md`
- プロジェクトの `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

ロードはフォルダ名とフロントマターの `name` による。Claude Code、Anthropic Claude Agent SDK、SkillKit（クロスエージェント）がすべてこのパターンに従っている。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`（TypeScript）と `claude-agent-sdk`（Python）がセッション開始時にスキルをロードし、ランタイム内で呼び出し可能な「エージェント」として公開する。エージェントループはユーザーがスキルを呼び出した時にスキルにディスパッチする。

### OpenAI Apps SDK

2025年10月に立ち上げ；MCPの上に直接構築されている。OpenAIの以前のコネクターとCustom GPT ActionsをOpenAIシングルデベロッパーサーフェスの下に統合した。Apps SDKのアプリは：

- MCPサーバー（ツール、リソース、プロンプト）。
- プラスChatGPTのUIのウィジェットメタデータ。
- プラスオプションのMCPアプリ `ui://` リソース（インタラクティブサーフェス用）。

同じプロトコル、より豊かなUX。

### SkillKitによるクロスエージェント移植性

SkillKitや同様のクロスエージェント配布レイヤーが単一のSKILL.mdを32以上のAIエージェント（Claude Code、Cursor、Codex、Gemini CLI、OpenCodeなど）それぞれのネイティブ形式に変換する。1つの真実のソース；多くの消費者。

### 3層スタック

| 層 | ファイル | ロードされる時 | 目的 |
|-------|------|-------------|---------|
| AGENTS.md | リポジトリルート | セッション開始時 | プロジェクトレベルの慣例 |
| SKILL.md | skillsディレクトリ | スキル呼び出し時 | 再利用可能なワークフロー |
| MCPサーバー | 外部プロセス | ツールが必要な時 | 呼び出し可能なアクション |

3つすべてが組み合わさる：エージェントがセッション開始時にAGENTS.mdを読み、ユーザーがスキルを呼び出し、スキルの命令がMCPツール呼び出しを含み、エージェントがMCPクライアント経由でディスパッチする。

## 使ってみる

`code/main.py` はstdlibのSKILL.mdパーサーとローダーを提供する。`./skills/` 配下のスキルを発見し、YAMLフロントマターとマークダウン本文を解析し、スキル名をキーにした辞書を生成する。そして `release-notes-writer` を名前で呼び出すエージェントループをシミュレートする。

確認すべきポイント：

- YAMLフロントマターが最小限のstdlibパーサーで解析される（`pyyaml` の依存なし）。
- スキル本文がそのまま保存され、呼び出し時にエージェントがシステムプロンプトの先頭に追加する。
- プログレッシブディスクロージャーがオンデマンドで参照ファイルをプルする `read_subresource` 関数でデモされる。

## 成果物を出す

このレッスンでは `outputs/skill-agent-bundle.md` を生成する。ワークフローが与えられると、このスキルはエージェントをまたいで移植可能な組み合わせ型SKILL.md + AGENTS.md + MCPサーバー設計図バンドルを生成する。

## 演習

1. `code/main.py` を実行する。`skills/` の下に2番目のスキルを追加し、ローダーがそれを認識することを確認する。

2. このコースのリポジトリ用のAGENTS.mdを書く。テストコマンド、スタイル慣例、フェーズ13のメンタルモデルを含める。

3. チームの内部ドキュメントの多ステップワークフローをSKILL.mdに移植する。Claude Codeでロードされることを確認する。

4. スキルをCursorとCodexのネイティブルール形式に手動で変換する。形式間の差分を数える — これがSkillKitが自動化する変換サーフェスだ。

5. AnthropicのAgent Skillsブログ投稿を読む。Claude Agent SDKにこのレッスンのローダーがカバーしていない機能を1つ特定する。（ヒント：エージェントのサブ呼び出し。）

## キーワード

| 用語 | よく言われる表現 | 実際の意味 |
|------|----------------|------------------------|
| SKILL.md | "スキルファイル" | エージェントランタイムによってロードされるYAMLフロントマターとマークダウン本文 |
| AGENTS.md | "リポジトリルートエージェントコンテキスト" | セッション開始時に読まれるプロジェクトレベルの慣例ファイル |
| プログレッシブディスクロージャー | "サブリソースの遅延ロード" | スキル本文が必要な時だけプルされるファイルを参照 |
| フロントマター | "上部のYAMLブロック" | `---` デリミタ内のメタデータ（名前、説明） |
| Claude Agent SDK | "Anthropicのスキルランタイム" | `@anthropic-ai/claude-agent-sdk`、スキルをロードしてルーティング |
| OpenAI Apps SDK | "MCP + ウィジェットメタ" | MCPの上に構築されたOpenAIのデベロッパーサーフェス |
| スキル発見 | "ファイルシステムスキャン" | 既知のディレクトリをSKILL.mdで走査し、名前でキー付け |
| クロスエージェント移植性 | "1つのスキル多くのエージェント" | 1つのSKILL.mdをSkillKitスタイルのツールで32以上のエージェントに変換 |
| Agent Skill | "移植可能なノウハウ" | MCPのツール概念の外の再利用可能なタスクテンプレート |
| Apps SDK | "MCP + ChatGPT UI" | MCPに統合されたコネクターとCustom GPT |

## 参考資料

- [Anthropic — Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025年12月のリリース
- [Anthropic — Agent Skills docs](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.mdのフォーマットリファレンス
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — ChatGPT向けのMCPベースのデベロッパープラットフォーム
- [agents.md](https://agents.md/) — AGENTS.mdのフォーマットと採用リスト
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — 公式スキルの例
