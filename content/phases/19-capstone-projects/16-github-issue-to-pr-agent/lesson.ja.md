# キャップストーン 16 — GitHub Issue-to-PR 自律エージェント

> AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codexクラウド、Google Julesはすべて同じ2026年のプロダクト形式を出荷している: Issueにラベルを付ければPRが生まれる。クラウドサンドボックスでエージェントを実行し、テストが通ることを確認し、理由付き・レビュー準備完了のPRを投稿する。難しい部分は、リポジトリのビルド環境を自動的に再現すること、クレデンシャルの漏洩を防ぐこと、リポジトリごとの予算を適用すること、エージェントが強制プッシュできないようにすることだ。このキャップストーンでは、セルフホスト版を構築し、ホスト型代替案とのコストとパス率を比較する。

**演習フェーズ:** P11 · P13 · P14 · P15 · P17

## 問題

非同期クラウドコーディングエージェントは、インタラクティブコーディングエージェント（キャップストーン01）とは別の製品カテゴリだ。UXはGitHubラベルだ。Issueに`@agent fix this`とラベルを付けると、ワーカーがクラウドサンドボックスで起動し、リポジトリをクローンし、テストを実行し、ファイルを編集し、確認し、エージェントの理由付きとともにPRを開く。インタラクティブループなし、ターミナルなし。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codexクラウド、Google Jules、Factory Droidsはすべてこれに収束している。

エンジニアリング上の課題は具体的だ: 環境の再現（エージェントはキャッシュされた開発イメージなしにリポジトリをゼロから構築しなければならない）、不安定なテスト（再実行または分離が必要）、クレデンシャルのスコーピング（最小限の細粒度パーミッションを持つGitHub App）、リポジトリごとの1日当たりの予算適用、強制プッシュ禁止ポリシー。このキャップストーンは、パス率、コスト、ホスト型代替案に対する安全性を測定する。

## コンセプト

トリガーはGitHubウェブフック（IssueラベルまたはPRコメント）だ。ディスパッチャーはECS FargateまたはLambdaにワークをキューイングする。ワーカーはリポジトリをリポジトリから推論された汎用Dockerfile（言語、フレームワーク）でDaytonaまたはE2Bサンドボックスに引き込む。エージェントはClaude Opus 4.7またはGPT-5.4-Codexに対してmini-swe-agentまたはSWE-agent v2ループを実行する。反復する: コードを読む、修正を提案する、パッチを適用する、テストを実行する。

確認がゲーティングステップだ。PRを開く前に、サンドボックスでフルCIが通らなければならない。カバレッジデルタが計算され、閾値を超えて負の場合、PRは開かれるが`needs-review`ラベルが付く。エージェントはPR説明として理由付けを投稿し、レビュー担当者がフォローアップのためにpingできる`@agent`スレッドも投稿する。

安全性は2つの異なるGitHubサーフェスを通じてスコープされる: Appは`workflows: read`と狭いリポジトリコンテンツ/PRスコープを持つ短命インストールトークンを提供する。ブランチプロテクション（これを実行できる唯一のサーフェス）は「mainへの直接書き込みなし」と「強制プッシュなし」を適用し、アプリはバイパスリストに追加されない。`.github/workflows`へのパスでスコープされた読み取り専用アクセスは実際のGitHub App プリミティブではないため、エージェントのファイル編集のアローリストはワーカーでそれを適用しなければならない。リポジトリごとの1日当たりの予算上限はディスパッチャーで適用される（例: リポジトリごとに1日最大5 PR、1 PRあたり$20）。

## アーキテクチャ

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## スタック

- トリガー: 細粒度トークンを持つGitHub App；Lambda またはFly.io経由のウェブフックレシーバー
- ワーカー: ECS Fargateタスク（またはGitHub Actionsセルフホストランナー）
- サンドボックス: タスクごとのDaytonaデブコンテナーまたはE2Bサンドボックス
- エージェントループ: Claude Opus 4.7 / GPT-5.4-Codex上のmini-swe-agentベースラインまたはSWE-agent v2
- 検索: tree-sitterリポジトリマップ + ripgrep
- 確認: サンドボックス内フルCI + カバレッジデルタゲート
- 可観測性: Langfuse（PR本文にリンクされたPRごとのトレースアーカイブ）
- 予算: リポジトリごとの1日当たりドル上限；リポジトリごとの最大PR数

## 実装する

1. **GitHub App。** 細粒度インストールトークン: Issues読み書き、pull_requests書き込み、コンテンツ読み書き、ワークフロー読み取り。ブランチプロテクション（これを実行できる唯一のサーフェス）は「mainへの直接プッシュなし」と「強制プッシュなし」を適用し、アプリはバイパスリストに含まれない。ワーカーは提案されたdiffのアローリストチェックとして`.github/workflows`への書き込みなしを適用する（GitHub Appのパーミッションはパスでスコープされないため）。

2. **ウェブフックレシーバー。** Lambda関数がIssueラベル/PRコメントウェブフックを受け付ける。ラベル`@agent fix this`でフィルタリングする。SQSにキューイングする。

3. **ディスパッチャー。** SQSからタスクをポップする。リポジトリごとの1日当たり予算を適用する。リポジトリURL、Issue本文、新しいDaytonaサンドボックスとともにECS Fargateタスクを起動する。

4. **環境推論。** 言語（Python、Node、Go、Rust）とパッケージマネージャー（uv、pnpm、go mod、cargo）を検出する。存在しない場合はDockerfileをその場で生成する。

5. **エージェントループ。** Claude Opus 4.7でmini-swe-agentまたはSWE-agent v2。ツール: ripgrep、tree-sitterリポジトリマップ、read_file、edit_file、run_tests、git。ハードリミット: $20コスト、30分ウォールクロック、30エージェントターン。

6. **確認。** ループ終了後、サンドボックスでフルテストスイートを実行する。jacoco/coverage.pyでカバレッジデルタを計算する。CIが赤の場合: 停止し、PRを開かない。カバレッジが2%以上低下した場合: `needs-review`ラベルでPRを開く。

7. **PR投稿。** エージェントブランチをプッシュする。GitHub APIでPRを開く（タイトル、理由付け、diff要約、トレースURL、コスト、ターン）。

8. **クレデンシャルハイジーン。** ワーカーは短命なGitHub Appインストールトークンで動作する。ログはアーカイブ前にシークレットを除去する。

9. **評価。** 難易度が異なる30のシード内部Issue。パス率、PRクオリティ（diffサイズ、スタイル、カバレッジ）、コスト、レイテンシーを測定する。同じIssueでCursor Background AgentsおよびAWS Remote SWE Agentsと比較する。

## 使ってみる

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## 成果物を出す

`outputs/skill-issue-to-pr.md`が成果物だ。ラベル付きIssueを制限されたコストとスコープされたクレデンシャルでレビュー準備完了のPRに変換するGitHub App + 非同期クラウドワーカー。

| 重み | 基準 | 測定方法 |
|:-:|---|---|
| 25 | 30 Issueのパス率 | エンドツーエンドの成功（CI緑 + カバレッジOK） |
| 20 | PRクオリティ | diffサイズ、カバレッジデルタ、スタイル適合性 |
| 20 | 解決IssueあたりのコストとレイテンシーI | $ とウォールクロック/PR |
| 20 | 安全性 | スコープトークン、リポジトリごとの予算、強制プッシュなし、クレデンシャルハイジーン |
| 15 | オペレーターUX | 理由付けコメント、リトライアフォーダンス、@メンションフォローアップ |
| **100** | | |

## 演習

1. 「不安定テスト修正」モードを追加する: ラベル`@agent stabilize-flake TestX`がサンドボックスで50回テストを実行し、安定化させる最小限の変更を提案する。

2. 3つの共有Issueでコスト対Cursor Background Agentsを比較する。どのツールがどこで勝つかを報告する。

3. 予算ダッシュボードを構築する: リポジトリごとの1日当たりのコスト、ユーザーごとのコスト。異常時にアラート。

4. 「ドライラン」モードを構築する: CIを実行せずにドラフトPRを開き、レビュー担当者が安価に計画を検討できるようにする。

5. 保持ポリシーを追加する: マージなしで7日以上経過したPRブランチを自動的に削除する。

## キーワード

| 用語 | 一般的な言い方 | 実際の意味 |
|------|-----------------|------------------------|
| GitHub App | "スコープされたボットID" | 細粒度パーミッション + 短命インストールトークンを持つApp |
| 非同期クラウドエージェント | "バックグラウンドエージェント" | クラウドサンドボックスで動作する非インタラクティブなワーカー（ターミナルではない） |
| 環境推論 | "Dockerfile合成" | 言語 + パッケージマネージャーの検出、不在時のDockerfile生成 |
| 確認 | "サンドボックス内CI" | PRを開く前にワーカー内でフルテストスイートを実行する |
| カバレッジデルタ | "カバレッジ保存" | ベースからエージェントブランチへのテストカバレッジ%の変化 |
| リポジトリごとの予算 | "1日の上限" | ディスパッチャーで適用されるドルとPR数のキャップ |
| 理由付け | "PR本文の説明" | 何が変更されたかと理由のエージェントの要約；PR本文で必須 |

## 参考資料

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 非同期クラウドエージェントの標準参照
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI参照
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — 商用代替案
- [OpenAI Codex (cloud)](https://openai.com/codex) — ホスト型競合
- [Google Jules](https://jules.google) — Googleのホスト型バージョン
- [Factory Droids](https://www.factory.ai) — 別の商用参照
- [GitHub App documentation](https://docs.github.com/en/apps) — スコープされたボットID
- [Daytona cloud sandboxes](https://daytona.io) — 参照サンドボックス
