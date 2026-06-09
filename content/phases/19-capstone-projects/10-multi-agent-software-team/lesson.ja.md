# キャップストーン 10 — マルチエージェントソフトウェアエンジニアリングチーム

> SWE-AFのファクトリーアーキテクチャ・MetaGPTのロールベースプロンプティング・AutoGen 0.4の型付きアクターグラフ・CognitionのDevin・FactoryのDroidsはすべて同じ2026年の形に収束した：アーキテクトが計画し、Nコーダーが並行するワークツリーで作業し、レビュアーがゲートし、テスターが検証する。並行するワークツリーはウォールクロックをスループットに変換する。共有ステートとハンドオフプロトコルが失敗サーフェスになる。キャップストーンはチームを構築し、SWE-bench Proで評価し、どのハンドオフが壊れてどのくらいの頻度で起きるかをレポートすることだ。

**演習するフェーズ:** P11 · P13 · P14 · P15 · P16 · P17

## 問題

シングルエージェントのコーディングハーネスは大きなタスクで天井に達する。個々のエージェントが弱いからではなく、200kトークンのコンテキストがアーキテクチャの計画と4つの並行するコードベーススライスとレビュアーのコメントとテスト出力を同時に保持できないからだ。マルチエージェントファクトリーは問題を分割する：アーキテクトがプランを所有し、コーダーが並行するワークツリーで実装を所有し、レビュアーがゲートし、テスターが検証する。SWE-AFの「ファクトリー」アーキテクチャ・MetaGPTのロール・AutoGenの型付きアクターグラフ — 3つのフレーミングすべてが同じ形を説明する。

失敗サーフェスはハンドオフだ。アーキテクトがコーダーが実装できないものを計画する。コーダーが競合するdiffを生成する。レビュアーがハルシネーションされた修正を承認する。テスターがまだ書いているコーダーと競合する。これらのチームの1つを構築し、50件のSWE-bench ProのIssueで実行し、すべてのハンドオフを追跡し、ポストモーテムを公開する。

## コンセプト

ロールは型付きエージェントだ。**アーキテクト**（Claude Opus 4.7）がIssueを読み、プランを書き、明示的なインターフェースを持つサブタスクに分割する。**コーダー**（Claude Sonnet 4.7、N個の並行インスタンス、それぞれが`git worktree` + Daytonaサンドボックス内）がサブタスクを独立して実装する。**レビュアー**（GPT-5.4）がマージされたdiffを読み、承認または特定の変更を要求する。**テスター**（Gemini 2.5 Pro）がテストスイートを単独で実行し、アーティファクト付きで合格/不合格をレポートする。

コミュニケーションは共有タスクボード（ファイルバックドまたはRedis）を通じて行われる。各ロールは処理を許可されたタスクを消費する。ハンドオフはA2AプロトコルタイプのメッセージだA。調整の関心事：マージ競合の解決（コーディネーターロールまたは自動3ウェイマージ）、共有ステートの同期（コーダーが開始するとプランは凍結される；リプランは別のイベントだ）、レビュアーのゲートキーピング（レビュアーは自分自身の変更や自分が提案した変更を承認できない）。

トークン増幅は隠れたコストだ。すべてのロール境界はサマリープロンプトとハンドオフコンテキストを追加する。40ターンのシングルエージェント実行が4つのロールにわたって160の合計ターンになる。ルーブリックは特にトークン効率対シングルエージェントのベースラインを重み付けする。なぜなら問題は「マルチエージェントは機能するか」ではなく「ドルあたりで勝てるか」だからだ。

## アーキテクチャ

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## スタック

- オーケストレーション: 共有ステート + エージェントごとのサブグラフを持つLangGraph
- メッセージング: 型付きエージェント間メッセージのためのA2Aプロトコル（Google 2025）
- モデル: Opus 4.7（アーキテクト）、Sonnet 4.7（コーダー）、GPT-5.4（レビュアー）、Gemini 2.5 Pro（テスター）
- ワークツリー分離: コーダーごとに`git worktree add` + Daytonaサンドボックス
- マージコーディネーター: カスタムの3ウェイマージ + LLM仲介の競合解決
- 評価: SWE-bench Pro（50件のIssue）、SWE-AFシナリオ、ユニットテスト用HumanEval++
- 可観測性: ロールタグ付きスパンを持つLangfuse、エージェントごとのトークン会計
- デプロイ: 各ロールを別々のDeployment + バックログでのHPAとしてK8s

## 実装する

1. **タスクボード。** 型付きメッセージを持つファイルバックドJSONL：`plan_request`、`subtask`、`diff_ready`、`review_needed`、`test_needed`、`approved`、`rejected`、`replan_needed`。エージェントはタグをサブスクライブする。

2. **アーキテクト。** GitHub Issueを読み、明示的なサブタスクインターフェース（触れるファイル、パブリック関数、テストへの影響）を必要とするプランテンプレートでOpus 4.7を実行する。サブタスクのDAGを持つ1つの`plan_request`を出力する。

3. **コーダー。** N個の並行ワーカー、それぞれがボードから1つのサブタスクを取得する。それぞれが新しい`git worktree add`ブランチ + Daytonaサンドボックスを生成する。サブタスクを実装する。パッチ + テストデルタと一緒に`diff_ready`を出力する。

4. **マージコーディネーター。** すべてのコーダー完了時に、Nブランチをステージングブランチに3ウェイマージする。ファイルレベルの重複がある場合のみLLM仲介の競合解決。

5. **レビュアー。** GPT-5.4がマージされたdiffを読む。自身が作成したdiffを承認できない。特定の変更要求を関連するコーダーにルーティングした`approved`（ノーオペレーション）または`review_feedback`を出力する。

6. **テスター。** Gemini 2.5 Proがクリーンなサンドボックスでテストスイートを実行する。アーティファクトをキャプチャする。スタックトレース付きで`test_passed`または`test_failed`を出力する。失敗したテストは失敗したサブタスクを所有するコーダーにループバックする。

7. **ハンドオフ会計。** ロール境界を越えるすべてのメッセージはペイロードサイズと使用されたモデルを持つLangfuseのスパンを得る。サブタスクごとのトークン増幅（コーダートークン + レビュアートークン + テスタートークン + アーキテクトシェア / コーダートークン）を計算する。

8. **評価。** 50件のSWE-bench Pro Issueで実行する。シングルエージェントのベースライン（単一のワークツリーにある1つのSonnet 4.7）と比較してpass@1と解決済みIssueあたりのコストを比較する。

9. **ポストモーテム。** 各失敗したIssueについて、壊れたハンドオフを特定する（プランが曖昧すぎる、マージ競合、レビュアーの偽承認、テスターのフレーク）。ハンドオフ失敗のヒストグラムを生成する。

## 使ってみる

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## 成果物を出す

`outputs/skill-multi-agent-team.md`が成果物。Issue URLと並行レベルを与えると、チームはロールごとのトークン会計を持つマージ準備完了のPRを生成する。

| 配点 | 基準 | 測定方法 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 一致した50件のIssueサブセット、pass@1 |
| 20 | 並行スピードアップ | シングルエージェントのベースライン対ウォールクロック |
| 20 | レビュー品質 | 注入されたバグプローブでの偽承認率 |
| 20 | トークン効率 | シングルエージェント対解決済みIssueあたりの総トークン数 |
| 15 | 調整エンジニアリング | マージ競合解決、ハンドオフ失敗ヒストグラム |
| **100** | | |

## 演習

1. 実行中のdiffに明らかなバグを注入する（メインボディの前に余分な`return None`）。レビュアーの偽承認率を測定する。偽承認が5%以下になるまでレビュアープロンプトを調整する。

2. コーダーを2人に減らす（アーキテクト + コーダー + レビュアー + テスター、コーダーは2つのサブタスクを順次実行）。ウォールクロックと合格率を比較する。

3. マージコーディネーターを単一ライター制約（サブタスクが互いに独立したファイルセットに触れる）に置き換える。アーキテクトへの計画の負担を測定する。

4. レビュアーをGPT-5.4からClaude Opus 4.7に交換する。偽承認率とトークンコストのデルタを測定する。

5. 5番目のロールを追加する：ドキュメンター（Haiku 4.5）。レビューの後、変更ログのエントリを生成する。ドキュメントの品質が追加のトークン消費を正当化するかを測定する。

## キーワード

| 用語 | 一般的な呼び方 | 実際の意味 |
|------|-----------------|------------------------|
| 並行ワークツリー | 「隔離されたブランチ」 | コーダーごとに新しいワーキングツリーを生成する`git worktree add` |
| タスクボード | 「共有メッセージバス」 | エージェントがサブスクライブする型付きメッセージのファイルまたはRedisストア |
| ハンドオフ | 「ロール境界」 | 1つのロールのコンテキストから別のロールのコンテキストに渡るメッセージ |
| トークン増幅 | 「マルチエージェントのオーバーヘッド」 | ロールにわたる総トークン / 同じタスクのシングルエージェントトークン |
| A2Aプロトコル | 「エージェント間」 | 型付きエージェント間メッセージのGoogleの2025年仕様 |
| マージコーディネーター | 「インテグレーター」 | 3ウェイマージを実行し競合を仲介するコンポーネント |
| 偽承認 | 「レビュアーのハルシネーション」 | レビュアーが既知のバグを含むdiffを承認する |

## 参考資料

- [SWE-AFファクトリーアーキテクチャ](https://github.com/Agent-Field/SWE-AF) — リファレンス2026年のマルチエージェントファクトリー
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — ロールベースのマルチエージェントフレームワーク
- [AutoGen v0.4](https://github.com/microsoft/autogen) — Microsoftの型付きアクターフレームワーク
- [Cognition AI（Devin）](https://cognition.ai) — リファレンスプロダクト
- [Factory Droids](https://www.factory.ai) — 代替リファレンスプロダクト
- [Google A2Aプロトコル](https://developers.google.com/agent-to-agent) — エージェント間メッセージング仕様
- [git worktreeドキュメント](https://git-scm.com/docs/git-worktree) — 分離の基盤
- [SWE-bench Pro](https://www.swebench.com) — 評価ターゲット
