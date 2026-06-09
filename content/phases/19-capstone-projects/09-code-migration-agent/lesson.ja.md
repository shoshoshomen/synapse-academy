# キャップストーン 09 — コード移行エージェント（リポジトリレベルの言語/ランタイムアップグレード）

> AmazonのMigrationBench（Java 8から17へ）とGoogleのApp Engine Py2-to-Py3マイグレーターが2026年の基準を設定した。ModerneのOpenRewriteはスケールで決定論的なAST書き換えを行う。GritはcodemodスタイルのDSLで同じ問題をターゲットにする。本番パターンは両方を組み合わせる：安全な書き換えのための決定論的な基盤にあいまいなケース用のエージェントレイヤー、ブランチごとのビルド用のサンドボックス、PRが開く前に緑に変わるテストハーネス。キャップストーンは50の実際のリポジトリを移行し、失敗の分類法を持つ合格率を公開することだ。

**演習するフェーズ:** P5 · P7 · P11 · P13 · P14 · P15 · P17

## 問題

大規模なコード移行は2026年のコーディングエージェントの最もクリーンな本番アプリケーションの1つだ。グラウンドトゥルースは明確だ（移行後にテストスイートは通るか？）、報酬はリアルだ（Java-8フリートの移行はヘッドカウントスケールのプロジェクトだ）、ベンチマークは公開されている（MigrationBench 50リポジトリサブセット）。ModerneのOpenRewriteは決定論的な側面を処理する。エージェントレイヤーはOpenRewriteのレシピができないすべてを処理する：あいまいな書き換え、ビルドシステムのドリフト、ロングテール構文、推移的な依存関係の破壊。

Java 8リポジトリ（またはPython 2リポジトリ）を取り込んで緑のCIの移行ブランチを生成するエージェントを構築する。合格率・テストカバレッジの保存・リポジトリあたりのコストを測定し、失敗の分類法を構築する。決定論的のみのベースラインとの並列比較がエージェントの価値が実際にどこにあるかを教えてくれる。

## コンセプト

パイプラインには2つのレイヤーがある。**決定論的基盤**（JavaにはOpenRewrite、PythonにはlibCST）は機械的な書き換えの大部分を安全に実行する：インポート、メソッドシグネチャ、null安全編集、try-with-resources、非推奨のAPI置き換え。これは高速で監査可能なdiffを生成する。**エージェントレイヤー**（Claude Opus 4.7とGPT-5.4-Codexを使ったOpenAI Agents SDKまたはLangGraph）はレシピができないケースを処理する：ビルドファイルのアップグレード（Maven/Gradle/pyproject）、推移的な依存関係の競合、テストのフレーク、カスタムアノテーション。

各リポジトリはターゲットランタイムがプリインストールされたDaytonaサンドボックスを得る。エージェントは反復する：ビルドを実行し、失敗を分類し、修正を適用し、再実行する。ハード制限：リポジトリあたり30分、$8、20のエージェントターン。すべてのテストが通りカバレッジデルタが負でなければ、ブランチはPRを開く。そうでなければ、リポジトリはエビデンス付きの失敗クラスに分類される。

失敗の分類法が成果物だ。50のリポジトリにわたって何が壊れたか？推移的な依存関係？カスタムアノテーション？ビルドツールのバージョン？移行と無関係のテストフレーク？各クラスは数と例示のdiffを得る。将来のレシピ作者は上位3つをターゲットにできる。

## アーキテクチャ

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## スタック

- 決定論的基盤: OpenRewrite（Java）またはlibcst（Python）
- エージェント: Claude Opus 4.7 + GPT-5.4-Codexを使ったOpenAI Agents SDKまたはLangGraph
- サンドボックス: ブランチごとのDaytonaデブコンテナ、ターゲットランタイムプリインストール（Java 17 / Python 3.12）
- ビルドシステム: Maven、Gradle、uv（Python）
- ベンチマーク: Amazon MigrationBench 50リポジトリサブセット（Java 8から17）、Google App Engine Py2-to-Py3リポジトリ
- テストハーネス: 並行ランナー、Jacoco（Java）またはcoverage.py（Python）経由のカバレッジ
- 可観測性: すべてのdiffチャンクを持つリポジトリごとのトレースバンドルを持つLangfuse
- ダッシュボード: クラスごとの数と例示のdiffを持つ失敗分類法ダッシュボード

## 実装する

1. **レシピパス。** まずOpenRewrite（Java）またはlibcst（Python）のレシピを実行する。機械的な移行の70-80%をキャッチする。「recipe」コミットとしてコミットする。

2. **ビルドトライアル。** Daytonaサンドボックス：ターゲットランタイムをインストールし、ビルドを実行する。緑の場合はテストへスキップ。赤の場合はエージェントへ渡す。

3. **エージェントループ。** ツール付きLangGraph：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。エージェントは失敗を（dep、構文、テスト、ビルドツール）に分類し、ターゲットを絞った修正を適用する。再実行する。

4. **予算上限。** リポジトリあたり30分のウォールクロック、$8のコスト、20のエージェントターン。違反があれば現在のdiffと一緒に「budget_exhausted」に分類して停止する。

5. **テスト + カバレッジゲート。** ビルドが緑になった後、テストスイートを実行する。ベースリポジトリとのカバレッジを比較する。カバレッジが2%以上下落した場合、「coverage_regression」に分類する。

6. **PRのオープン。** 成功時に、ブランチをプッシュし、適用されたレシピとエージェントが作成したコミットのサマリーを含むdiffとともにPRを開く。

7. **失敗の分類法。** 各失敗したリポジトリについて、クラスでタグ付けする：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。ダッシュボードを構築する。

8. **50リポジトリの実行。** MigrationBenchサブセット全体で実行する。クラスごとの合格率・リポジトリあたりのコスト・カバレッジ保存・決定論的のみのベースラインとの比較をレポートする。

## 使ってみる

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## 成果物を出す

`outputs/skill-migration-agent.md`が成果物。リポジトリを与えると、決定論的なレシピを実行してからエージェントループを実行し、緑の移行ブランチを生成するか、リポジトリを分類法クラスに分類する。

| 配点 | 基準 | 測定方法 |
|:-:|---|---|
| 25 | MigrationBench合格率 | 50リポジトリサブセットのpass@1 |
| 20 | テストカバレッジ保存 | ベースに対する平均カバレッジデルタ |
| 20 | 移行リポジトリあたりのコスト | 成功した実行での$/リポジトリ |
| 20 | エージェント/決定論的ツールの統合 | OpenRewriteが処理した修正対エージェントが作成した修正の割合 |
| 15 | 失敗分析のまとめ | 例示付きの分類法の完全性 |
| **100** | | |

## 演習

1. エージェントなしでマイグレートパイプラインを実行する（OpenRewriteのみ）。合格率をフルパイプラインと比較する。エージェント単独が違いをもたらすケースを特定する。

2. 「lint-clean」チェックを実装する：移行後、スタイルリンター（Javaにはspotless、PythonにはRuff）を実行する。新しいlintエラーが現れればPRを失敗させる。カバレッジ保存されているがスタイルが後退した率を測定する。

3. 「最小差分」オプティマイザーを追加する：エージェントのブランチがテストを通過した後、第2パスで不必要な変更を削除する。差分サイズの削減をレポートする。

4. 3番目の移行に拡張する：Node 18からNode 22へ。サンドボックスラッピングを再利用し；レシピレイヤーをカスタムcodemodに交換する。

5. ファーストグリーンビルドまでの時間（TTFGB）をUXメトリクスとして測定する。ターゲット：p50 10分以下。

## キーワード

| 用語 | 一般的な呼び方 | 実際の意味 |
|------|-----------------|------------------------|
| 決定論的基盤 | 「レシピエンジン」 | OpenRewrite / libcst：安全性保証を持つ宣言的なAST書き換え |
| codemod | 「コード変更プログラム」 | ソースコードを機械的に変更する書き換えルール |
| ビルドドリフト | 「ツールバージョンのずれ」 | メジャーバージョン間の微妙なMaven / Gradle / uvの動作変化 |
| 失敗クラス | 「分類法バケット」 | リポジトリが移行しなかったラベル付きの理由：dep、構文、テスト、ビルドツール、予算 |
| カバレッジデルタ | 「カバレッジ保存」 | ベースから移行ブランチへのテストカバレッジ%の変化 |
| エージェントターン | 「ツール呼び出しラウンド」 | エージェントループ内の1つの計画→実行→観察サイクル |
| 予算の枯渇 | 「上限に達した」 | リポジトリが30分/$8/20ターンの制限を通過せずに消費した |

## 参考資料

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 2026年のカノニカルベンチマーク
- [Moderne.io OpenRewriteプラットフォーム](https://www.moderne.io) — 決定論的基盤リファレンス
- [OpenRewriteドキュメント](https://docs.openrewrite.org) — レシピオーサリング
- [Grit.io](https://www.grit.io) — 代替codemod DSL
- [OpenAIサンドボックス移行クックブック](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK リファレンス
- [Google App Engine Py2からPy3マイグレーター](https://cloud.google.com/appengine) — 代替移行ベンチマーク
- [libcst](https://github.com/Instagram/LibCST) — Python決定論的基盤
- [Daytonaサンドボックス](https://daytona.io) — リファレンスブランチごとのサンドボックス
