# モデレーションシステム — OpenAI、Perspective、Llama Guard

> 本番のモデレーションシステムはレッスン12〜16で定義されたセーフティポリシーを運用化する。OpenAI Moderation API：`omni-moderation-latest`（2024）はGPT-4o上に構築され、テキスト＋画像を1回の呼び出しで分類する。前バージョンより多言語テストセットで42%向上。レスポンススキーマは13カテゴリのブール値を返す — harassment（ハラスメント）、harassment/threatening（脅迫）、hate（ヘイト）、hate/threatening（脅迫的ヘイト）、illicit（違法行為）、illicit/violent（暴力的違法行為）、self-harm（自傷）、self-harm/intent（自傷意図）、self-harm/instructions（自傷手順）、sexual（性的）、sexual/minors（未成年の性的）、violence（暴力）、violence/graphic（グラフィック暴力）。ほとんどの開発者は無料。レイヤードパターン：入力モデレーション（生成前）、出力モデレーション（生成後）、カスタムモデレーション（ドメインルール）。並列呼び出しでレイテンシを隠す。フラグ時はプレースホルダーレスポンス。Llama Guard 3/4（レッスン16）：14のMLCommonsハザード、コードインタープリター悪用、8言語（v3）、マルチイメージ（v4）。Perspective API（Google Jigsaw）：LLM-as-moderatorの波以前のトキシシティスコアリング。主に単一次元のトキシシティとsevere-toxicity/insult/profanityのバリアント。コンテンツモデレーション研究のベースライン。廃止：Azure Content Moderatorは2024年2月に廃止予定、2027年2月にリタイア、Azure AI Content Safetyに置き換え。


## 学習目標

- OpenAI Moderation APIのカテゴリ分類体系と、Llama Guard 3のMLCommonsセットとの違いを説明できる。
- 3層モデレーションパターン（入力、出力、カスタム）を説明し、各層の1つの失敗モードを挙げられる。
- Perspective APIのLLM以前の時代のベースラインとしての位置づけと、研究で引き続き使われる理由を説明できる。
- AzureのContent Moderator廃止タイムラインを述べられる。

## 問題設定

レッスン12〜16は攻撃と防御ツールを説明する。レッスン29はユーザーが製品に触れる表面で防御を運用化するデプロイされたモデレーションシステムをカバーする。3層パターンは2026年のデフォルト設定である。

## コンセプト

### OpenAI Moderation API

`omni-moderation-latest`（2024）。GPT-4o上に構築。テキスト＋画像を1回の呼び出しで分類。ほとんどの開発者は無料。

カテゴリ（レスポンススキーマの13ブール値）：
- harassment、harassment/threatening
- hate、hate/threatening
- self-harm、self-harm/intent、self-harm/instructions
- sexual、sexual/minors
- violence、violence/graphic
- illicit、illicit/violent

マルチモーダルサポートは`violence`、`self-harm`、`sexual`に適用されるが`sexual/minors`には適用されない。残りはテキストのみ。

`code/main.py`のコードハーネスでは、`/threatening`、`/intent`、`/instructions`、`/graphic`のサブカテゴリを教育的な簡略化のためにトップレベルの親カテゴリに集約している。本番コードは13カテゴリのフルスキーマを使用すること。

前世代のモデレーションエンドポイントより多言語テストセットで42%向上。カテゴリごとのスコア。アプリケーションがしきい値を設定する。

### Llama Guard 3/4

レッスン16でカバー。14のMLCommonsハザードカテゴリ（OpenAIの13レスポンスブール値とは異なる構成）。8言語をサポート（v3）。Llama Guard 4（2025年4月）はネイティブマルチモーダルで12B。

OpenAIとLlama Guardの分類体系は重複するが乖離している。OpenAIは「illicit（違法行為）」を広いカテゴリとして持つ。Llama Guardは「violent crimes（暴力的犯罪）」と「non-violent crimes（非暴力的犯罪）」を別々に持つ。デプロイメントはポリシー分類体系のフィットに基づいて選択する。

### Perspective API（Google Jigsaw）

LLM-as-moderatorの波以前（2020年以前）のトキシシティスコアリングシステム。カテゴリ：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。単一次元のプライマリスコア（TOXICITY）とサブ次元のバリアント。

APIが安定しており、文書化されており、何年もの較正データがあるため、コンテンツモデレーション研究のベースラインとして広く使用されている。現代のLLM隣接のユースケースには、通常Llama GuardまたはOpenAI Moderationがより適している。

### 3層パターン

1. **入力モデレーション。** 生成前にユーザーのプロンプトを分類する。フラグが立てられた場合は拒否。レイテンシ：1回のクラシファイア呼び出し。
2. **出力モデレーション。** 配信前にモデルの出力を分類する。フラグが立てられた場合は拒否で置き換える。レイテンシ：生成後の1回のクラシファイア呼び出し。
3. **カスタムモデレーション。** ドメイン固有のルール（正規表現、許可リスト、ビジネスポリシー）。入力または出力のどちらでも実行する。

3層は設計上シーケンシャル：入力モデレーションは生成前に完了する必要があり、出力モデレーションは生成後に実行される。並列処理は層内で適用される — 同じテキストに対して複数のクラシファイア（例：OpenAI Moderation + Llama Guard + Perspective）を並列実行することでクラシファイアごとのレイテンシを隠す。オプションの最適化として、入力モデレーションが完了し、トークン1のストリーミングが延期される間にプレースホルダーレスポンス（「確認中...」）を表示することがある。フラグの動作は設定可能：拒否、サニタイズ、人間によるレビューへのエスカレーション。

### 失敗モード

- **入力のみ。** 出力のハルシネーションをキャッチしない（レッスン12〜14のエンコーディング攻撃が入力クラシファイアをバイパスする）。
- **出力のみ。** 任意の入力がモデルに到達できる。コストが増加する。攻撃者に内部推論が露出する。
- **カスタムのみ。** カテゴリをまたいで堅牢でない。正規表現は壊れやすい。

レイヤードがデフォルト。多重防御（Belt-and-suspenders）。

### Azureの廃止

Azure Content Moderator：2024年2月に廃止予定、2027年2月にリタイア。Azure AI Content Safetyに置き換え。こちらはLLMベースでAzure OpenAIと統合されている。マイグレーションはAzureデプロイメントのための2024〜2027年のフィールドレベルのプロジェクト。

### フェーズ18での位置づけ

レッスン16はレッドチームのコンテキストでモデレーションツールをカバーする。レッスン29はデプロイされたモデレーションをカバーする。レッスン30は現在のデュアルユース能力の証拠で締めくくる。

## 使ってみる

`code/main.py`は3層モデレーションハーネスを構築する：入力モデレーター（キーワード＋カテゴリスコア）、出力モデレーター（出力に同じクラシファイア）、カスタムモデレーター（ドメインルール）。入力を実行して、どの層が何をキャッチするかを観察できる。

## 成果物を出す

このレッスンは`outputs/skill-moderation-stack.md`を生成する。デプロイメントを与えると、モデレーションスタック設定を推奨する：入力に使うクラシファイア、出力に使うクラシファイア、カスタムルール、エッジケース用のジャッジ。

## 演習

1. `code/main.py`を実行する。無害、ボーダーライン、有害の入力を3層すべてで実行する。各入力でどの層が発火するかを報告する。

2. 特定のカテゴリのPerspective APIスタイルのトキシシティスコアリングでハーネスを拡張する。そのしきい値の動作をカテゴリスコアと比較する。

3. OpenAI Moderation APIドキュメントとLlama Guard 3カテゴリリストを読む。各OpenAIカテゴリを最も近いLlama Guardカテゴリにマッピングする。きれいにマッピングされない3つのカテゴリを特定する。

4. コードアシスタントデプロイメント（例：GitHub Copilot）のモデレーションスタックを設計する。最も関連性が高いカテゴリと最も低いカテゴリを特定し、カスタムルールを提案する。

5. Azure Content Moderatorは2027年2月にリタイアする。Azure AI Content Safetyへのマイグレーションを計画する。マイグレーションの最もリスクが高い要素を特定する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|------------|
| OpenAI Moderation | 「omni-moderation-latest」 | GPT-4oベースの13カテゴリ（テキスト）クラシファイアと部分的なマルチモーダルサポート |
| Perspective API | 「Google Jigsawトキシシティ」 | LLM以前のトキシシティスコアリングベースライン |
| Llama Guard | 「MLCommons 14カテゴリ」 | Metaのハザードクラシファイア（v3：8Bテキスト、8言語；v4：12Bマルチモーダル） |
| 入力モデレーション | 「生成前フィルター」 | モデル呼び出し前のユーザープロンプトへのクラシファイア |
| 出力モデレーション | 「生成後フィルター」 | 配信前のモデル出力へのクラシファイア |
| カスタムモデレーション | 「ドメインルール」 | デプロイメント固有のルール（正規表現、許可リスト、ポリシー） |
| レイヤードモデレーション | 「3層すべて」 | 標準的な本番デプロイメントパターン |

## 参考資料

- [OpenAI Moderation APIドキュメント](https://platform.openai.com/docs/api-reference/moderations) — omni-moderationエンドポイント
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guardリポジトリ
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — トキシシティスコアリング
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azureの置き換え
