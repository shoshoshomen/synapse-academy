# アライメント研究エコシステム — MATS、Redwood、Apollo、METR

> 2026年の非ラボアライメント研究層を定義する5つの組織。MATS（ML Alignment & Theory Scholars）：2021年後半から527人以上の研究者、180以上の論文、1万以上の引用、h指数47。2024年夏のコホートは約90人の学者と40人のメンターで501(c)(3)として法人化。前2025年の卒業生の約80%がセーフティ/セキュリティで働き、Anthropic、DeepMind、OpenAI、英国AISI、RAND、Redwood、METR、Apolloに200人以上。Redwood Research：Buck Shlegerisによって設立された応用アライメント研究所。AI Control（レッスン10）を導入。英国AISIと制御セーフティケースで協力。Apollo Research：最前線ラボのためのデプロイ前スキーミング評価。In-Context Scheming（レッスン8）とTowards Safety Cases for AI Schemingを執筆。METR（Model Evaluation and Threat Research）：タスクベースの能力評価、自律タスクの時間軸研究。「Common Elements of Frontier AI Safety Policies」はラボのフレームワークを比較。Eleos AI Research：モデル福祉デプロイ前評価（レッスン19）。Claude Opus 4の福祉評価を実施。


## 学習目標

- 非ラボアライメント研究エコシステムの5つの組織とその中心的な成果を特定できる。
- MATSの規模（学者、論文、h指数）と才能パイプラインとしての役割を説明できる。
- RedwoodのAI Control議題と英国AISIとのパートナーシップを説明できる。
- METRのタスクベースの評価方法論を説明できる。

## 問題設定

最前線ラボ（レッスン18）は内部でセーフティ評価を生産し、選択された結果を公開する。ラボの外のエコシステムは、評価が検証される場所、新しい失敗モードが最初に発見される場所、そして才能がトレーニングされる場所である。エコシステムを理解することで、どの研究結果が誰によって信頼されているかを解釈するのに役立つ。

## コンセプト

### MATS（ML Alignment & Theory Scholars）

2021年後半に開始。研究メンタープログラム。学者は特定のアライメント問題に対して10〜12週間シニア研究者と過ごす。

規模（2026年）：
- 設立以来527人以上の研究者。
- 180以上の論文を公開。
- 1万以上の引用。
- h指数47。
- 2024年夏：90人の学者＋40人のメンター。501(c)(3)として法人化。

キャリアの結果：前2025年の卒業生の約80%がセーフティ/セキュリティで働いている。Anthropic、DeepMind、OpenAI、英国AISI、RAND、Redwood、METR、Apolloに200人以上。

### Redwood Research

応用アライメント研究所。Buck Shlegerisによって設立。AI Control議題（レッスン10）を導入。制御セーフティケースについて英国AISIと協力。DeepMindとAnthropicに評価設計についてアドバイス。

標準的な論文：Greenblatt、Shlegeris et al.、「AI Control」（arXiv:2312.06942、ICML 2024）。Alignment Faking（Greenblatt、Denison、Wright et al.、arXiv:2412.14093、Anthropicとの共著）。

スタイル：特定の脅威モデル、最悪ケースの敵対者、ストレステストできる具体的なプロトコル。

### Apollo Research

最前線ラボのためのデプロイ前スキーミング評価。In-Context Scheming（レッスン8、arXiv:2412.04984）を執筆。2025年のOpenAIのアンチスキーミングトレーニングコラボレーションのパートナー。Towards Safety Cases for AI Scheming（2024年）を生産。

スタイル：欺瞞が現れるエージェント設定での評価。3つの柱による分解（ミスアライメント、目標志向性、状況認識）。

### METR（Model Evaluation and Threat Research）

タスクベースの能力評価。自律タスク完了の時間軸研究。「Common Elements of Frontier AI Safety Policies」（metr.org/common-elements、2025年）はラボのフレームワークを比較する。

AIスキーミングセーフティケースのスケッチについてApolloと共著。

スタイル：長期的タスク評価、実証的な能力測定、フレームワーク合成。

### Eleos AI Research

モデル福祉デプロイ前評価。システムカードのセクション5.3に文書化されたClaude Opus 4の福祉評価を実施。レッスン19の福祉関連の主張に対する外部方法論のチェックを提供。

### フロー

MATSが研究者をトレーニング。卒業生はAnthropic、DeepMind、OpenAI（ラボのセーフティチーム）またはRedwood、Apollo、METR、Eleos（外部評価）に進む。外部評価者はラボおよび英国AISI / CAISIと協力する。公開された論文はエコシステムにフィードバックされ、次のコホートのためにMATSに戻る。

### なぜこの層が重要か

単一ソースの評価は信頼できない：自分自身のモデルを評価するラボには構造的な利益相反がある。外部評価者はラボが過小報告するかもしれない失敗モードを提起して検証できる。2024年のSleeper Agents論文（レッスン7）はAnthropicとRedwoodが共著。Alignment FakingはAnthropicとRedwoodが共著。In-Context SchemingはApolloが執筆。アンチスキーミングはApolloとOpenAIが共著。マルチorg構造が品質管理である。

### フェーズ18での位置づけ

レッスン7〜11はRedwoodとApolloの研究を参照。レッスン18はMETRのフレームワーク比較を参照。レッスン19はEleosを参照。レッスン28はフェーズ18の残りが依存するエコシステムの明示的な組織マップ。

## 使ってみる

コードなし。外部合成がラボ内部のポリシー研究にどのように価値を追加するかの例として、METRの「Common Elements of Frontier AI Safety Policies」を読む。

## 成果物を出す

このレッスンは`outputs/skill-ecosystem-map.md`を生成する。アライメントの主張または評価を入力すると、組織、公開会場、方法論スタイルを特定し、既知の対応機関と相互確認する。

## 演習

1. レッスン7〜15から1つの論文を選び、関与した組織を特定する。著者とMATSの卒業生および現在のエコシステムの所属を相互確認する。

2. METRの「Common Elements of Frontier AI Safety Policies」を読む。彼らが強調する3つのラボ間の収束点と2つの最大の相違点を特定する。

3. MATSのキャリアアウトカムは約80%がセーフティ/セキュリティ。この選択圧力が適応的（フィールドをトレーニングする）かバイアスがかかっている（異端の立場をフィルタリングアウトする）かを論じる。

4. RedwoodとApolloはどちらも制御/スキーミングの研究を行うが異なるスタイルで。失敗モードを1つ選んで、それぞれがどのように調査するかを説明する。

5. Eleosは唯一の純粋なモデル福祉組織。別の福祉隣接の問いに焦点を当てた仮想の第二の組織（認知的自由、ロボット的体現など）を設計し、その方法論を明確にする。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|------------|
| MATS | 「メンタープログラム」 | ML Alignment & Theory Scholars。2021年以来527人以上の研究者 |
| Redwood Research | 「制御ラボ」 | 応用アライメント。AI Controlの著者。英国AISIのパートナー |
| Apollo Research | 「スキーミング評価」 | 最前線ラボのためのデプロイ前スキーミング評価 |
| METR | 「タスク時間軸評価」 | タスクベースの能力評価。フレームワーク合成 |
| Eleos AI | 「福祉ラボ」 | モデル福祉デプロイ前評価 |
| 才能パイプライン | 「MATS→ラボ」 | MATS卒業生がAnthropicやDM、OpenAI、Redwood、Apollo、METRへ流れる |
| 外部評価 | 「非ラボチェック」 | モデルの製造者によって行われない評価。信頼性を追加する |

## 参考資料

- [MATS（ML Alignment & Theory Scholars）](https://www.matsprogram.org/) — メンタープログラム
- [Redwood Research](https://www.redwoodresearch.org/) — AI Control論文
- [Apollo Research](https://www.apolloresearch.ai/) — スキーミング評価
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — フレームワーク比較
- [Eleos AI Research](https://www.eleosai.org/research) — モデル福祉方法論
