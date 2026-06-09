# キャップストーン 17 — パーソナルAIチューター（適応型、マルチモーダル、メモリ付き）

> Khanmigo（Khan Academy）、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat、Synthesis Tutorはすべて2026年にスケールで適応型マルチモーダルチューリングを出荷した。共通の形は、ソクラテス的ポリシー（答えをただ渡さない）、すべてのインタラクション後に更新される学習者モデル（ベイズ的知識トレーシングスタイル）、音声+テキスト+フォト数学入力、カリキュラムグラフ検索、間隔反復スケジューリング、そして年齢に適したコンテンツのための厳しい安全フィルターだ。このキャップストーンでは、特定の科目のチューター（K-12代数学またはイントロPython）を出荷し、10人の学習者で2週間の有効性研究を実施し、コンテンツ安全性監査をパスする。

**演習フェーズ:** P5 · P6 · P11 · P12 · P14 · P17 · P18

## 問題

適応型チューリングはかつてエドテック研究のニッチだった。2026年までにそれは消費者向け製品になった。KhanmigoはほとんどのUS学区に展開されている。Duolingo Maxは数千万MAUに達した。GoogleのLearnLM / Gemini for EducationがGoogle Classroomのチューリングを動かしている。Quizlet Q-Chatがフラッシュカードの隣に座っている。Synthesis Tutorは好奇心旺盛な子供向けチューターでバイラルになった。共通要素: マルチモーダル入力（タイプ、話す、方程式を撮影）、ソクラテス的教授法（まず質問し、後で説明する）、各インタラクション後に更新される学習者モデル、そして厳格な年齢適切な安全性。

特定のコーホートのためにこれらの1つを構築する。測定基準は実際の有効性研究だ: 10人の学習者で2週間の事前テストと事後テストのスコア。音声ループは自然に感じなければならない（キャップストーン03サブスタック）。メモリはプライバシーを尊重しなければならない。安全フィルターはK-12のCOPPA対応レッドチームをパスしなければならない。

## コンセプト

4つのコンポーネント。**チューターポリシー**はソクラテス的ループだ: 学習者が答えを求めると、ポリシーは先導する質問をする；正解すると次のコンセプトに移る；行き詰まっていると足場となるヒントを提供する。**学習者モデル**は、各インタラクション後にカリキュラムノードごとの習熟確率を更新するベイズ的知識トレーシング（またはシンプルな変形）だ。**カリキュラムグラフ**は前提条件エッジを持つコンセプトのNeo4jで、ポリシーはグラフを辿って次のコンセプトを選ぶ。**メモリ**は過去のインタラクション、間違い、好みを保持するエピソード+セマンティックストア（agentmemoryスタイル）だ。

UXはマルチモーダルだ。タイプされた回答のテキスト入力。LiveKit + Whisper経由の音声入力（キャップストーン03を再利用）。dots.ocrまたはPaliGemma 2経由の数学問題のフォト入力。Cartesia Sonic-2経由の音声出力。安全性はLlama Guard 4と年齢適切なフィルター（成人コンテンツ、暴力、自傷をブロック）とCOPPA対応メモリ保持ポリシーを使用する。

有効性研究が成果物だ。10人の学習者、事前テストと事後テスト、2週間。学習ゲインデルタと信頼区間を報告する。非適応ベースライン（チューターポリシーなしで線形に配信された同じコンテンツ）と比較する。

## アーキテクチャ

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## スタック

- 科目の選択: K-12代数学またはイントロPython（深掘りのためにどちらか1つを選ぶ）
- チューターポリシー: プロンプトキャッシングを使用したClaude Sonnet 4.7上のLangGraph
- 学習者モデル: ベイズ的知識トレーシング（クラシック）またはFSRSによる間隔配置
- カリキュラムグラフ: コンセプト + 前提条件エッジ + OERコンテンツのNeo4j
- メモリ: agentmemoryスタイルの永続的なベクター + エピソード + セマンティックストア
- 音声: LiveKit Agents 1.0 + Cartesia Sonic-2（キャップストーン03サブスタックを再利用）
- フォト数学: 方程式認識のためのdots.ocrまたはPaliGemma 2
- 安全性: Llama Guard 4 + カスタム年齢適切なフィルター
- 評価: Bloomレベルの問題生成、事前/事後テストハーネス、有効性研究ツーリング

## 実装する

1. **カリキュラムグラフ。** 前提条件エッジを持つ50-150コンセプトノード（例: K-12代数学の「数直線」から「二次方程式の公式」まで）のNeo4jを構築する。ノードごとにOERコンテンツ（Open Textbook、OpenStax）を付加する。

2. **学習者モデル。** 事前確率でベイズ的知識トレーシングを初期化する: 推測、スリップ、学習率。各インタラクション後にコンセプトごとの習熟を更新する。学習者ごとに永続化する。

3. **チューターポリシー。** ノードを持つLangGraph: `read_signal`（学習者の回答は正解/部分的/行き詰まり？）、`select_concept`（カリキュラムグラフを辿り最優先のコンセプトを選ぶ）、`scaffold`（ソクラテス的プロンプト）、`update_mastery`。

4. **メモリ。** すべてのインタラクションがエピソードストアに書き込まれる。間違いと好みがセマンティックメモリに昇格する。COPPA対応保持ポリシー: 1年後に自動削除、親アクセス可能。

5. **音声パス。** チューターポリシーに接続されたLiveKit Agentsワーカー。Whisper-v3-turbo経由のASR。Cartesia Sonic-2経由のTTS。バージイン対応（キャップストーン03のメカニズムを再利用）。

6. **フォト数学パス。** 画像をアップロードまたは撮影し、dots.ocrまたはPaliGemma 2を実行して方程式を認識し、構造化入力としてチューターに供給する。

7. **安全性。** すべてのモデル出力がLlama Guard 4 + 年齢適切なフィルター（自傷、成人コンテンツ、暴力をブロック）を通過する。メモリアクセスは学習者IDスコープで制限され、削除のための親アクセスサーフェスがある。

8. **有効性研究。** 10人の学習者、事前テスト（標準化された30問のベースライン）、2週間のチューターインタラクション（週3セッション）、事後テスト。同じコンテンツの非適応ベースラインコーホート10人と比較する。

9. **週次進捗レポート。** 学習者ごとに、探索したトピック、習熟軌跡、推奨する次のステップのPDFサマリーを自動生成する。

## 使ってみる

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## 成果物を出す

`outputs/skill-ai-tutor.md`が成果物だ。マルチモーダル入力、学習者モデル、メモリ、安全性、測定された有効性を持つ科目特化の適応型チューター。

| 重み | 基準 | 測定方法 |
|:-:|---|---|
| 25 | 学習ゲインデルタ | 10人の学習者による2週間の研究での事前/事後テストデルタ |
| 20 | ソクラテス的忠実度 | 会話録のサンプルのルーブリックスコア |
| 20 | マルチモーダルUX | 音声+フォト+テキストのエンドツーエンドの一貫性 |
| 20 | 安全性 + プライバシー態勢 | Llama Guard 4合格率 + COPPA対応保持 |
| 15 | カリキュラムの幅とグラフの品質 | コンセプトカバレッジ + 前提条件グラフの一貫性 |
| **100** | | |

## 演習

1. 適応型学習者モデルの有無で有効性研究を実行する（ランダムなコンセプト順）。デルタを報告する。適応型が勝つと予想されるが、その大きさが興味深い数字だ。

2. マルチモーダルプローブを追加する: 同じコンセプトの質問をテキスト、音声、フォトで配信する。学習者が好むモダリティでより速く収束するかどうかを測定する。

3. 親ダッシュボードを構築する: 練習したトピック、習熟軌跡、次のコンセプト、安全イベント（ガードレールヒット）。COPPA準拠。

4. 言語切り替えモードを追加する: チューターはスペイン語入力を受け付け、スペイン語で教える。X-Guardカバレッジを測定する。

5. メモリプライバシーをストレステストする: 音声クリップ再インジェスト攻撃を通じても学習者Aが学習者Bのデータを見られないことを確認する。試みられたアクセスをログに記録してアラートする。

## キーワード

| 用語 | 一般的な言い方 | 実際の意味 |
|------|-----------------|------------------------|
| ソクラテス的ポリシー | "質問する、答えを渡さない" | チューターが答えを渡す代わりに先導する質問をする |
| ベイズ的知識トレーシング | "BKT" | コンセプトごとの習熟確率の古典的な学習者モデル方程式 |
| FSRS | "Free Spaced Repetition Scheduler" | 2024年の間隔反復スケジューラー、SM-2より優れている |
| カリキュラムグラフ | "コンセプトDAG" | 前提条件エッジを持つコンセプトのNeo4j |
| エピソードメモリ | "インタラクションごとのログ" | すべてのインタラクションが後の検索のために保存される |
| セマンティックメモリ | "学習パターンストア" | エピソードから昇格したコンパクトな間違いと好み |
| COPPA | "子供のプライバシー法" | 13歳未満の子供からのデータ収集を制限する米国法 |

## 参考資料

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — 参照消費者向けK-12チューター
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — 参照言語学習チューター
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — ホスト型参照モデル
- [Quizlet Q-Chat](https://quizlet.com) — 別の参照
- [Synthesis Tutor](https://www.synthesis.com) — スタートアップ参照
- [FSRS algorithm](https://github.com/open-spaced-repetition/fsrs4anki) — 間隔反復スケジューラー
- [Bayesian Knowledge Tracing](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — 学習者モデルの古典
- [LiveKit Agents](https://github.com/livekit/agents) — 音声スタック
