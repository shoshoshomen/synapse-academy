# キャップストーン 12 — 動画理解パイプライン（シーン・QA・検索）

> Twelve LabsがMarengo + Pegasusを製品化した。VideoDBがCRUD-for-video APIを出荷した。AI2のMolmo 2がオープンVLMチェックポイントを公開した。Geminiの長コンテキストは数時間の動画をネイティブに処理できる。TimeLens-100Kがスケールでの時間的グラウンディングを定義した。2026年のパイプラインは確立している：シーン分割・シーンごとのキャプション+埋め込み・トランスクリプトアライメント・マルチベクトルインデックス・そして（start, end）タイムスタンプとフレームプレビュー付きで回答するクエリ。キャップストーンは100時間をインジェストし・公開ベンチマークを実施し・カウントとアクション質問でのハルシネーションを計測することだ。

**演習するフェーズ:** P4 · P6 · P7 · P11 · P12 · P17

## 問題

長編動画QAは2026年スケールで最も帯域幅を消費するマルチモーダル問題だ。Gemini 2.5 Proは2時間の動画をネイティブに読めるが、100時間の動画をクエリ可能なコーパスにインジェストするにはシーンレベルのインデックスが依然として必要だ。本番環境の形はシーン分割（TransNetV2またはPySceneDetect）・VLMによるシーンごとのキャプション生成（Gemini 2.5・Qwen3-VL-Max・またはMolmo 2）・トランスクリプトアライメント（単語タイムスタンプ付きWhisper-v3-turbo）・キャプション・フレーム埋め込み・トランスクリプトを並べて格納するマルチベクトルインデックスを組み合わせる。クエリパイプラインは（start, end）タイムスタンプとフレームプレビュー付きで回答する。

ベンチマークは公開されている（ActivityNet-QA・NeXT-GQA）に加え、自前の100クエリカスタムセットを使う。カウントとアクション型質問でのハルシネーションは既知の困難な失敗クラスであり、キャップストーンで明示的に計測する。

## コンセプト

インジェスト時に3つのパイプラインが並行して動く。**シーン分割**が動画をシーンに切り出す。**VLMキャプショニング**がシーンごとのキャプションとキーフレームからのフレーム埋め込みを生成する。**ASRアライメント**が単語レベルのタイムスタンプを生成する。3つのストリームは（scene_id・時間範囲）で結合される。各シーンはマルチベクトルインデックス（Qdrant）に3種類のベクトルを持つ：キャプション埋め込み・キーフレーム埋め込み・トランスクリプト埋め込み。

クエリ時に自然言語の質問が3つのベクトル全てに対して発火し、結果はRRFでマージされ、テンポラルグラウンディングアダプター（TimeLensスタイル）が上位シーン内の（start, end）ウィンドウを精度向上させる。VLMシンセサイザー（Gemini 2.5 ProまたはQwen3-VL-Max）がクエリ+上位シーン+切り出したフレームを受け取り、引用タイムスタンプとフレームプレビュー付きで回答する。

ハルシネーションの計測が重要だ。カウント（「部屋に入る人は何人？」）とアクション型（「シェフはかき混ぜる前に注ぐか？」）の質問は著しく信頼性が低い。記述的な質問とは別に精度を報告する。

## アーキテクチャ

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## スタック

- シーン分割: TransNetV2（2024-26年最先端）またはPySceneDetect
- ASR: 単語タイムスタンプ付きfaster-whisper経由のWhisper-v3-turbo
- VLMキャプショナー+回答者: Gemini 2.5 ProまたはQwen3-VL-MaxまたはMolmo 2
- テンポラルグラウンディング: TimeLens-100K学習アダプターまたはVideoITG
- インデックス: マルチベクトル対応Qdrant（キャプション/フレーム/トランスクリプト）
- UI: HTMLビデオプレーヤーとシーンサムネイル付きNext.js 15
- 評価: ActivityNet-QA・NeXT-GQA・カスタム100質問手動ラベル付きセット
- ハルシネーションベンチマーク: 手動ラベル付きのカウントとアクション型サブセット

## 実装する

1. **インジェストウォーカー。** YouTube URLまたはローカルMP4を受け付ける。必要に応じて720pにダウンスケールする。`{video_id, file_path}`を保存する。

2. **シーン分割。** TransNetV2またはPySceneDetectを実行して`[{scene_id, start_ms, end_ms, keyframe_path}]`を生成する。目標100時間：約6k-8kシーン。

3. **ASRパス。** 音声にWhisper-v3-turboを実行し；単語レベルタイムスタンプをエクスポート；シーンごとのトランスクリプトスライスに分割する。

4. **VLMキャプショニング。** シーンごとにGemini 2.5 Pro（またはQwen3-VL-Max）をキーフレームと短いキャプションテンプレートで呼び出す。キャプション+フレーム埋め込みを生成する。

5. **マルチベクトルインデックス。** 3つの名前付きベクトルを持つQdrantコレクション。ペイロード：`{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **クエリ。** 自然言語の質問が3つのデンスクエリを発火し；reciprocal rank fusionでマージ；top-k=5シーン。

7. **テンポラルグラウンディング。** 上位シーンにTimeLensスタイルアダプターを実行してシーン内の（start, end）ウィンドウを精度向上させる。

8. **VLMシンセ。** クエリ+上位3シーンのクリップ（画像または短いクリップとして）+トランスクリプトでGemini 2.5 Proを呼び出す。`(video_id, start_ms, end_ms)`の引用を必須とする。

9. **評価。** ActivityNet-QAとNeXT-GQAを実行する。100クエリのカスタムセットを構築する。全体精度+クラス別内訳（カウント・アクション・記述）を報告する。

## 使ってみる

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## 成果物を出す

`outputs/skill-video-qa.md`が成果物。YouTube URLまたはアップロードされた動画が与えられると、パイプラインはシーンをインデックスし、タイムスタンプ引用付きで質問に回答する。

| 配点 | 基準 | 測定方法 |
|:-:|---|---|
| 25 | テンポラルグラウンディングIoU | ホールドアウトグラウンディングセットでの交差/和集合 |
| 20 | QA精度 | NeXT-GQAとカスタム100クエリ |
| 20 | インジェストスループット | 費やしたドルあたりの動画時間 |
| 20 | UIと引用UX | タイムスタンプリンク・サムネイルストリップ・フレームへのジャンプ |
| 15 | ハルシネーション率 | カウントとアクション型の精度を別々に |
| **100** | | |

## 演習

1. キャプショニングパスでGemini 2.5 ProをQwen3-VL-Maxに置き換える。人間が評価した50シーンサンプルでのキャプション品質のデルタを報告する。

2. シーンごとのフレーム埋め込みをマルチベクトルの代わりに1つのプールされたベクトルに削減する。リトリーバルのリグレッションを計測する。

3. 「カウント厳密」モードを構築する：シンセサイザーが各カウントされたインスタンスをタイムスタンプ付きで抽出し、ユーザーがクリックして検証する。ユーザー検証がハルシネーションを減らすか計測する。

4. インジェストコストをベンチマークする：3つのVLMの選択肢で費やしたドルあたりの動画時間。スイートスポットを選ぶ。

5. スピーカー分離されたトランスクリプトを追加する：音声にpyannoteスピーカー分離を実行し、スピーカーごとのトランスクリプトを埋め込む。「アリスがXについて何を言ったか」クエリを実証する。

## キーワード

| 用語 | 一般的な呼び方 | 実際の意味 |
|------|-----------------|------------------------|
| シーン分割 | 「ショット検出」 | ショット境界で動画をシーンに切り出すこと |
| マルチベクトルインデックス | 「キャプション+フレーム+トランスクリプト」 | 表現ごとに名前付きベクトルを持つQdrantコレクション |
| テンポラルグラウンディング | 「いつ正確に起きたか」 | クエリの回答の（start, end）ウィンドウを精度向上させること |
| フレーム埋め込み | 「視覚的表現」 | キーフレームのベクトル埋め込み；シーンの視覚的類似性に使用 |
| RRF融合 | 「Reciprocal rank fusion」 | 複数のランキングリストを統合する戦略；古典的なハイブリッドリトリーバルの技法 |
| カウントハルシネーション | 「誤カウント」 | VLMの「XはいくつあるかKが？」質問での既知の失敗モード |
| ActivityNet-QA | 「動画QAベンチマーク」 | 長編動画QA精度ベンチマーク |

## 参考資料

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — オープンVLMチェックポイント
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — スケールでのテンポラルグラウンディング
- [Gemini Video long-context](https://deepmind.google/technologies/gemini) — ホスト型リファレンス
- [VideoDB](https://videodb.io) — CRUD-for-video APIリファレンス
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — 商用リファレンス
- [TransNetV2](https://github.com/soCzech/TransNetV2) — シーン分割モデル
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — クラシックなオープン代替
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — リファレンス評価ベンチマーク
