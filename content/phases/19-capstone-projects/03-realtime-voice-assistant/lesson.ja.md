# キャップストーン 03 — リアルタイム音声アシスタント（ASRからLLMへ、そしてTTSへ）

> 正しく感じる音声エージェントは、エンドツーエンドのレイテンシーが800ms以下で、話し終えたことを認識し、割り込みを処理し、音声を止めることなくツールを呼び出せる。Retell、Vapi、LiveKit Agents、Pipecatはすべて2026年にこの基準を達成している。ストリーミングASR、ターン検出器、ストリーミングLLM、ストリーミングTTSを各ホップで積極的なレイテンシー予算とともにWebRTCを通じてすべて接続することで実現している。構築して、WER・MOS・誤カット率を測定し、パケットロス下で実行せよ。

**演習するフェーズ:** P6 · P7 · P11 · P13 · P14 · P17

## 問題

音声は2025-2026年に最も速く動くAI UXカテゴリとなった。技術的な天井は四半期ごとに下がった。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0、Pipecat 0.0.70はすべて800ms以下の初回音声出力を実現可能にした。基準はレイテンシーだけではない。インタラクションの感触だ：ユーザーの発話を切らないこと、切られないこと、文の途中での割り込みから回復すること、会話の途中でツールを呼び出しても音声を止めないこと、不安定なモバイルネットワークに耐えること。

3つのRESTコールをつなぎ合わせてはそこに到達できない。アーキテクチャはエンドツーエンドでパイプライン化されたストリーミングだ。構築すれば失敗モードが見えてくる：電話音声用に調整されたVADがバックグラウンドテレビで発火する、決して来ない句読点を待つターン検出器、音声を出力する前に400msバッファリングするTTS。キャップストーンは、これらを負荷下で1つずつ修正してレイテンシーと品質のレポートを公開することだ。

## コンセプト

パイプラインには5つのストリーミングステージがある：**音声入力**（ブラウザまたはPSTNからのWebRTC）、**ASR**（Deepgram Nova-3またはfaster-whisperからのストリーミング部分文字起こし）、**ターン検出**（VADと完了の手がかりを探して部分文字起こしを読む小さなターン検出モデル）、**LLM**（ターンが完了したと判断されたらすぐにストリーミングトークン）、**TTS**（最初のLLMトークンから約200ms以内にストリーミング音声出力）。

3つの横断的な関心事。**割り込み（barge-in）**：エージェントが話している間にユーザーが話し始めると、TTSはキャンセルされ、ASRはすぐに再開する。**ツール使用**：会話の途中での関数呼び出し（天気、カレンダー）はサイドチャンネルで実行され、音声を止めてはならない；レイテンシーが300msを超えるとエージェントは確認トークン（「少々お待ちください...」）を事前出力する。**バックプレッシャー**：パケットロス下では部分文字起こしを保持し、VADはスピーチゲートのしきい値を上げ、エージェントは未確認メッセージの上から話すことを避ける。

測定基準は定量的だ。15 dB SNRのHamming VADベンチマークでWER 8%以下。100回測定した通話でのp50初回音声出力800ms以下。誤カット率3%以下。TTSのMOSが4.2以上。単一のg5.xlargeで50の同時通話。これらの数字が成果物だ。

## アーキテクチャ

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## スタック

- トランスポート: LiveKit Agents 1.0（WebRTC）とTwilio PSTNゲートウェイ；Pipecat 0.0.70を代替フレームワークとして
- ASR: Deepgram Nova-3（ストリーミング、300ms以下の初回パーシャル）またはfaster-whisper Whisper-v3-turboセルフホスト
- VAD: Silero VAD v5とLiveKitターン検出器（部分文字起こしを読む小さなトランスフォーマー）
- LLM: 緊密な統合のためのOpenAI GPT-4o-realtime、Gemini 2.5 Flash Live、またはカスケードClaude Haiku 4.5（ストリーミング補完、別音声パス）
- TTS: Cartesia Sonic-2（最低初回バイト）、ElevenLabs Flash v3、またはセルフホスト用オープンソースOrpheus
- ツール: 天気/カレンダー/予約用FastMCPサイドチャンネル；ツールが300msを超えると埋め草を事前出力
- 可観測性: OpenTelemetry音声スパン、音声再生付きLangfuse音声トレース
- デプロイ: セルフホストWhisper + Orpheus用単一g5.xlarge（24GB VRAM）；最低レイテンシー用ホスト型API

## 実装する

1. **WebRTCセッション。** LiveKitルームとマイク音声をストリーミングするWebクライアントを立ち上げる。サーバー側では、ルームに参加するエージェントワーカーをアタッチする。

2. **ASRストリーミング。** 20ms PCMフレームをDeepgram Nova-3（またはGPU上のfaster-whisper）に供給する。部分と最終の文字起こしをサブスクライブする。パーシャルごとのレイテンシーをログに記録する。

3. **VADとターン検出器。** フレームストリームでSilero VAD v5を実行する。スピーチ終了イベントで、最新の部分文字起こしに対してLiveKitターン検出器を発火する。VADが500msの無音を示し、ターン検出器が完了スコア > 0.6 のときのみ「ターン完了」をコミットする。

4. **LLMストリーム。** ターン完了時に、実行中の会話と最終文字起こしを使ってLLM呼び出しを開始する。トークンをストリーム出力する。最初のトークンでTTSに引き渡す。

5. **TTSストリーム。** Cartesia Sonic-2は音声チャンクをストリームバックする。最初のチャンクは最初のLLMトークンから200ms以内にサーバーを出なければならない。チャンクをLiveKitルームに出力；クライアントはWebRTCジッターバッファーで再生する。

6. **割り込み（barge-in）。** TTSの再生中にVADが新しいユーザーの音声を検出すると、TTSストリームをすぐにキャンセルし、残りのLLM出力を破棄し、ASRを再起動する。`tts_canceled`スパンを公開する。

7. **ツールサイドチャンネル。** 天気とカレンダーを関数呼び出しツールとして登録する。呼び出されたとき、呼び出しを並列に発火する；300ms以内に解決しない場合、LLMに「少々お待ちください、確認します」を埋め草として出力させる；ツールが返ったら再開する。

8. **評価ハーネス。** 100件の通話を録音する。WER（保留中の文字起こしと比較）、誤カット率（ユーザーが文の途中にいる間にTTSがキャンセルされた）、p50初回音声出力、TTSのMOS（人間またはNISQA）、ジッターロステスト（パケットの3%をドロップ）を計算する。

9. **負荷テスト。** 合成発信者で単一のg5.xlargeに50の同時通話を発生させる。持続的なp95初回音声出力を測定する。

## 使ってみる

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## 成果物を出す

`outputs/skill-voice-agent.md`が成果物。ドメイン（カスタマーサポート、スケジューリング、キオスク）を与えると、測定基準に合わせて調整されたASR/VAD/LLM/TTSパイプラインを持つLiveKitエージェントを立ち上げる。ルーブリック：

| 配点 | 基準 | 測定方法 |
|:-:|---|---|
| 25 | エンドツーエンドレイテンシー | 100件の録音通話でのp50初回音声出力800ms以下 |
| 20 | ターンテイク品質 | Hamming VADベンチマークでの誤カット率3%以下 |
| 20 | ツール使用の正確さ | 音声を止めることなく正しいデータを返す会話途中のツール呼び出し |
| 20 | パケットロス下での信頼性 | 3%パケットドロップが注入されたときのWERとターンテイクの安定性 |
| 15 | 評価ハーネスの完全性 | 公開設定を使った再現可能な測定 |
| **100** | | |

## 演習

1. Deepgram Nova-3をg5.xlarge上のfaster-whisper v3 turboに交換する。レイテンシーとWERのギャップを測定する。CPU対GPU の判断が重要な箇所を特定する。

2. 割り込み調停ポリシーを追加する：ツール呼び出し中にユーザーが割り込んだ場合、エージェントは何をするか？3つのポリシー（ハードキャンセル、ツール完了後に停止、次のターンをキュー）を比較する。

3. 敵対的なターン検出器テストを実行する：ユーザーに文の途中で長い一時停止を取らせる。誤カットを最小化しながら900msを超えないようVAD無音しきい値とターン検出スコアしきい値を調整する。

4. 同じエージェントをTwilio経由でPSTNに展開する。PSTNの初回音声出力をWebRTCと比較する。ジッターバッファーとコーデックの違いを説明する。

5. 非英語言語（日本語、スペイン語）の音声活動検出を追加する。Silero VAD v5の誤トリガー率を言語固有のファインチューンと比較して測定する。

## キーワード

| 用語 | 一般的な呼び方 | 実際の意味 |
|------|-----------------|------------------------|
| ターン検出 | 「発話の終わり」 | VADの無音と部分文字起こしを与えられ、ユーザーが話し終えたと判断する分類器 |
| 割り込み（barge-in） | 「割り込み処理」 | VADが新しいユーザーの音声を検出したときにTTSの再生中にキャンセルする |
| 初回音声出力 | 「レイテンシー」 | ユーザーが話し終えてから最初の音声パケットがサーバーを出るまでの時間 |
| VAD | 「スピーチゲート」 | 音声フレームを音声か無音かに分類するモデル；Silero VAD v5が2026年のデフォルト |
| ジッターバッファー | 「音声スムージング」 | ネットワークの変動を吸収するためにパケットを一時的に保持するクライアント側バッファー |
| 埋め草 | 「確認トークン」 | ツールが遅い場合に無音を避けるためにエージェントが出力する短いフレーズ |
| MOS | 「平均オピニオンスコア」 | 音声品質の知覚的評価；NISQAが自動プロキシ |

## 参考資料

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — リファレンスWebRTCエージェントフレームワーク
- [Pipecat](https://github.com/pipecat-ai/pipecat) — 代替Python優先ストリーミングエージェントフレームワーク
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 統合音声モデルのリファレンス
- [Deepgram Nova-3ドキュメント](https://developers.deepgram.com/docs) — ストリーミングASRリファレンス
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VADリファレンスモデル
- [Cartesia Sonic-2](https://docs.cartesia.ai) — 低レイテンシーTTSリファレンス
- [Retell AIアーキテクチャ](https://docs.retellai.com) — 本番音声エージェントアーキテクチャ
- [Vapi.ai本番スタック](https://docs.vapi.ai) — 代替本番リファレンス
