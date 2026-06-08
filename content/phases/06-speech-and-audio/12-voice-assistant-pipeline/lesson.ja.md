# 音声アシスタントパイプラインの構築 — フェーズ6の集大成

> レッスン01〜11のすべてを組み合わせる。聞いて、推論して、話し返す音声アシスタントを構築する。2026年においてこれは解決済みのエンジニアリング問題であり、研究上の問題ではない——しかし統合の詳細が出荷できるかどうかを決定する。


## 問題

エンドツーエンドのアシスタントを構築する:

1. マイク入力をキャプチャする（16 kHzモノラル）。
2. ユーザーの発話の開始/終了を検出する。
3. ストリーミングで書き起こす。
4. ツール（タイマー・天気・カレンダー）を呼び出せるLLMにトランスクリプトを渡す。
5. LLMのテキストをTTSにストリーミングする。
6. ユーザーに音声で返答する。
7. ユーザーが応答の途中で割り込んだら止まる。

遅延目標: ユーザーが発話を終えてからラップトップCPUで最初のTTS音声バイトを800 ms以内に。品質目標: 言葉の聞き逃しなし、無音でのトランスクリプト幻覚なし、音声クローンの漏れなし、プロンプトインジェクション成功なし。

## 概念

![音声アシスタントパイプライン: マイク→VAD→STT→LLM+ツール→TTS→スピーカー](../assets/voice-assistant.svg)

### 7つのコンポーネント

1. **音声キャプチャ。** マイク→16 kHzモノラル→20 msチャンク。Pythonでは通常 `sounddevice`、本番ではネイティブのAudioUnit/ALSA/WASAPI。
2. **VAD（レッスン11）。** Silero VAD @ 閾値0.5、最小音声250 ms、無音ハングオーバー500 ms。「開始」と「終了」を通知する。
3. **ストリーミングSTT（レッスン4〜5）。** Whisper-streaming・Parakeet-TDT・またはDeepgram Nova-3（API）。部分的＋最終的なトランスクリプト。
4. **ツール呼び出し付きLLM。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。ツール用JSONスキーマ。トークンをストリーミング。
5. **ストリーミングTTS（レッスン7）。** Kokoro-82M（最速オープン）またはCartesia Sonic（商用）。20 LLMトークン後にTTSを開始する。
6. **再生。** スピーカー出力；低帯域ネットワーク向けにOpusエンコード。
7. **中断ハンドラ。** TTS再生中にVADが発火したら、再生を止め、LLMをキャンセルし、STTを再起動する。

### 遭遇する3つの失敗モード

1. **最初の単語のクリッピング。** VADが少し遅れて起動する。ユーザーの「hey」が失われる。開始閾値は0.5ではなく0.3に設定すること。
2. **応答途中での割り込み混乱。** ユーザーが割り込んだ後もLLMが生成し続け、アシスタントがユーザーの上で話す。VAD→LLMキャンセルを接続すること。
3. **無音での幻覚。** Whisperが無音のウォームアップフレームで「ご視聴ありがとうございました」を出力する。常にVADゲーティングを行うこと。

### 2026年の本番リファレンススタック

| スタック | 遅延 | ライセンス | 備考 |
|---------|------|----------|------|
| LiveKit＋Deepgram＋GPT-4o＋Cartesia | 350〜500 ms | 商用API | 2026年の業界標準 |
| Pipecat＋Whisper-streaming＋GPT-4o＋Kokoro | 500〜800 ms | ほぼオープン | DIY向け |
| Moshi（フルデュプレックス） | 200〜300 ms | CC-BY 4.0 | 単一モデル；異なるアーキテクチャ、レッスン15 |
| Vapi / Retell（マネージド） | 300〜500 ms | 商用 | 最速の立ち上げ；カスタマイズ制限あり |
| Whisper.cpp＋llama.cpp＋Kokoro-ONNX | オフライン | オープン | プライバシー / エッジ |

## 実装する

### ステップ1: チャンク化したマイクキャプチャ（疑似コード）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### ステップ2: VADゲート付きターンキャプチャ

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### ステップ3: ストリーミングSTT→LLM→TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### ステップ4: LLMループ内のツール呼び出し

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### ステップ5: 中断処理

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 使ってみる

ハードウェアなしでもパイプラインの形が見られるように、スタブモデルで7つのコンポーネントをすべて接続した実行可能なシミュレーションは `code/main.py` を参照すること。実際の実装にはスタブを以下に置き換えること:

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` または `openai-whisper`
- `openai` (`gpt-4o`) または `anthropic`
- `kokoro` または `cartesia`
- I/O用の `sounddevice`

## 落とし穴

- **PIIを永続的にログに記録する。** フルターンの音声はほとんどの法域でPIIだ。30日間の保持、保存時の暗号化。
- **割り込みなし。** ユーザーは必ず割り込む。アシスタントは話し続けてはならない。
- **ブロッキングTTS。** 同期TTSはイベントループをブロックする。非同期または別スレッドを使用すること。
- **ツール呼び出しのエラーハンドリングなし。** ツールは失敗する。LLMはエラーを受け取って1回リトライし、その後潔く劣化すること。
- **過剰な幻覚フィルタ。** フィルタしすぎるとアシスタントが「それはお手伝いできません」を繰り返す。フィルタ不足だと何でも言う。ホールドアウトセットでキャリブレーションすること。
- **ウェイクワードオプションなし。** 常時聴取はプライバシーリスクだ。ウェイクワードゲートを追加すること（PorcupineまたはopenWakeWord）。

## 成果物を出す

`outputs/skill-voice-assistant-architect.md` として保存する。予算＋スケール＋言語＋コンプライアンス制約に基づいて完全なスタック仕様を作成する。

## 演習

1. **簡単。** `code/main.py` を実行する。スタブモジュールでエンドツーエンドの1ターンをシミュレートし、ステージごとの遅延を出力する。
2. **中級。** STTスタブを事前録音済みの `.wav` の実際のWhisperモデルに置き換える。WERとエンドツーエンドの遅延を測定する。
3. **難しい。** ツール呼び出しを追加する: `get_weather`（任意のAPI）と `set_timer` を実装する。LLMをツールを通じてルーティングし、ユーザーが「5分のタイマーをセットして」と言うと正しい関数が発火して確認の音声応答が返ることを確認する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------|
| ターン | ユーザー＋アシスタントの往復 | VADで区切られた1ユーザーの発話＋1 LLM-TTS応答。 |
| 割り込み | ユーザーの中断 | アシスタントが話している間ユーザーが話す；アシスタントは止まる。 |
| ウェイクワード | 「ねえアシスタント」 | 短いキーワード検出器；Porcupine・Snowboy・openWakeWord。 |
| エンドポインティング | ターンの終了 | ユーザーが話し終えたというVAD＋最小無音の判断。 |
| プレロール | 発話前のバッファ | VADが発火する前の200〜400 msの音声を保持して最初の単語のクリッピングを防ぐ。 |
| ツール呼び出し | 関数呼び出し | LLMがJSONを出力；ランタイムがディスパッチ；結果をループ内に返す。 |

## 参考資料

- [LiveKit — 音声エージェントクイックスタート](https://docs.livekit.io/agents/) — 本番品質のリファレンス。
- [Pipecat — 音声エージェントの例](https://github.com/pipecat-ai/pipecat) — DIY向けフレームワーク。
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — マネージドな音声ネイティブパス。
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — フルデュプレックスリファレンス（レッスン15）。
- [Porcupineウェイクワード](https://picovoice.ai/products/porcupine/) — ウェイクワードゲーティング。
- [Anthropic — ツール使用ガイド](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM関数呼び出し。
