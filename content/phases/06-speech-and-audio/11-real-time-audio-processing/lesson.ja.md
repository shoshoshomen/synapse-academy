# リアルタイム音声処理

> バッチパイプラインはファイルを処理する。リアルタイムパイプラインは次の20ミリ秒が到着する前に現在の20ミリ秒を処理する。すべての会話型AI・放送スタジオ・電話ボットはこの遅延バジェットで生死が決まる。


## 問題

生き生きとした音声アシスタントが欲しい。人間の会話のターンテイク遅延は約230 msだ。500 msを超えるとロボット的に感じる；1500 msを超えると壊れているように感じる。2026年における完全な**聞く→理解する→応答する→話す**ループのバジェット:

| ステージ | バジェット |
|---------|---------|
| マイク→バッファ | 20 ms |
| VAD | 10 ms |
| ASR（ストリーミング） | 150 ms |
| LLM（最初のトークン） | 100 ms |
| TTS（最初のチャンク） | 100 ms |
| レンダリング→スピーカー | 20 ms |
| **合計** | **約400 ms** |

Moshi（Kyutai、2024年）は200 msのフルデュプレックスを達成した。GPT-4o-realtime（2024年）は約320 msだ。2022年のカスケードパイプラインは2500 msで出荷されていた。10倍の改善は3つのテクニックから来た: (1)あらゆる場所でのストリーミング、(2)部分的な結果を使った非同期パイプライン化、(3)中断可能な生成。

## 概念

![リングバッファ・VADゲート・中断を伴うストリーミング音声パイプライン](../assets/real-time.svg)

**フレーム / チャンク / ウィンドウ。** リアルタイム音声は固定サイズのブロックとして流れる。一般的な選択: 20 ms（16 kHzで320サンプル）。下流のすべてがこのケイデンスに追いつかなければならない。

**リングバッファ。** 固定サイズの循環バッファ。プロデューサースレッドが新しいフレームを書き込み、コンシューマースレッドが読み取る。ホットパスでのアロケーションを防ぐ。サイズ≈最大遅延×サンプルレート；2秒の16 kHzリングは32,000サンプルだ。

**VAD（音声区間検出）。** 誰も話していないときに下流の処理をゲートする。Silero VAD 4.0（2024年）はCPUの単一スレッドで30 msフレームあたり1 ms未満で動作する。`webrtcvad` は古い代替だ。

**ストリーミングASR。** 音声が届くにつれて部分的なトランスクリプトを出力するモデル。ストリーミングモードのParakeet-CTC-0.6B（NeMo、2024年）は320 msの遅延で2〜5% WERを達成する。Whisper-Streaming（Macháček et al.、2023年）はほぼストリーミングに近い約2秒の遅延でWhisperをチャンク化する。

**中断。** アシスタントが話している最中にユーザーが話すと、(a)割り込みを検出し、(b)TTSを止め、(c)残りのLLM出力を破棄しなければならない。すべて100 ms以内に行わないと、ユーザーは聴覚障害のあるアシスタントと感じる。

**WebRTC Opusトランスポート。** 20 msフレーム、48 kHz、アダプティブビットレート8〜128 kbps。ブラウザとモバイルの標準。LiveKit・Daily.co・Pionが2026年の音声アプリ構築スタックだ。

**ジッターバッファ。** ネットワークパケットは順序が狂うか遅延する。ジッターバッファは並べ替えて平滑化する；小さすぎると可聴ギャップ、大きすぎると遅延。典型的に60〜80 ms。

### よくある落とし穴

- **スレッドの競合。** PythonのGIL＋重いモデルが音声スレッドを飢餓状態にする。Cコールバック音声ライブラリ（sounddevice、PortAudio）を使用し、Pythonをホットパスから外すこと。
- **サンプルレート変換の遅延。** パイプライン内のリサンプリングに5〜20 msかかる。あらかじめリサンプルするかゼロ遅延リサンプラーを使用すること（PolyPhase、`soxr_hq`）。
- **TTSのプライミング。** Kokoroのような高速TTSでも最初のリクエストで100〜200 msのウォームアップがある。最初の実際のターンの前にモデルをキャッシュしてダミー実行でウォームアップすること。
- **エコーキャンセリング。** AECなしでは、TTS出力がマイクに入力されてASRがボット自身の声を転写する。WebRTC AEC3がオープンソースのデフォルトだ。

## 実装する

### ステップ1: リングバッファ

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

容量が最大バッファリング遅延を決める。16 kHzで32,000サンプル = 2秒。

### ステップ2: VADゲート

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

本番ではSilero VADに置き換える:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### ステップ3: ストリーミングASR

```python
# NeMo経由のParakeet-CTC-0.6Bストリーミング
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### ステップ4: 中断ハンドラ

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # 割り込み
        # その後ストリーミングASRに渡す

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

非同期I/OとキャンセラブルなTTSストリーミングに依存する。WebRTCのpeerconnection.stop()がオーディオトラックの標準的な方法だ。

## 使ってみる

2026年のスタック:

| レイヤー | 選択 |
|---------|------|
| トランスポート | LiveKit（WebRTC）またはPion（Go） |
| VAD | Silero VAD 4.0 |
| ストリーミングASR | Parakeet-CTC-0.6BまたはWhisper-Streaming |
| LLMの最初のトークン | Groq・Cerebras・vLLM-streaming |
| ストリーミングTTS | KokoroまたはElevenLabs Turbo v2.5 |
| エコーキャンセリング | WebRTC AEC3 |
| エンドツーエンドネイティブ | OpenAI Realtime APIまたはMoshi |

## 落とし穴

- **安全のために500 msバッファリング。** バッファは遅延の下限だ。小さくすること。
- **スレッドを固定しない。** UIより優先度の低いスレッドで音声コールバックを実行 = 負荷下でグリッチ。
- **TTSチャンクが小さすぎる。** 200 ms未満のチャンクはボコーダのアーティファクトが可聴になる。320 msチャンクがスイートスポットだ。
- **ジッターバッファなし。** 実際のネットワークはジッタがある；平滑化なしではポップが生じる。
- **単一ショットのエラー処理。** 音声パイプラインはクラッシュプルーフでなければならない。1つの例外がセッションを終了させる。

## 成果物を出す

`outputs/skill-realtime-designer.md` として保存する。ステージごとの具体的な遅延バジェットを持つリアルタイム音声パイプラインを設計する。

## 演習

1. **簡単。** `code/main.py` を実行する。リングバッファ＋エネルギーVADをシミュレートし、偽の10秒ストリームのステージ遅延を出力する。
2. **中級。** `sounddevice` を使ってマイクを20 msフレームで処理し、各フレームでVAD状態を出力するパススルーループを構築する。
3. **難しい。** `aiortc` を使ってフルデュプレックスのエコーテストを構築する: ブラウザ→WebRTC→Python→WebRTC→ブラウザ。1 kHzパルスでガラスツーガラス遅延を測定する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------|
| リングバッファ | 循環キュー | 音声フレーム用の固定サイズ・ロックフリー（またはSPSCロック）FIFO。 |
| VAD | 無音ゲート | 音声対非音声をマークするモデルまたはヒューリスティック。 |
| ストリーミングASR | リアルタイムSTT | 音声が届くにつれて部分的なテキストを出力；有界なルックアヘッド。 |
| ジッターバッファ | ネットワークスムーザー | 順序が狂ったパケットを並べ替えるキュー；典型的に60〜80 ms。 |
| AEC | エコーキャンセリング | スピーカーからマイクへのフィードバックパスを減算する。 |
| 割り込み | ユーザーの中断 | TTS中にユーザーの音声を検出；再生をキャンセルしなければならない。 |
| フルデュプレックス | 双方向同時 | ユーザーとボットが同時に話せる；MoshiはフルデュプレックスだA。 |

## 参考資料

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — チャンク化されたほぼストリーミングのWhisper。
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — フルデュプレックス200 ms遅延。
- [LiveKit Agentsフレームワーク（2024年）](https://docs.livekit.io/agents/) — 本番音声エージェントのオーケストレーション。
- [Silero VADリポジトリ](https://github.com/snakers4/silero-vad) — 1 ms未満のVAD、Apache 2.0。
- [WebRTC AEC3の論文](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — オープンソースのエコーキャンセリング。
