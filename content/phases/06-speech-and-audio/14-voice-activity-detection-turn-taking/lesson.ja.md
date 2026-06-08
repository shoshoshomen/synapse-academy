# 音声区間検出とターンテイク — Silero・Cobra、そしてフラッシュトリック

> すべての音声エージェントは2つの判断で生死が決まる: ユーザーは今話しているか、そして話し終えたか？VADが最初に答える。ターン検出（VAD＋無音ハングオーバー＋意味的エンドポイントモデル）が2番目に答える。どちらかを間違えると、アシスタントがユーザーを遮断するか、永遠に黙らなくなる。


## 問題

音声エージェントが20 msのチャンクごとに行う3つの異なる判断:

1. **このフレームは音声か？** — VAD。バイナリ、フレームごと。
2. **ユーザーは新しい発話を始めたか？** — オンセット検出。
3. **ユーザーは話し終えたか？** — エンドポインティング（ターン終了）。

単純な答え（エネルギー閾値）はあらゆるノイズで失敗する——交通・キーボード・群衆の騒音。2026年の答え: Silero VAD（オープン、深層学習）＋ターン検出モデル（意味的エンドポインティング）＋VADキャリブレーションされた無音ハングオーバー。

## 概念

![VADカスケード: エネルギー→Silero→ターン検出器→フラッシュトリック](../assets/vad-turn-taking.svg)

### 3ティアVADカスケード

**ティア1: エネルギーゲート。** 最も安価。RMSを-40 dBFSで閾値処理する。明らかな無音はフィルタリングするが、閾値を超えるすべてのノイズで発火する。

**ティア2: Silero VAD**（2020〜2026年、MIT）。100万パラメータ。6000以上の言語で学習。単一CPUスレッドで30 msチャンクあたり約1 msで動作する。5% FPRでTPR 87.7%。オープンソースのデフォルト。

**ティア3: 意味的ターン検出器。** LiveKitのターン検出モデル（2024〜2026年）または独自の小型分類器。「文中の間」と「話し終わった」を区別する。イントネーション＋最近の単語という言語的コンテキストを使用し、無音だけではない。

### 主要なパラメータとそのデフォルト

- **閾値。** Sileroは確率を出力する；音声を > 0.5（デフォルト）または > 0.3（敏感）で分類する。低い閾値 = 最初の単語のクリッピングが少ない、偽陽性が多い。
- **最小音声持続時間。** 250 ms未満の音声を拒否する——通常は咳や椅子の音だ。
- **無音ハングオーバー（エンドポインティング）。** VADが0に戻った後、ターン終了を宣言する前に500〜800 ms待つ。短すぎるとユーザーを中断する。長すぎると遅く感じる。
- **プレロールバッファ。** VADが発火する前の300〜500 msの音声を保持する。「ねえ」がクリッピングされるのを防ぐ。

### フラッシュトリック（Kyutai 2025年）

ストリーミングSTTモデルにはルックアヘッド遅延がある（Kyutai STT-1Bで500 ms、STT-2.6Bで2.5秒）。通常、トランスクリプトのために音声終了後その時間待つ必要がある。フラッシュトリック: VADが音声終了を検出したら、**STTにフラッシュシグナルを送ってすぐに出力を強制する**。STTは約4倍のリアルタイムで処理するので、500 msのバッファが約125 msで終了する。

エンドツーエンド: 125 ms VAD＋フラッシュSTT = 会話的な遅延。

### 2026年のVAD比較

| VAD | TPR @ 5% FPR | 遅延 | ライセンス |
|-----|--------------|------|----------|
| WebRTC VAD（Google、2013年） | 50.0% | 30 ms | BSD |
| Silero VAD（2020〜2026年） | 87.7% | 約1 ms | MIT |
| Cobra VAD（Picovoice） | 98.9% | 約1 ms | 商用 |
| pyannoteセグメンテーション | 95% | 約10 ms | MITに近い |

Sileroが適切なデフォルトだ。Cobraはコンプライアンス/精度のアップグレードだ。エネルギーのみのVADは2026年の本番に居場所がない。

## 実装する

### ステップ1: エネルギーゲート

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### ステップ2: PythonでのSilero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### ステップ3: ターン終了のステートマシン

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### ステップ4: フラッシュトリックのスケルトン

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

これが機能するにはSTT（Kyutai・Deepgram・AssemblyAI）がフラッシュをサポートしている必要がある。Whisperストリーミングはブロックベースで常にチャンクを待つため、これには対応していない。

## 使ってみる

| 状況 | VADの選択 |
|------|---------|
| オープン・高速・汎用 | Silero VAD |
| 商業コールセンター | Cobra VAD |
| オンデバイス（スマートフォン） | Silero VAD ONNX |
| 研究/話者分離 | pyannoteセグメンテーション |
| ゼロ依存のフォールバック | WebRTC VAD（レガシー） |
| ターン終了品質が必要 | Silero＋LiveKitターン検出器を重ねる |

目安: 他に選択肢がない場合を除いて、エネルギーのみのVADは絶対に出荷しないこと。

## 落とし穴

- **固定閾値。** 静かな場所では機能するが、ノイジーな場所では失敗する。オンデバイスでキャリブレーションするかSileroに切り替えること。
- **短すぎる無音ハングオーバー。** エージェントが文の途中でユーザーを遮断する。500〜800 msが会話音声のスイートスポットだ。
- **長すぎるハングオーバー。** 遅く感じる。対象ユーザーでA/Bテストすること。
- **プレロールバッファなし。** ユーザー音声の最初の200〜300 msが失われる。常に転がるプレロールを保持すること。
- **意味的エンドポインティングの無視。** 「えーと、考えます...」は長い間を含む。ユーザーは考えの途中で遮断されることを嫌う。LiveKitのターン検出器または同様のものを使用すること。

## 成果物を出す

`outputs/skill-vad-tuner.md` として保存する。ワークロードに対してVADモデル・閾値・ハングオーバー・プレロール・ターン検出戦略を選択する。

## 演習

1. **簡単。** `code/main.py` を実行する。音声＋無音＋音声＋咳のシーケンスをシミュレートし、3つのVADティアをテストする。
2. **中級。** `silero-vad` をインストールし、5分の録音を処理し、閾値をチューニングして最初の単語のクリッピングと偽トリガーの両方を最小化する。精度/再現率を報告する。
3. **難しい。** 小型ターン検出器を構築する: Silero VAD＋最後の10単語の埋め込みに対する3層MLP（sentence-transformersを使用）。手作業でラベル付けしたターン終了データセットで学習する。Silero単独より10% F1を上回ること。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------|
| VAD | 音声検出器 | フレームごとのバイナリ: これは音声か？ |
| ターン検出 | エンドポインティング | VAD＋無音ハングオーバー＋意味的エンドポイント。 |
| 無音ハングオーバー | 音声後の待機 | ターン終了を宣言するまでの待機時間；500〜800 ms。 |
| プレロール | 発話前バッファ | VADが発火する前の300〜500 msの音声を保持。 |
| フラッシュトリック | Kyutaiのハック | VAD→STTフラッシュ→500 msの代わりに125 msの遅延。 |
| 意味的エンドポイント | 「止めるつもりだったか？」 | 無音だけでなく単語を見るMLの分類器。 |
| TPR @ FPR 5% | ROCのポイント | 標準的なVADベンチマーク；Sileroで87.7%、WebRTCで50%。 |

## 参考資料

- [Silero VAD](https://github.com/snakers4/silero-vad) — リファレンスのオープンVAD。
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — 商用精度リーダー。
- [Kyutai — Unmute＋フラッシュトリック](https://kyutai.org/stt) — 200 ms未満のエンジニアリングトリック。
- [LiveKit — ターン検出](https://docs.livekit.io/agents/logic/turns/) — 本番の意味的エンドポインティング。
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — レガシーベースライン。
- [pyannoteセグメンテーション](https://github.com/pyannote/pyannote-audio) — 話者分離グレードのセグメンテーション。
