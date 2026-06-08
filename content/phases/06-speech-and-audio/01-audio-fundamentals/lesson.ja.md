# 音声の基礎 — 波形・サンプリング・フーリエ変換

> 波形は生の信号だ。スペクトログラムは表現形式だ。Mel特徴量はML向けの形式だ。現代のASR・TTSパイプラインはこの階段を上り下りしており、最初の段はサンプリングとフーリエを理解することにある。


## 問題

マイクは圧力対時間の信号を生成する。ニューラルネットはテンソルを入力として受け取る。その間には慣習の積み重ねがあり、それを破ると無音のバグが生まれる——モデルの学習は問題なく進むのにWERが倍になったり、TTSがノイズを出力したり、音声クローンシステムが話者ではなくマイクを記憶したりする。

音声システムのバグはすべて、以下の3つの問いにさかのぼる:

1. データはどのサンプルレートで録音されており、モデルは何を期待しているか？
2. 信号にエイリアシングが生じているか？
3. 生のサンプルを扱っているのか、それとも周波数表現を扱っているのか？

これらを正しく理解すれば、フェーズ6の残りは対処可能だ。間違えると、Whisper-Large-v4でさえゴミを出力する。

## 概念

![波形・サンプリング・DFT・周波数ビンの可視化](../assets/audio-fundamentals.svg)

**波形。** `[-1.0, 1.0]` の範囲の浮動小数点数からなる1次元配列。サンプル番号でインデックス付けされる。秒に変換するにはサンプルレートで割る: `t = n / sr`。16 kHzの10秒クリップは160,000個の浮動小数点数の配列だ。

**サンプルレート (sr)。** 1秒あたりのサンプル数。2026年における代表的なレート:

| レート | 用途 |
|--------|------|
| 8 kHz | 電話・レガシーVOIP。ナイキスト周波数4 kHzで子音が失われる。ASRには不向き。 |
| 16 kHz | ASR標準。Whisper・Parakeet・SeamlessM4T v2はいずれも16 kHzを入力とする。 |
| 22.05 kHz | 旧世代モデルのTTSボコーダー学習用。 |
| 24 kHz | 現代のTTS (Kokoro・F5-TTS・xTTS v2)。 |
| 44.1 kHz | CDオーディオ・音楽。 |
| 48 kHz | 映画・プロオーディオ・高品質TTS (VALL-E 2・NaturalSpeech 3)。 |

**ナイキスト-シャノン定理。** サンプルレート `sr` は `sr/2` までの周波数を曖昧なく表現できる。この `sr/2` の境界を*ナイキスト周波数*と呼ぶ。ナイキスト周波数を超えるエネルギーは*エイリアシング*を引き起こし——低い周波数に折り畳まれ——信号を汚染する。ダウンサンプリングの前には必ずローパスフィルタをかけること。

**ビット深度。** 16ビットPCM（符号付きint16、範囲±32,767）は汎用的な交換フォーマットだ。音楽には24ビット、内部DSP処理には32ビット浮動小数点。`soundfile` などのライブラリはint16を読み込むが、`[-1, 1]` のfloat32配列として公開する。

**フーリエ変換。** 有限信号はすべて、異なる周波数の正弦波の和で表せる。離散フーリエ変換 (DFT) は `N` 個のサンプルに対し `N` 個の複素係数を計算する——周波数ビン1つあたり1係数だ。`ビン k` は周波数 `k · sr / N` Hz に対応する。大きさはその周波数の振幅、角度は位相を表す。

**FFT。** 高速フーリエ変換: `N` が2のべき乗のときにDFTを `O(N log N)` で計算するアルゴリズム。すべてのオーディオライブラリが内部でFFTを使用している。16 kHzにおける1024サンプルのFFTは、0〜8 kHzを15.6 Hzの分解能でカバーする512個の有効周波数ビンを生成する。

**フレーミング＋窓関数。** クリップ全体にFFTをかけるのではない。重なり合う*フレーム*（典型的には25 msの窓、10 msのホップ）に切り刻み、各フレームに窓関数（Hann、Hamming）を掛けてエッジの不連続性を除去し、各フレームにFFTをかける。これが短時間フーリエ変換 (STFT) だ。レッスン02はここから続く。

## 実装する

### ステップ1: クリップの読み込みと波形のプロット

`code/main.py` は依存関係なしのデモとして標準ライブラリの `wave` モジュールのみを使用する。本番環境では `soundfile` または `torchaudio.load`（いずれも `(waveform, sr)` タプルを返す）を使用する:

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### ステップ2: 基本原理から正弦波を合成する

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

16 kHzで1秒間の440 Hz正弦波（コンサートA）は16,000個の浮動小数点数だ。16ビットPCMエンコードで `wave.open(..., "wb")` を使って書き込む。

### ステップ3: DFTを手計算する

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)` — `N=256` での正しさの確認には使えるが、実際の音声には無意味だ。実際のコードは `numpy.fft.rfft` または `torch.fft.rfft` を呼び出す。

### ステップ4: 主要周波数を求める

大きさのピークのインデックス `k_star` は周波数 `k_star * sr / N` に対応する。440 Hz正弦波でこれを実行すると、ビン `440 * N / sr` にピークが現れるはずだ。

### ステップ5: エイリアシングを実証する

10 kHzで7 kHzの正弦波をサンプリングする（ナイキスト = 5 kHz）。7 kHzのトーンはナイキスト周波数を超えており、`10 - 7 = 3 kHz` に折り返される。FFTのピークは3 kHzに現れる。これがエイリアシングの古典的なデモであり、すべてのDAC/ADCにブリックウォールローパスフィルタが搭載されている理由だ。

## 使ってみる

2026年に実際に使用するスタック:

| タスク | ライブラリ | 理由 |
|--------|-----------|------|
| WAV/FLAC/OGGの読み書き | `soundfile` (libsndfileラッパー) | 最速・安定・float32を返す。 |
| リサンプリング | `torchaudio.transforms.Resample` または `librosa.resample` | アンチエイリアシングが内蔵されている。 |
| STFT / Mel | `torchaudio` または `librosa` | GPUフレンドリー・PyTorchエコシステム。 |
| リアルタイムストリーミング | `sounddevice` または `pyaudio` | クロスプラットフォームのPortAudioバインディング。 |
| ファイルの検査 | `ffprobe` または `soxi` | CLI・高速・sr/チャンネル/コーデックを報告。 |

判断基準: **他の何より先にサンプルレートを合わせること**。WhisperはFloat32の16 kHzモノラルを期待する。44.1 kHzのステレオを渡すと、モデルのバグのように見えるゴミが出力される。

## 成果物を出す

`outputs/skill-audio-loader.md` として保存する。このスキルは音声入力が下流モデルの期待に合致しているかを確認し、合致しない場合に正しくリサンプリングするのに役立つ。

## 演習

1. **簡単。** 16 kHzで220 Hz + 440 Hz + 880 Hzを1秒混合して合成する。DFTを実行する。期待されるビンに3つのピークが現れることを確認する。
2. **中級。** 48 kHzで3秒の声を録音する。`torchaudio.transforms.Resample`（アンチエイリアシングあり）を使って16 kHzにダウンサンプリングし、次に素朴な間引き（3サンプルごと）を使って16 kHzにダウンサンプリングする。両方のFFTを比較する。エイリアシングはどこに現れるか？
3. **難しい。** `math` とステップ3のDFTのみを使ってSTFTをゼロから構築する。フレームサイズ400、ホップ160、Hann窓。`matplotlib.pyplot.imshow` で大きさをプロットする。これがレッスン02のスペクトログラムだ。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------|
| サンプルレート | 1秒あたりのサンプル数 | ADCが信号を測定する周波数（Hz）。 |
| ナイキスト | 表現できる最大周波数 | `sr/2`；これを超えるエネルギーはエイリアシングする。 |
| ビット深度 | 各サンプルの解像度 | `int16` = 65,536段階；`float32` = `[-1, 1]` での24ビット精度。 |
| DFT | 数列のフーリエ変換 | `N` サンプル → `N` 個の複素周波数係数。 |
| FFT | 高速DFT | `O(N log N)` アルゴリズム；`N` が2のべき乗を要求する。 |
| ビン | 周波数の列 | `k · sr / N` Hz；分解能 = `sr / N`。 |
| STFT | スペクトログラムの基礎 | フレーミング＋窓関数をかけたFFTの時系列。 |
| エイリアシング | 奇妙な周波数の幽霊 | ナイキスト周波数を超えるエネルギーが低いビンに折り返される。 |

## 参考資料

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — サンプリング定理の基礎となる論文。
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — 無料のDSP教科書の定番。
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) — コード付きの実践的な解説。
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — 現実の音声がなぜきれいな正弦波でないかの参考書。
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 10分で周波数ビンの直感を身につける。
