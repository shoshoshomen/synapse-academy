# スペクトログラム・Melスケール・音声特徴量

> ニューラルネットは生の波形をうまく扱えない。スペクトログラムを扱う。Melスペクトログラムはさらに相性がいい。2026年のすべてのASR・TTS・音声分類器は、この単一の前処理の選択にかかっている。


## 問題

10秒の16 kHzクリップを取り上げる。これは `[-1, 1]` の160,000個の浮動小数点数で、「犬の吠え声」や「catという単語」というラベルとほぼ無相関だ。生の波形には情報が含まれているが、モデルが簡単に取り出せる形ではない。100 ms離れて発音された2つの同じ音素は、生のサンプルではまったく異なる。

スペクトログラムはこれを解決する。人間の知覚が無視する時間的な細部（マイクロ秒のジッター）を折りたたみ、知覚が注目する構造（どの周波数が約10〜25 msの時間窓でエネルギーを持つか）を保持する。

Melスペクトログラムはさらに踏み込む。人間はピッチを対数的に知覚する: 100 Hzと200 Hzの「距離感」は1000 Hzと2000 Hzと「同じ」に聞こえる。Melスケールは周波数軸をこれに合わせて変形する。Mel正規化されたスペクトログラムは、2010年から2026年にかけて音声MLで最も重要な特徴量だ。

## 概念

![波形→STFT→Melスペクトログラム→MFCCの階段](../assets/mel-features.svg)

**STFT（短時間フーリエ変換）。** 波形を重なり合うフレームに分割する（典型的: 25 ms窓、10 msホップ = 16 kHzで400サンプル / 160サンプル）。各フレームに窓関数（デフォルトはHann；Hammingはトレードオフがわずかに異なる）を掛ける。各フレームにFFTをかける。大きさのスペクトルを `(n_frames, n_freq_bins)` の形状の行列に積み上げる。これがスペクトログラムだ。

**対数大きさ。** 生の大きさは5〜6桁にわたる。`log(|X| + 1e-6)` または `20 * log10(|X|)` でダイナミックレンジを圧縮する。すべての本番パイプラインは生の大きさではなく対数大きさを使用する。

**Melスケール。** Hz単位の周波数 `f` はMel `m` に次式でマッピングされる: `m = 2595 * log10(1 + f / 700)`。このマッピングは1 kHz以下でほぼ線形、以上でほぼ対数的だ。0〜8 kHzをカバーする80個のMelビンが標準的なASR入力だ。

**Melフィルタバンク。** Melスケールで等間隔に配置された三角フィルタのセット。各フィルタは隣接するFFTビンの加重和だ。STFTの大きさにフィルタバンク行列を掛けると、1回の行列積でMelスペクトログラムが得られる。

**対数Melスペクトログラム。** `log(mel_spec + 1e-10)`。WhisperへのInput。ParakeetへのInput。SeamlessM4TへのInput。2026年の汎用音声フロントエンド。

**MFCC。** 対数Melスペクトログラムに離散コサイン変換（DCT タイプII）を適用し、最初の13係数を保持する。特徴量を無相関化してさらに圧縮する。CNNやTransformerが生の対数Melで追いついた2015年頃まで支配的な特徴量だった。話者認識（x-vectors・ECAPA）では今も使用されている。

**解像度のトレードオフ。** FFTが大きいほど周波数分解能が上がるが時間分解能が下がる。25 ms / 10 msは音声ML標準；音楽は50 ms / 12.5 ms；過渡的な検出（ドラムヒット・閉鎖音）は5 ms / 2 ms。

## 実装する

### ステップ1: 波形をフレームに分割する

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

`frame_len=400, hop=160` で10秒の16 kHzクリップから998フレームが得られる。

### ステップ2: Hann窓

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

FFTの前に要素ごとに掛ける。ゼロでないエンドポイントでの切り捨てによるスペクトル漏れを除去する。

### ステップ3: STFTの大きさ

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

本番では `torch.stft` または `librosa.stft`（FFTベース、ベクトル化）を使用する。ここのループは教育目的で、`code/main.py` の短いクリップで動作する。

### ステップ4: Melフィルタバンク

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

`n_fft=400` で0〜8 kHzをカバーする80個のMelは `(80, 201)` 行列を生成する。`(n_frames, 201)` のSTFT大きさにその転置を掛けると `(n_frames, 80)` のMelスペクトログラムが得られる。

### ステップ5: 対数Mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

一般的な代替: `librosa.power_to_db`（参照正規化dB）、`10 * log10(power + eps)`。WhisperはdB-Melではなく対数Melを期待する（WhisperのREADMEに記載の、より複雑なクリップ＋正規化ルーティンも参照）。

### ステップ6: MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

各対数MelフレームにDCTを適用し、最初の13係数を保持する。これがMFCC行列だ。最初の係数は通常省く（全体エネルギーをエンコードするため）。

## 使ってみる

2026年のスタック:

| タスク | 特徴量 |
|--------|--------|
| ASR (Whisper・Parakeet・SeamlessM4T) | 80個の対数Mel、10 msホップ、25 ms窓 |
| TTS音響モデル (VITS・F5-TTS・Kokoro) | 80個のMel、5〜12 msホップで細かい時間制御 |
| 音声分類 (AST・PANNs・BEATs) | 128個の対数Mel、10 msホップ |
| 話者埋め込み (ECAPA-TDNN・WavLM) | 80個の対数MelまたはSSL生波形 |
| 音楽 (MusicGen・Stable Audio 2) | EnCodec離散トークン（Melではない） |
| キーワードスポッティング | 小型デバイス向け40個のMFCC |

目安: **音楽以外を扱うなら、まず80個の対数Melから始めよ。** 逸脱する場合は正当な理由が必要だ。

## 2026年でも起きる落とし穴

- **Mel数の不一致。** 80個のMelで学習し、128個のMelで推論する。サイレントな失敗。両端で特徴量の形状をログに記録すること。
- **上流でのサンプルレートの不一致。** 22.05 kHzで計算したMelは16 kHzとは異なる見た目になる。特徴量抽出の前にサンプルレートを修正すること。
- **dBと対数の混同。** WhisperはdB-Melではなく対数Melを期待する。HFのパイプラインは自動検出するが、自作コードはしない。
- **正規化のドリフト。** 学習時に発話ごとの正規化、推論時にグローバル正規化。WERを倍にする本番バグだ。
- **パディングからの漏れ。** クリップの末尾をゼロパディングすると末尾フレームで平坦なスペクトルが生じる。対称にパディングするか複製すること。

## 成果物を出す

`outputs/skill-feature-extractor.md` として保存する。このスキルはモデルのターゲットに合わせて特徴量の種類・Mel数・フレーム/ホップ・正規化を選択する。

## 演習

1. **簡単。** `code/main.py` を実行する。チャープ（200 → 4000 Hzにスイープする周波数）を合成し、フレームごとのargmax Melビンを出力する。（オプション）プロットしてスイープと一致することを確認する。
2. **中級。** `n_mels` を `{40, 80, 128}`、`frame_len` を `{200, 400, 800}` で再実行する。時間軸のシャープなピーク帯域幅を測定する。どの組み合わせがチャープを最もよく解像するか？
3. **難しい。** `power_to_db` を実装し、AudioMNISTの小型CNN分類器のASR精度を (a) 生の対数Mel、(b) `ref=max` のdB-Mel、(c) MFCC-13 + delta + delta-delta で比較する。Top-1精度を報告する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------|
| フレーム | スライス | 1つのFFTに渡される25 msの波形チャンク。 |
| ホップ | ストライド | 連続フレーム間のサンプル数；ASRデフォルトは10 ms。 |
| 窓 | Hann/Hammingのもの | フレームのエッジをゼロに近づけるポイントワイズ乗数。 |
| STFT | スペクトログラム生成器 | フレーミング＋窓関数をかけたFFT；時間×周波数の行列を生成。 |
| Mel | 変形された周波数 | 対数知覚スケール；`m = 2595·log10(1 + f/700)`。 |
| フィルタバンク | 行列 | STFTをMelビンに射影する三角フィルタ。 |
| 対数Mel | Whisperの入力 | `log(mel_spec + eps)`；2026年の標準。 |
| MFCC | 昔ながらの特徴量 | 対数MelのDCT；13係数、無相関化。 |

## 参考資料

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — MFCCの論文。
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 元のMelスケール。
- [OpenAI — Whisperのソース、log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — リファレンス実装を読むこと。
- [librosa特徴量抽出のドキュメント](https://librosa.org/doc/main/feature.html) — `mfcc`・`melspectrogram`・ホップ/窓のリファレンス。
- [NVIDIA NeMo — 音声前処理](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — Parakeet + Canaryモデルの本番規模パイプライン。
