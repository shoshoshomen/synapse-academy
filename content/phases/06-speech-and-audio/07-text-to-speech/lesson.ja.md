# テキスト音声合成（TTS）— TacotronからF5とKokoroまで

> ASRは音声をテキストに変換し、TTSはテキストを音声に変換する。2026年のスタックは3段階だ: テキスト→トークン、トークン→Mel、Mel→波形。各段階にはラップトップに収まるデフォルトモデルがある。


## 問題

文字列がある: 「Please remind me to water the plants at 6 pm.（植物に水やりするよう6 pmに教えて）」。3秒の音声クリップが必要だ——自然に聞こえ、正しいプロソディ（ポーズ・強調）を持ち、「plants」を正しい母音で発音し、ライブ音声アシスタントとしてCPUで300 ms未満で動作する。さらに声を切り替え、コードスイッチングされた入力（「6 pmに教えて、大丈夫？」）を処理し、名前を誤って発音しない必要がある。

現代のTTSパイプラインはこのようになっている:

1. **テキストフロントエンド。** テキストを正規化し（日付・数字・メール）、音素またはサブワードトークンに変換し、プロソディ特徴量を予測する。
2. **音響モデル。** テキスト→Melスペクトログラム。Tacotron 2（2017年）、FastSpeech 2（2020年）、VITS（2021年）、F5-TTS（2024年）、Kokoro（2024年）。
3. **ボコーダ。** Mel→波形。WaveNet（2016年）、WaveRNN、HiFi-GAN（2020年）、BigVGAN（2022年）、2024年以降の神経コーデックボコーダ。

2026年には音響＋ボコーダの分割がエンドツーエンドの拡散とフローマッチングモデルで曖昧になってきている。しかし3段階のメンタルモデルはデバッグに依然有効だ。

## 概念

![Tacotron・FastSpeech・VITS・F5/Kokoroの比較](../assets/tts.svg)

**Tacotron 2（2017年）。** seq2seq: 文字埋め込み→BiLSTMエンコーダ→位置感応アテンション→自己回帰LSTMデコーダがMelフレームを出力する。遅く（AR）、長いテキストでは不安定。依然ベースラインとして引用される。

**FastSpeech 2（2020年）。** 非自己回帰的。デュレーション予測器が各音素が何フレームのMelを持つかを出力する。1パス、Tacotronより10倍高速。自然さはやや失われる（単調なアライメント）が至る所で使われている。

**VITS（2021年）。** エンコーダ＋フローベースデュレーション＋HiFi-GANボコーダを変分推論でエンドツーエンドで結合学習する。高品質で単一モデル。2022〜2024年の主要オープンソースTTS。バリアント: YourTTS（多話者ゼロショット）、XTTS v2（2024年、Coqui）。

**F5-TTS（2024年）。** フローマッチング上の拡散Transformer。自然なプロソディ、5秒のリファレンス音声でゼロショット音声クローン。2026年のオープンソースTTSリーダーボードのトップ。3.35億パラメータ。

**Kokoro（2024年）。** 小型（8200万パラメータ）、CPU実行可能、リアルタイム使用で最高クラスの英語TTS。クローズドな英語専用語彙、Apache-2.0。

**OpenAI TTS-1-HD・ElevenLabs v2.5・Google Chirp-3。** 商業的な最高水準。ElevenLabs v2.5の感情タグ（「[ひそひそ声]」「[笑い声]」）とキャラクターボイスが2026年の音声書籍制作を席巻している。

### ボコーダの進化

| 時代 | ボコーダ | 遅延 | 品質 |
|------|---------|------|------|
| 2016年 | WaveNet | オフラインのみ | リリース時SOTA |
| 2018年 | WaveRNN | ほぼリアルタイム | 良好 |
| 2020年 | HiFi-GAN | 100×リアルタイム | ほぼ人間レベル |
| 2022年 | BigVGAN | 50×リアルタイム | 話者/言語をまたいで汎化 |
| 2024年 | SNAC・DAC（神経コーデック） | ARモデルに統合 | 離散トークン、ビット効率的 |

2026年には多くの「TTS」モデルがテキストから波形までエンドツーエンドだ；Melスペクトログラムは内部表現だ。

### 評価

- **MOS（平均意見スコア）。** 1〜5スケール、クラウドソーシング。依然ゴールドスタンダード；非常に時間がかかる。
- **CMOS（比較MOS）。** A対Bの選好。アノテーションあたりの信頼区間が狭い。
- **UTMOS・DNSMOS。** 参照なしの神経MOSプレディクター。リーダーボードで使用。
- **ASR経由CER（文字エラーレート）。** TTS出力をWhisperで処理し、入力テキストに対してCERを計算する。理解度の代理指標。
- **SECS（話者埋め込みコサイン類似度）。** 音声クローンの品質。

2026年のLibriTTS test-cleanでの数値:

| モデル | UTMOS | CER（Whisper経由） | サイズ |
|-------|-------|------------------|--------|
| 地面真実 | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 3.35億 |
| XTTS v2 | 3.81 | 3.5% | 4.70億 |
| VITS | 3.62 | 3.1% | 2500万 |
| Kokoro v0.19 | 3.87 | 1.8% | 8200万 |
| Parler-TTS Large | 3.76 | 2.8% | 23億 |

## 実装する

### ステップ1: 入力を音素化する

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

音素は汎用的な橋渡しだ。VITSレベル以下の品質のものに生のテキストを渡すことは避ける。

### ステップ2: Kokoroを実行する（2026年のCPUデフォルト）

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = アメリカ英語
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32テンソル, sr=24000
```

オフライン実行、単一ファイル、8200万パラメータ。

### ステップ3: F5-TTSで音声クローンを実行する

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

5秒のリファレンスクリップとそのトランスクリプトを渡すと、F5はプロソディとティンバーをクローンする。

### ステップ4: HiFi-GANボコーダをゼロから作る

チュートリアルスクリプトに収めるには大きすぎるが、形状は:

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4つのアップサンプルブロック、合計256倍でMelレートから音声レートへ
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> 波形
```

学習: 敵対的（短いウィンドウの判別器）＋Mel再構成損失＋特徴マッチング損失。コモディティ化済み——`hifi-gan` リポジトリまたはnvidia-NeMoの事前学習済みチェックポイントを使用すること。

### ステップ5: 完全なパイプライン（疑似コード）

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 使ってみる

2026年のスタック:

| 状況 | 選択 |
|------|------|
| リアルタイム英語音声アシスタント | Kokoro（CPU）またはXTTS v2（GPU） |
| 5秒リファレンスからの音声クローン | F5-TTS |
| 商業キャラクターボイス | ElevenLabs v2.5 |
| 音声書籍のナレーション | ElevenLabs v2.5またはXTTS v2＋ファインチューニング |
| 低リソース言語 | 対象言語の5〜20時間のデータでVITSを学習 |
| 表現力豊か / 感情タグ | ElevenLabs v2.5またはStyleTTS 2ファインチューニング |

2026年のオープンソースリーダー: **品質ならF5-TTS、効率ならKokoro**。Tacotronは歴史家でない限り使わないこと。

## 落とし穴

- **テキスト正規化なし。** 「Dr. Smith」は「Doctor」か「Drive」か？「2026」は「twenty twenty six」か「two zero two six」か？音素化の前に正規化すること。
- **OOVの固有名詞。** 「Ghumare」→「ghyu-mair」？未知トークンのフォールバックグラフィーム音素変換モデルを用意すること。
- **クリッピング。** ボコーダ出力はほとんどクリッピングしないが、推論時のMelスケーリングの不一致で±1.0を超えることがある。常に `np.clip(wav, -1, 1)` を行うこと。
- **サンプルレートの不一致。** Kokoroは24 kHzで出力するが、下流パイプラインが16 kHzを期待している→リサンプリングするかエイリアシングが生じる。

## 成果物を出す

`outputs/skill-tts-designer.md` として保存する。特定のボイス・遅延・言語ターゲットのTTSパイプラインを設計する。

## 演習

1. **簡単。** `code/main.py` を実行する。おもちゃの語彙から音素辞書を構築し、音素ごとのデュレーションを推定し、偽の「Mel」スケジュールを出力する。
2. **中級。** Kokoroをインストールし、同じ文章をボイス `af_bella` と `am_adam` で合成する。音声の長さと主観的品質を比較する。
3. **難しい。** 自分の声の5秒リファレンスクリップを録音する。F5-TTSでクローンする。リファレンスとクローン出力のSECSを報告する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------|
| 音素 | 音の単位 | 抽象的な音のクラス；英語には39個（ARPABet）。 |
| デュレーション予測器 | 各音素の長さ | 非ARモデルの出力；音素ごとの整数フレーム数。 |
| ボコーダ | Mel→波形 | MelスペクトログラムをRAWサンプルにマッピングするニューラルネット。 |
| HiFi-GAN | 標準ボコーダ | GANベース；2020〜2024年の主流。 |
| MOS | 主観品質 | 人間の評価者による1〜5の平均意見スコア。 |
| SECS | 音声クローン指標 | 対象とクローンの話者埋め込み間のコサイン類似度。 |
| F5-TTS | 2024年オープンソースSOTA | フローマッチング拡散；ゼロショットクローン。 |
| Kokoro | CPU英語リーダー | 8200万パラメータモデル、Apache 2.0。 |

## 参考資料

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — seq2seqベースライン。
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — エンドツーエンドフローベース。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 現在のオープンソースSOTA。
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — 2026年でも使われているボコーダ。
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — 2024年のCPUフレンドリーな英語TTS。
