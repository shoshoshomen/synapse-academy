# 音声Transformer — Whisperアーキテクチャ

> 音声は時間に対する周波数の画像だ。WhisperはメルスペクトログラムをInputとして食べ、テキストを返すViTだ。


## 問題

Whisper（OpenAI、Radford et al. 2022年）以前、最先端の自動音声認識（ASR）はwav2vec 2.0とHuBERTを意味していた——自己教師付き特徴抽出器とファインチューニングされたヘッド。高品質だが、高価なデータパイプライン、ドメインに脆弱。多言語音声認識には言語ファミリーごとに別々のモデルが必要だった。

Whisperは3つの賭けをした：

1. **すべてで学習する。** 97言語にわたってインターネットから収集した弱ラベル付き680,000時間の音声。クリーンな学術コーパスなし。音素ラベルなし。
2. **マルチタスク単一モデル。** タスクトークンを通じて転写、翻訳、音声活動検出、言語ID、タイムスタンピングを共同で学習した1つのデコーダ。
3. **標準のエンコーダ-デコーダTransformer。** エンコーダはログメルスペクトログラムを消費する。デコーダはテキストトークンを自己回帰的に生成する。ボコーダなし、CTCなし、HMMなし。

結果：Whisper large-v3はクリーンなラベル付きデータがゼロの言語のアクセント、ノイズ、言語にわたって堅牢だ。2026年、すべてのオープンソース音声アシスタントとほとんどの商業的なものでデフォルトの音声フロントエンドだ。

## 概念

![Whisperパイプライン：音声 → メル → エンコーダ → デコーダ → テキスト](../assets/whisper.svg)

### ステップ1 — リサンプリング + ウィンドウ

16 kHzの音声。30秒にクリップ/パディング。ログメルスペクトログラムを計算：80メルビン、10 msストライド → 〜3,000フレーム × 80特徴。これがWhisperが見る「入力画像」だ。

### ステップ2 — 畳み込みステム

カーネル3、ストライド2の2つのConv1D層が3,000フレームを1,500に削減する。多くのパラメータを追加せずにシーケンス長を半分にする。

### ステップ3 — エンコーダ

1,500タイムステップにわたる24層（largeの場合）のTransformerエンコーダ。Sinusoidal位置エンコーディング、Self-Attention、GELU FFN。1,500 × 1,280の隠れ状態を生成する。

### ステップ4 — デコーダ

24層のTransformerデコーダ。GPT-2の語彙に少数の音声特有の特殊トークンを追加したBPE語彙からトークンを自己回帰的に生成する。

### ステップ5 — タスクトークン

デコーダプロンプトはモデルに何をすべきかを伝える制御トークンで始まる：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

または

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

モデルはこの規則で学習された。プレフィックスでタスクを制御する。2026年の指示ファインチューニングの等価物、音声に適用された。

### ステップ6 — 出力

ログ確率閾値付きビームサーチ（幅5）。`<|notimestamps|>`トークンが存在しない場合、タイムスタンプは音声0.02秒ごとに予測される。

### Whisperのサイズ

| モデル | パラメータ | 層 | d_model | ヘッド | VRAM（fp16） |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | 〜1 GB |
| Base | 74M | 6 | 512 | 8 | 〜1 GB |
| Small | 244M | 12 | 768 | 12 | 〜2 GB |
| Medium | 769M | 24 | 1024 | 16 | 〜5 GB |
| Large | 1550M | 32 | 1280 | 20 | 〜10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | 〜10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | 〜6 GB（4層デコーダ） |

Large-v3-turbo（2024年）はデコーダを32層から4層に削減した。WERポイント1未満の後退で8倍速いデコーディング。そのデコードスピードの解放が、Whisper-turboが2026年のリアルタイム音声エージェントのデフォルトである理由だ。

### Whisperがしないこと

- 話者ダイアリゼーション（誰が話しているか）なし。そのためにはpyannoteと組み合わせる。
- ネイティブなリアルタイムストリーミングなし——30秒ウィンドウは固定だ。現代のラッパー（`faster-whisper`、`WhisperX`）はVAD + オーバーラップを通じてストリーミングをボルトオンする。
- 外部チャンキングなしでは30秒を超える長形式コンテキストなし。実際には人間のスピーチが転写のために長距離コンテキストを必要とすることはめったにないため、うまく機能する。

### 2026年の状況

| タスク | モデル | 備考 |
|------|-------|-------|
| 英語ASR | Whisper-turbo、Moonshine | Moonshineはエッジで4倍速い |
| 多言語ASR | Whisper-large-v3 | 97言語 |
| ストリーミングASR | faster-whisper + VAD | 150 msレイテンシ目標達成可能 |
| TTS | Piper、XTTS-v2、Kokoro | エンコーダ-デコーダパターン、ただしWhisper形状 |
| 音声 + 言語 | AudioLM、SeamlessM4T | テキストトークン + 音声トークンが1つのTransformer内 |

## 実装する

`code/main.py`を参照。Whisperは学習しない——ログメルスペクトログラムパイプライン + タスクトークンプロンプトフォーマッターを構築する。それらが本番で実際に触れる部分だ。

### ステップ1：音声を合成する

16 kHzでサンプリングされた440 Hzの1秒正弦波を生成する。16,000サンプル。

### ステップ2：ログメルスペクトログラム（簡略化）

完全なメルスペクトログラムはFFTが必要だ。`librosa`を必要とせずにパイプラインを示す簡略化されたフレーミング + フレームごとのエネルギーバージョンを行う：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

フレーム = 25 ms、ホップ = 10 ms。Whisperのウィンドウと一致する。教育目的でフレームごとのエネルギーがメルビンの代わりとなる。

### ステップ3：30秒にパディング

Whisperは常に30秒チャンクを処理する。スペクトログラムを3,000フレームにパディング（またはクリップ）する。

### ステップ4：プロンプトトークンを構築する

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

それがタスク制御サーフェス全体だ。4トークンのプレフィックス。

## 使ってみる

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

より速い、OpenAI互換：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026年にWhisperを選ぶ場合：**

- 1つのモデルで多言語ASR。
- ノイズの多い多様な音声の堅牢な転写。
- 最も速いスタートポイントの研究/プロトタイプASR。

**他のものを選ぶ場合：**

- エッジでの超低レイテンシストリーミング——Moonshineがマッチした品質でWhisperを上回る。
- <200 msが必要なリアルタイム会話AI——専用ストリーミングASR。
- 話者ダイアリゼーション——WhisperはこれをしないのでPyannoteをボルトオンする。

## 成果物を出す

`outputs/skill-asr-configurator.md`を参照。このスキルは、新しい音声アプリケーションのためのASRモデル・デコーディングパラメータ・前処理パイプラインを選択する。

## 演習

1. **易.** `code/main.py`を実行せよ。10 msホップで16 kHzの1秒信号のフレーム数が〜100フレームであることを確認せよ。30秒の場合：〜3,000フレーム。
2. **中.** `numpy.fft`を使って完全なログメルスペクトログラムを構築せよ。`librosa.feature.melspectrogram(n_mels=80)`と数値誤差内で一致する80メルビンを確認せよ。
3. **難.** ストリーミング推論を実装せよ：音声を2秒オーバーラップの10秒ウィンドウにチャンクし、各チャンクでWhisperを実行し、転写をマージせよ。5分のポッドキャストサンプルでシングルパスとの単語エラー率を測定せよ。

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| メルスペクトログラム（Mel spectrogram） | 「音声画像」 | 2D表現：一方の軸に周波数ビン、もう一方に時間フレーム；セルあたりのログスケールエネルギー。 |
| ログメル（Log-mel） | 「Whisperが見るもの」 | logを通したメルスペクトログラム；人間の音量知覚を近似する。 |
| フレーム（Frame） | 「1つの時間スライス」 | 25 msのサンプルウィンドウ；10 msストライドで重複。 |
| タスクトークン（Task token） | 「音声のプロンプトプレフィックス」 | デコーダプロンプトの`<\|transcribe\|>` / `<\|translate\|>`のような特殊トークン。 |
| 音声活動検出（VAD） | 「スピーチを見つける」 | ASR前の沈黙を除去するゲート；コストを大幅に削減。 |
| CTC | 「Connectionist Temporal Classification」 | アライメントフリー学習のためのクラシックASR損失；WhisperはCTCを使わない。 |
| Whisper-turbo | 「小さいデコーダ、完全なエンコーダ」 | large-v3エンコーダ + 4層デコーダ；8倍速いデコーディング。 |
| Faster-whisper | 「本番ラッパー」 | CTranslate2再実装；int8量子化；OpenAIリファレンスより4倍速い。 |

## 参考資料

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper論文。
- [OpenAI Whisperリポジトリ](https://github.com/openai/whisper) — リファレンスコード + モデルウェイト。〜400行で`whisper/model.py`を読むとConv1Dステム + エンコーダ + デコーダが最初から最後まで見られる。
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — ステップ5〜6で説明したビームサーチ + タスクトークンロジックがここにある；500行、完全に読める。
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — 前身；一部の設定では依然としてSOTA特徴。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 本番ラッパー、リファレンスより4倍速い。
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024年エッジフレンドリーASR、Whisper形状だがより小さい。
- [HuggingFaceブログ — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — メルスペクトログラム前処理とトークンタイムスタンプ処理を含む標準ファインチューニングレシピ。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — レッスンのアーキテクチャ図を反映した完全な実装（エンコーダ、デコーダ、クロスアテンション、生成）。
