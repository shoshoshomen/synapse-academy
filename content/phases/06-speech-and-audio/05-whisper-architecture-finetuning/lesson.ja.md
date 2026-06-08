# Whisper — アーキテクチャとファインチューニング

> Whisperは30秒窓のTransformerエンコーダ-デコーダで、多言語の弱ラベル付き音声-テキストペア68万時間で学習されている。1つのアーキテクチャ、複数のタスク、99言語でロバスト。2026年のASRリファレンスだ。


## 問題

2022年9月にOpenAIが公開したWhisperは、ASRモデルとして初めてコモディティとして提供された: 音声を貼り付けてテキストを得る、99言語、ノイズにロバスト、ラップトップで動作する。2024年までにOpenAIはLarge-v3とTurboバリアントを提供し、2026年にはWhisperがポッドキャストの文字起こしから音声アシスタントやYouTube字幕まで、あらゆるもののデフォルトベースラインになっている。

しかしWhisperはブラックボックスとして永遠に扱えるパイプラインではない。ドメインシフトがパフォーマンスを低下させる——技術用語、話者のアクセント、固有名詞、短いクリップ、無音。以下を知る必要がある:

1. 内部で何が起きているか。
2. チャンク化・ストリーミング・長時間音声を正しく渡す方法。
3. ファインチューニングが必要な時とその方法。

## 概念

![Whisperエンコーダ-デコーダ・タスク・チャンク化推論・ファインチューニング](../assets/whisper.svg)

**アーキテクチャ。** 標準的なTransformerエンコーダ-デコーダ。

- 入力: 30秒の対数Melスペクトログラム、80 Mel、10 msホップ → 3000フレーム。より短いクリップはゼロパディング、より長いクリップはチャンク化される。
- エンコーダ: conv-ダウンサンプル（ストライド2）＋ `N` Transformerブロック。Large-v3: 32レイヤー、1280次元、20ヘッド。
- デコーダ: 因果的セルフアテンション＋エンコーダ出力へのクロスアテンションを持つ `N` Transformerブロック。エンコーダと同サイズ。
- 出力: 51,865トークンの語彙に対するBPEトークン。

Large-v3は15.5億パラメータ。Turboは4レイヤーのデコーダを使用し（32から削減）、WERへの影響1%未満で遅延を8倍削減している。

**プロンプトフォーマット。** Whisperはデコーダプロンプトの特殊トークンで誘導されるマルチタスクモデルだ:

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — 言語タグ；翻訳対書き起こしの動作を強制する。
- `<|transcribe|>` または `<|translate|>` — 任意言語の入力から英語出力に翻訳するか、そのまま書き起こすか。
- `<|notimestamps|>` — 単語レベルのタイムスタンプをスキップする（高速化）。

プロンプトが1つのモデルで多くのタスクを可能にする。`<|en|>` を `<|fr|>` に変えるとフランス語を書き起こす。

**30秒窓。** すべてが30秒に固定されている。より長いクリップはチャンク化が必要；短いクリップはパディングされる。窓はネイティブではストリーミングされない——これがWhisperX・Whisper-Streaming・faster-whisperが存在する理由だ。

**対数Mel正規化。** `(log_mel - mean) / std`（統計量はWhisper自身の学習コーパスから）。Whisperの前処理（`whisper.audio.log_mel_spectrogram`）を使用しなければならず、`librosa.feature.melspectrogram` ではない。

### 2026年のバリアント

| バリアント | パラメータ数 | 遅延（A100） | WER（LibriSpeech-clean） |
|----------|------------|-------------|------------------------|
| Tiny | 3900万 | 1×リアルタイム | 5.4% |
| Base | 7400万 | 1× | 4.1% |
| Small | 2.44億 | 1× | 3.0% |
| Medium | 7.69億 | 1× | 2.7% |
| Large-v3 | 15.5億 | 2× | 1.8% |
| Large-v3-turbo | 8.09億 | 8× | 1.58% |
| Whisper-Streaming（2024年） | 15.5億 | ストリーミング | 2.0% |

### ファインチューニング

2026年の標準的なワークフロー:

1. 対象ドメインの音声と整合したトランスクリプトを10〜100時間収集する。
2. `generate_with_loss` コールバックとともに `transformers.Seq2SeqTrainer` を実行する。
3. パラメータ効率的: アテンション層の `q_proj`・`k_proj`・`v_proj` に対するLoRAでGPUメモリを4倍削減し、WERへの影響は0.3未満だ。
4. 10時間未満の場合はエンコーダを凍結し、デコーダのみをチューニングする。
5. Whisper自身のトークナイザーとプロンプトフォーマットを使用し、トークナイザーを交換しない。

コミュニティの実績: 医療口述の20時間でMediumをファインチューニングすると、医療語彙でのWERが12%から4.5%に下がる。アイスランド語の4時間でTurboをファインチューニングするとWERが18%から6%に下がる。

## 実装する

### ステップ1: Whisperをそのまま実行する

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # 連鎖する繰り返しを防ぐ
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

常にオーバーライドすべきデフォルト: `temperature=0.0`（サンプリングはデフォルトで0.0→0.2→0.4...のフォールバックチェーン）、`condition_on_previous_text=False`（連鎖する幻覚問題を防ぐ）、`no_speech_threshold=0.6`（無音検出）。

### ステップ2: チャンク化した長時間録音

```python
# whisperxは長時間録音で単語レベルのタイムスタンプを得る2026年のリファレンス
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperXは(1) Silero VADゲーティング、(2) wav2vec 2.0による単語レベルのアライメント、(3) `pyannote.audio` による話者分離を追加する。本番文字起こしの2026年の定番ツールだ。

### ステップ3: LoRAでファインチューニングする

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> 学習可能パラメータ約300万 / 合計8.09億
```

その後、標準的なTrainerループ。1000ステップごとにチェックポイント。ホールドアウトデータでWERを評価する。

### ステップ4: 各レイヤーが何を学習するか調べる

```python
# デコード中のクロスアテンション重みを取得して、デコーダが何に注目するかを見る。
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: レイヤー × ヘッド × ステップ × ソース長
```

ヒートマップで可視化すると、デコーダのステップがエンコーダフレームをスキャンする対角線のアライメントが見える。その対角線がWhisperの単語タイムスタンプの概念だ。

## 使ってみる

2026年のスタック:

| 状況 | 選択 |
|------|------|
| 英語・オフライン全般 | `whisperx` 経由のLarge-v3-turbo |
| モバイル / エッジ | Whisper-Tiny量子化（int8）またはMoonshine |
| 多言語長時間録音 | `whisperx` 経由のLarge-v3＋話者分離 |
| 低リソース言語 | LoRAでMediumまたはTurboをファインチューニング |
| ストリーミング（2秒遅延） | Whisper-StreamingまたはParakeet-TDT |
| 単語レベルのタイムスタンプ | WhisperX（wav2vec 2.0による強制アライメント） |

`faster-whisper`（CTranslate2バックエンド）は2026年の最速CPU＋GPU推論ランタイムで、同一出力でバニラの4倍高速だ。

## 2026年でも起きる落とし穴

- **無音でテキストが幻覚生成される。** Whisperはキャプション付きで学習されているため「ご視聴ありがとうございました！」「チャンネル登録！」歌詞などを生成する。常にVADゲーティングをかけること。
- **`condition_on_previous_text` の連鎖。** 1つの幻覚が後続の窓を汚染する。流暢さが必要でない限り `False` に設定すること。
- **短いクリップのパディング。** 2秒のクリップを30秒にパディングすると末尾の無音で幻覚が生じる。`pad=False` またはVADゲーティングを使用すること。
- **間違ったMel統計。** WhisperのMelの代わりにlibrosaのMelを使うとほぼランダムな出力になる。`whisper.audio.log_mel_spectrogram` を使用すること。

## 成果物を出す

`outputs/skill-whisper-tuner.md` として保存する。特定のドメイン向けにWhisperのファインチューニングまたは推論パイプラインを設計する。

## 演習

1. **簡単。** `code/main.py` を実行する。Whisperスタイルのプロンプトをトークン化し、デコードされた形状のバジェットを計算し、10分クリップのチャンクスケジュールを出力する。
2. **中級。** `faster-whisper` をインストールし、10分のポッドキャストを文字起こしして人間のトランスクリプトと比較してWERを計算する。`language="auto"` と `language="en"` 強制を比較する。
3. **難しい。** HF `datasets` を使って、Whisperが苦手な言語（例: ウルドゥー語）を選び、2時間のデータでLoRAを使ってMediumを2エポックファインチューニングし、WERの変化を報告する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------|
| 30秒窓 | Whisperの制限 | ハードな入力上限；より長い音声はチャンク化が必要。 |
| SOT | トランスクリプト開始 | `<\|startoftranscript\|>` がデコーダプロンプトを開始する。 |
| タイムスタンプトークン | 時間的アライメント | 0.02秒ごとのオフセットが51kの語彙の特殊トークン。 |
| Turbo | 高速バリアント | 4デコーダレイヤー、8倍高速、WER低下1%未満。 |
| WhisperX | 長時間録音ラッパー | VAD＋Whisper＋wav2vecアライメント＋話者分離。 |
| LoRAファインチューニング | 効率的なチューニング | アテンションに低ランクアダプタを追加；全パラメータの約0.3%を学習。 |
| 幻覚 | サイレントな失敗 | Whisperがノイズ/無音から流暢な英語を生成する。 |

## 参考資料

- [Radford et al. (2022). Whisperの論文](https://arxiv.org/abs/2212.04356) — 元のアーキテクチャと学習レシピ。
- [OpenAI (2024). Whisper Large-v3-turboのリリース](https://github.com/openai/whisper/discussions/2363) — 4レイヤーデコーダ、8倍高速化。
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — 長時間録音・単語アライメント・話者分離。
- [Systran — faster-whisperのリポジトリ](https://github.com/SYSTRAN/faster-whisper) — CTranslate2バックエンド、4倍高速。
- [HuggingFace — Whisperファインチューニングチュートリアル](https://huggingface.co/blog/fine-tune-whisper) — LoRA / フルFTの標準的なウォークスルー。
