# 音声評価 — WER・MOS・UTMOS・MMAU・FAD、そしてオープンリーダーボード

> 測定できないものは出荷できない。このレッスンでは2026年のすべての音声タスクの指標を名付ける: ASR（WER・CER・RTFx）、TTS（MOS・UTMOS・SECS・ASRラウンドトリップWER）、音声言語（MMAU・LongAudioBench）、音楽（FAD・CLAP）、話者（EER）。さらに比較のためのリーダーボードも。


## 問題

すべての音声タスクには複数の指標があり、それぞれが異なる軸を測定する。間違った指標を使うと、ダッシュボードでは素晴らしく見えるが本番では最悪なモデルを出荷することになる。2026年の標準リスト:

| タスク | 主要指標 | 副次指標 |
|--------|---------|---------|
| ASR | WER | CER · RTFx · 最初のトークン遅延 |
| TTS | MOS / UTMOS | SECS · ASRラウンドトリップWER · CER · TTFA |
| 音声クローニング | SECS（ECAPA コサイン） | MOS · CER |
| 話者検証 | EER | minDCF · 運用点でのFAR/FRR |
| 話者分離 | DER | JER · 話者混乱 |
| 音声分類 | top-1 · mAP | マクロF1 · クラスごとの再現率 |
| 音楽生成 | FAD | CLAP · リスニングパネルMOS |
| 音声言語モデル | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| ストリーミングS2S | 遅延P50/P95 | WER · MOS |

## 概念

![音声評価マトリクス — 指標vsタスクvs2026年リーダーボード](../assets/eval-landscape.svg)

### ASR指標

**WER（Word Error Rate: 単語誤り率）。** `(S + D + I) / N`。スコアリング前に小文字化・句読点削除・数字正規化を行う。`jiwer` またはOpenAIの `whisper_normalizer` を使用する。5%未満 = 読み上げ音声での人間パリティ。

**CER（Character Error Rate: 文字誤り率）。** 同じ式で文字レベル。単語分節が曖昧な声調言語（北京語・広東語）に使用される。

**RTFx（逆リアルタイム係数）。** ウォールクロック秒あたりに処理された音声秒数。高いほど良い。Parakeet-TDTは3380倍に達する。Whisper-large-v3は約30倍。

**最初のトークン遅延。** 音声入力から最初のトランスクリプトトークンまでのウォールクロック時間。ストリーミングには重要。Deepgram Nova-3: 約150 ms。

### TTS指標

**MOS（Mean Opinion Score: 平均主観評価点）。** 1〜5の人間評価。ゴールドスタンダードだが遅い。サンプルあたり20人以上のリスナー、モデルあたり100以上のサンプルを集める。

**UTMOS（2022〜2026年）。** 学習済みMOS予測器。標準ベンチマークで人間MOSと約0.9の相関。F5-TTS: UTMOS 3.95；グラウンドトゥルース: 4.08。

**SECS（Speaker Encoder Cosine Similarity: 話者エンコーダコサイン類似度）。** 音声クローニング用。参照とクローン出力のECAPAエンベディングコサイン。> 0.75 = 認識可能なクローン。

**ASRラウンドトリップWER。** TTS出力に対してWhisperを実行し、入力テキストに対してWERを計算する。了解可能性の後退を検出する。2026年SOTA: CER < 2%。

**TTFA（Time-to-First-Audio: 最初の音声までの時間）。** ウォールクロック遅延。Kokoro-82M: 約100 ms；F5-TTS: 約1秒。

### 音声クローニング固有

**SECS＋MOS＋CER** のトリプル。SECSが高くMOSが低いクローンはティンバーは合っているが不自然；逆は自然な声だが話者が違う。

### 話者検証

**EER（Equal Error Rate: 等誤り率）。** 誤受け入れ率が誤拒否率に等しい閾値。VoxCeleb1-OでのECAPA: 0.87%。

**minDCF（最小検出コスト）。** 選択した運用点（多くの場合FAR=0.01）での重み付きコスト。EERより本番に関連性がある。

### 話者分離

**DER（Diarization Error Rate: 話者分離誤り率）。** `(FA + Miss + Confusion) / total_speaker_time`。見逃し発話＋誤警報発話＋話者混乱、それぞれを割合として。AMIミーティング: 録音状態が良好なリアルな環境でDER 10〜20%。pyannote 3.1 + Precision-2商用: 良好な録音でDER < 10%。

**JER（Jaccard Error Rate）。** DERの代替；短いセグメントバイアスに対してロバスト。

### 音声分類

マルチラベル: 全クラスの**mAP（mean Average Precision: 平均適合率）**。AudioSet: BEATs-iter3で0.548 mAP。

排他的マルチクラス: **top-1、top-5精度**。Speech Commands v2: 99.0% top-1（Audio-MAE）。

不均衡: **マクロF1**＋**クラスごとの再現率**。クラスごとに報告する——集計精度はどのクラスが失敗しているかを隠す。

### 音楽生成

**FAD（Fréchet Audio Distance: フレシェ音声距離）。** 本物vs生成音声のVGGishエンベディング分布間の距離。MusicCapsでのMusicGen-small: 4.5。MusicLM: 4.0。低いほど良い。

**CLAPスコア。** CLAPエンベディングを使用したテキスト-音声アライメントスコア。> 0.3 = 妥当なアライメント。

**リスニングパネルMOS。** 消費者向け音楽での最終評価。Suno v5のTTS ArenaでELO 1293（ペアの人間の好みから）。

### 音声言語ベンチマーク

**MMAU（Massive Multi-Audio Understanding）。** 1万件の音声QAペア。

**MMAU-Pro。** 1800件の難しい項目、4カテゴリ: 音声/サウンド/音楽/マルチ音声。4択でのランダムチャンス25%。Gemini 2.5 Proの全体で約60%；マルチ音声はすべてのモデルで約22%。

**LongAudioBench。** 数分間のクリップと意味的クエリ。Audio Flamingo NextがGemini 2.5 Proを上回る。

**AudioCaps / Clotho。** キャプショニングベンチマーク。SPICE・CIDEr・FENSE指標。

### ストリーミング音声対音声

**遅延P50/P95/P99。** ユーザー発話終了から最初の音声応答までのウォールクロック時間。Moshi: 200 ms；GPT-4o Realtime: 300 ms。

**WER/MOS** 出力に対して。

**割り込み応答性。** ユーザーの割り込みからアシスタントのミュートまでの時間。目標 < 150 ms。

### 2026年のリーダーボード

| リーダーボード | トラック | URL |
|------------|--------|-----|
| Open ASR Leaderboard（HF） | 英語＋多言語＋長文 | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena（HF） | 英語TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS＋STT、ペア投票からのELO | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM推論 | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | 話者認識 | `voxsrc.github.io` |
| MMAU音楽サブセット | 音楽LALM | （MMAU内） |
| HEARベンチマーク | 自己教師あり音声 | `hearbenchmark.com` |

## 実装する

### ステップ1: 正規化付きWER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### ステップ2: TTSラウンドトリップWER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### ステップ3: 音声クローニングのSECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### ステップ4: 音楽生成のFAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### ステップ5: 話者検証のEER（レッスン6と同じコード）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 使ってみる

すべてのデプロイに固定された評価ハーネスをペアにして、すべてのモデル更新で実行する。3つの基本ルール:

1. **スコアリング前に正規化する。** 小文字化・句読点削除・数字展開。正規化ルールを報告する。
2. **平均ではなく分布を報告する。** 遅延のP50/P95/P99。分類のクラスごとの再現率。MMAUのカテゴリごと。
3. **1つの標準的な公開ベンチマークを実行する。** 本番データが異なっていても、Open ASR / TTS Arena / MMAUでの報告でレビュアーがリンゴとリンゴを比較できる。

## 落とし穴

- **UTMOSの外挿。** VCTKスタイルのクリーン音声で学習；ノイズの多い/クローン/感情的な音声では評点が悪い。
- **MOSパネルバイアス。** Amazon Mechanical Turkの20人の作業者 ≠ 20人のターゲットユーザー。重要度が高い場合はドメインパネルにお金を払うこと。
- **FADは参照セットに依存する。** モデル間で同じ参照分布に対して比較すること。
- **集計WER。** 全体5% WERは訛りのある音声での30% WERを隠す場合がある。人口統計スライスごとに報告すること。
- **公開ベンチマークの飽和。** ほとんどのフロンティアモデルは標準ベンチマークで上限に近い。自分のトラフィックを反映した社内ホールドアウトセットを構築すること。

## 成果物を出す

`outputs/skill-audio-evaluator.md` として保存する。任意の音声モデルリリースのための指標・ベンチマーク・報告形式を選択する。

## 演習

1. **簡単。** `code/main.py` を実行する。おもちゃの入力でWER/CER/EER/SECS/FADっぽい/MMAUっぽいを計算する。
2. **中級。** TTSラウンドトリップWERハーネスを構築する。KokoroまたはF5-TTS出力をWhisperで処理する。50のプロンプトでWERを計算する。WER > 10%のプロンプトにフラグを立てる。
3. **難しい。** レッスン10のLALM選択肢をMMAU-Pro音声＋マルチ音声サブセット（各50項目）でスコアリングする。カテゴリごとの精度を報告し、公開された数値と比較する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------|
| WER | ASRスコア | 正規化後の単語レベルで `(S+D+I)/N`。 |
| CER | 文字WER | 声調言語や文字レベルシステム用。 |
| MOS | 人間の意見 | 1〜5評価；20人以上のリスナー × 100サンプル。 |
| UTMOS | ML MOSプレディクタ | 学習済みモデル；人間MOSと約0.9の相関。 |
| SECS | 音声クローン類似度 | 参照とクローンのECAPAコサイン。 |
| EER | 話者検証スコア | FAR = FRRとなる閾値。 |
| DER | 話者分離スコア | (FA + Miss + Confusion) / 合計。 |
| FAD | 音楽生成品質 | VGGishエンベディングのフレシェ距離。 |
| RTFx | スループット | ウォールクロック秒あたりの音声秒数。 |

## 参考資料

- [jiwer](https://github.com/jitsi/jiwer) — 正規化ユーティリティ付きWER/CERライブラリ。
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — 学習済みMOS予測器。
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — 音楽生成の標準。
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026年のライブランキング。
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — 人間投票TTSリーダーボード。
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — LALM推論リーダーボード。
- [HEARベンチマーク](https://hearbenchmark.com/) — 音声SSLベンチマーク。
