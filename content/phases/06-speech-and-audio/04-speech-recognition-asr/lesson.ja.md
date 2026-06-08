# 音声認識（ASR）— CTC・RNN-T・Attention

> 音声認識はすべてのタイムステップでの音声分類であり、英語と無音を知っているシーケンスモデルによって結びつけられている。CTC・RNN-T・Attentionの3つの方法がある。1つを選んでその理由を理解せよ。


## 問題

10秒の16 kHzクリップがある。目標は「turn on the kitchen lights（キッチンの照明をつけて）」という文字列だ。課題は構造的なものだ: 音声フレームは文字と1対1で対応しない。「okay」という単語は200 msかもしれないし1200 msかもしれない。沈黙が発話を区切る。一部の音素は他より長い。出力トークンの数は事前にわからない。

3つの定式化がこれを解決する:

1. **CTC（接続性時系列分類）。** 特殊な*ブランク*を含むフレームごとのトークン確率を出力する。デコード時に繰り返しとブランクを折りたたむ。非自己回帰的で高速。wav2vec 2.0・MMSが使用している。
2. **RNN-T（再帰型ニューラルネットワークトランスデューサー）。** 結合ネットワークがエンコーダフレームと前のトークンを条件に次のトークンを予測する。ストリーミング可能。GoogleのオンデバイスASR・NVIDIA Parakeetが使用している。
3. **Attentionエンコーダ-デコーダ。** エンコーダが音声を隠れ状態に圧縮し、デコーダがクロスアテンションで自己回帰的にトークンを生成する。Whisper・SeamlessM4Tが使用している。

2026年、LibriSpeech test-cleanのSOTA WERはParakeet-TDT-1.1B（NVIDIA）で1.4%、Whisper-Large-v3-turboで1.58%だ。差は小さいが、デプロイ上の差は大きい。

## 概念

![3つのASR定式化: CTC・RNN-T・Attention-Encoder-Decoder](../assets/asr-formulations.svg)

**CTCの直感。** エンコーダに `V+1` トークン（V文字＋ブランク）に対する `T` フレームレベルの分布を出力させる。長さ `U < T` の目標文字列 `y` に対して、`y` に折りたたまれるフレームのアラインメントはすべてカウントされる。CTC損失はそのようなすべてのアラインメントを合計する。推論: フレームごとのargmax、繰り返しを折りたたみ、ブランクを除去する。

利点: 非自己回帰的、ストリーミング可能、ルックアヘッドなし。欠点: *条件付き独立性の仮定* — 各フレームの予測は他から独立しているため、内部言語モデルがない。ビームサーチや浅い融合による外部言語モデルで修正する。

**RNN-Tの直感。** トークン履歴を埋め込む*予測器*ネットワークと、予測器の状態とエンコーダフレームを組み合わせて `V+1` の結合分布に変換する*ジョイナー*を追加する（`+1` はnull / 非出力）。CTCが無視した条件付き依存性を明示的にモデル化する。各ステップが過去のフレームと過去のトークンにのみ依存するため、ストリーミング可能。

利点: ストリーミング可能＋内部言語モデル。欠点: 学習がより複雑でメモリを消費する（3次元損失格子）；RNN-T損失カーネルはそれ自体が1つのライブラリカテゴリだ。

**Attentionエンコーダ-デコーダ。** 対数Melフレームに対するエンコーダ（6〜32 Transformerレイヤー）。デコーダ（6〜32 Transformerレイヤー）がエンコーダ出力に対してクロスアテンションし、自己回帰的にトークンを生成する。アライメント制約なし——アテンションは音声のどこでも見られる。チャンク化Whisper-Streaming（2024年）を除いてストリーミングは不可。

利点: オフラインASRで最高品質、標準的なseq2seqツールで学習が容易。欠点: 自己回帰的な遅延は出力長に比例する；エンジニアリングなしではストリーミングできない。

### WER: 唯一の数値

**単語エラーレート** = `(S + D + I) / N`（S=置換、D=削除、I=挿入、N=参照単語数）。単語レベルのレーベンシュタイン編集距離と一致する。低いほど良い。WER 20%超は一般的に使用不可；5%未満は読み上げ音声での人間同等。2026年の標準ベンチマーク数値:

| モデル | LibriSpeech test-clean | LibriSpeech test-other | サイズ |
|-------|------------------------|------------------------|--------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 11億パラメータ |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 8.09億 |
| Canary-1B Flash | 1.48% | 2.87% | 10億 |
| Seamless M4T v2 | 1.7% | 3.5% | 23億 |

これらはすべてエンコーダ-デコーダまたはRNN-Tベースだ。純粋なCTCシステム（wav2vec 2.0）はtest-cleanで約1.8〜2.1%だ。

## 実装する

### ステップ1: 貪欲なCTCデコード

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: フレームごとの確率ベクトルのリスト
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

2つのルール: 連続する繰り返しを折りたたみ、ブランクを除去する。例: `a a _ _ a b b _ c` → `a a b c`。

### ステップ2: ビームサーチCTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

本番では言語モデル融合付きのプレフィックスツリービームサーチを使用する；これは概念的なスケルトンだ。

### ステップ3: WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### ステップ4: Whisperでの推論

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026年の最強汎用ASRを1行で実現。24 GBのGPUで約20倍リアルタイムで動作する。

### ステップ5: ParakeetまたはWav2Vec 2.0でのストリーミング

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

ストリーミングASRはチャンク化されたエンコーダアテンションとキャリーオーバーステートが必要だ；それをサポートするライブラリを使用する（Parakeet用NeMo、`transformers` パイプラインの `chunk_length_s` など）。

## 使ってみる

2026年のスタック:

| 状況 | 選択 |
|------|------|
| 英語・オフライン・最高品質 | Whisper-large-v3-turbo |
| 多言語・ロバスト | SeamlessM4T v2 |
| ストリーミング・低遅延 | Parakeet-TDT-1.1BまたはRiva |
| エッジ・モバイル・500ms未満の遅延 | Whisper-Tiny量子化またはMoonshine（2024年） |
| 長時間録音 | VADベースのチャンク化によるWhisper（WhisperX） |
| ドメイン特化（医療・法律） | wav2vec 2.0のファインチューニング＋ドメイン言語モデル融合 |

## 2026年でも起きる落とし穴

- **VADなし。** 無音でWhisperを実行すると幻覚が生じる（「ご視聴ありがとうございました！」）。常にVADでゲーティングすること。
- **文字・単語・サブワードのWERの混同。** 正規化後（小文字、句読点除去）の単語レベルWERを報告すること。
- **言語識別のドリフト。** WhisperのLIDはノイジーなクリップを日本語やウェールズ語に誤ルーティングする；わかっている場合は `language="en"` を強制すること。
- **チャンク化なしの長いクリップ。** Whisperは30秒の窓を持つ。それ以上のものには `chunk_length_s=30, stride=5` を使用すること。

## 成果物を出す

`outputs/skill-asr-picker.md` として保存する。デプロイ先に応じてモデル・デコード戦略・チャンク化・言語モデル融合を選択する。

## 演習

1. **簡単。** `code/main.py` を実行する。手作りのCTC出力を貪欲デコードし、参照に対してWERを計算する。
2. **中級。** ステップ2のプレフィックスツリービームサーチを正しく実装する（ブランクのマージルールを考慮する）。10個の合成データサンプルで貪欲デコードと比較する。
3. **難しい。** `whisper-large-v3-turbo` を [LibriSpeech test-clean](https://www.openslr.org/12) で使用する。最初の100発話のWERを計算する。公表数値と比較する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------|
| CTC | ブランクトークン損失 | すべてのフレームからトークンへのアラインメントの周辺化；非AR。 |
| RNN-T | ストリーミング損失 | CTC＋次トークン予測器；語順を扱う。 |
| Attentionエンコーダ-デコーダ | Whisperスタイル | エンコーダ＋クロスアテンションデコーダ；オフライン品質が最高。 |
| WER | 報告すべき数値 | 単語レベルの `(S+D+I)/N`。 |
| ブランク | 空白 | CTCの特殊トークン；「このフレームは出力なし」を意味する。 |
| 言語モデル融合 | 外部言語モデル | ビームサーチで重み付けされた言語モデルの対数確率を加算する。 |
| VAD | 無音ゲート | 音声区間検出；非音声を除去する。 |

## 参考資料

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTCの論文。
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — RNN-Tの論文。
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 2022年の標準論文；v3-turboの拡張は2024年。
- [NVIDIA NeMo — Parakeet-TDTカード](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — 2026年のOpen ASRリーダーボードトップ。
- [Hugging Face — Open ASRリーダーボード](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 25以上のモデルにわたるライブベンチマーク。
