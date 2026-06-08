# 音声分類 — MFCCのk-NNからASTとBEATsまで

> 「犬の吠え声かサイレンか」から「これは何語か」まで、すべて音声分類だ。特徴量はMelだ。アーキテクチャは10年ごとに変わる。評価はAUC・F1・クラスごとの再現率のままだ。


## 問題

10秒のクリップが手元にある。知りたいのは「これは何か？」だ。都市音（サイレン・ドリル・犬）、音声コマンド（yes/no/stop）、言語識別（en/es/ar）、話者の感情（怒り/中立）、環境音（室内/屋外、雑音）。これらはすべて*音声分類*であり、2026年のベースラインアーキテクチャは成熟している: 対数Mel → CNNまたはTransformer → softmax。

核心的な難しさはネットワークではない。データだ。音声データセットはクラス不均衡が激しく、ドメインシフト（クリーンかノイジーか）が大きく、ラベルノイズが多い（「都市の雑音」と「レストランの騒音」の境界は誰が決めたのか？）。問題の80%はキュレーション・データ拡張・評価であり、CNNをTransformerに入れ替えることではない。

## 概念

![音声分類の階段: MFCCのk-NNからASTとBEATsまで](../assets/audio-classification.svg)

**MFCCのk-NN（1990年代のベースライン）。** クリップごとにMFCCを平坦化し、ラベル付きバンクとのコサイン類似度を計算し、上位K個の多数決を返す。クリーンで小さいデータセット（Speech Commands・ESC-50）では驚くほど強力だ。GPUなしで動作する。

**対数MelのCNN 2D（2015〜2019年）。** `(T, n_mels)` の対数Melを画像として扱う。ResNet-18またはVGGスタイルを適用する。時間軸をグローバル平均プール。クラスに対してsoftmax。2026年のほとんどのKaggleコンペでも依然ベースラインだ。

**音声スペクトログラムTransformer、AST（2021〜2024年）。** 対数Melをパッチ化（例: 16×16パッチ）し、位置埋め込みを追加し、ViTに入力する。教師あり学習のAudioSetでSOTA（mAP 0.485）。

**BEATsとWavLM-base（2024〜2026年）。** 数百万時間の自己教師あり事前学習。必要だった教師ありデータの1〜10%でタスクをファインチューニング。2026年では非音声オーディオのデフォルトの出発点だ。BEATs-iter3はAudioSetでASTより1〜2 mAP高く、計算量は1/4だ。

**Whisperエンコーダを固定バックボーンとして使用（2024年）。** Whisperのエンコーダを取り、デコーダを外し、線形分類器を付ける。音声拡張なしで言語識別や単純なイベント分類でほぼSOTAに達する。「フリーランチ」ベースラインだ。

### クラス不均衡が本当の課題

ESC-50: 50クラス、各40クリップ — バランス良好、簡単。UrbanSound8K: 10クラス、10:1の不均衡。AudioSet: 100,000:1のロングテールを持つ632クラス。有効なテクニック:

- 学習中のバランスサンプリング（評価では行わない）。
- Mixup: 2つのクリップ（とそのラベル）を線形補間してデータ拡張。
- SpecAugment: ランダムな時間帯域と周波数帯域をマスク。シンプルで重要。

### 評価

- 多クラス排他的（Speech Commands）: top-1精度、top-5精度。
- 多クラス多ラベル（AudioSet・UrbanSoundスタイル）: 平均精度（mAP）。
- 大きな不均衡: クラスごとの再現率 + マクロF1。

2026年に知っておくべき数値:

| ベンチマーク | ベースライン | 2026年SOTA | 出典 |
|------------|------------|-----------|------|
| ESC-50 | 82%（AST） | 97.0%（BEATs-iter3） | BEATsの論文（2024年） |
| AudioSet mAP | 0.485（AST） | 0.548（BEATs-iter3） | HEARリーダーボード2026 |
| Speech Commands v2 | 98%（CNN） | 99.0%（Audio-MAE） | HEAR v2の結果 |

## 実装する

### ステップ1: 特徴量を抽出する

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### ステップ2: 固定長のサマリ

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

シンプルだが強力: 時間方向の平均＋分散で13係数MFCCから26次元の固定埋め込みが得られる。即座に実行できる。2017年まで、ESC-50ではNNのベースラインより優れていた。

### ステップ3: k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### ステップ4: 対数MelのCNNへアップグレード

PyTorchで:

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

300万パラメータ。単一RTX 4090でESC-50を約10分で学習。精度80%以上。

### ステップ5: 2026年のデフォルト — BEATsをファインチューニングする

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

BEATsには `microsoft/BEATs-base` を `beats` ライブラリ経由で使用する；transformers APIは同じ形状だ。

## 使ってみる

2026年のスタック:

| 状況 | 出発点 |
|------|--------|
| 小規模データセット（1000クリップ未満） | MFCCの平均のk-NN（ベースライン）＋音声拡張 |
| 中規模データセット（1K〜100K） | BEATsまたはASTのファインチューニング |
| 大規模データセット（100K超） | ゼロから学習またはWhisperエンコーダのファインチューニング |
| リアルタイム・エッジ | 40個のMFCC CNN、int8量子化（KWSスタイル） |
| 多ラベル（AudioSet） | BEATs-iter3 + BCE損失 + mixup + SpecAugment |
| 言語識別 | MMS-LID・SpeechBrain VoxLingua107ベースライン |

判断基準: **新規モデルではなく固定バックボーンから始めよ**。BEATsのヘッドをファインチューニングすれば数時間でSOTAの95%に達する。

## 成果物を出す

`outputs/skill-classifier-designer.md` として保存する。音声分類タスクに対してアーキテクチャ・データ拡張・クラスバランス戦略・評価指標を選択する。

## 演習

1. **簡単。** `code/main.py` を実行する。4クラスの合成データセット（異なるピッチの純音）でMFCCのk-NNベースラインを学習する。混同行列を報告する。
2. **中級。** `summarize` を [平均、分散、歪度、尖度] に置き換える。4次モーメントのプーリングは同じ合成データセットで平均＋分散を上回るか？
3. **難しい。** `torchaudio` を使ってESC-50のfold 1でCNN 2Dを学習する。5分割交差検証の精度を報告する。SpecAugment（時間マスク=20、周波数マスク=10）を追加してその差を報告する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------|
| AudioSet | オーディオのImageNet | Googleの200万クリップ・632クラスの弱ラベルYouTubeデータセット。 |
| ESC-50 | 小規模分類ベンチマーク | 50クラス×40クリップの環境音。 |
| AST | 音声スペクトログラムTransformer | 対数MelパッチのViT；2021年SOTA。 |
| BEATs | 自己教師あり音声 | Microsoftのモデル；iter3は2026年時点でAudioSetをリード。 |
| Mixup | ペアのデータ拡張 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | マスクベースのデータ拡張 | スペクトログラムのランダムな時間帯域と周波数帯域をゼロにする。 |
| mAP | 主要な多ラベル指標 | クラスと閾値にわたる平均適合率。 |

## 参考資料

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 2021〜2024年の代表的アーキテクチャ。
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — 2024年以降のデフォルト。
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — 主要な音声拡張。
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — 50クラスのベンチマーク。
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 632クラスのYouTube分類体系；依然としてゴールドスタンダード。
