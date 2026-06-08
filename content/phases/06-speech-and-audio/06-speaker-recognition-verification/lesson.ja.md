# 話者認識と話者照合

> ASRは「何を言ったか？」を問う。話者認識は「誰が言ったか？」を問う。数学は同じに見える——埋め込みとコサイン——だが、本番のすべての判断は単一のEERの数値にかかっている。


## 問題

ユーザーがパスフレーズを言う。知りたいのは: これは本人が主張している人物か（*照合*、1:1）、それとも登録バンクの最初の人物か（*識別*、1:N）？あるいはどちらでもないのか——これは未知の話者か（*オープンセット*）？

2018年以前: GMM-UBM＋i-vector。妥当なEERだがチャンネルシフト（電話とラップトップ）や感情に対して脆弱。2018〜2022年: x-vector（角度マージンで学習したTDNNバックボーン）。2022年以降: ECAPA-TDNNとWavLM-largeの埋め込み。2026年にはフィールドは3つのモデルと1つの指標に支配されている。

その指標が **EER**——等エラーレートだ。誤受け入れ率=誤拒否率となる決定閾値を設定する。その交点がEERだ。すべての論文、すべてのリーダーボード、すべての調達の呼び声に使われる。

## 概念

![登録＋照合パイプラインと埋め込み＋コサイン＋EER](../assets/speaker-verification.svg)

**パイプライン。** 登録: 対象話者の5〜30秒を録音し、固定次元の埋め込みを計算する（ECAPA-TDNNで192次元、WavLM-largeで256次元）。照合: テスト発話の埋め込みを得る；コサイン類似度を計算する；閾値と比較する。

**ECAPA-TDNN（2020年、2026年でも主流）。** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network。スクイーズエキサイテーション付き1D convブロック、マルチヘッドアテンションプーリング、その後192次元への線形層。Additive Angular Margin損失（AAM-softmax）でVoxCeleb 1+2（2,700話者、110万発話）を学習。

**WavLM-SV（2022年以降）。** 事前学習済みWavLM-large SSLバックボーンをAAM損失でファインチューニング。品質は高いが遅い——300+ MB対15 MB。

**x-vector（ベースライン）。** TDNN＋統計プーリング。古典的；CPU/エッジで依然有用。

**AAM-softmax。** 角度空間で余白 `m` を追加した標準softmax: 正解クラスに `cos(θ + m)`。クラス間の角度分離を強制する。典型的な `m=0.2`、スケール `s=30`。

### スコアリング

- 登録とテストの埋め込み間の**コサイン**。閾値ベースの決定。
- **PLDA（確率的LDA）。** 同一話者対異なる話者に対して閉形式の尤度比を持つ潜在空間に埋め込みを射影する。EERを10〜20%削減するためにコサインの上に追加される。2020年以前の標準；現在はクローズドセットの設定でのみ使用。
- **スコア正規化。** `S-norm` または `AS-norm`: 各スコアをインポスターの平均と標準偏差のコホートに対して正規化する。クロスドメイン評価に必須。

### 知っておくべき数値（2026年）

| モデル | VoxCeleb1-O EER | パラメータ数 | スループット（A100） |
|-------|-----------------|------------|------------------|
| x-vector（古典的） | 3.10% | 500万 | 400×RT |
| ECAPA-TDNN | 0.87% | 1500万 | 200×RT |
| WavLM-SV large | 0.42% | 3.16億 | 20×RT |
| Pyannote 3.1セグメンテーション＋埋め込み | 0.65% | 600万 | 100×RT |
| ReDimNet（2024年） | 0.39% | 2400万 | 100×RT |

### 話者分離

複数話者のクリップで「誰がいつ話したか」。パイプライン: VAD → セグメント → 各セグメントを埋め込み → クラスタリング（凝集的またはスペクトル） → 境界を平滑化。現代のスタック: `pyannote.audio` 3.1。これは話者セグメンテーション＋埋め込み＋クラスタリングを1つの呼び出しで実現する。2026年のSOTA DERはAMIで約15%（2022年の23%から低下）。

## 実装する

### ステップ1: MFCC統計からのトイ埋め込み

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26次元
```

SOTAには程遠い——教育目的のみ。`code/main.py` はこれを合成話者データの概念実証として使用する。

### ステップ2: コサイン類似度＋閾値

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### ステップ3: 類似度ペアからEERを計算する

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

(EER、EER時の閾値) を返す。両方を報告すること。

### ステップ4: SpeechBrainによる本番環境

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# 登録: 3〜5個のクリーンなサンプルの埋め込みを平均する
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# 照合
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPAの典型的な閾値；自分のデータでチューニングする
```

### ステップ5: pyannoteで話者分離する

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 使ってみる

2026年のスタック:

| 状況 | 選択 |
|------|------|
| クローズドセット1:1照合、エッジ | ECAPA-TDNN＋コサイン閾値 |
| オープンセット照合、クラウド | WavLM-SV＋AS-norm |
| 話者分離（会議・ポッドキャスト） | `pyannote/speaker-diarization-3.1` |
| アンチスプーフィング（リプレイ/ディープフェイク検出） | AASISTまたはRawNet2 |
| 小型組込み（KWS＋登録） | Titanet-Small（NeMo） |

## 落とし穴

- **チャンネルの不一致。** VoxCeleb（Web動画）で学習したモデル≠電話の音声。常に対象チャンネルで評価すること。
- **短い発話。** テスト音声が3秒未満だとEERが急激に悪化する。
- **ノイジーな登録。** ノイジーな登録1つがアンカーを汚染する。クリーンなサンプルを3つ以上使用し平均すること。
- **条件をまたぐ固定閾値。** 常に対象ドメインのホールドアウトdevセットで閾値をチューニングすること。
- **非正規化された埋め込みに対するコサイン。** まずL2正規化を行う；さもないとL2ノルムが支配する。

## 成果物を出す

`outputs/skill-speaker-verifier.md` として保存する。モデル・登録プロトコル・閾値チューニング計画・不正防止策を選択する。

## 演習

1. **簡単。** `code/main.py` を実行する。合成「話者」（異なるトーンプロファイル）を作成し、登録し、100ペアのトライアルリストでEERを計算する。
2. **中級。** SpeechBrain ECAPAをVoxCeleb1の30発話（5話者×6発話）で使用する。コサイン対PLDAでEERを計算する。
3. **難しい。** `pyannote.audio` を使って完全な登録→話者分離→照合パイプラインを構築する。AMI devセットでDERを評価する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------|
| EER | メインの指標 | 誤受け入れ=誤拒否となる閾値。 |
| 照合 | 1:1 | 「これはAliceか？」 |
| 識別 | 1:N | 「誰が話しているか？」 |
| オープンセット | 未知の可能性あり | テストセットに未登録の話者が含まれる場合がある。 |
| 登録 | 参照登録 | 話者のリファレンス埋め込みを計算する。 |
| AAM-softmax | 損失関数 | 加算角度マージン付きsoftmax；クラスタ分離を強制。 |
| PLDA | 古典的スコアリング | 確率的LDA；埋め込みの上での尤度比スコアリング。 |
| DER | 話者分離指標 | 話者分離エラーレート——ミス＋偽アラーム＋混乱。 |

## 参考資料

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — 古典的な深層埋め込み論文。
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — 2020〜2026年の主要アーキテクチャ。
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — SVと話者分離のためのSSLバックボーン。
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — 本番話者分離＋埋め込みスタック。
- [VoxCelebリーダーボード（2026年更新）](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — モデル間の最新EER順位。
