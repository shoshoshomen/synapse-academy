# 音声スプーフィング対策と音声透かし — ASVspoof 5・AudioSeal・WaveVerify

> 音声クローンは防御より速く普及した。2026年の本番音声システムには2つのものが必要だ: 本物vs偽の音声を分類する検出器（AASIST、RawNet2）と、圧縮・編集後も生き残る透かし（AudioSeal）。音声クローンを出荷するならこの両方を出荷すること。そうしないなら出荷するな。


## 問題

3つの関連する防御:

1. **スプーフィング対策/ディープフェイク検出。** 音声クリップが与えられたとき、それは合成か本物か？ASVspoofベンチマーク（ASVspoof 2019 → 2021 → 5）がゴールドスタンダードだ。
2. **音声透かし。** 後で検出器が抽出できる感知不能なシグナルを生成音声に埋め込む。AudioSeal（Meta）とWavMarkがオープンな選択肢だ。
3. **認証済みプロベナンス。** 音声ファイル＋メタデータの暗号署名。C2PA / Content Authenticity Initiative。

検出は協力しない攻撃者に対処する。透かしはコンプライアンスに対処する——AI生成音声はそのように識別できるべきだ。2026年には両方が必要だ。

## 概念

![スプーフィング対策vs透かしvsプロベナンス — 3層防御](../assets/spoofing-watermark.svg)

### ASVspoof 5 — 2024〜2025年のベンチマーク

前エディションからの主な変更点:

- **クラウドソースデータ**（スタジオクリーンではない）— リアルな条件。
- **約2000話者**（以前は約100）。
- **32種の攻撃アルゴリズム。** TTS＋音声変換＋敵対的摂動。
- **2トラック。** カウンターメジャー（CM）スタンドアロン検出；バイオメトリクスシステム向けスプーフィング耐性ASV（SASV）。

ASVspoof 5での最先端: 約7.23% EER。古いASVspoof 2019 LA: 0.42% EER。実世界のデプロイ: 実際のクリップで5〜10% EERを見込む。

### AASISTとRawNet2 — 検出モデルファミリー

**AASIST**（2021年、2026年まで更新）。スペクトル特徴量に対するグラフアテンション。ASVspoof 5カウンターメジャータスクでの現SOTA。

**RawNet2。** 生波形に対する畳み込みフロントエンド＋TDNNバックボーン。シンプルなベースライン；ファインチューニングで依然として競争力がある。

**NeXt-TDNN＋SSL特徴量。** 2025年バリアント: ECAPAスタイル＋WavLM特徴量＋フォーカルロス。ASVspoof 2019 LAで0.42% EERを達成。

### AudioSeal — 2024年の透かしデフォルト

MetaのAudioSeal（2024年1月、2024年12月v0.2）。主な設計:

- **局所化。** 16 kHzサンプル解像度（1/16000秒）でフレームごとに透かしを検出する。
- **生成器＋検出器を共同学習。** 生成器は感知不能なシグナルを埋め込むことを学習；検出器は拡張を通じてそれを見つけることを学習する。
- **ロバスト。** MP3/AAC圧縮・EQ・速度シフト±10%・ノイズ混合+10 dB SNRでも生き残る。
- **高速。** 検出器は485倍リアルタイムで動作；WavMarkより1000倍高速。
- **容量。** 16ビットペイロード（モデルID・生成タイムスタンプ・ユーザーIDをエンコード可能）を各発話に埋め込める。

### WavMark

AudioSeal前のオープンベースライン。可逆ニューラルネットワーク、32ビット/秒。問題点:

- 同期のブルートフォース探索が遅い。
- ガウスノイズやMP3圧縮で除去可能。
- リアルタイムに不向き。

### WaveVerify（2025年7月）

AudioSealの弱点——特に時間的操作（逆転・速度変更）——に対処する。FiLMベースの生成器＋Mixture-of-Experts検出器を使用。標準的な攻撃ではAudioSealと競争力があり；時間的編集を処理する。

### 攻撃者が利用するギャップ

AudioMarkBenchより: 「ピッチシフト下では、すべての透かしでビット回収精度が0.6未満になり、ほぼ完全な除去を示す。」**ピッチシフトが普遍的な攻撃だ。** 2026年の透かしはいずれも積極的なピッチ変更に完全にロバストではない。これがAASISTのような検出を透かしとともに必要とする理由だ。

### C2PA / Content Authenticity Initiative

ML技術ではなく——マニフェスト形式だ。音声ファイルは作成ツール・著者・日付に関する暗号署名されたメタデータを持つ。Audobox / Seamlessが使用する。プロベナンスには良い；悪意ある行為者が再エンコードしてメタデータを削除すると何もできない。

## 実装する

### ステップ1: シンプルなスペクトル特徴量検出器（おもちゃ）

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

合成音声は異常に平坦な高周波エネルギーを持つことが多い。本番の検出器はこれではなくAASISTを使用する。しかし直感は正しい。

### ステップ2: AudioSealの埋め込みと検出

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: [0, 1] の float — 透かし存在の確率
# decoded_payload: 16ビット；埋め込まれたペイロードと照合する
```

### ステップ3: 評価 — EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### ステップ4: 本番統合

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

すべての生成に: (1) 透かし、(2) 署名済みマニフェスト、(3) 保持ポリシー準拠の監査ログを含める。

## 使ってみる

| ユースケース | 防御 |
|------------|------|
| TTSや音声クローンの出荷 | すべての出力にAudioSeal埋め込み（非交渉）|
| バイオメトリック音声ロック解除 | AASIST＋ECAPAアンサンブル；ライブネスチャレンジ |
| コールセンター詐欺検出 | 着信コールの20%サンプルにAASIST |
| ポッドキャスト真正性 | アップロード時にC2PA署名、AI生成ならAudioSeal |
| 研究/検出器学習 | ASVspoof 5学習/開発/評価セット |

## 落とし穴

- **透かしを埋めるが検出器を実行しない。** 無意味。検出器をCIに組み込むこと。
- **キャリブレーションなしの検出。** ASVspoof LAで学習したAASISTは過学習する；実世界の精度は低下する。自分のドメインでキャリブレーションすること。
- **ピッチシフトギャップ。** 積極的なピッチシフトはほとんどの透かしを除去する。検出のフォールバックを持つこと。
- **メタデータ削除・再ホスティング。** C2PAは再エンコードで容易にバイパスできる。暗号的防御＋知覚的防御（透かし）を常に一緒に追加すること。
- **ライブネスの検出への依存。** ユーザーにランダムなフレーズを言わせる。リプレイ攻撃を防ぐがリアルタイムクローンには対応できない。

## 成果物を出す

`outputs/skill-spoof-defender.md` として保存する。音声生成デプロイのための検出モデル・透かし・プロベナンスマニフェスト・運用プレイブックを選択する。

## 演習

1. **簡単。** `code/main.py` を実行する。合成音声に対するおもちゃの検出器＋おもちゃの透かし埋め込み/検出。
2. **中級。** `audioseal` をインストールし、TTS出力に16ビットペイロードを埋め込み、再デコードする。ノイズで音声を劣化させ、ビット回収精度を測定する。
3. **難しい。** RawNet2またはAASISTをASVspoof 2019 LAでファインチューニングする。EERを測定する。F5-TTSで生成されたクリップのホールドアウトセットでテストする——ドメイン外検出がどのように劣化するか確認する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------|
| ASVspoof | ベンチマーク | 隔年チャレンジ；2024年 = ASVspoof 5。 |
| CM（カウンターメジャー） | 検出器 | 分類器: 本物の音声vs合成/変換音声。 |
| SASV | 話者検証＋CM | 統合バイオメトリクス＋スプーフィング検出。 |
| AudioSeal | Metaの透かし | 局所化、16ビットペイロード、WavMarkより485倍高速。 |
| ビット回収精度 | 透かしの生存 | 攻撃後に回収されたペイロードビットの割合。 |
| C2PA | プロベナンスマニフェスト | 作成/著作に関する暗号メタデータ。 |
| AASIST | 検出器ファミリー | グラフアテンションベースのスプーフィング対策SOTA。 |

## 参考資料

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — 現在のベンチマーク。
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — 透かしのデフォルト。
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — 時間的攻撃のためのMoE検出器。
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — SOTA検出バックボーン。
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — ロバスト性評価。
- [C2PA仕様](https://c2pa.org/specifications/specifications/) — プロベナンスマニフェスト形式。
