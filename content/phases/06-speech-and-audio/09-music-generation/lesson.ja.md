# 音楽生成 — MusicGen・Stable Audio・Suno、そしてライセンス大地震

> 2026年の音楽生成: Suno v5とUdio v4が商業面で圧倒；MusicGen・Stable Audio Open・ACE-Stepがオープンソースをリードする。技術的な問題はほぼ解決されている。法的な問題（Warner Music 5億ドルの和解、UMGとの和解）が2025〜2026年にこの分野を再形成した。


## 問題

テキスト→30秒から4分の音楽クリップ、歌詞・ボーカル・構成付き。3つのサブ問題:

1. **インストゥルメンタル生成。** 「lo-fiヒップホップドラムとウォームなキーズ」→音声。MusicGen・Stable Audio・AudioLDM。
2. **歌生成（ボーカル＋歌詞付き）。** 「雨のテキサスの夜についてのカントリーソング」→完全な歌。Suno・Udio・YuE・ACE-Step。
3. **条件付き / コントロール可能。** 既存クリップの拡張、ブリッジの再生成、ジャンル変更、ステム分離、インペインティング。Udiaのインペインティング＋ステム分離が2026年に追いつくべき機能だ。

## 概念

![音楽生成: トークンLM対拡散、2026年のモデルマップ](../assets/music-generation.svg)

### 神経コーデックトークンに対するトークンLM

Metaの**MusicGen**（2023年、MIT）と多くの派生モデル: テキスト/メロディー埋め込みを条件に、EnCodecトークン（32 kHz、4つのコードブック）を自己回帰的に予測し、EnCodecでデコードする。3〜33億パラメータ。強力なベースライン；30秒を超えると苦しむ。

**ACE-Step**（オープンソース、2026年4月リリースの40億XL）は完全な歌の歌詞条件付き生成のためにこれを拡張する。Sunoに最も近いオープンコミュニティの存在だ。

### MelまたはLatentに対する拡散

**Stable Audio（2023年）**と**Stable Audio Open（2024年）**: 圧縮音声に対するLatent拡散。ループ・音響デザイン・アンビエントテクスチャに優れる。構造化された完全な歌は苦手だ。

**AudioLDM / AudioLDM2**: T2Iスタイルのlatent拡散でテキストから音声、音楽・効果音・スピーチに汎化。

### ハイブリッド（本番）— Suno・Udio・Lyria

クローズドウェイト。おそらくARコーデックLM＋拡散ベースのボコーダと専用の声/ドラム/メロディーヘッド。Suno v5（2026年）はELO 1293で品質リーダー。Udio v4はインペインティング＋ステム分離（ベース・ドラム・ボーカルを個別ダウンロード）を追加する。

### 評価

- **FAD（フレシェ音声距離）。** VGGishまたはPANNsの特徴量を使った生成音声と実音声の分布間の埋め込みレベルの距離。低いほど良い。MusicGen small: MusicCapsで4.5 FAD；SOTA約3.0。
- **音楽性（主観的）。** 人間の選好。Suno v5 ELO 1293がリード。
- **テキスト-音声アライメント。** プロンプトと出力のCLAPスコア。
- **音楽性のアーティファクト。** オフビートの遷移・ボーカルフレーズのドリフト・30秒を超えた構成の崩壊。

## 2026年のモデルマップ

| モデル | パラメータ数 | 長さ | ボーカル | ライセンス |
|-------|------------|------|---------|----------|
| MusicGen-large | 33億 | 30秒 | なし | MIT |
| Stable Audio Open | 12億 | 47秒 | なし | Stability非商用 |
| ACE-Step XL（2026年4月） | 40億 | 2分以上 | あり | Apache-2.0 |
| YuE | 70億 | 2分以上 | あり、多言語 | Apache-2.0 |
| Suno v5（クローズド） | ? | 4分 | あり、ELO 1293 | 商用 |
| Udio v4（クローズド） | ? | 4分 | あり＋ステム | 商用 |
| Google Lyria 3（クローズド） | ? | リアルタイム | あり | 商用 |
| MiniMax Music 2.5 | ? | 4分 | あり | 商用API |

## 法的状況（2025〜2026年）

- **Warner MusicとSunoの和解。** 5億ドル。WMGはSuno上でのAIの肖像・音楽権・ユーザー生成トラックの監視権を持つ。Udiaに対するUMGとの同様の和解も。
- **EU AI法**＋**カリフォルニア州SB 942**: AI生成音楽は開示が必要。
- **RiffusionとMusicGen**はMITライセンスでコンプライアンスの問題がない一方、商用ボーカルもない。

安全に出荷できるパターン:

1. インストゥルメンタルのみを生成する（MusicGen・Stable Audio Open・MIT/CC0の出力）。
2. 商用APIを使用する（Suno・Udio・ElevenLabs Music）で1生成ごとのライセンス。
3. 所有または許諾されたカタログで学習する（ほとんどの企業はここに行き着く）。
4. 電子透かし＋メタデータで生成物をタグ付けする。

## 実装する

### ステップ1: MusicGenで生成する

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

3サイズ: `small`（3億、高速）、`medium`（15億）、`large`（33億）。Smallは「アイデアが機能するか」を確認するのに十分だ。

### ステップ2: メロディー条件付け

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melodyはクロマグラムを受け取り、音色を変えながらメロディーを保持する。「このメロディーを弦楽四重奏で」といった用途に便利だ。

### ステップ3: FAD評価

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

VGGish埋め込みの距離を計算する。ジャンルレベルの回帰テストに便利；人間のリスナーの代わりにはならない。

### ステップ4: LLM-音楽ワークフローに追加する

レッスン7〜8のアイデアと組み合わせる:

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 使ってみる

| 目標 | スタック |
|------|---------|
| インストゥルメンタルサウンドデザイン | Stable Audio Open |
| ゲーム/アダプティブ音楽 | Google Lyria RealTime（クローズド） |
| ボーカル付き完全な歌（商用） | 明示的なライセンスありのSuno v5またはUdio v4 |
| ボーカル付き完全な歌（オープン） | ACE-Step XLまたはYuE |
| 短い広告ジングル | 口ずさんだリファレンスにメロディー条件付けしたMusicGen |
| ミュージックビデオの背景音楽 | MusicGen＋Stable Video Diffusion |

## 2026年でも起きる落とし穴

- **著作権ロンダリングのプロンプト。** 「Taylor Swiftのスタイルの歌」——商用のSuno/Udiaは今はフィルタリングするが、オープンモデルはしない。独自のフィルターリストを追加すること。
- **30秒を超えた繰り返し/ドリフト。** ARモデルはループする。複数の生成をクロスフェード、または構造的な一貫性のためにACE-Stepを使用すること。
- **テンポのドリフト。** モデルはBPMからずれていく。プロンプトにBPMタグを使用し、librosaの `beat_track` でポストフィルタリングすること。
- **ボーカルの明瞭度。** Sunoは優れている；オープンモデルは歌詞がしばしばぼやけている。歌詞が重要なら商用APIを使用するかファインチューニングすること。
- **モノラル出力。** オープンモデルはモノラルまたは偽ステレオを生成する。適切なステレオ再構成でアップグレードすること（ezst・Cartesiaのステレオ拡散）。

## 成果物を出す

`outputs/skill-music-designer.md` として保存する。音楽生成デプロイのためにモデル・ライセンス戦略・長さ/構成計画・開示メタデータを選択する。

## 演習

1. **簡単。** `code/main.py` を実行する。ASCII記号として「生成的な」コード進行＋ドラムパターンを生成する——音楽生成の漫画版だ。MIDIレンダラーがあれば再生できる。
2. **中級。** `audiocraft` をインストールし、MusicGen-smallで4つのジャンルプロンプトにわたって10秒のクリップを生成し、リファレンスジャンルセットに対してFADを測定する。
3. **難しい。** ACE-Step（またはMusicGen-melody）を使って同じメロディーの3つのバリエーションを異なる音色プロンプトで生成する。プロンプトとのアライメントを確認するためにCLAP類似度を計算する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------|
| FAD | 音声のFID | 実音声と生成音声の埋め込み分布間のフレシェ距離。 |
| クロマグラム | ピッチとしてのメロディー | フレームごとの12次元ベクトル；メロディー条件付けの入力。 |
| ステム | 楽器トラック | 分離されたベース/ドラム/ボーカル/メロディーのWAVファイル。 |
| インペインティング | セクションの再生成 | 時間窓をマスク；モデルがそこだけ再生成する。 |
| CLAP | テキスト-音声CLIP | 対照的な音声-テキスト埋め込み；テキスト-音声アライメントを評価。 |
| EnCodec | 音楽コーデック | MusicGenが使用するMetaの神経コーデック；32 kHz、4コードブック。 |

## 参考資料

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — オープンな自己回帰ベンチマーク。
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — サウンドデザインのデフォルト。
- [ACE-Step](https://github.com/ace-step/ACE-Step) — オープンな40億パラメータの完全な歌生成器、2026年4月。
- [Suno v5のプラットフォームドキュメント](https://suno.com) — 商用品質リーダー。
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — 音楽と効果音のためのlatent拡散。
- [WMG-Suno和解の報道](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — 2025年11月の先例。
