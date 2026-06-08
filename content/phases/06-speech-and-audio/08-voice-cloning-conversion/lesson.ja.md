# 音声クローンと音声変換

> 音声クローンはあなたのテキストを他の人の声で読み上げる。音声変換はあなたの声を他の人の声に書き換えながら、言っている内容を保持する。どちらも同じ分解に依存している: 話者のアイデンティティをコンテンツから分離する。


## 問題

2026年、5秒の音声クリップがあれば、コンシューマーGPUで誰でも高品質な声のクローンを作れる。ElevenLabs・F5-TTS・OpenVoice v2・VoiceBoxはすべてゼロショットまたは数ショットのクローンを提供している。この技術は恩恵（アクセシビリティTTS・吹き替え・支援ボイス）であり、武器（詐欺電話・政治的ディープフェイク・知的財産の盗用）でもある。

密接に関連する2つのタスク:

- **音声クローン（TTS側）:** テキスト＋5秒のリファレンス音声 → その声での音声。
- **音声変換（音声側）:** ソース音声（人物AがXを言う）＋人物Bのリファレンス音声 → BがXを言う音声。

どちらも波形を（コンテンツ、話者、プロソディ）に因数分解し、一方のソースのコンテンツと他方の話者を組み合わせる。

2026年に遵守しなければならない制約: **EUのAI法（2026年8月施行）とカリフォルニア州AB 2905（2025年施行）では電子透かしと同意ゲートが法的に義務付けられている**。パイプラインは不可聴の電子透かしを出力し、非同意のクローンを拒否しなければならない。

## 概念

![音声クローン対音声変換: 因数分解・話者の交換・再結合](../assets/voice-cloning.svg)

**ゼロショットクローン。** 数千の話者で学習されたモデルに5秒のクリップを渡す。話者エンコーダがクリップを話者埋め込みにマッピングし、TTSデコーダがその埋め込みとテキストを条件に生成する。

使用例: F5-TTS（2024年）、YourTTS（2022年）、XTTS v2（2024年）、OpenVoice v2（2024年）。

**数ショットのファインチューニング。** 対象の声を5〜30分録音する。ベースモデルをLoRAで1時間ファインチューニングする。品質は「まあまあ」から「区別がつかない」レベルに跳ね上がる。CoquiとElevenLabsはどちらもこのパターンをサポートしており、コミュニティはF5-TTSで使用している。

**音声変換（VC）。** 2つのファミリー:

- **認識-合成。** ASR的なモデルを実行してコンテンツ表現を抽出し（例: ソフト音素ポステリオール、PPG）、ターゲット話者埋め込みで再合成する。言語とアクセントに対してロバスト。KNN-VC（2023年）、Diff-HierVC（2023年）が使用している。
- **ディスエンタングルメント。** ボトルネックでコンテンツ・話者・プロソディを潜在空間で分離するオートエンコーダを学習する。推論時に話者埋め込みを交換する。品質は低いが高速。AutoVC（2019年）、VITS-VCバリアントが使用している。

**神経コーデックベースのクローン（2024年以降）。** VALL-E・VALL-E 2・NaturalSpeech 3・VoiceBox——音声をSoundStream / EnCodecからの離散トークンとして扱い、コーデックトークンに対して大きな自己回帰またはフローマッチングモデルを学習する。短いプロンプトでElevenLabs相当の品質。

### 倫理は後付けでない

**電子透かし。** PerTh（Perth）とSilentCipher（2024年）は不可聴な形で約16〜32ビットのIDを埋め込む。再エンコード・ストリーミング・一般的な編集を経ても生き残る。本番対応のオープンソース。

**同意ゲート。** すべてのクローン出力に検証可能な同意記録をペアリングしなければならない。「私、Rohitは2026-04-22に、この声をX目的に使用することを許可します」。改ざん防止ログに保存する。

**検出。** AASIST・RawNet2・Wav2Vec2-AASISTが検出器として提供されている。ASVspoof 2025チャレンジでは、ElevenLabs・VALL-E 2・Bark出力に対する最先端の検出器のEERは0.8〜2.3%だった。

### 数値（2026年）

| モデル | ゼロショット？ | SECS（ターゲット類似度） | WER（理解度） | パラメータ数 |
|-------|-------------|----------------------|------------|------------|
| F5-TTS | はい | 0.72 | 2.1% | 3.35億 |
| XTTS v2 | はい | 0.65 | 3.5% | 4.70億 |
| OpenVoice v2 | はい | 0.70 | 2.8% | 2.20億 |
| VALL-E 2 | はい | 0.77 | 2.4% | 3.70億 |
| VoiceBox | はい | 0.78 | 2.1% | 3.30億 |

SECS > 0.70はほとんどのリスナーにとってターゲットと区別がつかないのが一般的だ。

## 実装する

### ステップ1: 認識-合成で分解する（main.pyのコードのみのデモ）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念的にはシンプル；実装の大部分は `tts_model` と話者エンコーダにある。

### ステップ2: F5-TTSでゼロショットクローン

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

リファレンストランスクリプトは音声と完全に一致しなければならない；不一致はアライメントを壊す。

### ステップ3: KNN-VCで音声変換

```python
import torch
from knnvc import KNNVC  # 2023年モデル, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VCはWavLMを実行してソースとターゲットプールのフレームごとの埋め込みを抽出し、各ソースフレームをプール内の最近傍に置き換える。ノンパラメトリックで、1分のターゲット音声で機能する。

### ステップ4: 電子透かしを埋め込む

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # ペイロードバイトを返す
```

約32ビットのペイロード、MP3再エンコードと軽いノイズ後も検出可能。

### ステップ5: 同意ゲート

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "署名された同意が必要"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 使ってみる

2026年のスタック:

| 状況 | 選択 |
|------|------|
| 5秒ゼロショットクローン、オープンソース | F5-TTSまたはOpenVoice v2 |
| 商業本番クローン | ElevenLabs Instant Voice Clone v2.5 |
| 音声変換（書き換え） | KNN-VCまたはDiff-HierVC |
| 多話者ファインチューニング | StyleTTS 2＋話者アダプタ |
| クロス言語クローン | XTTS v2またはVALL-E X |
| ディープフェイク検出 | Wav2Vec2-AASIST |

## 落とし穴

- **リファレンストランスクリプトの不一致。** F5-TTSなどはリファレンステキストがリファレンス音声と句読点まで完全に一致することを要求する。
- **残響のあるリファレンス。** エコーがクローンを破壊する。ドライに、マイクに近い状態で録音すること。
- **感情の不一致。** 「陽気な」リファレンスで学習するとすべてのクローンが陽気になる。リファレンスの感情をターゲットの用途に合わせること。
- **言語の漏れ。** 英語話者をクローンしてフランス語を話させるとアクセントが残ることが多い；クロス言語モデル（XTTS、VALL-E X）を使うこと。
- **電子透かしなし。** 2026年8月からEUでは出荷不可。

## 成果物を出す

`outputs/skill-voice-cloner.md` として保存する。同意ゲート＋電子透かし＋品質目標を含むクローンまたは変換パイプラインを設計する。

## 演習

1. **簡単。** `code/main.py` を実行する。交換前後で2つの「話者」のコサインを計算することで話者埋め込みの交換を実証する。
2. **中級。** OpenVoice v2を使って自分の声をクローンする。リファレンスとクローンのSECSを測定する。Whisper経由でCERを測定する。
3. **難しい。** SilentCipherの電子透かしを20個のクローンに適用し、128 kbpsのMP3エンコード＋デコードを通してペイロードを検出する。ビット回収精度を報告する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------|
| ゼロショットクローン | 5秒あれば十分 | 事前学習済みモデル＋話者埋め込み；学習不要。 |
| PPG | 音素ポステリオーグラム | 言語に依存しないコンテンツ表現として使われるフレームごとのASRポステリオール。 |
| KNN-VC | 最近傍変換 | 各ソースフレームをターゲットプールの最近傍フレームで置き換える。 |
| 神経コーデックTTS | VALL-Eスタイル | EnCodec/SoundStreamトークンに対するARモデル。 |
| 電子透かし | 不可聴の署名 | 再エンコード後も生き残るビット列を音声に埋め込む。 |
| SECS | クローンの忠実度 | ターゲットとクローンの話者埋め込み間のコサイン。 |
| AASIST | ディープフェイク検出器 | グラフアテンションベースのアンチスプーフィングモデル；合成音声を検出。 |

## 参考資料

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — オープンソースSOTAのゼロショットクローン。
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) および [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — 神経コーデックTTS。
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — ディスエンタングルメントベースの音声変換。
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — 検索ベースのVC。
- [SilentCipher (2024) — 音声電子透かし](https://github.com/sony/silentcipher) — 本番対応の32ビット音声電子透かし。
- [ASVspoof 2025の結果](https://www.asvspoof.org/) — 検出器vs合成器の軍拡競争、2026年更新。
