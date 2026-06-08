# 音声言語モデル — Qwen2.5-Omni・Audio Flamingo・GPT-4o Audio

> 2026年の音声言語モデルはスピーチ＋環境音＋音楽を横断して推論する。Qwen2.5-Omni-7BはMMau-ProでGPT-4o Audioに匹敵する。Audio Flamingo NextはLongAudioBenchでGemini 2.5 Proを上回る。オープンとクローズドの差は実質的に解消された——ただし、全員がランダムに近いスコアを出すマルチ音声タスクを除いて。


## 問題

5秒の音声がある: 犬が吠え、誰かが「stop!（止まれ！）」と叫び、次に沈黙。役立つ問いは複数の軸にわたる:

- **書き起こし。** 「何が言われたか？」——ASRの領域。
- **意味的な推論。** 「その人は危険にさらされているか？」——吠え声＋叫び＋沈黙の共同理解が必要。
- **音楽の推論。** 「メロディーを演奏している楽器は何か？」
- **長い音声の検索。** 「この90分の講義のどこで講師が勾配降下法を説明したか？」

1つのプロンプトでこれらすべてに答える単一モデルが**音声言語モデル**（LALM / ALM）だ。純粋なASRとは別: LALMはトランスクリプトだけでなく自由形式の自然言語で回答を生成する。

## 概念

![音声言語モデル: 音声エンコーダ＋プロジェクタ＋LLMデコーダ](../assets/alm-architecture.svg)

### 3コンポーネントテンプレート

2026年のすべてのLALMは同じスケルトンを持つ:

1. **音声エンコーダ。** Whisperエンコーダ・BEATs・CLAP・WavLM・またはモデルごとのカスタムエンコーダ。
2. **プロジェクタ。** 音声エンコーダの特徴量をLLMのトークン埋め込み空間にブリッジする線形またはMLP。
3. **LLM。** Llama / Qwen / Gemmaベースのデコーダ。インターリーブされたテキスト＋音声トークンを受け取り；テキストを生成する。

学習:

- **ステージ1。** エンコーダ＋LLMを固定；ASR / キャプショニングデータでプロジェクタのみを学習する。
- **ステージ2。** 指示追従の音声タスク（QA・推論・音楽理解）でフル / LoRAファインチューニング。
- **ステージ3（オプション）。** 音声入力/音声出力で音声デコーダを追加する。Qwen2.5-OmniとAF3-Chatがこれを行う。

### 2026年のモデルマップ

| モデル | バックボーン | 音声エンコーダ | 出力モダリティ | アクセス |
|-------|------------|-------------|-------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | カスタム＋Whisper | テキスト＋音声 | Apache-2.0 |
| Qwen3-Omni | Qwen3 | カスタム | テキスト＋音声 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | テキスト | NVIDIA非商用 |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | テキスト | NVIDIA非商用 |
| SALMONN | Vicuna | Whisper＋BEATs | テキスト | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | テキスト | Apache-2.0 |
| GAMA | Llama | AST＋Q-Former | テキスト | Apache-2.0 |
| Gemini 2.5 Flash/Pro（クローズド） | Gemini | 独自 | テキスト＋音声 | API |
| GPT-4o Audio（クローズド） | GPT-4o | 独自 | テキスト＋音声 | API |

### ベンチマークの実態（2026年）

**MMAU-Pro。** スピーチ / サウンド / 音楽 / 混合をカバーする1800個のQAペア。マルチ音声サブセット含む。

| モデル | 全体 | スピーチ | サウンド | 音楽 | マルチ音声 |
|-------|------|---------|--------|------|----------|
| Gemini 2.5 Pro | 約60% | 73.4% | 51.9% | 64.9% | 約22% |
| Gemini 2.5 Flash | 約57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | 約20% |
| Audio Flamingo 3 | 約54% | — | — | — | — |
| Audio Flamingo Next | LongAudioBenchでSOTA | — | — | — | — |

**マルチ音声の列は誰にとっても打撃的だ。** 4択の多肢選択でランダムは25%；ほとんどのモデルはそのあたりのスコアだ。LALMは依然として2つのクリップを比較するのに苦労している。

### 2026年にLALMが役立つ場面

- **コールセンター録音のコンプライアンス監査。** 「エージェントは必要な開示を言及したか？」
- **アクセシビリティ。** 聴覚障害のあるユーザーへのサウンドイベントの説明（単なる書き起こしではない）。
- **コンテンツモデレーション。** 暴力的な言語＋脅迫的なトーン＋背景文脈の検出。
- **ポッドキャスト / 会議のチャプタリング。** 話者ターンだけでなく意味的なサマリ。
- **音楽カタログの分析。** 「Bセクションで転調するすべてのトラックを見つける。」

### まだ役立たない場面

- 細かい音楽理論（コードレベル以下）。
- 長い会話での話者帰属推論（10分を超えると劣化）。
- マルチ音声の比較（22〜26%はほぼランダム）。
- リアルタイムストリーミング推論（ほとんどはオフラインのバッチ推論）。

## 実装する

### ステップ1: Qwen2.5-Omniに問い合わせる

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### ステップ2: プロジェクタパターン

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

これだけだ。プロジェクタは通常1〜3つの線形レイヤーだ。ASRペア（音声→トランスクリプト）で学習するのがステージ1の事前学習タスクだ。

### ステップ3: MMAU / LongAudioBenchのベンチマーク

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

カテゴリごとに（スピーチ / サウンド / 音楽 / マルチ音声）別々に報告する。集計数値はモデルがどこで失敗するかを隠す。

## 使ってみる

| タスク | 2026年の選択 |
|--------|------------|
| 自由形式の音声QA（オープン） | Qwen2.5-Omni-7B |
| 長い音声で最良のオープン | Audio Flamingo Next |
| 最良のクローズド | Gemini 2.5 Pro |
| 音声入力/音声出力エージェント | Qwen2.5-OmniまたはGPT-4o Audio |
| 音楽推論 | Audio Flamingo 3または2（音楽専門のAF-CLAP） |
| コールセンター監査 | ポリシードキュメントに対するRAGを伴うGemini 2.5 Pro（API経由） |

## 落とし穴

- **マルチ音声への過信。** タスクに「どのクリップにXがあるか」が必要なら、ランダムレベルのパフォーマンスは現実だ。
- **長い音声の劣化。** 10分を過ぎると、ほとんどのモデルの話者帰属が壊れる。先に話者分離（レッスン6）してからサマリを作成すること。
- **無音での幻覚。** WhisperエンコーダーをInheritするLALMでも同じWhisperスタイルの問題が発生する。VADゲーティングを行うこと。
- **ベンチマークのチェリーピッキング。** ベンダーのブログ記事は最良のカテゴリを強調する。自分でMMau-Proのマルチ音声サブセットを実行すること。

## 成果物を出す

`outputs/skill-alm-picker.md` として保存する。特定の音声理解タスクに対してLALM＋ベンチマークサブセット＋出力モダリティ（テキスト対音声）を選択する。

## 演習

1. **簡単。** `code/main.py` を実行して、(音声埋め込み、テキストトークン)→出力トークンのおもちゃのプロジェクタパターン＋偽のLALMルーティングを見る。
2. **中級。** Qwen2.5-Omni-7BをMMau-Proの100個のスピーチ項目でスコアリングする。論文で報告された数値と比較する。
3. **難しい。** 最小限の音声キャプショニングベースラインを構築する: BEATsエンコーダ＋2レイヤープロジェクタ＋固定Llama-3.2-1B。AudioCapsでプロジェクタのみをファインチューニングする。Clotho-AQAでSALMONNと比較する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------|
| LALM | 音声版ChatGPT | 音声エンコーダ＋プロジェクタ＋LLMデコーダ。 |
| プロジェクタ | アダプタ | 音声特徴量をLLM埋め込み空間にマッピングする小さなMLP。 |
| MMAU | ベンチマーク | スピーチ・サウンド・音楽にわたる1万件の音声QAペア。 |
| MMAU-Pro | より難しいMMAU | 1800件のマルチ音声/推論重視の問題。 |
| LongAudioBench | 長時間評価 | 意味的なクエリを含む数分のクリップ。 |
| 音声入力/音声出力 | 音声ネイティブ | テキストを経由せずモデルが音声を取り込んで音声を出力する。 |

## 参考資料

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — リファレンスアーキテクチャ。
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — 音声入力音声出力。
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — オープンな長時間音声リーダー。
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA。
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — デュアルエンコーダの先駆者。
- [MMAU-Proリーダーボード](https://mmaubenchmark.github.io/) — ライブ2026年ランキング。
