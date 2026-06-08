# ストリーミング音声対音声 — Moshi・Hibiki・フルデュプレックス対話

> 2024〜2026年は音声AIを再定義した。Moshiは200 msの遅延で同時に聞きながら話す単一モデルを出荷した。HibikiはS2S翻訳をチャンクごとに行う。どちらもASR → LLM → TTSパイプラインを統合されたフルデュプレックスアーキテクチャ（Mimiコーデックトークン上）に置き換えた。これが新しいリファレンス設計だ。


## 問題

レッスン11＋12で構築されたすべての音声エージェントには、VAD発火・STT処理・LLM推論・TTS生成という根本的な遅延の下限（300〜500 ms）がある。各ステージには独自の最小遅延がある。チューニングと並列化はできるが、パイプラインの形が上限を決める。

Moshi（Kyutai、2024〜2026年）は異なる問いを立てる: パイプラインがなかったらどうなるか？ 1つのモデルが音声を継続的に受け取って音声を出力し、テキストを必須ステージではなく中間的な「内なる独り言」として扱うとしたら？

答えは**フルデュプレックス音声対音声**だ。理論的遅延は160 ms（Mimiフレーム80 ms＋アコースティック遅延80 ms）。単一L4 GPUで実用的な遅延は200 ms。これはベストインクラスのパイプライン型音声エージェントの半分だ。

## 概念

![Moshiアーキテクチャ: 2つの並列Mimiストリーム＋内なる独り言テキスト](../assets/moshi-hibiki.svg)

### Moshiアーキテクチャ

**入力。** 2つのMimiコーデックストリーム、どちらも12.5 Hz × 8コードブック:

- ストリーム1: ユーザー音声（Mimi符号化、常に到着）
- ストリーム2: Moshi自身の音声（Moshiが生成）

**Transformer。** 70億パラメータのTemporal Transformerが両ストリームとテキスト「内なる独り言」ストリームを処理する。80 msステップごとに:

1. 最新のユーザーMimiトークン（8コードブック）を消費する。
2. 直前のMoshi Mimiトークン（8コードブック、生成済みの分）を消費する。
3. 次のMoshiテキストトークン（内なる独り言）を生成する。
4. 次のMoshi Mimiトークン（Depth Transformerを介して8コードブック）を生成する。

3つのストリーム——ユーザー音声・Moshi音声・Moshiテキスト——が並列で動作する。Moshiは話しながらユーザーを聞ける；ユーザーが割り込んだら自分で中断できる；主な発話を中断せずにバックチャネル（「うんうん」）できる。

**Depth Transformer。** フレーム内では、8つのコードブックは並列に予測されない——コードブック間の依存関係がある。小型の2層「depth transformer」が80 ms以内で順番に予測する。これはAR コーデックLMの標準的な因数分解だ（VALL-E・VibeVoiceでも使用）。

### 内なる独り言テキストが役立つ理由

明示的なテキストなしでは、モデルはアコースティックストリームで言語を暗黙的にモデル化する必要がある。Moshiの洞察: 音声と並んでテキストトークンを出力するように強制する。テキストストリームは本質的にMoshiが言っていることのトランスクリプトだ。これにより意味的一貫性が向上し、言語モデルヘッドの交換が容易になり、トランスクリプトが無料で得られる。

### Hibiki: ストリーミング音声対音声翻訳

同じアーキテクチャで翻訳ペアを学習したもの。ソース音声を入力し、ターゲット言語の音声を継続的に出力する。Hibiki-Zero（2026年2月）は単語レベルでアラインされた学習データを不要にする——文レベルのデータ＋遅延最適化のためのGRPO強化学習を使用。

当初4言語ペアをサポート；約1000時間で新しい言語に適応できる。

### Kyutaiスタック全体（2026年）

- **Moshi** — フルデュプレックス対話（フランス語優先、英語は良好）
- **Hibiki / Hibiki-Zero** — 同時音声翻訳
- **Kyutai STT** — ストリーミングASR（500 msまたは2.5秒のルックアヘッド）
- **Kyutai Pocket TTS** — CPUで動作する1億パラメータTTS（2026年1月）
- **Unmute** — これらを組み合わせた公開サーバー上の完全パイプライン

L40S GPUでのスループット: 3倍リアルタイムで64セッションの同時処理。

### Sesame CSM — 近縁モデル

Sesame CSM（2025年）は同様のアイデアを使用——Mimiコーデックヘッドを持つLlama-3バックボーン。しかしCSMは単方向（コンテキスト＋テキストを受け取り音声を生成）であり、Moshiのフルデュプレックス機能とは異なる。市場で最高の「音声プレゼンス」TTSだ；Moshiのフルデュプレックス能力とは別物だ。

### 2026年の性能数値

| モデル | 遅延 | ユースケース | ライセンス |
|-------|------|------------|---------|
| Moshi | 200 ms（L4） | フルデュプレックス英語/フランス語対話 | CC-BY 4.0 |
| Hibiki | 12.5 Hzフレームレート | フランス語↔英語ストリーミング翻訳 | CC-BY 4.0 |
| Hibiki-Zero | 同じ | 5言語ペア、アラインデータ不要 | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | コンテキスト条件付きTTS | Apache-2.0 |
| GPT-4o Realtime | 約300 ms | クローズド、OpenAI API | 商用 |
| Gemini 2.5 Live | 約350 ms | クローズド、Google API | 商用 |

## 実装する

### ステップ1: インターフェース

MoshiはWebSocketサーバーを公開し、Mimi符号化音声の80 msチャンクを受け取り、Mimi符号化音声の80 msチャンクを返す。双方向に。継続的に。

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### ステップ2: フルデュプレックスループ

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

両方向が同時に動作する。Python asyncioまたはRust futuresが標準的なトランスポートだ。

### ステップ3: 学習目標（概念的）

各80 msフレーム `t` に対して:

- 入力: `user_mimi[0..t]`、`moshi_mimi[0..t-1]`、`moshi_text[0..t-1]`
- 予測: `moshi_text[t]`、次に `moshi_mimi[t, codebook_0..7]`

テキストは音声より先に予測される（内なる独り言）；音声はdepth transformer内でコードブック順に予測される。

### ステップ4: Moshiが勝る場面と勝らない場面

Moshiが勝る場面:

- 安価なハードウェアでエンドツーエンド250 ms未満。
- 自然なバックチャネルと中断。
- パイプラインのグルーコードなし。

Moshiが勝らない場面:

- ツール呼び出し（学習されていない；別のLLMパスが必要）。
- 長い推論（MoshiはClaude/GPT-4ではなく約80億パラメータの対話モデル）。
- ニッチなトピックでの事実精度。
- ほとんどの本番エンタープライズユースケース（2026年でもパイプラインを使用）。

## 使ってみる

| 状況 | 選択 |
|------|------|
| 最低遅延の音声コンパニオン | Moshi |
| ライブ翻訳コール | Hibiki |
| 音声デモ/研究 | Moshi、CSM |
| ツール付きエンタープライズエージェント | パイプライン（レッスン12）、Moshiではない |
| コンテキスト内カスタム音声TTS | Sesame CSM |
| 任意言語の音声対音声 | GPT-4o RealtimeまたはGemini 2.5 Live（商用） |

## 落とし穴

- **限定的なツール呼び出し。** Moshiは対話モデルであり、エージェントフレームワークではない。ツールにはパイプラインと組み合わせること。
- **特定の音声コンディショニング。** Moshiは学習済みの単一ペルソナを使用；クローニングは別の学習実行が必要。
- **言語カバレッジ。** フランス語＋英語は優秀；他は限定的。Hibiki-Zeroは役立つが、学習データは依然として必要。
- **リソースコスト。** Moshiのフルセッションはスロット全体を占有する；安価な共有テナントデプロイのパターンではない。

## 成果物を出す

`outputs/skill-duplex-pipeline.md` として保存する。理由を添えて、音声エージェントのワークロードに対してパイプラインかフルデュプレックスアーキテクチャかを選択する。

## 演習

1. **簡単。** `code/main.py` を実行する。2ストリーム＋内なる独り言アーキテクチャをシンボリックにシミュレートする。
2. **中級。** HuggingFaceからMoshiをプルし、サーバーを起動して1回会話をテストする。ユーザー発話終了からMoshi応答開始までのウォールクロック遅延を測定する。
3. **難しい。** レッスン12のパイプラインエージェントと、20の一致したテスト発話でMoshiのP50遅延を比較する。パイプラインがアーキテクチャ的に依然として勝る場合をまとめる。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------|
| フルデュプレックス | 同時に聞いて話す | 同じモデルで2つの音声ストリームが同時に動作する。 |
| 内なる独り言 | モデルのテキストストリーム | Moshiが音声出力と並んでテキストトークンを出力する。 |
| Depth Transformer | コードブック間予測器 | 80 msフレーム内で8コードブックを予測する小型Transformer。 |
| Mimi | KyutaiのコーデックMoshi | 12.5 Hz × 8コードブック；セマンティック+アコースティック；Moshiを支える。 |
| ストリーミングS2S | リアルタイム音声→音声 | パイプラインステージなしのチャンクごとの翻訳/対話。 |
| バックチャネル | 「うんうん」の反応 | Moshiはターンを中断せず小さな確認音を出せる。 |

## 参考資料

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — 論文。
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — アラインデータなしのストリーミング翻訳。
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — CSM仕様。
- [Kyutai — Moshiリポジトリ](https://github.com/kyutai-labs/moshi) — インストール＋サーバー。
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — クローズドな商用の同等品。
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — 内部のSTT/TTSフレームワーク。
