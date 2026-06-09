# 音声エージェント: Pipecat と LiveKit

> 音声エージェントは 2026 年に第一級の本番カテゴリです。Pipecat は Python のフレームベースパイプライン（VAD → STT → LLM → TTS → トランスポート）を提供します。LiveKit Agents は WebRTC 経由で AI モデルをユーザーにブリッジします。本番のレイテンシターゲットはプレミアムスタックで 450〜600ms エンドツーエンドです。


## 学習目標

- Pipecat のフレームベースパイプラインを説明できる: DOWNSTREAM（ソース→シンク）と UPSTREAM（コントロール）。
- 標準的な音声パイプラインのステージと Pipecat がサポートするトランスポートを説明できる。
- LiveKit Agents の2つの音声エージェントクラス（MultimodalAgent、VoicePipelineAgent）とそれぞれの適した場面を説明できる。
- 2026 年の本番レイテンシ要件とそれがアーキテクチャの選択にどう影響するかを要約できる。

## 問題設定

音声エージェントは TTS を追加したテキストループではありません。レイテンシ予算は厳しく（約 600ms）、部分的な音声がデフォルトで、ターン検出はモデルであり、トランスポートは電話 SIP から WebRTC まで多岐にわたります。フレームベースパイプライン（Pipecat）を構築するか、プラットフォーム（LiveKit）を活用するかのどちらかです。

## コンセプト

### Pipecat（pipecat-ai/pipecat）

- Python のフレームベースパイプラインフレームワーク。
- `Frame` → `FrameProcessor` チェーン。
- 2つのフロー方向:
  - **DOWNSTREAM** — ソース → シンク（音声入力、TTS 出力）。
  - **UPSTREAM** — フィードバックとコントロール（キャンセル、メトリクス、バージインターラプト）。
- `PipelineTask` はイベント（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）とメトリクス/トレーシング/RTVI 用のオブザーバーでライフサイクルを管理します。

典型的なパイプライン:

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

トランスポート: Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows は構造化会話（ステートマシン）を追加します。Pipecat Cloud はマネージドランタイムです。

### LiveKit Agents（livekit/agents）

- WebRTC 経由で AI モデルをユーザーにブリッジします。
- 主要コンセプト: `Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 2つの音声エージェントクラス:
  - **MultimodalAgent** — OpenAI Realtime またはそれに相当するものを介した直接音声。
  - **VoicePipelineAgent** — STT → LLM → TTS カスケード。テキストレベルのコントロールを提供。
- トランスフォーマーモデルによるセマンティックなターン検出。
- ネイティブ MCP 統合。
- SIP を介した電話対応。
- LiveKit Inference による API キー不要の 50 以上のモデル。プラグイン経由でさらに 200 以上。

### 商用プラットフォーム

Vapi（最適化されたプレミアムスタックで 450〜600ms）と Retell（180 回のテスト呼び出し全体で 約 600ms エンドツーエンド）はこれらの上に構築されています。WebRTC チームなしでマネージドな音声スタックが欲しい場合はプラットフォームを選んでください。

### このパターンが失敗するケース

- **バージイン処理なし.** ユーザーが割り込む。エージェントが話し続ける。Pipecat では UPSTREAM キャンセルフレームが必要です。LiveKit でも同等のものが必要です。
- **STT の信頼度を無視する.** 信頼度の低いトランスクリプトが正典として LLM に渡される。信頼度でゲートするか確認を求めてください。
- **TTS の中断カットオフ.** パイプラインが発話中にキャンセルされた場合、TTS はそれを知るか音声をカットする必要があります。
- **レイテンシ予算を無視する.** すべてのコンポーネントが 50〜200ms を追加します。出荷前にチェーンを合計してください。

### 2026 年の典型的なレイテンシ

- VAD: 20〜60ms
- STT 部分: 100〜250ms
- LLM ファーストトークン: 150〜400ms
- TTS 最初の音声: 100〜200ms
- トランスポート RTT: 30〜80ms

エンドツーエンド 450〜600ms がプレミアムです。800〜1200ms は一般的。1500ms 超えると壊れているように感じます。

## 実装する

`code/main.py` はフレームベースの玩具パイプラインです:

- `Frame` タイプ（音声、トランスクリプト、テキスト、tts_audio、コントロール）。
- `process(frame)` を持つ `Processor` インターフェース。
- スクリプト化されたプロセッサーとして5ステージパイプライン（VAD → STT → LLM → TTS → トランスポート）。
- バージインターラプトを示す UPSTREAM キャンセルフレーム。

実行:

```
python3 code/main.py
```

トレースは通常フローと、TTS を発話中に停止させるバージインターラプトキャンセルを示します。

## 使ってみる

- **Pipecat** は完全なコントロール用 — カスタムプロセッサー、Python ファースト、プラガブルなプロバイダー。
- **LiveKit Agents** は WebRTC ファーストのデプロイメントと電話対応に。
- **Vapi / Retell** は WebRTC チームなしのホスト型音声エージェントに。
- **OpenAI Realtime / Gemini Live** は直接音声入出力（MultimodalAgent）に。

## 成果物を出す

`outputs/skill-voice-pipeline.md` は VAD + STT + LLM + TTS + トランスポートとバージイン処理を持つ Pipecat 形状の音声パイプラインの足場を提供します。

## 演習

1. 玩具パイプラインにメトリクスオブザーバーを追加します: ステージごとに1秒あたりのフレーム数をカウントします。レイテンシはどこで蓄積しますか？
2. 信頼度ゲート付き STT を実装します: 閾値以下の場合「もう一度言っていただけますか？」とリクエストします。
3. セマンティックなターン検出を追加します: シンプルなルール — トランスクリプトが「？」で終わる場合がターン終了。
4. Pipecat のトランスポートドキュメントを読みます。標準ライブラリのトランスポートを SmallWebRTCTransport の設定（スタブ）に交換します。
5. 同じクエリで OpenAI Realtime と STT+LLM+TTS カスケードを比較計測します。テキストレベルのコントロールはどのレイテンシコストをもたらしますか？

## キーワード

| 用語 | 一般的な言い方 | 実際の意味 |
|------|----------------|------------|
| Frame | "イベント" | パイプライン内のデータの型付き単位（音声、トランスクリプト、テキスト、コントロール） |
| Processor | "パイプラインステージ" | process(frame) を持つハンドラー |
| DOWNSTREAM | "前方フロー" | ソースからシンクへ: 音声入力、音声出力 |
| UPSTREAM | "フィードバックフロー" | コントロール: キャンセル、メトリクス、バージイン |
| VAD | "音声活動検出" | ユーザーが話しているときを検出する |
| セマンティックターン検出 | "スマートなターン終了" | ユーザーが発話を終えたかどうかのモデルベースの判断 |
| MultimodalAgent | "直接音声エージェント" | 音声入力、音声出力。中間にテキストなし |
| VoicePipelineAgent | "カスケードエージェント" | STT + LLM + TTS。テキストレベルのコントロール |

## 参考資料

- [Pipecat ドキュメント](https://docs.pipecat.ai/getting-started/introduction) — フレームベースパイプライン、プロセッサー、トランスポート
- [LiveKit Agents ドキュメント](https://docs.livekit.io/agents/) — WebRTC + 音声プリミティブ
- [Vapi](https://vapi.ai/) — マネージド音声プラットフォーム
- [Retell AI](https://www.retellai.com/) — マネージド音声、レイテンシベンチマーク済み
