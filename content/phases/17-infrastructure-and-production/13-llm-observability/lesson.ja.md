# LLMオブザーバビリティスタック選定

> 2026年のオブザーバビリティ市場は2つのカテゴリーに分かれる。開発プラットフォーム（LangSmith、Langfuse、Comet Opik）はオブザーバビリティをevals、プロンプト管理、セッションリプレイとバンドルする。ゲートウェイ/インストゥルメンテーションツール（Helicone、SigNoz、OpenLLMetry、Phoenix）はテレメトリーに注力する。LangfuseはMITライセンスのコアで強力なOSSバランスを持つ（クラウド無料ティアで月50Kイベント）。PhoenixはElastic License 2.0のもとでOpenTelemetryネイティブ — ドリフト/RAG可視化に優れているが永続的な本番バックエンドではない。Arize AXはゼロコピーIceberg/Parquet統合を使用しモノリシックオブザーバビリティより約100倍安価だと主張する。LangSmithはLangChain/LangGraphをリードし、$39/ユーザー/月、セルフホストはEnterpriseのみ。Heliconeはプロキシベースで15〜30分のセットアップ、月10万リクエスト無料だがエージェントトレースの深さは少ない。新興の本番パターン：ゲートウェイ（Helicone/Portkey）＋evalプラットフォーム（Phoenix/TruLens）をOpenTelemetryで連結。


## 学習目標

- 開発プラットフォーム（バンドル：evals＋プロンプト＋セッション）とゲートウェイ/テレメトリーツール（トレース＋メトリクスのみ）を区別する。
- 6つの主要ツール（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）をライセンス、価格、スイートスポットのユースケースにマッピングする。
- ゲートウェイツールと別のevalプラットフォームを組み合わせるOpenTelemetryグルーパターンを説明する。
- 2026年のコスト差別化要因（Arize AXのゼロコピーアプローチ対モノリシックインジェスト）と大まかな100倍の乗数を挙げる。

## 問題

LLM機能を出荷した。動作している。プロンプト失敗、ツールループ、レイテンシリグレッション、コストスパイク、プロンプトキャッシュヒット率についての可視性がない。「LLMオブザーバビリティ」でGoogleすると3つの価格帯で同じ問題を解決すると主張する8つのツールが出てくる。

同じ問題を解決しているわけではない。LangSmithは「このLangGraphの実行がなぜ失敗したか？」に答える。Phoenixは「RAGパイプラインがドリフトしているか？」に答える。Heliconeは「どのアプリがトークンを燃やしているか？」に答える。Langfuseは「全体をセルフホストできるか？」に答える。異なるツール、異なるオーディエンス。

選択には4つの軸がある：スタック（LangChain？生のSDK？マルチベンダー？）、ライセンス許容度（MITのみ？Elastic OK？商用も可？）、予算（無料ティア？月$100？月$1000？）、セルフホスト（必須？あれば良い？不要？）。

## コンセプト

### 2つのカテゴリー

**開発プラットフォーム**はオブザーバビリティをevals、プロンプト管理、データセットバージョニング、セッションリプレイとバンドルする。実験を実行し、どのプロンプトが機能したかを確認し、新しいプロンプトを古い勝者に対してデータセット回帰する。LangSmith、Langfuse、Comet Opik。

**ゲートウェイ/テレメトリーツール**は推論呼び出しをインストゥルメントする — プロンプト、レスポンス、トークン、レイテンシ、モデル、コスト。Helicone、SigNoz、OpenLLMetry、Phoenix。ミニマリスト。OpenTelemetryを介して別のevalツールと組み合わせることができる。

### Langfuse — OSSバランス

- コアはApache / MITライセンス；Docker経由でセルフホスト。
- クラウド無料ティア：月50Kイベント。有料：チームで月$29。
- Evals、プロンプト管理、トレース、データセット。4つの開発プラットフォーム機能をすべて合理的にカバー。
- スイートスポット：LangSmithクラスの機能が必要だがセルフホストまたはOSSライセンスが必須の場合。

### Phoenix（Arize）— テレメトリーファースト、OpenTelemetryネイティブ

- Elastic License 2.0；セルフホストは簡単。
- RAGとドリフトの可視化に優れている。埋め込み空間散布図はファーストクラスとして搭載。
- 永続的な本番バックエンドとして設計されていない — 主に開発時のオブザーバビリティ。
- スイートスポット：RAGパイプライン開発、ドリフトデバッグ、本番用に別のゲートウェイと組み合わせる。

### Arize AX — スケールプレイ

- 商用。Iceberg/Parquet経由のゼロコピーデータレイク統合。
- スケール時にモノリシックオブザーバビリティ（Datadogクラス）より約100倍安価と主張。数学：トレースをS3上の自分のParquetに保存；Arizeが直接読み込む。
- スイートスポット：日間10M以上のトレース、既存データレイク、Datadogの価格設定なしにLLM固有のダッシュボードが必要な場合。

### LangSmith — LangChain/LangGraphファースト

- 商用、$39/ユーザー/月。セルフホストはEnterpriseのみ。
- LangChainとLangGraphスタックでベストインクラス。どちらでもない場合は魅力が薄い。
- スイートスポット：LangChainにコミットしたチーム、支払い意欲あり。

### Helicone — プロキシベースの最小実行可能

- `OPENAI_API_BASE`をHeliconeプロキシに変更するだけで15〜30分のセットアップ。
- MITライセンス；月10万リクエスト無料、有料$20/月〜。
- フェイルオーバー、キャッシング、レート制限を含む — ゲートウェイとしても機能。
- エージェント/マルチステップトレースの深さは少ない。
- スイートスポット：クイックスタート、シングルスタックアプリ、ゲートウェイ＋オブザーバビリティを一つで必要な場合。

### Opik（Comet）— OSS開発プラットフォーム

- Apache 2.0、完全OSS。
- Cometのヘリテージを持つLangfuseと同様の機能セット。
- スイートスポット：Cometを既に使用しているMLチームが同じペインでLLMオブザーバビリティを必要とする場合。

### SigNoz — OpenTelemetryファーストの完全APM

- Apache 2.0。OpenTelemetry経由でLLMを含む一般APMを処理。
- スイートスポット：サービスとLLM呼び出し全体での統合オブザーバビリティ。

### グルー：OpenTelemetry + GenAIセマンティック規約

OpenTelemetryは2025年後半にGenAIセマンティック規約を公開した（`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。OTelを消費するツールは相互運用できる。新興の本番パターン：

1. すべてのLLM呼び出しからGenAI規約付きOTelを送出する。
2. 日常業務のためにゲートウェイ（Helicone / Portkey）にルーティングする。
3. リグレッションのためにevalプラットフォーム（Phoenix / Langfuse）にデュアル送信する。
4. Arize AXまたはDuckDB経由の長期分析のためにデータレイク（Iceberg）にアーカイブする。

### 罠：間違ったレイヤーでのインストゥルメンテーション

エージェントフレームワーク内でインストゥルメントする（例：LangSmithトレースを追加する）と、そのフレームワークに結合される。HTTP/OpenAI-SDKレイヤーでインストゥルメントする（OpenLLMetryやゲートウェイ経由）とポータブルだ。

### サンプリング — すべてを保持できない

日間100万リクエスト超では、フルトレース保持はLLM呼び出しより高コストになる。ルールでサンプリングする：エラーは100%、高コストは100%、成功は5%。集計は常に保持；ロングテールは生データを保持する。

### 覚えておくべき数値

- Langfuseの無料クラウド：月50Kイベント。
- LangSmith：$39/ユーザー/月。
- Heliconeの無料：月10万リクエスト。
- Arize AXの主張：スケール時にモノリシックより約100倍安価。
- OpenTelemetry GenAI規約：2025年出荷、2026年広く採用。

## 使ってみる

`code/main.py`は保持戦略（100%インジェスト、サンプリング、サンプリング＋エラー）にわたって100万トレースの日をシミュレートする。各戦略のストレージコストと失われるものを報告する。

## 成果物を出す

このレッスンは`outputs/skill-observability-stack.md`を作成する。スタック、スケール、予算、ライセンス姿勢を与えると、ツールを選ぶ。

## 演習

1. LangChainを使用するチームがOSSセルフホストオブザーバビリティを求めている。LangfuseかOpikを選んで理由を述べる。
2. 日間500万トレースでDatadogが月$150Kを見積もる。Arize AXの損益分岐点を計算する。
3. 組織のガイドラインがすべてのLLM呼び出しに義務付けるべきOpenTelemetry GenAI属性セットを設計する。
4. Phoenixだけで本番に十分かを論じる。いつ不十分になるか？
5. Heliconeは20msのプロキシオーバーヘッドがある。P99 TTFT 300msでそれは許容できるか？SLAが100msの場合は？

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|----------------|------------|
| OpenLLMetry | 「LLMのためのOTel」 | LLM向けのオープンソースOpenTelemetryインストゥルメンテーション |
| GenAI規約 | 「OTel属性」 | LLM呼び出し用の標準OTel属性名 |
| LangSmith | 「LangChainオブザーバビリティ」 | LangChainエコシステムにバンドルされた商用プラットフォーム |
| Langfuse | 「OSS LangSmith」 | 同様の機能セットを持つMIT OSS |
| Phoenix | 「Arize開発ツール」 | OpenTelemetryネイティブの開発/evalプラットフォーム |
| Arize AX | 「スケールオブザーバビリティ」 | 商用ゼロコピーIceberg/Parquetオブザーバビリティ |
| Helicone | 「プロキシオブザーバビリティ」 | LLMテレメトリー＋ゲートウェイ機能を収集するHTTPプロキシ |
| Opik | 「Comet LLM」 | CometのApache 2.0 OSS開発プラットフォーム |
| セッションリプレイ | 「トレースの再実行」 | ツール呼び出しを含む完全なエージェントセッションの再生 |
| Eval | 「オフラインテスト」 | ラベル付きデータセットに対して候補モデル/プロンプトを実行する |

## 参考資料

- [SigNoz — 2026年トップLLMオブザーバビリティツール](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX代替分析](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Langfuse、LangSmith、Helicone、Phoenixのセットアップ](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAIセマンティック規約](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
