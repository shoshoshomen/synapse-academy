# AIゲートウェイ — LiteLLM、Portkey、Kong AI Gateway、Bifrost

> ゲートウェイはアプリとモデルプロバイダーの間に位置する。コアフィーチャーはプロバイダールーティング、フォールバック、リトライ、レート制限、シークレット参照、オブザーバビリティ、ガードレールだ。2026年のマーケット分布：**LiteLLM**はMIT OSSで100以上のプロバイダー、OpenAI互換だが、公開ベンチマークで約2,000 RPS周辺で崩壊する（8 GBメモリ、持続負荷下でのカスケード障害）；Python、<500 RPS、開発/プロトタイピングに最適。**Portkey**はコントロールプレーンポジショニング（ガードレール、PII除去、ジェイルブレーク検出、監査証跡）、2026年3月にApache 2.0オープンソース化、レイテンシオーバーヘッド20〜40 ms、本番ティア$49/月。**Kong AI Gateway**はKong Gateway上に構築——Kongが同じ12 CPUで実施したベンチマーク：Portkeyより228%速く、LiteLLMより859%速い；$100/モデル/月の価格設定（Plusティアで最大5モデル）；既にKongを使っている場合のエンタープライズ向け。**Bifrost**（Maxim AI）——設定可能なバックオフを持つ自動リトライ、OpenAI 429時のAnthropicへのフォールバック。**Cloudflare / Vercel AIゲートウェイ** — マネージド、ゼロオペレーション、基本的なリトライ。データレジデンシーがセルフホスト決定の原動力；PortkeyとKongはOSS＋オプションのマネージドで中間に位置する。


## 学習目標

- 6つのコアゲートウェイフィーチャーを列挙する（ルーティング、フォールバック、リトライ、レート制限、シークレット、オブザーバビリティ、ガードレール）。
- 4つの2026年ゲートウェイ（LiteLLM、Portkey、Kong AI、Bifrost）をスケール上限とユースケースにマッピングする。
- Kongベンチマーク（Portkeyより228%速く、LiteLLMより859%速い）を引用し、>500 RPSで重要な理由を説明する。
- データレジデンシーとオペレーションバジェットを考慮してセルフホストかマネージドかを選択する。

## 問題の背景

プロダクトがOpenAI、Anthropic、セルフホストのLlamaを呼び出している。各プロバイダーは異なるSDK、エラーモデル、レート制限、認証スキームを持つ。フェイルオーバー（OpenAIが429した場合はAnthropicを試す）、一元化されたクレデンシャルストア、統合されたオブザーバビリティ、テナントごとのレート制限が欲しい。

アプリ層でこれを再実装するとすべてのサービスがすべてのプロバイダーに結合される。ゲートウェイ層はこれを1つのプロセスに集約し、プロバイダーにファンアウトする1つのAPI（一般的にOpenAI互換）を提供する。

## コンセプト

### 6つのコアフィーチャー

1. **プロバイダールーティング** — OpenAI、Anthropic、Gemini、セルフホストなどを1つのAPIの背後に。
2. **フォールバック** — 429、5xx、品質失敗時に別の場所でリトライ。
3. **リトライ** — 指数バックオフ、制限された試行回数。
4. **レート制限** — テナントごと、キーごと、モデルごと。
5. **シークレット参照** — 実行時にvaultからクレデンシャルを取得（アプリには決して含めない）。
6. **オブザーバビリティ** — OTel＋GenAI属性（フェーズ17・13）＋コスト配分。
7. **ガードレール** — PII除去、ジェイルブレーク検出、許可トピックフィルター。

### LiteLLM — MIT OSS、Python

- 100以上のプロバイダー、OpenAI互換、ルーター設定、フォールバック、基本的なオブザーバビリティ。
- Kongのベンチマークで約2,000 RPSで崩壊；8 GBメモリフットプリント、持続負荷下でのカスケード障害。
- 最適用途：Pythonアプリ、<500 RPS、開発/ステージングゲートウェイ、実験的なルーティング。
- コスト：OSSは$0；クラウドの無料ティアあり。

### Portkey — コントロールプレーンポジショニング

- 2026年3月からApache 2.0 OSSに。ガードレール、PII除去、ジェイルブレーク検出、監査証跡。
- リクエストあたり20〜40 msのレイテンシオーバーヘッド。
- リテンション＋SLA付き本番ティアは$49/月。
- 最適用途：ガードレール＋オブザーバビリティをバンドルした規制産業。

### Kong AI Gateway — スケールの選択

- Kong Gateway上に構築（成熟したAPIゲートウェイ製品、lua+OpenResty）。
- 12CPU相当でのKong自身のベンチマーク：Portkeyより228%速く、LiteLLMより859%速い。
- 価格：$100/モデル/月、Plusティアで最大5モデル。
- 最適用途：既にKong使用；>1,000 RPS；ライセンス取得を厭わない。

### Bifrost（Maxim AI）

- 設定可能なバックオフを持つ自動リトライ。
- OpenAI 429時のAnthropicへのフォールバックが標準的なレシピ。
- 新しい参入者；商用。

### Cloudflare AI Gateway / Vercel AI Gateway

- マネージド、ゼロオペレーション。基本的なリトライとオブザーバビリティ。
- 最適用途：Cloudflare/Vercel上のエッジサービングJavaScriptアプリ。
- ガードレールとレート制限ではKong/Portkeyより機能が限られる。

### セルフホスト対マネージド

データレジデンシーが強制要因だ。ヘルスケアと金融はデフォルトでセルフホスト（LiteLLM、Portkey OSS、またはKong）。コンシューマープロダクトはデフォルトでマネージド（Cloudflare AIゲートウェイ）または中間ティア（Portkey管理）。ハイブリッド：規制されたテナントにはセルフホスト、その他にはマネージド。

### レイテンシバジェット

- LiteLLM：通常5〜15 msのオーバーヘッド。
- Portkey：20〜40 msのオーバーヘッド。
- Kong：3〜8 msのオーバーヘッド。
- Cloudflare/Vercel：1〜3 msのオーバーヘッド（エッジの優位性）。

ゲートウェイのレイテンシはTTFTに直接加わる。TTFT P99 < 100 ms SLAにはKongまたはCloudflare。P99 < 500 msなら何でも。

### レート制限のセマンティクスが重要

シンプルなトークンバケットは中程度のスケールまで機能する。マルチテナントにはスライディングウィンドウ＋バースト許容＋テナントごとのティアリングが必要。LiteLLMはトークンバケット；Kongはスライディングウィンドウ；Portkeyはティア化されている。

### ゲートウェイ＋オブザーバビリティ＋ルーティングの組み合わせ

フェーズ17・13（オブザーバビリティ）＋16（モデルルーティング）＋19（ゲートウェイ）は本番では同じ層だ。3つすべてをカバーする1つのツールを選ぶか、慎重にワイヤリングする：ほとんどの2026年のデプロイメントはHelicone（オブザーバビリティ）またはPortkey（ガードレール）とKong（スケール）を別々の役割で組み合わせている。

### 覚えておくべき数値

- LiteLLM：約2,000 RPSで崩壊、8 GBメモリ。
- Portkey：20〜40 msのオーバーヘッド；2026年3月からApache 2.0。
- Kong：Portkeyより228%速く、LiteLLMより859%速い。
- Kongの価格：$100/モデル/月、Plusティアで5モデルまで。
- Cloudflare/Vercel：エッジで1〜3 msのオーバーヘッド。

## 使ってみる

`code/main.py`は、429/5xxインジェクション下で3つのプロバイダーにわたるフォールバック付きゲートウェイルーティングをシミュレートする。レイテンシ、リトライ率、フォールバックヒット率をレポートする。

## 成果物を出す

このレッスンでは`outputs/skill-gateway-picker.md`を生成する。スケール、オペレーション姿勢、コンプライアンス、レイテンシバジェットを考慮してゲートウェイを選択する。

## 演習

1. `code/main.py`を実行する。OpenAI→Anthropic→セルフホストのフォールバックを設定する。5%のプロバイダーエラー率でのヒット率はどうなるか？
2. SLAがベースライン300 msでTTFT P99 < 200 ms。どのゲートウェイがバジェット内に収まるか？
3. ヘルスケアの顧客がセルフホスト＋PII除去＋監査を必要とする。Portkey OSSかKongかを選ぶ。
4. LiteLLM対Kong：どのRPS上限でチームは移行すべきか？
5. マルチテナントSaaSのレート制限ポリシーを設計する：フリーティア、トライアルティア、有料ティア。トークンバケットかスライディングウィンドウか？

## キーワード

| 用語 | よく言われる表現 | 実際の意味 |
|------|----------------|------------|
| Gateway | 「APIブローカー」 | アプリとプロバイダーの間に位置するプロセス |
| LiteLLM | 「MITのもの」 | Python OSS、100以上のプロバイダー、2K RPSで崩壊 |
| Portkey | 「ガードレールゲートウェイ」 | コントロールプレーン＋オブザーバビリティ、Apache 2.0 |
| Kong AI Gateway | 「スケールのもの」 | Kong Gateway上に構築、ベンチマークリーダー |
| Bifrost | 「Maximのゲートウェイ」 | リトライ＋Anthropicフォールバックレシピ |
| Cloudflare AI Gateway | 「エッジマネージド」 | エッジデプロイのマネージドゲートウェイ、ゼロオペレーション |
| PII redaction | 「データスクラブ」 | モデルに送信前に正規表現＋NERでマスク |
| Jailbreak detection | 「プロンプトインジェクションガード」 | ユーザー入力の分類器 |
| Audit trail | 「規制ログ」 | すべてのLLM呼び出しの不変記録 |
| Token-bucket | 「シンプルなレート制限」 | リフィルベースのレートリミッター |
| Sliding-window | 「精密なレート制限」 | タイムウィンドウ型のレートリミッター；より良い公平性 |

## 参考資料

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
