# マネージドLLMプラットフォーム — Bedrock、Vertex AI、Azure OpenAI

> 3つのハイパースケーラー、3つの異なる戦略。AWS Bedrockはモデルマーケットプレイスだ — Claude、Llama、Titan、Stability、CohereをひとつのAPIで提供する。Azure OpenAIはOpenAIとの独占パートナーシップに加え、専用キャパシティのためのProvisioned Throughput Units（PTU）を持つ。Vertex AIはGeminiファーストで、長コンテキストとマルチモーダルの最良の選択肢だ。2026年にArtificial AnalysisはAzure OpenAIの中央値レイテンシが約50ms、BedrockはLlama 3.1 405B相当で約75msと計測している — PTUが差を生む理由は、専用キャパシティがオンデマンドの共有より有利だからだ。「どれが最速か」ではなく「どのモデルカタログとFinOps体制が自社プロダクトに合うか」が判断基準だ。このレッスンでは、感覚ではなくトレードオフを書き出した上で選択する方法を教える。


## 学習目標

- 3つのプラットフォーム戦略（マーケットプレイス vs 独占 vs Geminiファースト）を挙げ、各製品ユースケースに対応させる。
- Azure OpenAIにおけるProvisioned Throughput Units（PTU）が何をもたらすか、そしてオンデマンドBedrockが405Bスケールで通常約25ms遅い理由を説明する。
- 各プラットフォームのFinOpsアトリビューション体制を図示する（Bedrockのアプリケーション推論プロファイル vs Vertexのプロジェクト別チーム vs AzureのスコープとPTU予約）。
- 「2プロバイダー最小ポリシー」を書き下し、2026年において単一ベンダーのロックインが高コストな失敗である理由を説明する。

## 問題

あなたはプロダクトにClaude 3.7 Sonnetを選んだ。次は配信が必要だ。Anthropic APIを直接呼ぶか、AWS Bedrock経由で呼ぶか、ゲートウェイを使うか。直接APIが最もシンプル。BedrockはBAA、VPCエンドポイント、IAM、CloudWatchアトリビューションを追加する。ゲートウェイはフェイルオーバー、統合課金、プロバイダー横断のレート制限を追加する。

より深い問いはカタログだ。同じプロダクト内でClaude、Llama、Geminiが必要な場合、BedrockとVertex AIとAzure OpenAIを同時に使わない限り、1ヶ所ですべてを購入できない。ハイパースケーラーは互換ではない — それぞれが異なるモデルレイヤーに賭けている。

このレッスンでは3つの賭け、レイテンシ差、FinOps差、ロックインリスクをマップする。

## コンセプト

### 3つの戦略

**AWS Bedrock** — マーケットプレイス。Claude（Anthropic）、Llama（Meta）、Titan（AWS自社）、Stability（画像）、Cohere（エンベディング）、Mistral、さらに画像とエンベディングのサブカタログ。ひとつのAPI、ひとつのIAM体制、ひとつのCloudWatchエクスポート。Bedrockの賭けは、顧客が単一モデルより選択肢を求めているという信念だ。

**Azure OpenAI** — 独占パートナーシップ。AzureデータセンターでGPT-4 / 4o / 5 / oシリーズ、DALL·E、Whisper、OpenAIモデルのファインチューニングが使える。「Azure OpenAIサービス」カタログにOpenAI以外のモデルはない — それらはAzure AI Foundry（別製品）に入る。Azureの賭けは、OpenAIがフロンティアに留まり続け、顧客がその特定の関係についてエンタープライズコントロールを望むという信念だ。

**Vertex AI** — Geminiファースト、それ以外はセカンド。Gemini 1.5 / 2.0 / 2.5 FlashとPro、加えてModel Garden（サードパーティ）。Vertexの賭けはマルチモーダル長コンテキスト — 100万トークンのGeminiコンテキストが差別化要因だ。

### スケールにおけるレイテンシ差

Artificial Analysisは継続的なベンチマークを実行している。同等のLlama 3.1 405Bデプロイメント（共有オンデマンド）で、Azure OpenAIの中央値ファーストトークンレイテンシは約50ms、Bedrockは約75ms。この差はAWSの失敗ではなく、キャパシティモデルの違いだ。AzureはPTU（Provisioned Throughput Units）を販売しており、テナントのGPUキャパシティを予約する。Bedrockの同等品（Provisioned Throughput）は存在するが、1ユニット当たり約21ドル/時から始まり、ほとんどの顧客は共有オンデマンドに留まる。

オンデマンド共有キャパシティは他のすべての顧客トラフィックと競合する。専用キャパシティはそうではない。プロダクトSLAがP99でTTFT < 100msの場合、Azure上でPTUを購入するか、Bedrock Provisioned Throughputを購入するか、デフォルトの変動を受け入れるしかない。

### Provisioned Throughputの経済学

Azure PTU：推論コンピュートの予約ブロック。予測可能なワークロードに対してオンデマンドと比較して最大約70%の節約。トラフィックに関わらず1時間ごとに固定費用 — アイドル時でも予約料を支払う。損益分岐点は通常、約40〜60%の持続的利用率だ。

Bedrock Provisioned Throughput：モデルとリージョンによって1時間あたり21〜50ドル。同様の計算 — 損益分岐点はピーク利用率の約半分。月次コミットメントが必要。

VertexのプロビジョンドキャパシティはGemini SKUごとに販売される。価格はモデルとリージョンによって異なり、公式に広告されていることが少ない。

### FinOps体制 — 真の差別化要因

**Bedrock アプリケーション推論プロファイル**はマーケットプレイスで最もクリーンなアトリビューションだ。プロファイルに`team`、`product`、`feature`タグを付け、すべてのモデル呼び出しをそこに通すと、CloudWatchが後処理なしでプロファイルごとのコストを分解する。2025年に追加され、現在も最も細粒度のハイパースケーラーネイティブ手段だ。

**Vertex**のアトリビューションはプロジェクト別チームにラベルを全リソースに付ける方式だ。各チームをGCPプロジェクトとして模型化し、すべてのリソースにラベルを付け、BigQuery Billing Export + DataStudioでロールアップする。作業は多いが、BigQueryによりコストデータへの任意のSQLが使える。

**Azure**はサブスクリプション/リソースグループのスコープとタグに依存し、PTU予約をファーストクラスのコストオブジェクトとして扱う。タグはリクエストからではなくリソースグループから継承されるため、リクエスト単位のアトリビューションにはApplication Insightsのカスタムメトリクスか、ヘッダーをスタンプするゲートウェイが必要だ。

パターン：Bedrockがネイティブで最もクリーン、VertexはBigQuery経由で最も柔軟、Azureはインストゥルメントしないかぎりほとんどが不透明。

### ロックインは2026年のリスク

単一ハイパースケーラーのコミットメントは、ひとつのモデルが支配していた頃は問題なかった。2026年にフロンティアは月次で動く — ある四半期はClaude 3.7、次はGemini 2.5、その次はGPT-5。ひとつのプラットフォームにロックされると、フロンティアの3分の2を締め出される。

動いているチームが採用するパターン：プロダクトクリティカルなLLM呼び出しに最低2プロバイダー。Bedrock + Azure OpenAIが一般的な組み合わせ — 一方からClaude、もう一方からGPT、両者間のフェイルオーバー、同一ゲートウェイ。ゲートウェイが最適ルーティングするのでコスト上昇は無視できる一方、障害時（2025年1月のAzure OpenAIインシデント、AWS us-east-1停止など）の可用性向上は決定的だ。

### データレジデンシー、BAA、規制産業

Bedrock：ほとんどのリージョンでBAA。VPCエンドポイント。ガードレール。フィンテックのデフォルト。
Azure OpenAI：HIPAA、SOC 2、ISO 27001。EU データレジデンシー。規制されたエンタープライズのデフォルト。
Vertex：HIPAA、GDPR、リージョン別データレジデンシー。Google Cloudのコンプライアンススタック。

3つすべてが基本的なチェックボックスを満たす。違いはデータ保持ポリシー、ログの扱い方、そして不正利用監視がトラフィックを読むかどうか（デフォルトはほとんどオプトイン。エンタープライズはオプトアウト可能）。

### 覚えておくべき数値

- Llama 3.1 405B相当のAzure OpenAIの中央値TTFT：約50ms（PTUあり）
- Bedrockオンデマンドの中央値TTFT：約75ms
- Bedrock Provisioned Throughput：1ユニット当たり21〜50ドル/時
- Azure PTU損益分岐点：約40〜60%の持続的利用率
- 高利用率でオンデマンド比PTU節約率：最大70%

## 使ってみる

`code/main.py`は合成ワークロードで3つのプラットフォームを比較する — オンデマンドとPTUの経済性、TTFTの変動、コストアトリビューションの精度をモデル化する。PTUが報われる場面と、マーケットプレイスのモデルの幅がTTFT差を上回る場面を確認するために実行する。

## 成果物を出す

このレッスンでは`outputs/skill-managed-platform-picker.md`を作成する。ワークロードプロファイル（必要なモデル、TTFT SLA、1日の量、コンプライアンス要件）に基づいて、主要プラットフォーム、フォールバック、FinOpsインストゥルメンテーション計画を推薦する。

## 演習

1. `code/main.py`を実行する。70Bクラスのモデルで、Azure PTUがオンデマンドを上回るのはどの持続的利用率か？損益分岐点を計算し、広告の40〜60%バンドと比較する。
2. プロダクトにClaude 3.7 SonnetとGPT-4oが必要だ。2プロバイダーデプロイを設計する — どちらのハイパースケーラーに何を配置するか、どのゲートウェイを前面に置くか、フェイルオーバーポリシーは何か？
3. 規制されたヘルスケア顧客が、BAA、US-Eastデータレジデンシー、P99 TTFT 100ms以下を要求する。プラットフォームを選び、3つの具体的な機能で正当化する。
4. Bedrockの請求が今月、トラフィック変化なしで4倍になった。アプリケーション推論プロファイルなしで原因を見つけるにはどうするか？プロファイルありではどのくらいかかるか？
5. Azure OpenAIとBedrockの価格ページを読む。月1億トークンのClaudeワークロードで、Anthropic API直接、Bedrockオンデマンド、Bedrock Provisioned Throughputのどれが安いか？

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|----------------|------------|
| Bedrock | 「AWS LLMサービス」 | Claude、Llama、Titan、Mistral、Cohereをまたぐモデルマーケットプレイス |
| Azure OpenAI | 「AzureのChatGPT」 | エンタープライズコントロール付きのAzureデータセンター内独占OpenAIモデル |
| Vertex AI | 「GoogleのLLM」 | Geminiファーストプラットフォームとサードパーティモデル用Model Garden |
| PTU | 「専用キャパシティ」 | Provisioned Throughput Unit — 予約済み推論GPU、時間単位で課金 |
| アプリケーション推論プロファイル | 「Bedrockタグ付け」 | タグ付きのプロダクト別コスト/使用プロファイル、CloudWatchネイティブ |
| Model Garden | 「Vertexカタログ」 | Vertex AIのサードパーティモデルセクション、Geminiとは別 |
| 2プロバイダー最小 | 「LLM冗長性」 | クリティカルなすべてのLLMパスを2つ以上のハイパースケーラーで実行するポリシー |
| BAA | 「HIPAA書類」 | Business Associate Agreement。PHI取り扱いに必要。3つすべてが提供 |
| 不正利用監視 | 「ログ監視者」 | プロンプト/出力に対するプロバイダー側の安全スキャン。エンタープライズはオプトアウト可 |

## 参考資料

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) — 正式レートカードとProvisioned Throughput価格
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — PTU経済性とレートカード
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) — GeminiティアとModel Gardenの追加料金
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/) — プロバイダー横断の継続的レイテンシ・スループットベンチマーク
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO Guide 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — エンタープライズ意思決定フレームワーク
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — アトリビューションメカニズムの並列比較
