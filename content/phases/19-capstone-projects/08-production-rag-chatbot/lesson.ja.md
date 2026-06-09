# キャップストーン 08 — 規制されたバーティカル向けの本番RAGチャットボット

> Harvey、Glean、Mendable、LlamaCloudはすべて2026年に同じ本番形状を運用している。docklingまたはUnstructuredとビジュアル用のColPaliで取り込む。ハイブリッド検索。bge-reranker-v2-gemmaでリランク。60-80%のヒット率でプロンプトキャッシュを使ってClaude Sonnet 4.7で合成する。Llama Guard 4とNeMo Guardrailsでガード。LangfuseとPhoenixで監視する。200問のゴールデンセットでRAGASでグレード付け。規制されたドメイン（法律、臨床、保険）で1つを構築し、キャップストーンはゴールデンセット・レッドチーム・ドリフトダッシュボードを通過することだ。

**演習するフェーズ:** P5 · P7 · P11 · P12 · P17 · P18

## 問題

規制されたドメインのRAG（法的契約、臨床試験プロトコル、保険ポリシー）は、ROIが明確で賭け金が具体的であるため2026年に最も多く出荷される本番形状だ。Harvey（Allen & Overy）は法律向けに構築した。Mendableは開発者ドキュメントの形を出荷する。Gleanはエンタープライズ検索をカバーする。パターンは：高品質に取り込み、リランク付きのハイブリッドで検索し、引用の強制とプロンプトキャッシュで合成し、複数の安全レイヤーでガードし、ドリフトを継続的に監視する。

難しい部分はモデルではない。管轄意識のあるコンプライアンス（HIPAA、GDPR、SOC2）、引用レベルの監査可能性、コスト制御（ヒット率が高い場合プロンプトキャッシュが60-90%の割引をもたらす）、RAGASの忠実度によるハルシネーション検出、ソースドキュメントがインデックスが追いつかずに更新されるときのドリフト検出だ。このキャップストーンは、200問のゴールデンセットとレッドチームスイートと一緒にすべてを出荷することを要求する。

## コンセプト

パイプラインには2つの側面がある。**取り込み**：docklingまたはUnstructuredが構造化文書をパース；ColPaliが視覚的に豊かなものを処理；チャンクはサマリー・タグ・ロールベースアクセスラベルを得る。ベクターはpgvector + pgvectorscale（50M以下のベクター）またはQdrant Cloudに入る；スパースBM25が並行して実行される。**会話**：LangGraphはメモリとマルチターンを処理；各クエリはハイブリッド検索を実行し、bge-reranker-v2-gemma-2bでリランクし、Claude Sonnet 4.7（プロンプトキャッシュ付き）で合成し、出力をLlama Guard 4とNeMo Guardrailsを通して渡し、引用アンカー付きの応答を出力する。

評価スタックには4つのレイヤーがある。正確性のための**ゴールデンセット**（引用付きの200のラベル付きQ/A）。安全性のための**レッドチーム**（ジェイルブレーク、PII抽出試み、オフドメイン質問）。ターンごとの自動**RAGAS**（忠実度/回答関連性/コンテキスト精度）。週次の検索品質とハルシネーションスコアを監視する**ドリフトダッシュボード**（Arize Phoenix）。

プロンプトキャッシュがコストレバーだ。Claude 4.5+とGPT-5+はシステムプロンプト+取得されたコンテキストのキャッシュをサポートする。60-80%のヒット率で、クエリあたりのコストが3-5倍下がる。パイプラインは高いキャッシュヒット率を達成するために安定したプレフィックス（システムプロンプト+リランクされたコンテキストが先）のために設計されなければならない。

## アーキテクチャ

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## スタック

- 取り込み: 構造化文書用のUnstructured.ioまたはdocling；視覚的に豊かなPDF用のColPali
- ベクターDB: 50M以下のベクター用pgvector + pgvectorscale；それ以上はQdrant Cloud
- スパース: フィールド重み付きTantivy BM25
- オーケストレーション: LlamaIndex Workflows（取り込み）+ LangGraph（会話）
- リランカー: セルフホストbge-reranker-v2-gemma-2bまたはホストVoyage rerank-2
- LLM: プロンプトキャッシュ付きClaude Sonnet 4.7；フォールバックとしてセルフホストLlama 3.3 70B
- 評価: RAGAS 0.2オンライン、ハルシネーションとジェイルブレークスイート用DeepEval
- 可観測性: アノテーションキュー付きのセルフホストLangfuse；ドリフト用Arize Phoenix
- ガードレール: Llama Guard 4入力/出力分類器、NeMo Guardrails v0.12ポリシー、Presidio PII除去
- コンプライアンス: チャンクのロールベースアクセスラベル；GDPR/HIPAA用の管轄タグ

## 実装する

1. **取り込み。** コーパス（真剣なビルドには1000-10000ドキュメント）をUnstructuredまたはdocklingでパースする。スキャン/視覚的に重いページについては、ColPaliを経由させる。サマリー・ロールラベル・管轄タグ付きのチャンクを生成する。

2. **インデックス。** 密な埋め込み（Voyage-3またはNomic-embed-v2）をpgvector + pgvectorscaleに入れる。Tantivy経由のBM25サイドインデックス。ペイロードとしてロールと管轄フィルター。

3. **ハイブリッド検索。** まずロール+管轄でフィルター；次に密とBM25を並行して実行；逆数ランク融合でマージ；上位20をリランカーへ；上位5を合成へ。

4. **プロンプトキャッシュで合成。** キャッシュヘッダーにシステムプロンプトと静的ポリシー；キャッシュ拡張としてリランクされたコンテキスト；未キャッシュサフィックスとしてユーザーの質問。定常状態で60-80%のキャッシュヒット率をターゲットにする。

5. **ガードレール。** 入力にLlama Guard 4；NeMo Guardrailsのレールがオフドメイン質問またはポリシーで禁止されたトピックをブロック；Presidioが出力の偶発的なPIIを除去；引用強制の後処理フィルター。

6. **ゴールデンセット。** ドメインエキスパートが（回答、引用）でラベル付けした200のQ/Aペア。正確な引用マッチ・回答の正確さ・忠実度（RAGAS）でエージェントをスコアリングする。

7. **レッドチーム。** 50の敵対的プロンプト：ジェイルブレーク（PAIR、TAP）、PII抽出試み、オフドメイン、クロス管轄漏洩。合格/不合格と重篤度でスコアリングする。

8. **ドリフトダッシュボード。** Arize PhoenixはnDCGと引用忠実度の週次検索品質を追跡する。5%の低下でアラート。

9. **コストレポート。** Langfuse：プロンプトキャッシュヒット率、クエリあたりのトークン、ステージ別$/クエリの内訳。

## 使ってみる

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 成果物を出す

`outputs/skill-production-rag.md`は成果物を説明する。コンプライアンスラベル付きでデプロイされ、ルーブリックを通過し、ライブドリフト監視で観察される規制ドメインのチャットボット。

| 配点 | 基準 | 測定方法 |
|:-:|---|---|
| 25 | RAGAS忠実度 + 回答関連性 | ゴールデンセット（200のQ/A）でのオンラインスコア |
| 20 | 引用の正確さ | 検証可能なソースアンカーを持つ回答の割合 |
| 20 | ガードレールのカバレッジ | Llama Guard 4の合格率 + ジェイルブレークスイートの結果 |
| 20 | コスト/レイテンシーエンジニアリング | プロンプトキャッシュヒット率、p95レイテンシー、$/クエリ |
| 15 | ドリフト監視ダッシュボード | 週次の検索品質トレンドを持つPhoenixライブダッシュボード |
| **100** | | |

## 演習

1. 別の管轄（例えばGDPRと並んでHIPAA）で2番目のコーパススライスを構築する。20問のクロス管轄プローブでロール+管轄フィルタリングがクロス漏洩を防ぐことを示す。

2. 1週間の本番トラフィックでプロンプトキャッシュヒット率を測定する。どのクエリがキャッシュプレフィックスを壊すかを特定する。再構成する。

3. 10kトークンのサマリーバッファを使ったマルチターンメモリを追加する。会話が長くなるにつれて忠実度が下がるかどうかを測定する。

4. Claude Sonnet 4.7をセルフホストのLlama 3.3 70Bに交換する。$/クエリと忠実度デルタを測定する。

5. 「不確か」モードを追加する：上位リランクスコアがしきい値を下回る場合、エージェントは回答する代わりに「自信のある引用がありません」と言う。偽の信頼の削減を測定する。

## キーワード

| 用語 | 一般的な呼び方 | 実際の意味 |
|------|-----------------|------------------------|
| プロンプトキャッシュ | 「キャッシュされたシステム+コンテキスト」 | Claude/OpenAI機能：キャッシュされたプレフィックストークンがヒットで60-90%割引 |
| RAGAS | 「RAG評価器」 | 忠実度・回答関連性・コンテキスト精度の自動スコアリング |
| ゴールデンセット | 「ラベル付き評価」 | 引用付きの200+の専門家ラベル付きQ/A；グラウンドトゥルース |
| 管轄タグ | 「コンプライアンスラベル」 | チャンクに付与されたGDPR/HIPAA/SOC2スコープ；検索フィルターで強制 |
| 引用忠実度 | 「根拠付き回答率」 | 検索可能なソーススパンに裏付けられた主張の割合 |
| ドリフト | 「検索品質の劣化」 | nDCGまたは引用スコアの週次変化；アラートしきい値5% |
| レッドチーム | 「敵対的評価」 | リリース前のジェイルブレーク・PII抽出・オフドメインプローブ |

## 参考資料

- [Harvey AI](https://www.harvey.ai) — リファレンス法律本番スタック
- [Gleanエンタープライズ検索](https://www.glean.com) — エンタープライズスケールでのリファレンスRAG
- [Mendableドキュメント](https://mendable.ai) — 開発者ドキュメントRAGリファレンス
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — マネージド取り込み
- [Anthropicプロンプトキャッシュ](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — コストレバーリファレンス
- [RAGAS 0.2ドキュメント](https://docs.ragas.io/) — カノニカルRAG評価フレームワーク
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — リファレンスドリフト可観測性
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026年の安全分類器
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — ポリシーレールフレームワーク
