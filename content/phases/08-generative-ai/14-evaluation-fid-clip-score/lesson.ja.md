# 評価指標：FID、CLIPスコア、人間の好み

> 生成モデルのあらゆるリーダーボードはFID、CLIPスコア、人間の好みアリーナでの勝率を引用する。各数値には、意欲的な研究者が操作できる失敗モードがある。失敗モードを知らなければ、本物の改善とゲーミングランとを区別できない。


## 問題

生成モデルは*サンプル品質*と*条件付け遵守*で評価される。どちらも閉形式の測定値を持たない。モデルは10,000枚の画像をレンダリングしなければならない；何かがそれらに数値を割り当てなければならない；モデルファミリー間、解像度間、アーキテクチャ間でその数値を信頼しなければならない。3つの指標が2014〜2026年の試練を生き残った：

- **FID（Fréchet Inception Distance）。** InceptionネットワークのFIDの特徴空間における2つの分布（実画像と生成画像）間の距離。低いほど良い。
- **CLIPスコア。** 生成画像のCLIP画像埋め込みとプロンプトのCLIPテキスト埋め込みのコサイン類似度。高いほど良い。プロンプト遵守を測る。
- **人間の好み。** 同じプロンプトで2つのモデルを対戦させ、人間（またはGPT-4クラスのモデル）により良い方を選ばせ、Eloスコアに集計する。

他にも見かけるもの：IS（Inception Score、ほぼ廃止）、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。それぞれが前のものの1つの失敗を修正する。

## 概念

![FID、CLIP、好み：3つの軸、異なる失敗モード](../assets/evaluation.svg)

### FID——サンプル品質

Heusel et al.（2017）。手順：

1. N枚の実画像とN枚の生成画像についてInception-v3特徴（2048次元）を抽出する。
2. 各プールにガウス分布をフィットする：平均 `μ_r, μ_g` と共分散 `Σ_r, Σ_g` を計算する。
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

解釈：特徴空間における2つの多変量ガウス分布間のFréchet距離。低い = より類似した分布。

失敗モード：
- **小さなNではバイアスがかかる。** FIDは特徴分布上の2乗平均——小さなNでは共分散を過小評価し、誤って低いFIDを出す。常にN ≥ 10,000を使う。
- **Inception依存。** Inception-v3はImageNetで訓練された。ImageNetから遠いドメイン（顔、アート、テキスト画像）は意味のないFIDを生成する。ドメイン固有の特徴抽出器を使う。
- **ゲーミング。** Inception事前分布への過学習は視覚的品質の改善なしに低いFIDを出す。CMMMDで対策する（後述）。

### CLIPスコア——プロンプト遵守

Radford et al.（2021）。生成画像＋プロンプトに対して：

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

30,000枚の生成画像で平均 → モデル間で比較可能なスカラー。

失敗モード：
- **CLIPのブラインドスポット。** CLIPは合成的な推論が弱い（「青い球体の上の赤い立方体」はしばしば失敗する）。モデルは複雑なプロンプトを本当に理解せずにCLIPスコアで高い評価を得ることがある。
- **短いプロンプトのバイアス。** 短いプロンプトは野生でより多くのCLIP画像マッチがある。長いプロンプトは機械的に低いCLIPスコアになる。
- **プロンプトゲーミング。** プロンプトに「高品質、4k、マスターピース」を含めるとCLIPスコアが膨らむが、画像テキストバインディングは改善されない。

CMMD（Jayasumana et al.、2024）はこれらのいくつかを修正する：Inceptionの代わりにCLIP特徴を使い、FréchetではなくMaximum-Mean Discrepancyを使う。微妙な品質差異の検出に優れる。

### 人間の好み——グラウンドトゥルース

プロンプトのプールを選ぶ。モデルAとモデルBで生成する。人間（または強力なLLM判定者）にペアを見せる。より良い方を選ばせる。勝利をEloまたはBradley-Terryスコアに集計する。ベンチマーク：

- **PartiPrompts（Google）**：1,600種の多様なプロンプト、12カテゴリ。
- **HPSv2**：107,000件の人間アノテーション、自動プロキシとして広く使われる。
- **ImageReward**：137,000件のプロンプト画像好みペア、MITライセンス。
- **PickScore**：Pick-a-Pic 260万の好みで訓練された。
- **Chatbot-Arenaスタイルの画像アリーナ**：https://imagearena.ai/ など。

失敗モード：
- **判定者の分散。** 非専門家と専門家は異なる好みを持つ。両方を使う。
- **プロンプト分布。** 厳選されたプロンプトは1つのファミリーを優遇する。常に文書化する。
- **LLM判定者の報酬ハッキング。** GPT-4判定者はきれいだが誤った出力に騙される。人間と三角測量する。

## 組み合わせて使う

プロダクションの評価レポートには以下を含めるべきだ：

1. ホールドアウトされた実分布に対する10,000〜30,000サンプルのFID（サンプル品質）。
2. 同じサンプルとプロンプトに対するCLIPスコア/CMMD（遵守度）。
3. 前のモデルとのブラインドアリーナでの勝率（全体的な好み）。
4. 失敗モード分析：50個のランダムサンプル出力を既知の問題（手の解剖学、テキストレンダリング、一貫したオブジェクト数）でフラグ。

単一の指標は嘘だ。3つの裏付け指標＋定性的レビューが主張になる。

## 実装する

`code/main.py` は合成「特徴ベクトル」（Inception特徴の代わりに4次元ベクトルを使う）でFID、CLIPスコア的、Elo集計を実装する。見られること：

- 小さなNと大きなNでのFID計算——バイアス。
- 特徴プール間のコサイン類似度としての「CLIPスコア」。
- 合成の好みストリームからのElo更新ルール。

### ステップ1：4行でFID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### ステップ2：CLIPスタイルのコサイン類似度

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### ステップ3：Elo集計

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 落とし穴

- **N=1,000でのFID。** ヒューリスティックはN=10,000未満では信頼できない。低いNのFIDを報告する論文はゲーミングしている。
- **解像度をまたいでFIDを比較する。** Inceptionの299×299へのリサイズが特徴分布を変える。同じ解像度でのみ比較する。
- **1シードで報告する。** 最低3シード実行する。標準偏差を報告する。
- **ネガティブプロンプトによるCLIPスコアの水増し。** 一部のパイプラインはプロンプトへの過学習でCLIPを高める。視覚的な彩度を確認する。
- **プロンプトの重複によるEloのバイアス。** 両モデルが訓練中にベンチマークプロンプトを見ていた場合、Eloは無意味だ。ホールドアウトされたプロンプトセットを使う。
- **人間評価のクラウドスキュー。** Prolific、MTurkのアノテーターは若年層/テック寄りにスキューする。採用されたアート/デザイン専門家と混ぜる。

## 使ってみる

2026年のプロダクション評価プロトコル：

| 柱 | 最低限 | 推奨 |
|--------|---------|-------------|
| サンプル品質 | ホールドアウト実データに対する10,000サンプルのFID | + 5,000サンプルのCMMMD + カテゴリごとのサブセットでのFID |
| プロンプト遵守 | 30,000サンプルのCLIPスコア | + HPSv2 + ImageReward + VQAスタイルの質問回答 |
| 好み | ベースラインとの200ペアのブラインド比較 | + 2,000ペアの人間 + LLM判定者 + Chatbot Arena |
| 失敗分析 | 50個の手動フラグ | 500個の手動フラグ + 自動安全分類器 |

1つのレポートに4本の柱全部 = 主張。1本だけ = マーケティング。

## 成果物を出す

`outputs/skill-eval-report.md` を保存する。スキルは新しいモデルチェックポイント＋ベースラインを取り、完全な評価計画を出力する：サンプルサイズ、指標、失敗モードプローブ、サインオフ基準。

## 演習

1. **易。** `code/main.py` を実行する。同じ合成分布でN=100対N=1,000でのFIDを比較する。バイアスの大きさを報告する。
2. **中。** 合成CLIPスタイルの特徴からCMMMDを実装する（式はJayasumana et al.、2024を参照）。FID対品質差の感度を比較する。
3. **難。** HPSv2の設定を再現する：Pick-a-Picのサブセットから1,000枚の画像プロンプトペアを取り、好みに基づいて小さなCLIPベースのスコアラーをファインチューンし、ホールドアウトセットとの一致度を測る。

## キーワード

| 用語 | 一般的な言い方 | 実際の意味 |
|------|-----------------|-----------------------|
| FID | 「Fréchet Inception Distance」 | 実画像と生成画像のInception特徴にフィットしたガウス分布のFréchet距離。 |
| CLIPスコア | 「テキスト画像類似度」 | CLIP画像とテキスト埋め込み間のコサイン類似度。 |
| CMMD | 「FIDの後継」 | CLIP特徴MMD；バイアスが少なく、ガウス仮定不要。 |
| IS | 「Inception Score」 | Exp KL(p(y|x) || p(y))；現代モデルとの相関が低く廃止。 |
| HPSv2 / ImageReward / PickScore | 「学習済み好みプロキシ」 | 人間の好みで訓練された小モデル；自動判定者として使われる。 |
| Elo | 「チェスのレーティング」 | ペアワイズの勝利のBradley-Terry集計。 |
| PartiPrompts | 「ベンチマークプロンプトセット」 | 12カテゴリにわたるGoogle厳選の1,600プロンプト。 |
| FD-DINO | 「自己教師あり代替」 | DINOv2特徴を使ったFD；ImageNet外ドメインに優れる。 |

## プロダクションノート：評価も推論ワークロードだ

10,000サンプルでFIDを実行するということは10,000枚の画像を生成することを意味する。単一のL4で1024²のSDXLベース50ステップでは約11時間の単一リクエスト推論だ。評価予算は現実であり、フレーミングはオフライン推論シナリオ（TTFTを無視してスループットを最大化する）そのものだ：

- **ハードにバッチし、レイテンシは忘れる。** オフライン評価 = メモリに収まる最大サイズでの静的バッチング。80GB H100で `num_images_per_prompt=8` の `pipe(...).images` は単一リクエストより4〜6倍速い実時間で動く。
- **実特徴をキャッシュする。** 実リファレンスセットに対するInception（FID）またはCLIP（CLIPスコア、CMMD）の特徴抽出は*一度だけ*実行し、`.npz` として保存する。評価ごとに再計算しない。

CI / リグレッションゲートには：PRごとに500サンプルのサブセットでFID＋CLIPスコアを実行する（約30分）；毎晩10,000のFID＋HPSv2＋Eloを完全実行する。

## 参考資料

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID論文。
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD。
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP。
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2。
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward。
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts。
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — 失敗モード調査。
