# 潜在拡散とStable Diffusion

> 512×512ピクセル空間での拡散は計算上の犯罪だ。Rombach et al.（2022）は、画像を生成するのに78万6千次元すべては必要ない——意味構造を捉えるのに十分な次元数と、残りのための別のデコーダがあれば十分だと気づいた。VAEの潜在空間の中で拡散を実行する。そのアイデアひとつがStable Diffusionだ。


## 問題

512²でのピクセル空間拡散は、U-Netが `[B, 3, 512, 512]` 形状のテンソルで実行されることを意味する。500Mパラメータの U-Netでは各サンプリングステップは約100 GFLOPs。50ステップで1画像あたり5 TFLOPS。10億枚の画像で訓練すると計算コストは天文学的だ。

それらのFLOPsの大部分は、損失のあるVAEが圧縮できる知覚的に重要でない詳細をネットを通じて押し込むことに費やされる。Rombachのアイデア：VAEを一度訓練し（*第1ステージ*）、それを凍結し、4チャンネル64×64の潜在空間（*第2ステージ*）で完全に拡散を実行する。同じU-Net。ピクセル数は1/16。同等の品質でFLOPsは約64倍少ない。

これがStable Diffusionのレシピだ。SD 1.x / 2.xは `64×64×4` 潜在変数上の860M U-Netを使い、SDXLは `128×128×4` 上の2.6B U-Netを使い、SD3はU-NetをフローマッチングのDiffusion Transformer（DiT）に交換した。Flux.1-dev（Black Forest Labs、2024）は12BパラメータのDiT-MMDiTを出荷する。すべてが同じ2ステージ基盤で動く。

## 概念

![潜在拡散：VAE圧縮＋潜在空間での拡散](../assets/latent-diffusion.svg)

**2つのステージ、個別に訓練。**

1. **ステージ1 — VAE。** エンコーダ `E(x) → z`、デコーダ `D(z) → x`。目標圧縮：各空間軸で8倍ダウンサンプル＋潜在サイズがピクセル数の約1/16になるようチャンネルを調整。損失 = 再構成（L1 + LPIPS知覚的）+ KL（小さい重みで `z` を強くGaussianに押し付けない、正確なサンプリングは不要だから）。多くの場合、デコードされた画像がシャープになるよう敵対的損失で訓練される。

2. **ステージ2 — `z` 上の拡散。** `z = E(x_real)` をデータとして扱う。`z_t` をデノイズするU-Net（またはDiT）を訓練する。推論時：拡散で `z_0` をサンプリングし、`x = D(z_0)` とする。

**テキスト条件付け。** 2つの追加コンポーネント。凍結されたテキストエンコーダ（SD 1.xにはCLIP-L、SD 2/XLにはCLIP-L+OpenCLIP-G、SD3とFluxにはT5-XXL）。クロスアテンション注入：各U-Netブロックが `[Q = 画像特徴、K = V = テキストトークン]` を取りそれらを混合する。トークンがテキストを画像に影響させる唯一の手段だ。

**損失関数はレッスン06と同一だ。** 同じDDPM / フローマッチングノイズのMSE。データドメインを交換するだけ。

## アーキテクチャのバリアント

| モデル | 年 | バックボーン | 潜在形状 | テキストエンコーダ | パラメータ |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L（77トークン） | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net＋リファイナー | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | 蒸留済み | 128×128×4 | 同上 | 1〜4ステップサンプリング |
| SD3 | 2024 | MMDiT（マルチモーダルDiT） | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | 蒸留済みMMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B、1〜4ステップ |

トレンド：U-NetをDiT（潜在パッチ上のTransformer）に置き換え、テキストエンコーダをスケールアップ（T5はプロンプト順守でCLIPを上回る）、潜在チャンネルを増やす（4→16でより多くの詳細の余裕）。

## 実装する

`code/main.py` は、おもちゃの1次元「VAE」（デモ用の恒等エンコーダ＋デコーダ；本物のVAEは畳み込みネットになる）をレッスン06のDDPMに重ね、クラス条件付きのclassifier-free guidanceを追加する。これは生の1次元値でも符号化された値でも同じ拡散損失が機能することを示す——鍵となる洞察だ。

### ステップ1：エンコーダ/デコーダ

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

本物のVAEには訓練済みのウェイトがある。教育目的には、この線形マップで十分に、拡散が元のデータ空間を気にせず `z` で動作することを示せる。

### ステップ2：`z` 空間での拡散

レッスン06と同じDDPM。ネットが見るデータは `z = E(x)` だ。`z_0` をサンプリングした後、`D(z_0)` でデコードする。

### ステップ3：Classifier-free guidance

訓練中、クラスラベルを10%の確率でドロップする（ヌルトークンに置き換える）。推論時に `ε_cond` と `ε_uncond` の両方を計算し：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = guidanceなし（完全な多様性）、`w = 3` = デフォルト、`w = 7+` = 飽和 / 過シャープ。

### ステップ4：テキスト条件付け（概念のみ、コードなし）

クラスラベルを凍結されたテキストエンコーダの出力に置き換える。クロスアテンション経由でテキスト埋め込みをU-Netに入力する：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

これがクラス条件付き拡散モデルとStable Diffusionの唯一の実質的な違いだ。

## 落とし穴

- **VAEスケールの不一致。** SD 1.xのVAEにはエンコード後にスケーリング定数（`scaling_factor ≈ 0.18215`）が適用される。これを忘れるとU-Netが大幅に間違った分散の潜在変数で訓練される。すべてのチェックポイントにこれが含まれている。
- **テキストエンコーダが静かに間違っている。** SD3には>=128トークンのT5-XXLが必要で、CLIPのみへのフォールバックは損失がある。常に `use_t5=True` を確認する、さもなくばプロンプト忠実度が落ちる。
- **潜在空間の混在。** SDXL、SD3、Fluxはすべて異なるVAEを使う。SDXL潜在変数で訓練されたLoRAはSD3では機能しない。Hugging Face diffusers 0.30+は不一致のチェックポイントの読み込みを拒否する。
- **CFGが高すぎる。** `w > 10` は飽和した油っぽい画像を生成し、多様性を犠牲にしてプロンプトに過学習する。スイートスポットは `w = 3-7`。
- **ネガティブプロンプトのリーク。** 空のネガティブプロンプトはヌルトークンになる；入力されたネガティブプロンプトは `ε_uncond` になる。これらは同じではない；一部のパイプラインは静かにヌルをデフォルトにする。

## 使ってみる

2026年のプロダクションスタック：

| 目標 | 推奨バックボーン |
|--------|----------------------|
| 狭いドメイン、ペアデータ、ゼロからモデルを訓練 | SDXLファインチューン（LoRA / フル）——出荷が最速 |
| オープンドメインテキスト→画像、オープンウェイト | Flux.1-dev（12B、Apache / 非商用）またはSD3.5-Large |
| 最速推論、オープンウェイト | Flux.1-schnell（1〜4ステップ、Apache）またはSDXL-Lightning |
| 最良のプロンプト順守、ホスト型 | GPT-Image / DALL-E 3（依然）、Midjourney v7、Imagen 4 |
| 編集ワークフロー | Flux.1-Kontext（2024年12月）——ネイティブに画像＋テキストを受け付ける |
| リサーチ、ベースライン | SD 1.5——古いが十分に研究されている |

## 成果物を出す

`outputs/skill-sd-prompter.md` を保存する。スキルはテキストプロンプト＋目標スタイルを受け取り、モデル＋チェックポイント、CFGスケール、サンプラー、ネガティブプロンプト、解像度、オプションのControlNet/IP-Adapterコンボ、ステップごとのQAチェックリストを出力する。

## 演習

1. **易。** `w ∈ {0, 1, 3, 7, 15}` でguidance付きで `code/main.py` を実行する。クラスごとの平均サンプルを記録する。どの `w` でクラスの平均が実データの平均を超えるか？
2. **中。** おもちゃの線形エンコーダを再構成損失付きのtanh-MLPエンコーダ/デコーダペアに交換する。新しい潜在変数で拡散を再訓練する。サンプル品質は変わるか？
3. **難。** diffusersで本物のStable Diffusion推論をセットアップする：`sdxl-base` を読み込み、CFG=7でEuler 30ステップを実行し、時間を計測する。次に `sdxl-turbo` をCFG=0で4ステップに切り替える。同じ題材で品質が異なる——何が変わったかとその理由を説明する。

## キーワード

| 用語 | 一般的な言い方 | 実際の意味 |
|------|-----------------|-----------------------|
| 第1ステージ | 「VAE」 | 訓練済みエンコーダ/デコーダペア；512²を64²に圧縮する。 |
| 第2ステージ | 「U-Net」 | 潜在空間上の拡散モデル。 |
| CFG | 「Guidanceスケール」 | `(1+w)·ε_cond - w·ε_uncond`；条件付け強度を調整。 |
| ヌルトークン | 「空のプロンプト埋め込み」 | `ε_uncond` に使われる無条件埋め込み。 |
| クロスアテンション | 「テキストが入る方法」 | 各U-NetブロックがK、VとしてテキストトークンにアテンドするMechanism。 |
| DiT | 「拡散Transformer」 | U-Netを潜在パッチ上のTransformerに置き換える；スケールがより良い。 |
| MMDiT | 「マルチモーダルDiT」 | SD3のアーキテクチャ：結合アテンション付きテキストと画像ストリーム。 |
| VAEスケーリング係数 | 「マジックナンバー」 | 拡散が単位分散空間で動作するよう潜在変数を約5.4で割る。 |

## プロダクションノート：8GBのコンシューマGPUでFlux-12Bを実行する

リファレンスFlux統合は正規の「8GBのコンシューマGPU、これを出荷できるか？」レシピだ。トリックはプロダクション推論文献が列挙する3つのノブのレシピを拡散DiTに適用することだ：

1. **段階的読み込み。** FluxにはVRAMに同時に存在する必要のない3つのネットワークがある：T5-XXLテキストエンコーダ（fp32で約10GB）、CLIP-L（小）、12B MMDiT、VAE。まずプロンプトをエンコードし、エンコーダを*削除*し、DiTを読み込み、デノイズし、DiTを*削除*し、VAEを読み込み、デコードする。コンシューマの8GB GPUは一度に1ステージしか収まらない。
2. **bitsandbytesによる4ビット量子化。** T5エンコーダとDiT両方に `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)` を使う。メモリを8倍削減し、品質の低下はAritiraのベンチマーク（ノートブックにリンク）によればテキスト→画像では知覚できない。
3. **CPUオフロード。** `pipe.enable_model_cpu_offload()` は各フォワードパスが進むにつれてモジュールをCPUとGPUの間で自動スワップする。レイテンシが10〜20%増加するが、パイプラインが少なくとも実行できるようになる。

メモリ計算：量子化後 `10 GB T5 / 8 = 1.25 GB`、量子化DiT `12 B params × 0.5 bytes = 約6 GB`、活性化分を加える。stas00の用語でこれはTP=1推論の極端な端——モデル並列なし、最大量子化。プロダクションではH100でTP=2またはTP=4を実行するだろう；シングルの開発ラップトップにはこのレシピだ。

## 参考資料

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion。
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL。
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3、MMDiT。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG。
- [Labs (2024). Flux.1 — Black Forest Labs発表](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1ファミリー。
- [Hugging Face Diffusersドキュメント](https://huggingface.co/docs/diffusers/index) — 上記すべてのチェックポイントのリファレンス実装。
