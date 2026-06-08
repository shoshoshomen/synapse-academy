# オートエンコーダと変分オートエンコーダ（VAE）

> 普通のオートエンコーダは圧縮して復元する。それは暗記だ。生成はしない。1つのトリックを加えれば——コードをガウス分布に見えるよう強制する——サンプラーが得られる。その1つのトリック、`z = μ + σ·ε` の再パラメータ化が、2026年に使われるすべての潜在拡散・フローマッチング画像モデルの入力にVAEが存在する理由だ。


## 問題

784ピクセルのMNIST数字を16個の数のコードに圧縮し、復元する。普通のオートエンコーダは再構成MSEを優秀にこなすが、コード空間は凸凹した乱雑な状態になる。コード空間のランダムな点を選んでデコードすると、ノイズが出てくる。サンプラーがない。圧縮モデルに過ぎない。

本当に欲しいのは：(a) コード空間が綺麗で滑らかなサンプリング可能な分布（たとえば等方性ガウス `N(0, I)`）、(b) サンプルからのデコードが妥当な数字を生成する、(c) エンコーダとデコーダが依然として良好に圧縮する。3つの目標、1つのアーキテクチャ、1つの損失。

Kingmaの2013年のVAEは、エンコーダが分布 `q(z|x) = N(μ(x), σ(x)²)` を出力するよう訓練し、KLペナルティでその分布を事前分布 `N(0, I)` に引き寄せ、デコードの前に `q(z|x)` から `z` をサンプリングすることでこれを解決する。推論時はエンコーダを捨て、`z ~ N(0, I)` をサンプリングしてデコードする。KLペナルティがコード空間を構造化させるものだ。

2026年においてVAEが単独で出荷されることはほとんどない——生の画像品質では拡散に圧倒された——しかし、すべての潜在拡散モデル（SD 1/2/XL/3、Flux、AudioCraft）のエンコーダとして選ばれ続けている。VAEを学べば、使っているすべての画像パイプラインの見えない第一層を学んだことになる。

## 概念

![オートエンコーダ対VAE：再パラメータ化トリック](../assets/vae.svg)

**オートエンコーダ。** `z = encoder(x)`、`x̂ = decoder(z)`、損失 = `||x - x̂||²`。コード空間は非構造化。

**VAEエンコーダ。** 2つのベクトルを出力：`μ(x)` と `log σ²(x)`。これらが `q(z|x) = N(μ, diag(σ²))` を定義する。

**再パラメータ化トリック。** `q(z|x)` からのサンプリングは微分不可能だ。サンプルを `z = μ + σ·ε`（`ε ~ N(0, I)`）と書き直す。これで `z` は `(μ, σ)` の決定論的な関数にノイズを加えたものになり——勾配が `μ` と `σ` を通って流れる。

**損失。** 証拠下限（ELBO）、2つの項：

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

再構成項が `x̂` を `x` に近づける。KL項が `q(z|x)` を事前分布に近づける。両者はトレードオフする。小さいβ（<1）= より鮮明なサンプル、コード空間がガウス分布に近くない。大きいβ（>1）= よりきれいなコード空間、よりぼやけたサンプル。β-VAE（Higgins 2017）がこのノブを有名にし、ディスエンタングルメント研究を牽引した。

**サンプリング。** 推論時：`z ~ N(0, I)` を引き、デコーダを通す。1回のフォワードパス——拡散のような反復サンプリングはない。

## 実装する

`code/main.py` はnumpyもtorchも使わない小さなVAEを実装する。入力は8次元の合成データで、8次元の2成分ガウス混合から引かれる。エンコーダとデコーダは単一の隠れ層MLPだ。tanh活性化、フォワードパス、損失、手書きのバックワードパスを実装する。プロダクション向けではなく——教育目的だ。

### ステップ1：エンコーダのフォワード

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

`σ` の代わりに `log σ²` を使うのは、ネットワークの出力が制約なし（softplus(σ) はσ≈0で勾配が死ぬ罠だ）。

### ステップ2：再パラメータ化とデコード

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### ステップ3：ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

両方の分布がガウス分布なので、閉形式の正確なKL。数値積分はしない。2026年においてもモンテカルロKL推定を使ったコードが出荷されている——理由もなく3倍遅い。

### ステップ4：生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

これが生成モデルだ。5行。

## 落とし穴

- **事後分布崩壊。** KL項が `q(z|x) → N(0, I)` へ積極的に押しすぎて `z` が `x` の情報を持たなくなる。修正：β-アニーリング（β=0から始めて1にランプアップ）、フリービット、または非活性次元のKLをスキップ。
- **ぼやけたサンプル。** ガウスデコーダ尤度はMSE再構成を意味し、L2に対してベイズ最適（平均）——多数の妥当な数字の平均はぼやけた数字になる。修正：離散デコーダ（VQ-VAE、NVAE）、またはVAEをエンコーダとしてのみ使い、潜在変数の上に拡散を積む（Stable Diffusionがやっていること）。
- **βが大きすぎる、早すぎる。** 事後分布崩壊を参照。β≈0.01から始めてランプアップする。
- **潜在次元が小さすぎる。** MNISTには16次元で十分、ImageNet 256²には256次元、ImageNet 1024²には2048次元。Stable DiffusionのVAEは512×512×3 → 64×64×4を圧縮する（空間面積の32倍ダウンサンプル、チャンネルの32倍）。

## 使ってみる

2026年のVAEスタック：

| 状況 | 選択 |
|-----------|------|
| 拡散用の画像潜在エンコーダ | Stable Diffusion VAE（`sd-vae-ft-ema`）またはFlux VAE |
| 音声潜在エンコーダ | Encodec（Meta）、SoundStream、またはDAC（Descript） |
| 動画潜在変数 | Soraの時空間パッチ、Latte VAE、WAN VAE |
| ディスエンタングル表現学習 | β-VAE、FactorVAE、TCVAE |
| 離散潜在変数（Transformerモデリング用） | VQ-VAE、RVQ（ResidualVQ） |
| 生成用連続潜在変数 | 普通のVAE、その後その潜在空間でフロー/拡散モデルを条件付け |

潜在拡散モデルはエンコーダとデコーダの間に拡散モデルが住んでいるVAEだ。VAEが粗い圧縮をし、拡散モデルが重作業をする。動画（VAE + 動画拡散DiT）と音声（Encodec + MusicGen Transformer）でも同じパターンが使われる。

## 成果物を出す

`outputs/skill-vae-trainer.md` を保存する。

スキルは：データセットプロファイル + 潜在次元目標 + 下流の用途（再構成、サンプリング、または潜在拡散入力）を受け取り、アーキテクチャの選択（普通/β/VQ/RVQ）、βスケジュール、潜在次元、デコーダ尤度（ガウス対カテゴリカル）、評価計画（再構成MSE、次元ごとのKL、`q(z|x)` と `N(0, I)` 間のフレシェ距離）を出力する。

## 演習

1. **易。** `code/main.py` で `β` を `0.01`、`0.1`、`1.0`、`5.0` に変更する。最終的な再構成MSEとKLを記録する。合成データにとってパレート最適なβはどれか？
2. **中。** ガウスデコーダ尤度をベルヌーイ尤度（クロスエントロピー損失）に置き換える。同じ合成データの二値化バージョンでサンプル品質を比較する。
3. **難。** `code/main.py` をミニVQ-VAEに拡張する：連続的な `z` をK=32エントリのコードブックでの最近傍ルックアップに置き換える。再構成MSEを比較し、何個のコードブックエントリが使用されるかを報告する（コードブック崩壊は現実だ）。

## キーワード

| 用語 | 一般的な言い方 | 実際の意味 |
|------|-----------------|-----------------------|
| オートエンコーダ | エンコード・デコードネットワーク | `x → z → x̂`、MSEを学習。生成的ではない。 |
| VAE | サンプラー付きAE | エンコーダが分布を出力し、KLペナルティがコード空間を形成する。 |
| ELBO | 証拠下限 | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`；`q = p(z\|x)` のとき等号。 |
| 再パラメータ化 | `z = μ + σ·ε` | 確率的ノードを決定論的＋純粋ノイズとして書き直す。サンプリングを通じた逆伝播を可能にする。 |
| 事前分布 | `p(z)` | 潜在変数の目標分布、通常は `N(0, I)`。 |
| 事後分布崩壊 | 「KL項が勝つ」 | エンコーダが `x` を無視して事前分布を出力する；デコーダが幻覚しなければならない。 |
| β-VAE | 調整可能なKL重み | `loss = recon + β·KL`。βが高いほどディスエンタングルされるがよりぼやける。 |
| VQ-VAE | 離散潜在変数 | 連続的な `z` を最近傍コードブックベクトルに置き換える；Transformerモデリングを可能にする。 |

## プロダクションノート：VAEは拡散サーバーの最もホットなパスだ

Stable Diffusion / Flux / SD3のパイプラインでは、VAEはリクエストごとに2回呼ばれる——エンコード時（img2img / インペインティングを行う場合）とデコード時。1024²では、デコードパスがパイプライン全体で最大の活性化メモリピークになることが多い。`128×128×16` の潜在変数を `1024×1024×3` にアップサンプルするからだ。2つの実践的な結果：

- **スライスまたはタイルでデコードする。** `diffusers` は `pipe.vae.enable_slicing()` と `pipe.vae.enable_tiling()` を公開している。タイリングは小さなシームアーティファクトと引き換えに `O(H·W)` の代わりに `O(tile²)` のメモリを使う。1024²以上のコンシューマGPUには必須だ。
- **デコーダはbf16、最終リサイズの数値はfp32。** SD 1.xのVAEはfp32でリリースされ、1024²以上でfp16にキャストすると*サイレントにNaNが生成される*。SDXLには `madebyollin/sdxl-vae-fp16-fix` が付属している——常にfp16修正バリアントを使うかbf16を使うこと。

## 参考資料

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAEの論文。
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — ディスエンタングルβ-VAE。
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE。
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — 最先端の画像VAE。
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion；エンコーダとしてのVAE。
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec、音声VAEの標準。
