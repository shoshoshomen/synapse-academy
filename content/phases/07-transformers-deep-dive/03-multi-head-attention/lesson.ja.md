# Multi-Head Attention

> 1つのアテンションヘッドは一度に1つの関係を学習する。8つのヘッドは8つを学習する。ヘッドはタダだ。もっと使おう。


## 問題

シングルのSelf-Attentionヘッドは1つのアテンション行列を計算する。その行列は1種類の関係を捉える——通常は学習信号が何であれ損失を最小化するものだ。データに主語-動詞一致、共参照、長距離談話、統語的チャンキングがすべて絡み合っている場合、シングルヘッドはそれらを単一のsoftmax分布に塗りつぶし、信号の半分を失う。

2017年のVaswani論文による解決策：それぞれ独自のQ、K、V射影を持つ複数のアテンション関数を並列に実行し、出力を連結する。各ヘッドは`d_model / n_heads`次元のより小さな部分空間で動作する。合計パラメータは同じ。表現力は上がる。

Multi-Head Attentionは2026年のすべてのTransformerが搭載するデフォルトだ。唯一の議論は*いくつ*のヘッドを使うか、そしてキーとバリューが射影を共有するかどうかだ（Grouped-Query Attention、Multi-Query Attention、Multi-head Latent Attention）。

## 概念

![Multi-Head Attentionの分割・アテンション・連結](../assets/multi-head-attention.svg)

**分割。** `(N, d_model)`の`X`を受け取る。Q、K、Vそれぞれを`(N, d_model)`に射影する。`d_head = d_model / n_heads`とすると`(N, n_heads, d_head)`に再形成する。`(n_heads, N, d_head)`に転置する。

**並列にアテンション。** 各ヘッド内でスケール済みドット積アテンションを実行する。各ヘッドは`(N, d_head)`を生成する。ヘッドは埋め込みの異なる部分空間で動作し、アテンション計算中は互いに通信しない。

**連結して射影。** ヘッドを`(N, d_model)`に重ね合わせ、`(d_model, d_model)`の学習済み出力行列`W_o`を掛ける。`W_o`はヘッドが混合できる場所だ。

**なぜうまくいくか。** 各ヘッドは他のヘッドと表現予算を奪い合わずに専門化できる。2019〜2024年のプロービング研究は明確なヘッドの役割を示している：位置ヘッド、前のトークンにアテンションを向けるヘッド、コピーヘッド、固有名詞ヘッド、帰納ヘッド（これがインコンテキスト学習の基盤）。

**2026年のバリエーションの系譜：**

| バリアント | Qヘッド | K/Vヘッド | 採用例 |
|---------|---------|-----------|---------|
| Multi-head（MHA） | N | N | GPT-2、BERT、T5 |
| Multi-query（MQA） | N | 1 | PaLM、Falcon |
| Grouped-query（GQA） | N | G（例：N/8） | Llama 2 70B、Llama 3+、Qwen 2+、Mistral |
| Multi-head latent（MLA） | N | 低ランクに圧縮 | DeepSeek-V2、V3 |

GQAは`N/G`倍のKVキャッシュメモリを削減しながらほぼ完全な品質を維持するため、現代のデフォルトだ。MLAはKとVをレイテント空間に圧縮し、計算時に逆投影することでさらに進む——FLOPsのコストがかかるが、はるかに多くのメモリを節約する。

## 実装する

### ステップ1：すでに持っているシングルヘッドアテンションからヘッドを分割する

レッスン02の`SelfAttention`を取り、分割/連結ペアでラップする。numpy実装は`code/main.py`を参照；ロジックは：

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

1つのreshapeと1つのtranspose。ループなし。これはPyTorchが`nn.MultiheadAttention`内部で行っていることとまったく同じだ。

### ステップ2：ヘッドごとにスケール済みドット積アテンションを実行する

各ヘッドはQ、K、Vの独自のスライスを取得する。アテンションはバッチ化行列積になる：

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

実際のハードウェアでは`Qh @ Kh.transpose(...)`は1つの`bmm`だ。GPUは`(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`という形の単一バッチ化行列積を見る。ヘッドを追加してもタダだ。

### ステップ3：Grouped-Query Attentionバリアント

変わるのはキーとバリューの射影だけだ。Qは`n_heads`グループを取得し、K、Vは`n_kv_heads < n_heads`グループを取得してマッチするために繰り返される：

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

推論時、KVキャッシュには`n_heads`コピーではなく`n_kv_heads`コピーのみが存在するためメモリを節約できる。Llama 3 70Bは64のクエリヘッドと8のKVヘッドを使用——8倍のキャッシュ縮小。

### ステップ4：各ヘッドが何を学習したかを探索する

4つのヘッドで短い文に対してMHAを実行する。各ヘッドに対して`(N, N)`のアテンション行列を出力する。ランダム初期化でも異なるヘッドが異なる構造を選ぶのを見られる——部分的にはシグナル、部分的には部分空間の回転対称性だ。

## 使ってみる

PyTorchの1行バージョン：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+のGQA：

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attentionはCUDAでFlash Attentionを自動ディスパッチする。
# GQAには、Qを形状(B, n_heads, N, d_head)として渡し、K,Vを
# (B, n_kv_heads, N, d_head)として渡す。PyTorchがrepeatを処理する。
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**ヘッド数は？** 2026年の本番モデルからの経験則：

| モデルサイズ | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| 小（〜125M） | 768 | 12 | 64 |
| ベース（〜350M） | 1024 | 16 | 64 |
| 大（〜1B） | 2048 | 16 | 128 |
| フロンティア（〜70B） | 8192 | 64 | 128 |

`d_head`はほぼ常に64か128に落ち着く。1つのヘッドが「見られる」情報量の単位だ。32を下回るとヘッドがスケーリング係数`sqrt(d_head)`と戦い始める；256を超えると「多くの小さな専門家」の利点が失われる。

## 成果物を出す

`outputs/skill-mha-configurator.md`を参照。このスキルは、パラメータ予算・シーケンス長・デプロイメントターゲットを考慮して、新しいTransformerのヘッド数・KVヘッド数・射影戦略を推奨する。

## 演習

1. **易.** `code/main.py`のMHAを取り、`d_model=64`固定で`n_heads`を1から16に変更せよ。合成コピータスクで小さな1層モデルの損失をプロットせよ。ヘッドを増やすと助けになるか、プラトーになるか、それとも悪化するか？
2. **中.** MQA（すべてのクエリヘッドで共有される1つのKVヘッド）を実装せよ。完全なMHAに対してパラメータ数がどれだけ減少するかを測定せよ。N=2048での推論時のKVキャッシュサイズがどれだけ縮小するかを計算せよ。
3. **難.** Mini版のMulti-head Latent Attentionを実装せよ：K、Vをランク`r`のレイテントに圧縮し、レイテントをKVキャッシュに保存し、アテンション時に展開する。キャッシュメモリが完全なMHAの1/8を下回りながら品質がバリデーションのpplで1ビット以内に収まる`r`はいくつか？

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| ヘッド（Head） | 「単一のアテンション回路」 | 独自のアテンション行列を持つ`d_head = d_model / n_heads`次元のQ/K/V射影。 |
| d_head | 「ヘッド次元」 | ヘッドごとの隠れ幅；本番ではほぼ常に64か128。 |
| 分割/結合（Split/combine） | 「reshapeトリック」 | アテンションの前後での`(N, d_model) ↔ (n_heads, N, d_head)`のreshape+転置。 |
| W_o | 「出力射影」 | ヘッドを連結した後に適用される`(d_model, d_model)`行列；ヘッドが混合する場所。 |
| MQA | 「1つのKVヘッド」 | Multi-Query Attention：単一の共有K/V射影。最小のKVキャッシュ、多少の品質低下。 |
| GQA | 「Llama 2以来のデフォルト」 | `n_kv_heads < n_heads`のGrouped-Query Attention；Qにマッチするように繰り返す。 |
| MLA | 「DeepSeekのトリック」 | Multi-head Latent Attention：K、Vを低ランクレイテントに圧縮し、アテンション時に展開。 |
| 帰納ヘッド（Induction head） | 「インコンテキスト学習の背後にある回路」 | 過去の出現を検出してその後に続くものをコピーするヘッドのペア。 |

## 参考資料

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 元のマルチヘッド仕様。
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA論文。
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — 学習後にMHAをGQAに変換する方法。
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLAとそれがMHA/GQAよりもキャッシュメモリで優れている理由。
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — ヘッドが実際に何をするかのメカニスティックな考察。
