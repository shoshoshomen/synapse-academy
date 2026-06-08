# 位置エンコーディング — Sinusoidal、RoPE、ALiBi

> アテンションは置換不変だ。「The cat sat on the mat」と「mat the on sat cat the」は位置信号なしで同じ出力を生成する。3つのアルゴリズムがそれを修正する——各々が「位置」の意味について異なる賭けをしながら。


## 問題

スケール済みドット積アテンションは順序に無頓着だ。アテンション行列`softmax(Q K^T / √d) V`はペアワイズ類似度から計算される。`X`の行をシャッフルすると、出力の行も同じようにシャッフルされる。アテンション内部は位置を気にしない。

それは bag-of-words モデルにおけるバグではない。言語、コード、音声、映像——順序が意味を持つあらゆるものに対しては致命的だ。

修正方法は、何らかの形で埋め込みに位置を注入することだ。3つの時代の答え：

1. **絶対正弦波（Vaswani 2017）。** 位置の`sin/cos`を埋め込みに加算する。シンプルで学習不要だが、学習済み長さを超える外挿が不得意。
2. **RoPE — Rotary Position Embeddings（Su 2021）。** Qとkベクトルを位置に比例した角度で回転させる。ドット積に*相対*位置を直接エンコードする。2026年の主流。
3. **ALiBi — Attention with Linear Biases（Press 2022）。** 埋め込みトリックを完全にスキップ；距離に基づいてアテンションスコアにヘッドごとの線形ペナルティを直接追加する。優れた長さ外挿。

2026年時点で、ほぼすべてのフロンティアオープンモデルはRoPEを使用している：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。一部の長コンテキストモデルはALiBiやその現代版を使用する。絶対正弦波は歴史的だ。

## 概念

![絶対正弦波 vs RoPE回転 vs ALiBi距離バイアス](../assets/positional-encoding.svg)

### 絶対正弦波

形状`(max_len, d_model)`の固定行列`PE`を事前計算する：

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

そしてアテンション前に`X' = X + PE[:N]`とする。各次元は異なる周波数の正弦波だ。モデルは位相パターンから位置を読むことを学習する。`max_len`を超えると失敗する：モデルは位置0〜2047だけを見たので位置2048で何が起こるか伝えられていない。

### RoPE

Q、Kベクトルを（埋め込みではなく）回転させる。次元のペア`(2i, 2i+1)`に対して：

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

位置`pos_k`のキーにも同じ回転を適用する。ドット積`q'_m · k'_n`は`(m - n)`のみの関数になる。つまり：**アテンションスコアは相対距離のみに依存する**。回転は絶対位置に基づいていたにもかかわらずだ。美しいトリックだ。

RoPEの拡張：`base`をスケールして（NTK-aware、YaRN、LongRoPE）、再学習なしに長いコンテキストに外挿できる。Llama 3はこの方法で8Kから128Kコンテキストに拡張した。

### ALiBi

埋め込みトリックをスキップ。アテンションスコアに直接バイアスを加える：

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

`m_h`はヘッド固有のスロープ（例：`1 / 2^(8·h/H)`）。近いトークンがブーストされ、遠いトークンがペナルティを受ける。学習時のコストなし。論文は長さ外挿が正弦波を上回り、元の学習済み長さでRoPEに匹敵することを示している。

### 2026年に何を選ぶか

| バリアント | 外挿 | 学習コスト | 採用例 |
|---------|---------------|---------------|---------|
| 絶対正弦波 | 劣る | タダ | 元のTransformer、初期BERT |
| 学習済み絶対 | なし | 微小 | GPT-2、GPT-3 |
| RoPE | スケーリングで良好 | タダ | Llama 2/3/4、Qwen 2/3、Mistral、DeepSeek-V3、Kimi |
| RoPE + YaRN | 優秀 | ファインチューニング段階 | Qwen2-1M、Llama 3.1 128K |
| ALiBi | 優秀 | タダ | BLOOM、MPT、Baichuan |

RoPEはアーキテクチャを変えずにアテンションに組み込めること、相対位置をエンコードすること、長コンテキストファインチューニングのためのクリーンなノブを`base`ハイパーパラメータで提供することから勝利した。

## 実装する

### ステップ1：正弦波エンコーディング

`code/main.py`を参照。4行の計算：

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

最初のアテンション層の前に埋め込み行列に追加する。

### ステップ2：Q、KにRoPEを適用する

RoPEはQとKに対してインプレースで動作する。各次元ペアに対して：

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

重要：位置`m`のQと位置`n`のKに同じ関数を適用する。それらのドット積はすべての座標ペアで`cos((m-n)·θ_i)`因子を受け取る。アテンションは無料で相対位置を学習する。

### ステップ3：ALiBiのスロープとバイアス

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # softmax前にアテンションスコアに追加する
```

softmaxの前にヘッド`h`のアテンションスコア行列の`(seq_len, seq_len)`に`bias[h]`を追加する。

### ステップ4：RoPEの相対距離プロパティを確認する

2つのランダムなベクトル`a、b`を取る。`(pos_a, pos_b)`で回転させる。次に`(pos_a + k, pos_b + k)`で回転させる。両方のドット積は浮動小数点誤差内で一致しなければならない。それがRoPEの全ポイントだ——絶対オフセットに不変で、相対ギャップのみが重要。

## 使ってみる

PyTorch 2.5+は`torch.nn.functional`にRoPEユーティリティを搭載している。ほとんどの本番コードはRoPEがアテンションカーネル内部に適用される`flash_attn`や`xformers`を使用する。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026年の長コンテキストトリック：**

- **NTK-awareインターポレーション。** コンテキストを4Kから16K以上に拡張する際に`base`を`base * (scale_factor)^(d/(d-2))`にスケール変更する。
- **YaRN。** 長いコンテキストでアテンションエントロピーを保持するよりスマートなインターポレーション。Llama 3.1 128Kで使用。
- **LongRoPE。** 進化的探索を使って次元ごとのスケール係数を選ぶMicrosoft 2024の手法。Phi-3-Longで使用。
- **位置インターポレーション + ファインチューニング。** 拡張係数で位置を縮小し、1〜5Bトークンでファインチューニングするだけ。驚くほど効果的。

## 成果物を出す

`outputs/skill-positional-encoding-picker.md`を参照。このスキルは、目標コンテキスト長・外挿ニーズ・学習予算を考慮して、新しいモデルのエンコーディング戦略を選択する。

## 演習

1. **易.** `max_len=512、d=128`の正弦波`PE`行列をヒートマップとしてプロットせよ。「次元インデックスが増えるほど縞が広くなる」パターンを確認せよ。
2. **中.** NTK-awareのRoPEスケーリングを実装せよ。長さ256のシーケンスで小さなLMを学習し、スケーリングありとなしで長さ1024でテストせよ。パープレキシティを測定せよ。
3. **難.** 同じアテンションモジュールにALiBiとRoPEを実装せよ。長さ512のシーケンスでコピータスクで4層Transformerを学習せよ。テスト時に2048に外挿せよ。劣化を比較せよ。

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| 位置エンコーディング（Positional encoding） | 「アテンションに順序を伝える」 | 位置をエンコードする埋め込みまたはアテンションに追加される任意の信号。 |
| 正弦波（Sinusoidal） | 「元のやつ」 | 幾何学的周波数の`sin/cos`を埋め込みに加算；外挿しない。 |
| RoPE | 「回転埋め込み」 | Qとkを位置に依存した角度で回転；ドット積が相対距離をエンコードする。 |
| ALiBi | 「線形バイアストリック」 | アテンションスコアに`-m·\|i-j\|`を加算；埋め込み不要で優れた外挿。 |
| base | 「RoPEのノブ」 | RoPEの周波数スケーラー；推論時コンテキストを拡張するために増加させる。 |
| NTK-aware | 「RoPEスケーリングトリック」 | コンテキストが拡張されたとき高周波数次元が圧縮されないよう`base`をスケール変更する。 |
| YaRN | 「高度なやつ」 | アテンションエントロピーを保持する次元ごとのインターポレーション+外挿。 |
| 外挿（Extrapolation） | 「学習済み長さを超えて動作する」 | 位置スキームが学習で見た`max_len`を超えて正しい出力を提供できるか？ |

## 参考資料

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 元の正弦波。
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE論文。
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi。
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 最先端のRoPEスケーリング。
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — MetaのLlama 2長コンテキスト論文。
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — Phi-3-LongとUse Itセクションで引用されているMicrosoftの手法。
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — すべてのRoPEスケーリングスキームの本番グレード実装（デフォルト、線形、動的、YaRN、LongRoPE、Llama-3）。
