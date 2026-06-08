# Attentionメカニズム — ブレークスルー

> デコーダーは圧縮された要約を目を細めて見ることをやめて、ソース全体を見始める。この後はすべてAttentionと工学だ。


## 問題の背景

レッスン09は計測された失敗で終わった。トイコピータスクで訓練されたGRUエンコーダー-デコーダーは長さ5での89%の精度から長さ80ではほぼ偶然と変わらないレベルまで落ちる。その理由は訓練のバグではなく構造的なものだ：エンコーダーが収集したすべての情報は一つの固定サイズの隠れ状態に収まらなければならず、デコーダーは他のものを一切見ない。

Bahdanau、Cho、Bengioは2014年に3行の修正を発表した。デコーダーに最終エンコーダー状態だけを与える代わりに、すべてのエンコーダー状態を保持する。各デコーダーステップで、エンコーダー状態の加重平均を計算し、その重みは「デコーダーは今エンコーダー位置`i`をどれだけ見る必要があるか？」を示す。その加重平均がコンテキストで、それは各デコーダーステップで変化する。

それがすべてのアイデアだ。Transformerはそれを拡張した。Self-attentionは単一のシーケンスに適用した。Multi-head attentionは並列に実行した。しかし2014年のバージョンはすでにボトルネックを解消し、それを得た後はTransformerへの転換は概念的なものではなく工学的なものだ。

## 概念

各デコーダーステップ`t`で：

1. 前のデコーダー隠れ状態`s_{t-1}`を**クエリ**として使う。
2. すべてのエンコーダー隠れ状態`h_1, ..., h_T`に対してスコアを付ける。エンコーダー位置ごとに1つのスカラー。
3. スコアをSoftmaxして、合計が1になるAttention重み`α_{t,1}, ..., α_{t,T}`を得る。
4. コンテキストベクトル`c_t = Σ α_{t,i} * h_i`。エンコーダー状態の加重平均。
5. デコーダーは`c_t`と前の出力トークンを受け取り、次のトークンを生成する。

加重平均がポイントだ。デコーダーが「Je」を「I」に翻訳する必要がある場合、「Je」に対するエンコーダー状態を高く重み付けし、他を低くする。「not」が必要な場合、「pas」を高く重み付けする。コンテキストベクトルが各ステップを形成する。

## 形状（誰もが引っかかること）

すべてのAttention実装が最初に間違える部分だ。ゆっくり読む。

| もの | 形状 | 注記 |
|------|------|------|
| エンコーダー隠れ状態 `H` | `(T_enc, d_h)` | BiLSTMなら `d_h = 2 * d_hidden` |
| デコーダー隠れ状態 `s_{t-1}` | `(d_s,)` | 1つのベクトル |
| Attentionスコア `e_{t,i}` | スカラー | エンコーダー位置ごとに1つ |
| Attention重み `α_{t,i}` | スカラー | 全`i`でSoftmax後 |
| コンテキストベクトル `c_t` | `(d_h,)` | エンコーダー状態と同じ形状 |

**Bahdanau（加算）スコア。** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}`は形状`(d_s,)`、`h_i`は形状`(d_h,)`。
- `W_a`は形状`(d_attn, d_s)`。`U_a`は形状`(d_attn, d_h)`。
- tanh内の和は形状`(d_attn,)`。
- `v_α`は形状`(d_attn,)`。`v_α`との内積がスカラーに潰れる。**これが`v_α`のすること。** 魔法ではない。Attention次元ベクトルをスカラースコアに変換する射影だ。

**Luong（乗法）スコア。** 3つのバリアント：

- `dot`: `e_{t,i} = s_t^T * h_i`。`d_s == d_h`が必要。ハード制約。エンコーダーが双方向ならスキップ。
- `general`: `e_{t,i} = s_t^T * W * h_i`、`W`の形状`(d_s, d_h)`。等次元制約を取り除く。
- `concat`: 本質的にBahdanau形式。最初の2つの方が安価なのでほとんど使われない。

**BahdanauとLuongの注意点。** BahdanauはDecoderの`s_{t-1}`（現在の単語を生成する*前*の状態）を使う。Luongは`s_t`（*後*の状態）を使う。これらを混同するとデバッグが非常に難しい微妙に間違った勾配が生まれる。どちらか一方の論文を選んでその規約に従う。

## 実装する

### ステップ1: 加算（Bahdanau）Attention

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

上記の表と形状を確認する。`encoder_states`は形状`(T_enc, d_h)`。`projected_enc`は形状`(T_enc, d_attn)`。`projected_dec`は形状`(d_attn,)`でブロードキャストされる。`combined`は形状`(T_enc, d_attn)`。`scores`は形状`(T_enc,)`。`weights`は形状`(T_enc,)`。`context`は形状`(d_h,)`。完成。

### ステップ2: LuongのdotとGeneral

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

それぞれ3行。これがLuongの論文が受け入れられた理由だ。ほとんどのタスクで同じ精度、コードははるかに少ない。

### ステップ3: 数値例

3つのエンコーダー状態（大まかに「cat」、「sat」、「mat」）と最初に最もアライメントするデコーダー状態が与えられると、Attention分布は位置0に集中する。デコーダー状態が最後にアライメントするようにシフトすると、Attentionは位置2に移動する。コンテキストベクトルが追跡する。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

最初の行が勝つ。次にデコーダー状態を3番目のエンコーダー状態に近づけて重みがシフトするのを見る。それだけだ。Attentionは明示的なアライメントだ。

### ステップ4: なぜこれがTransformerへの橋なのか

上記の言語をQ/K/Vに変換する：

- **クエリ** = デコーダー状態 `s_{t-1}`
- **キー** = エンコーダー状態（スコアを付ける対象）
- **バリュー** = エンコーダー状態（加重して合計する対象）

古典的なAttentionでは、キーとバリューは同じものだ。Self-attentionはそれらを分離する：シーケンスをそれ自身に対してクエリでき、KとVに異なる学習済み射影を使う。Multi-head attentionは異なる学習済み射影で並列に実行する。Transformerはそのステージ全体を何度も積み重ねてRNNを廃棄する。

数学は同じだ。形状は同じだ。BahdanauのAttentionからスケールドdot-product attentionへの教育的なジャンプはほとんど表記の問題だ。

## 使ってみる

PyTorchとTensorFlowはAttentionを直接提供する。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

これがTransformerのAttentionレイヤーだ。5位置のクエリバッチ、10位置のキー/バリューバッチ、それぞれ128次元、8ヘッド。`output`は新しいコンテキスト拡張クエリだ。`weights`は視覚化できる5x10のアライメント行列だ。

### 古典的なAttentionがまだ重要な場面

- 教育目的。シングルヘッド、シングルレイヤー、RNNベースのバージョンはすべての概念を見える形にする。
- Transformerが収まらないデバイス上のシーケンスタスク。
- 2014〜2017年の論文。Bahdanauの規約を知らないと誤って読む。
- MTにおける細粒度のアライメント分析。生のAttention重みはTransformerモデルでも解釈可能性ツールであり、読むにはそれが何かを知る必要がある。

### Attention重みを説明として使う罠

Attention重みは解釈可能に見える。位置を跨いで合計が1になる重みだ。プロットできる。高い値は「これを見た」を意味する。レビュアーはそれが好きだ。

しかし見た目ほど解釈可能ではない。JainとWallace（2019）は、いくつかのタスクでモデルの予測を変えることなくAttention分布を順列替えして任意の代替に置き換えられることを示した。アブレーションや反事実確認なしにAttention重みを推論の証拠として報告してはならない。

## 成果物を出す

`outputs/prompt-attention-shapes.md`として保存する：

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## 演習

1. **易しい。** パディングトークンのAttention重みがゼロになるように`softmax`マスキングを実装する。可変長シーケンスを持つバッチでテストする。
2. **普通。** Luongの`general`形式にマルチヘッドAttentionを追加する。`d_h`を`n_heads`グループに分割し、ヘッドごとにAttentionを実行し、連結する。シングルヘッドケースが前の実装と一致することを確認する。
3. **難しい。** BahdanauのAttentionを使ったGRUエンコーダー-デコーダーをレッスン09のトイコピータスクで訓練する。シーケンス長に対する精度をプロットする。Attentionなしのベースラインと比較する。長さが増すにつれてギャップが広がり、Attentionがボトルネックをリフトすることを確認する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------------------|
| Attention | ものを見る | バリューシーケンスの加重平均で、重みはクエリとキーの類似度から計算される。 |
| クエリ、キー、バリュー | QKV | 3つの射影：Qが質問し、Kがマッチするもの、Vが返すもの。 |
| 加算Attention | Bahdanau | フィードフォワードスコア：`v^T tanh(W q + U k)`。 |
| 乗法Attention | Luongのdot/general | スコアは`q^T k`または`q^T W k`。安価でほとんどのタスクで同じ精度。 |
| アライメント行列 | きれいな図 | `(T_dec, T_enc)`グリッドとしてのAttention重み。モデルが何に注目したかを見るために読む。 |

## 参考資料

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 論文。
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 3つのスコアバリアントとその比較。
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — 解釈可能性の注意点。
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — PyTorchによる実行可能なウォークスルー。
