# アテンション変形 — スライディングウィンドウ、スパース、差分

> Full Attentionは円だ。すべてのトークンがすべてのトークンを見て、メモリがその代償を払う。4つの変形は円の形を曲げて、コストの半分を取り戻す。


## 問題

Full Attentionはシーケンス長で`O(N²)`のメモリと`O(N²)`の計算コストがかかる。128Kコンテキストのつある Llama 3 70Bでは、レイヤーごとに160億のAttentionエントリ、それが80レイヤー分ある。Flash Attention（レッスン12）は`O(N²)`のアクティベーションメモリを隠すが、算術コストは変わらない——すべてのトークンはまだ他のすべてのトークンに対してAttentionを行う。

3つの変形クラスがAttention行列のトポロジー自体を変える：

1. **スライディングウィンドウAttention（SWA）。** 各トークンはフルプレフィックスではなく、隣接する固定ウィンドウにAttentionする。メモリと計算が`O(N · W)`に下がる（`W`はウィンドウ）。Gemma 2/3、Mistral 7Bの初期レイヤー、Phi-3-Long。
2. **スパース / ブロックAttention。** 選ばれたペア`(i, j)`のみがスコアリングされる；残りはゼロ重みに強制される。Longformer、BigBird、OpenAIスパーストランスフォーマー。
3. **差分Attention。** 別々のQ/K投影で2つのAttentionマップを計算し、一方から他方を引く。最初の数トークンに重みを流す「Attentionシンク」を排除する。MicrosoftのDIFF Transformer（2024年）。

これらは共存する。2026年のフロンティアモデルはしばしばそれらを混合する：ほとんどのレイヤーはSWA-1024、5番目ごとにグローバルFull Attention、そして検索をクリーンアップする差分ヘッドがいくつか。Gemma 3の5:1のSWA対グローバル比が現在の教科書デフォルトだ。

## 概念

### スライディングウィンドウAttention（SWA）

位置`i`の各クエリは`[i - W, i]`（因果的SWA）または`[i - W/2, i + W/2]`（双方向）の位置にのみAttentionする。ウィンドウ外のトークンはスコア行列で`-inf`を取得する。

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

`N = 8192`と`W = 1024`では、スコア行列は期待値で1024 × 8192の非ゼロ行を持つ——8倍の削減だ。

**SWAでKVキャッシュが縮小する。** レイヤーごとにKとVの最後の`W`トークンのみを保持する必要がある。Gemma-3風のコンフィグ（1024ウィンドウ、128Kコンテキスト）では、KVキャッシュが128倍下がる。

**品質コスト。** SWAのみのTransformerは長距離検索に苦労する。修正：SWAレイヤーをFull Attentionレイヤーと交互に配置する。Gemma 3は5:1のSWA:グローバルを使う。Mistral 7Bは因果的SWAスタックを使い、そこでは情報が重なり合うウィンドウを通じて「前方に流れる」——各レイヤーは有効受容野を`W`だけ拡張し、`L`レイヤー後にモデルは`L × W`トークン前まで参照できる。

### スパース / ブロックAttention

`N × N`のスパースパターンを事前に選ぶ。3つの標準的な形状：

- **ローカル + ストライド（OpenAIスパーストランスフォーマー）。** 最後の`W`トークンとその前の`stride`番目ごとのトークンにAttentionする。ローカルと長距離の両方を`O(N · sqrt(N))`計算でキャプチャ。
- **Longformer / BigBird。** ローカルウィンドウ + 全員にAttentionし全員からAttentionされるグローバルトークンの小さいセット（例：`[CLS]`） + ランダムスパースリンク。マッチした品質で経験的に2倍のコンテキスト。
- **Native Sparse Attention（DeepSeek、2025年）。** どの`(Q, K)`のブロックが重要かを学習する；カーネルレベルでゼロブロックをスキップ。FlashAttention互換。

スパースAttentionはカーネルエンジニアリングの話だ。数学はシンプル（スコア行列をマスクする）；利益はゼロエントリをSRAMにロードしないことから来る。FlashAttention-3と2026年のFlexAttention APIがPyTorchでカスタムスパースパターンをファーストクラスにする。

### 差分Attention（DIFF Transformer、2024年）

通常のAttentionには「Attentionシンク」問題がある：softmaxはすべての行が1に合計するように強制するので、何にもAttentionしたくないトークンは最初のトークン（または最初のいくつか）に重みを捨てる。これは本当のコンテンツに行くべき容量を奪う。

差分Attentionはこれを**2つの**Attentionマップを計算して引き算することで修正する：

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

`λ`は学習済みスカラー（通常0.5〜0.8）。A1は本物のコンテンツ重みをキャプチャ；A2はシンクをキャプチャ。引き算がシンクをキャンセルし、重みを関連するトークンに再配分する。

報告された結果（Microsoft 2024年）：5〜10%低いパープレキシティ、同じ学習長での1.5〜2倍の長い有効コンテキスト、鋭いneedle-in-haystackの検索。

### 変形の比較

| 変形 | 計算 | KVキャッシュ | full vs 品質 | 本番での使用 |
|---------|---------|----------|-----------------|----------------|
| Full Attention | O(N²) | O(N) レイヤーごと | ベースライン | すべてのモデルのデフォルトレイヤー |
| SWA（ウィンドウ1024） | O(N·W) | O(W) レイヤーごと | -0.1 ppl、グローバルレイヤーで良好 | Gemma 2/3、Phi-3-Long |
| ローカル + ストライドスパース | O(N·√N) | 混合 | SWA同等 | OpenAIスパーストランスフォーマー、Longformer |
| BigBird（ローカル + グローバル + ランダム） | O(N) おおよそ | 混合 | 2倍コンテキストでfullに匹敵 | 初期の長コンテキストBERT |
| Native Sparse（DeepSeek-V3.2） | O(N · アクティブフラクション) | O(N) | 0.05 ppl以内 | DeepSeek-V3.2、2025年 |
| 差分 | O(2·N²) | O(2N) | pplで-5〜-10% | DIFF Transformer、2026年初期モデル |

## 実装する

`code/main.py`を参照。おもちゃのシーケンスで、Full、SWA、ローカル+ストライド、差分Attentionを並べて比較する因果マスクコンパレーターを実装する。

### ステップ1：フル因果マスク（ベースライン）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

レッスン07のベースライン。下三角；対角線より上はゼロ重み。

### ステップ2：スライディングウィンドウ因果マスク

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

パラメータは1つ——`window`。`window >= n`のとき、フル因果Attentionに戻る。`window = 1`のとき、各トークンは自分自身にのみAttentionする。

### ステップ3：ローカル + ストライドスパースマスク

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

密なローカルウィンドウとシーケンスの先頭まで`stride`番目ごとのトークン。受容野は追加レイヤーでログステップで成長する。

### ステップ4：差分Attention

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

2つのAttentionパス、学習済み混合係数で引き算する。コードでは単一と差分のAttentionシンクのヒートマップを比較し、シンクの崩壊を観察する。

### ステップ5：KVキャッシュサイズ

各変形について`N = 131072`でレイヤーごとのキャッシュサイズを出力せよ。SWAとスパース変形は10〜100倍下がる。差分は2倍になる。メモリの支払いを意識的に行う。

## 使ってみる

2026年の本番パターン：

```python
from transformers import AutoModelForCausalLM
# Gemma 3はSWA（window=1024）とグローバルレイヤーを5:1で混合する。
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+のFlexAttentionはマスク関数を受け付ける：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

これはカスタムTritonカーネルにコンパイルされる。一般的なパターンでFlashAttention-3速度の10%以内、マスク関数はPythonのcallableだ。

**各変形を選ぶ時：**

- **Pure Full Attention** — 〜16Kコンテキストまでのすべてのレイヤー、または検索品質が最優先の時。
- **SWA + グローバルミックス** — 長コンテキスト（>32K）、学習と推論がメモリバウンド。32K以上の2026年デフォルト。
- **スパースブロックAttention** — カスタムカーネル、カスタムパターン。特殊なワークロード（検索、オーディオ）に限定。
- **差分Attention** — Attentionシンクの汚染が問題となるすべてのワークロード（長コンテキストRAG、needle-in-haystack）。

## 成果物を出す

`outputs/skill-attention-variant-picker.md`を参照。このスキルは、目標コンテキスト長・検索要件・学習/推論計算プロファイルを考慮して、新しいモデルのAttentionトポロジーを選択する。

## 演習

1. **易.** `code/main.py`を実行せよ。`window=4`のSWAが行ごとに最後の4トークン外のすべてをゼロにすることを確認せよ。`window=n`がフル因果Attentionをビット同一で再現することを確認せよ。
2. **中.** レッスン07のカプストーンの上に`window=1024`の因果SWAを実装せよ。tinyshakespeareで1,000ステップ学習せよ。Full Attentionに比べてバリデーション損失はどれだけ悪化するか？ピークメモリはどれだけ下がるか？
3. **難.** カプストーンモデルにGemma-3風の5:1レイヤーミックス（5 SWA、1 グローバル）を実装せよ。マッチしたパラメータでピュアSWAとピュアグローバルのベースラインに対して損失、メモリ、生成品質を比較せよ。
4. **難.** ヘッドごとに学習済み`λ`で差分Attentionを実装せよ。合成検索タスク（1つのニードル、2,000のディストラクター）で学習せよ。マッチしたパラメータでシングルAttentionベースラインに対する検索精度を測定せよ。

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| スライディングウィンドウAttention（SWA） | 「ローカルAttention」 | 各クエリは最後の`W`トークンにAttentionする；KVキャッシュは`O(W)`に縮小。 |
| 有効受容野（Effective receptive field） | 「モデルが後ろどこまで見えるか」 | ウィンドウ`W`の`L`レイヤーSWAスタックでは、最大`L × W`トークン。 |
| Longformer / BigBird | 「ローカル + グローバル + ランダム」 | 常に参照するグローバルトークンを少数持つスパースパターン；初期の長コンテキストアプローチ。 |
| Native Sparse Attention | 「DeepSeekのカーネルトリック」 | ブロックレベルのスパース性を学習；品質を保ちながらカーネルレベルでゼロブロックをスキップ。 |
| 差分Attention（Differential attention） | 「2つのマップ、1つが引き算」 | DIFF Transformer：Attentionシンクをキャンセルするために学習済み`λ`倍の2番目のAttentionマップを最初のものから引く。 |
| Attentionシンク（Attention sink） | 「重みがトークン0に流れる」 | Softmax正規化は行が1に合計するよう強制する；無関係なクエリが位置0に重みを捨てる。 |
| FlexAttention | 「マスク-as-Python」 | PyTorch 2.5+ APIが任意のマスク関数をFlashAttention形状のカーネルにコンパイル。 |
| レイヤータイプミックス（Layer type mix） | 「5:1のSWA対グローバル」 | より低いメモリで品質を保つためにスタック内でスパースとFull Attentionレイヤーを交互配置。 |

## 参考資料

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — 標準的なスライディングウィンドウ + グローバルトークン論文。
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — ローカル + グローバル + ランダム。
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAIのローカル+ストライドパターン。
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1のSWA:グローバルミックス。
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — 現在の教科書デフォルトのwindow=1024の5:1ミックス。
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer論文。
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2の学習済みスパースAttention。
- [PyTorch — FlexAttentionブログとドキュメント](https://pytorch.org/blog/flexattention/) — 使ってみるでのマスク-as-callableパターンのAPIリファレンス。
