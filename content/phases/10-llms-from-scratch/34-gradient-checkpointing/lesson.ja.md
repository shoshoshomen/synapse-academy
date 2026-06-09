# 勾配チェックポインティングと活性化再計算

> バックプロパゲーションはすべての中間活性化を保持する。70Bパラメータで128Kコンテキストの場合、ランクあたり3TBの活性化となる。チェックポインティングはFLOPsとメモリをトレードオフする: 保存する代わりに再計算する。問題はどのセグメントを削除するかであり、答えは「すべて」ではない。


## 問題

Transformerを学習すると、各層について逆伝播で微分されるすべての演算への入力が保存される: アテンション入力、Q/K/Vプロジェクション、Softmax出力、FFN入力、正規化出力、残差ストリーム。隠れ次元 `d`、シーケンス長 `L`、バッチ `B` の層では、これは層あたりおよそ `12 * B * L * d` 個の浮動小数点数になる。

`d=8192, L=8192, B=1` では、BF16で800MB/層。64層モデルでは51GBの活性化——マイクロバッチサイズを掛ける前、アテンションSoftmax中間値（ヘッドあたり `L^2`）を加える前、テンソル並列の部分コピーを考慮する前の話だ。

両面の代償: BF16の重みと最適化器の状態は80GBに収まるかもしれないが、活性化がそれを超えさせる。勾配チェックポインティング（別名: 活性化再計算）が標準的な解決策だ。ほとんどの活性化を削除し、逆伝播中にフォワードを再実行して取り戻す。コスト: 追加のFLOPs。利点: メモリはチェックポイントセグメント数と総層数の比で削減される。

単純に行うと、チェックポインティングはステップあたり約33%多いフォワードパスFLOPsを要する。Korthikanti et al.の「スマートセレクション」による選択的チェックポインティングを使えば、FLOP オーバーヘッドを5%未満に抑えながら5倍のメモリ削減ができる。FP8行列積、FSDPオフロード、Expert並列MoEではこれが本当に重要になる: メモリも無駄な計算コストも両方許容できない。

## 概念

### 逆伝播が実際に必要とするもの

`output = layer(input)`。逆伝播は `grad_input` と `grad_params` を求める。それらを計算するには以下が必要だ:

- `input`（線形層の `grad_params = input.T @ grad_output` を計算するため）
- 一部の活性化導関数の中間値（ReLU/GELU/Softmaxの導関数は活性化値に依存する）

フォワードパスはこれらをautogradグラフに自動的に保存する。すべての `tensor.retain_grad()` と入力を保持する必要があるすべての演算が参照を保持する。

### ナイーブな完全チェックポインティング

ネットワークを `N` セグメントに分割する。フォワード中は各セグメントの*入力*のみを保存する。逆伝播が中間値を必要とするとき、セグメントのフォワードパスを再実行して具現化してから微分する。

例: 32層Transformerを1層ずつ32セグメントに分割。

- メモリ: 32個の層入力（小さい）vs 32 × （層あたりの活性化量）（巨大）。
- 追加計算: セグメントあたり1回の追加フォワード、つまり合計で約33%多いフォワードFLOPs（逆伝播は2×フォワードなので、完全ステップは1+2=3ではなく1+1+2=4単位になる）。

これがオリジナルのChen et al. 2016のレシピだ: バランスをとるためにL層ごとに `sqrt(L)` 回のチェックポイント。L=64の場合、8チェックポイント。

### 選択的チェックポインティング（Korthikanti 2022）

すべての活性化が同じコストを持つわけではない。アテンションSoftmax出力は `B*L*L*heads` であり、シーケンス長とともに*二乗的*に増加する。FFN隠れ活性化は `B*L*4d` で線形に増加する。長いシーケンスではSoftmaxが支配的だ。

選択的チェックポインティングは安価に保存できる活性化（線形プロジェクション、残差）を保持し、高コストなもの（アテンション）のみを再計算する。最小のFLOPsで再計算しながら `O(L^2)` のメモリを節約できる。

Megatron-Coreはこれを「選択的」活性化再計算として実装している。ほとんどの2024年以降のフロンティア学習ランで使用されている。

### オフロード

再計算の代替: 活性化をフォワードと逆伝播の間にCPU RAMに転送する。PCIe帯域幅が必要; 再実体化のコストを超える帯域幅が空いている場合に有利。混合戦略が一般的: 一部の層をチェックポイント、他をオフロード。

FSDP2はオフロードをファーストクラスオプションとして搭載している。オフロードはGPUがメモリでボトルネックだがCPU-GPU転送に余裕がある場合に輝く。

### 再計算コストモデル

`L` 層のうち `k` 層ごとにチェックポイントするナイーブな場合のステップあたりFLOPs:

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

選択的チェックポインティングでは、層全体ではなくアテンションカーネルのみを再計算する:

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### メモリ削減モデル

層あたりの活性化量: `A`。`L` 層の場合、総活性化メモリ: `L * A`。

完全チェックポイント（セグメントサイズ1）: `L * input_volume` のみを保存（標準Transformerで約 `L * 1/10 A`）。節約は約 `9 * L * A * 1/10`。

`k` 層ごとのチェックポイント: `L/k * A` を保存し、さらにアクティブセグメント内のk-1層分を保存。

`k = sqrt(L)` の場合、メモリと再計算コストの両方が `sqrt(L)` でスケールする——一様コスト層に対する最適トレードオフ。

### チェックポインティングしない場合

- すでにインフライトのパイプラインステージの最内層。いずれにせよ完了する必要がある。
- ステージの計算を支配する最初と最後の層（Transformerでは稀）。
- FlashAttentionを使用しているアテンションカーネル——Flashはすでに高速にSoftmaxを再計算するため、層レベルの追加チェックポインティングの恩恵は少ない。

### 実装パターン

1. **関数ラッパー:** セグメントを `torch.utils.checkpoint.checkpoint(fn, input)` でラップする。PyTorchは `input` のみを保存し、逆伝播でその他すべてを再計算する。

2. **デコレータ方式:** 層をチェックポイント可能とラベル付け; トレーナーがどのセグメントをラップするかを設定時に決定する。

3. **手動明示的再計算:** 後退パスを自分で記述し、保存した入力でカスタム `recompute_forward` を呼び出す。

3つの方法はすべて同じ関数的結果を与える。ラッパーが標準的な慣用法だ。

### TP / PP / FP8との相互作用

- **テンソル並列:** チェックポイント入力は再計算時にギャザーまたは再スキャッターが必要; 通信コストを扱う。
- **パイプライン並列:** 典型的なパターンは各パイプラインステージのフォワードをチェックポイントし、逆順マイクロバッチが活性化メモリを再利用できるようにする。
- **FP8再計算:** 再計算中に更新されるamax履歴は元のフォワードと一致する必要がある。そうでないとFP8スケールがずれる。ほとんどのフレームワークはスケールをスナップショットする。

## 実装する

### ステップ1: セグメントを持つおもちゃモデル

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### ステップ2: すべての活性化を必要とするナイーブな逆伝播

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### ステップ3: k層ごとのチェックポイントメモリ

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### ステップ4: コストモデル

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### ステップ5: メモリ推定器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### ステップ6: 最適セグメントサイズ

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### ステップ7: 選択的チェックポイント判断

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 使ってみる

- **torch.utils.checkpoint**: `from torch.utils.checkpoint import checkpoint` — PyTorchの標準ラッパー。関数をラップし、入力のみを保存して逆伝播で再計算する。
- **Megatron-Core活性化再計算**: `selective`、`full`、`block` モードをサポート。2024年以降のフロンティア学習での標準。
- **FSDP2オフロード**: `module.to_empty(device="cpu")` と `offload_policy` を使ってFSDP2で活性化をCPUにシャードする（再計算の代替）。
- **DeepSpeed ZeRO-Offload**: チェックポインティングを補完する最適化器状態と活性化のCPUオフロード。

## 成果物を出す

このレッスンは `outputs/prompt-activation-recompute-policy.md` を生成する——モデルの設定（層数、隠れ次元、シーケンス長、バッチ）と利用可能なGPUメモリを取り込み、層ごとの再計算ポリシー（なし/選択的/完全/オフロード）を出力するプロンプトだ。

## 演習

1. 正確性を検証する。完全活性化の `model_forward` + `model_backward` と、セグメント化した `model_forward_checkpointed` + `model_backward_checkpointed` を実行する。パラメータ勾配はマシン精度まで同一でなければならない。

2. セグメントサイズ `k` を1から `L` まで変化させる。FLOPオーバーヘッドとメモリをプロットする。曲線の変曲点を求める。

3. 選択的チェックポインティングを実装する: アテンションモジュールの入力は保存するが中間値は保存しない。32層モデル（seq=8192）に対して、完全層チェックポインティングと比較したFLOPオーバーヘッドを測定する。

4. オフロードを追加する。セグメント入力をシミュレートされた「CPUバッファ」（別のリスト）に保存する。「PCIe帯域幅」を bytes/time として測定し、オフロードと再計算の損益分岐点を求める。

5. 実際のPyTorch Transformerを `torch.utils.checkpoint` あり/なしでベンチマーク。メモリ（`torch.cuda.max_memory_allocated` 経由）とステップ時間を測定する。

## キーワード

| 用語 | よく言われる表現 | 実際の意味 |
|------|----------------|-----------|
| 勾配チェックポインティング | 「フォワードをやり直してメモリを節約」 | セグメント入力のみを保存; 逆伝播中に中間値を再計算して勾配サポートテンソルを取得 |
| 活性化再計算 | 「チェックポインティングと同じ」 | 同じ技術のHPC風の名称 |
| セグメントサイズ（k） | 「チェックポイントあたりの層数」 | 中間値を削除して一緒に再実体化する層の数 |
| 選択的チェックポインティング | 「Korthikantiのトリック」 | 高コスト活性化（アテンションSoftmax）のみを再計算; 安価なものは保持 |
| 完全チェックポインティング | 「ナイーブな版」 | 各セグメントのすべての層の中間値を再計算 |
| ブロックチェックポインティング | 「粗い粒度」 | Transformerブロック全体をチェックポイント; 最大の粒度 |
| FLOPオーバーヘッド | 「計算税」 | ステップあたりの追加FLOPs = (再計算FLOPs) / (fwd + bwd FLOPs); ナイーブ33%、選択的5% |
| 活性化オフロード | 「CPUに転送」 | フォワード→逆伝播間に活性化をCPU RAMに移動; 再計算の代替 |
| sqrt-Lルール | 「古典的最適値」 | 一様コスト層の場合、最適なチェックポイント間隔は sqrt(L) 層 |
| アテンションSoftmax量 | 「O(L^2)問題」 | L^2 * ヘッド数 * バッチ浮動小数点数; 長いコンテキストで活性化メモリを支配 |

## 参考資料

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- 勾配チェックポインティングを形式化したオリジナル論文
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- 選択的活性化再計算と形式的コスト分析
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- 逆モード再実体化による代替の定数メモリアプローチ
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- 大規模での活性化オフロード
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) -- 標準API
- [Megatron-Core activation recomputation documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- 選択的、完全、ブロックモード
