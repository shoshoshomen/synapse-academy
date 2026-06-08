# 数値安定性

> 浮動小数点数は漏れのある抽象化です。学習中に必ず問題を引き起こしますが、予兆なくやってきます。


## 学習目標

- 最大値引き算トリックを使って数値的に安定なsoftmaxとlog-sum-expを実装する
- 浮動小数点演算におけるオーバーフロー、アンダーフロー、桁落ちを特定する
- 中心差分法を使って解析的勾配を数値的勾配と照合して検証する
- 学習においてbfloat16がfloat16より優先される理由と、損失スケーリングが勾配アンダーフローを防ぐ仕組みを説明する

## 問題の背景

モデルが3時間学習し、その後損失がNaNになりました。print文を追加します。ステップ9,000ではロジットは正常です。ステップ9,001で`inf`になります。ステップ9,002では勾配がすべて`nan`になり、学習は停止します。

あるいは：モデルは最後まで学習を完了しますが、精度が論文の主張より2%低いです。すべてを確認します。アーキテクチャは一致しています。ハイパーパラメータも一致しています。データも一致しています。問題は、論文がfloat32を使用し、あなたが適切なスケーリングなしでfloat16を使用したことです。32ビット分の丸め誤差の蓄積が静かに精度を食いつぶしていました。

あるいは：クロスエントロピー損失をゼロから実装します。小さなロジットでは動作します。ロジットが100を超えると`inf`を返します。`exp(100)`はfloat32が表現できる最大値を超えるためsoftmaxがオーバーフローしました。すべてのMLフレームワークは2行のトリックでこれを処理しています。あなたはそのトリックの存在を知らなかったのです。

数値安定性は理論的な懸念ではありません。学習が成功するか静かに失敗するかの違いです。デバッグするすべての深刻なMLのバグは、最終的には浮動小数点数に行き着きます。

## 概念

### IEEE 754：コンピュータが実数を格納する方法

コンピュータはIEEE 754規格に従って浮動小数点値として実数を格納します。浮動小数点数には3つのパーツがあります：符号ビット、指数、仮数（有効数）。

```
Float32レイアウト（合計32ビット）：
[1 符号] [8 指数] [23 仮数]

値 = (-1)^符号 * 2^(指数 - 127) * 1.仮数
```

仮数は精度を決定します（有効数字の桁数）。指数は範囲を決定します（数値がどれだけ大きくまたは小さくなれるか）。

```
形式      ビット  指数  仮数  十進桁数     範囲（概算）
float64    64     11    52    ~15-16      +/- 1.8e308
float32    32     8     23    ~7-8        +/- 3.4e38
float16    16     5     10    ~3-4        +/- 65,504
bfloat16   16     8     7     ~2-3        +/- 3.4e38
```

float32は約7桁の十進精度を提供します。つまり1.0000001と1.0000002は区別できますが、1.00000001と1.00000002は区別できません。7桁を超えると、すべて丸め誤差です。

float16は約3桁しかありません。表現できる最大値は65,504です。MLではロジット、勾配、活性化が日常的にこれを超えるため、恐ろしいほど小さい値です。

bfloat16はfloat16の範囲問題に対するGoogleの回答です。float32と同じ8ビット指数（同じ範囲、最大3.4e38）を持ちますが、仮数ビットは7つのみ（float16より精度が低い）です。ニューラルネットワークの学習では、精度より範囲が重要なため、通常はbfloat16が勝ちます。

### 0.1 + 0.2 が 0.3 に等しくない理由

0.1という数は2進浮動小数点数で正確に表現できません。2進数では循環小数になります：

```
0.1を2進数で表すと = 0.0001100110011001100110011... （無限に繰り返す）
```

Float32はこれを仮数の23ビットで打ち切ります。格納される値は約0.100000001490116です。同様に、0.2は約0.200000002980232として格納されます。それらの和は0.300000004470348であり、0.3ではありません。

```
Pythonで：
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

これがMLで重要な理由：

1. `if loss < threshold`のような損失比較が誤った答えを返すことがある
2. 多くの小さな値の累積（数千ステップにわたる勾配更新）が真の和からずれる
3. `==`で浮動小数点数を比較すると、チェックサムや再現性テストが失敗する

修正方法：浮動小数点数を`==`で比較しない。`abs(a - b) < epsilon`または`math.isclose()`を使用する。

### 桁落ち（Catastrophic Cancellation）

2つのほぼ等しい浮動小数点数を引き算すると、有効桁が打ち消し合い、丸め誤差だけが残って主要桁に昇格します。

```
a = 1.0000001    （float32では1.00000011920929として格納）
b = 1.0000000    （float32では1.00000000000000として格納）

真の差：  0.0000001
計算値：  0.00000011920929

相対誤差：19.2%
```

たった1回の引き算で19%の相対誤差です。MLでは、以下の場合に発生します：

- 大きな平均を持つデータの分散計算：`E[x^2] - E[x]^2`でE[x]が大きい場合
- ほぼ等しい対数確率の引き算
- 小さすぎるイプシロンを使った数値微分

修正方法：大きくてほぼ等しい数の引き算を避けるように式を変形する。分散にはWelfordアルゴリズムを使用するか、先にデータを中心化する。対数確率には、全体を通じて対数空間で作業する。

### オーバーフローとアンダーフロー

オーバーフローは結果が表現できないほど大きくなるときに発生します。アンダーフローは結果が（正の最小表現可能な数より）小さくなるときに発生します。

```
Float32の境界：
  最大値：          3.4028235e+38
  最小正規化正数：  1.175e-38
  最小非正規化正数：1.401e-45
  オーバーフロー：  3.4e38を超えるとinfになる
  アンダーフロー：  1.4e-45を下回ると0.0になる
```

`exp()`関数はMLにおけるオーバーフローの主要原因です：

```
exp(88.7)  = 3.40e+38   （float32にかろうじて収まる）
exp(89.0)  = inf         （オーバーフロー）
exp(-87.3) = 1.18e-38   （アンダーフローのすぐ上）
exp(-104)  = 0.0         （アンダーフローしてゼロになる）
```

`log()`関数は逆方向に問題が起きます：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      （問題なし）
log(1e-46) = -inf        （入力がゼロにアンダーフローしてlog(0) = -inf）
```

MLでは、`exp()`はsoftmax、sigmoid、確率計算に現れます。`log()`はクロスエントロピー、対数尤度、KLダイバージェンスに現れます。適切なトリックなしには`log(exp(x))`の組み合わせが地雷です。

### Log-Sum-Expトリック

`log(sum(exp(x_i)))`を直接計算するのは数値的に危険です。`x_i`が大きければ`exp(x_i)`がオーバーフローします。すべての`x_i`が非常に負なら、すべての`exp(x_i)`がゼロにアンダーフローして`log(0)`は`-inf`になります。

トリック：指数化する前に最大値を引き算します。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

なぜ機能するか：`max(x)`を引いた後、最大の指数は`exp(0) = 1`です。オーバーフローは不可能です。和の中の少なくとも1項が1なので、和は少なくとも1となり、`log(1) = 0`です。`-inf`へのアンダーフローも不可能です。

証明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    （cを加減する）
= log(sum(exp(x_i - c) * exp(c)))               （exp(a+b) = exp(a)*exp(b)）
= log(exp(c) * sum(exp(x_i - c)))               （exp(c)を括り出す）
= c + log(sum(exp(x_i - c)))                    （log(a*b) = log(a) + log(b)）
```

`c = max(x)`とすればオーバーフローが排除されます。

このトリックはMLのあらゆる場所に登場します：
- Softmax正規化
- クロスエントロピー損失の計算
- 系列モデルにおける対数確率の合計
- 混合ガウスモデル
- 変分推論

### Softmaxが最大値引き算トリックを必要とする理由

Softmaxはロジットを確率に変換します：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

トリックなしでは、[100, 101, 102]のロジットはオーバーフローを引き起こします：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
合計     = 2.99e44

これらはfloat32でオーバーフローする（float32のexp(88.7)がすでに限界）：
exp(100) = float32ではinf。
```

トリックを使うと、max(x) = 102を引き算：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
合計 = 1.503

softmax = [0.090, 0.245, 0.665]
```

確率は同一です。計算は安全です。これは最適化ではありません。正確性のための要件です。

### NaNとInf：検出と防止

`nan`（非数）と`inf`（無限大）は計算を通じてウイルス的に伝播します。勾配更新中に1つの`nan`が重みを`nan`にし、それが後続のすべての出力を`nan`にします。1ステップで学習は終わりです。

`inf`が現れる原因：
- 大きな正の数の`exp()`
- ゼロ除算：`1.0 / 0.0`
- 累積でのfloat32オーバーフロー

`nan`が現れる原因：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 負の数の`sqrt()`
- 負の数の`log()`
- 既存の`nan`を含む算術演算

検出：

```python
import math

math.isnan(x)       # xがnanならTrue
math.isinf(x)       # xが+infまたは-infならTrue
math.isfinite(x)    # xがnanでもinfでもなければTrue
```

防止策：

1. `exp()`の入力をクランプ：`exp(clamp(x, -80, 80))`
2. 分母にイプシロンを追加：`x / (y + 1e-8)`
3. `log()`内にイプシロンを追加：`log(x + 1e-8)`
4. 安定な実装を使用（log-sum-exp、安定なsoftmax）
5. 重みの爆発を防ぐための勾配クリッピング
6. デバッグ中は各フォワードパス後に`nan`/`inf`をチェック

### 数値的勾配チェック

解析的勾配（誤差逆伝播から得られる）にはバグがある可能性があります。数値的勾配チェックは有限差分法で勾配を計算することで検証します。

中心差分の公式：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

これはO(h^2)の精度で、前進差分`(f(x+h) - f(x)) / h`のO(h)よりはるかに優れています。

hの選択：大きすぎると近似が誤り。小さすぎると桁落ちが答えを破壊します。`h = 1e-5`〜`1e-7`が一般的です。

チェック：解析的勾配と数値的勾配の相対的な差を計算します。

```
相対誤差 = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

目安：
- 相対誤差 < 1e-7：完璧、勾配は正しい
- 相対誤差 < 1e-5：許容範囲、おそらく正しい
- 相対誤差 > 1e-3：何か問題がある
- 相対誤差 > 1：勾配が完全に間違っている

新しい層や損失関数を実装するときは必ず勾配をチェックしてください。PyTorchはこのために`torch.autograd.gradcheck()`を提供しています。

### 混合精度学習

現代のGPUには、float16の行列積を計算するための特殊なハードウェア（テンソルコア）があり、float32より2〜8倍高速です。混合精度学習はこれを活用します：

```
1. float32のマスターウェイトコピーを維持
2. フォワードパスをfloat16で実行（高速）
3. 損失をfloat32で計算（オーバーフロー防止）
4. バックワードパスをfloat16で実行（高速）
5. 勾配をfloat32にスケール
6. float32マスター重みを更新
```

純粋なfloat16学習の問題：勾配はしばしば非常に小さい（1e-8以下）。Float16は約6e-8以下のものをゼロにアンダーフローします。すべての勾配更新がゼロになるため、モデルが学習を止めます。

修正方法は損失スケーリングです：

```
1. 損失に大きなスケール係数を掛ける（例：1024）
2. バックワードパスが（損失 × 1024）の勾配を計算
3. すべての勾配が1024倍大きい（float16のアンダーフロー以上に押し上げる）
4. 重みを更新する前に勾配を1024で割る
5. 正味の効果：同じ更新、しかしアンダーフローなし
```

動的損失スケーリングはスケール係数を自動的に調整します。大きな値（65536）から始めます。勾配が`inf`にオーバーフローした場合、半分にします。Nステップがオーバーフローなしで通過した場合、2倍にします。

### bfloat16 vs float16：なぜbfloat16が学習に有利か

```
float16:   [1 符号] [5 指数]  [10 仮数]
bfloat16:  [1 符号] [8 指数]  [7 仮数]
```

float16はより高い精度（仮数10ビット対7ビット）を持ちますが、範囲が限られています（最大約65,504）。bfloat16は精度が低いですが、float32と同じ範囲（最大約3.4e38）を持ちます。

ニューラルネットワークの学習において：

- 学習のスパイク中に活性化とロジットが定期的に65,504を超えます。float16はオーバーフローしますが、bfloat16は処理できます。
- float16では損失スケーリングが必要ですが、bfloat16では通常不要です。その範囲が勾配の大きさのスペクトルをカバーするからです。
- bfloat16はfloat32の単純な切り捨てです：仮数の下位16ビットを削除するだけ。変換は自明で指数部において損失がありません。

float16は値が有界で精度が重要な推論に適しています。bfloat16は範囲が重要な学習に適しています。これがTPUと最新NVIDIAのGPU（A100、H100）がbfloat16のネイティブサポートを持つ理由です。

### 勾配クリッピング

勾配の爆発は、多くの層を通じて勾配が指数関数的に増大するときに発生します（RNN、深いネットワーク、トランスフォーマーで一般的）。1つの大きな勾配が1ステップですべての重みを破壊する可能性があります。

クリッピングの2つの種類：

**値によるクリッピング：**各勾配要素を独立にクランプします。

```
grad = clamp(grad, -max_val, max_val)
```

シンプルですが、勾配ベクトルの方向を変える可能性があります。

**ノルムによるクリッピング：**勾配ベクトル全体のノルムが閾値を超えないようにスケールします。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

勾配の方向を保持します。これが`torch.nn.utils.clip_grad_norm_()`の動作です。標準的な選択です。

典型的な値：トランスフォーマーでは`max_norm=1.0`、RLでは`max_norm=0.5`、シンプルなネットワークでは`max_norm=5.0`。

勾配クリッピングはハックではありません。セーフティメカニズムです。これなしでは、1つの外れ値バッチが何週間もの学習を台無しにするほど大きな勾配を生成する可能性があります。

### 数値安定化器としての正規化層

バッチ正規化、レイヤー正規化、RMS正規化は通常、学習収束を助ける正則化手法として紹介されます。しかし数値安定化器でもあります。

正規化なしでは、活性化は層を通じて指数関数的に増大したり減少したりする可能性があります：

```
層1：値が [0, 1] の範囲
層5：値が [0, 100] の範囲
層10：値が [0, 10,000] の範囲
層50：値が [0, inf] の範囲
```

正規化は各層で活性化を再中心化し再スケールします：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常1e-5）はすべての活性化が同一のときのゼロ除算を防ぎます。学習可能なパラメータ`gamma`と`beta`により、ネットワークが必要なスケールを復元できます。

これにより、ネットワーク全体を通じて値が数値的に安全な範囲に保たれ、フォワードパスでのオーバーフローとバックワードパスでの勾配爆発の両方を防ぎます。

### よくあるML数値バグ

**バグ：数エポック後に損失がNaNになる。**
原因：ロジットが大きくなりすぎてsoftmaxがオーバーフロー。または学習率が高すぎて重みが発散。
修正：安定なsoftmax（最大値引き算）を使用、学習率を下げ、勾配クリッピングを追加。

**バグ：損失がlog(クラス数)で止まっている。**
原因：モデルの出力がほぼ一様確率。多くの場合、勾配が消失しているかモデルがまったく学習していないことを意味する。
修正：データラベルが正しいか確認、損失関数を検証、死んだReLUをチェック。

**バグ：検証精度が期待より1〜3%低い。**
原因：適切な損失スケーリングなしの混合精度。勾配アンダーフローが静かに小さな更新をゼロにする。
修正：動的損失スケーリングを有効にするか、bfloat16に切り替える。

**バグ：一部の層の勾配ノルムが0.0。**
原因：死んだReLUニューロン（すべての入力が負）、またはfloat16アンダーフロー。
修正：LeakyReLUまたはGELUを使用、勾配スケーリングを使用、重みの初期化を確認。

**バグ：モデルが1つのGPUでは動くが別のGPUでは異なる結果を出す。**
原因：非決定論的な浮動小数点の累積順序。GPUの並列リダクションは異なるハードウェアで異なる順序で合計するが、浮動小数点の加算は結合的ではない。
修正：小さな差（1e-6）を受け入れるか、`torch.use_deterministic_algorithms(True)`を設定して速度低下を受け入れる。

**バグ：損失計算で`exp()`が`inf`を返す。**
原因：最大値引き算トリックなしに生のロジットを`exp()`に渡している。
修正：log-sum-expを内部実装している`torch.nn.functional.log_softmax()`を使用する。

**バグ：float32からfloat16に切り替えると学習が発散する。**
原因：float16は6e-8以下の勾配の大きさや65,504以上の活性化を表現できない。
修正：損失スケーリング付きの混合精度（AMP）を使用するか、代わりにbfloat16を使用する。

## 実装

### ステップ1：浮動小数点精度の限界を示す

```python
print("=== 浮動小数点精度 ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"差: {(0.1 + 0.2) - 0.3:.2e}")
```

### ステップ2：単純なsoftmax vs 安定なsoftmaxを実装

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"単純：  {softmax_naive(safe_logits)}")
print(f"安定：  {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"安定：  {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits)は[nan, nan, nan]を返す
```

### ステップ3：安定なlog-sum-expを実装

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"単純：  {logsumexp_naive(safe):.6f}")
print(f"安定：  {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"安定：  {logsumexp_stable(large):.6f}")
# logsumexp_naive(large)はinfを返す
```

### ステップ4：安定なクロスエントロピーを実装

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"単純：  {cross_entropy_naive(true_class, logits):.6f}")
print(f"安定：  {cross_entropy_stable(true_class, logits):.6f}")
```

### ステップ5：勾配チェック

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 活用

### 混合精度のシミュレーション

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 勾配クリッピング

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"元のノルム: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"クリップ後のノルム: {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"方向が保持されている: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf の検出

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"警告 {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

すべてのエッジケースを示した完全な実装は`code/numerical.py`を参照してください。

## 成果物

このレッスンでは以下を生成します：
- 安定なsoftmax、log-sum-exp、クロスエントロピー、勾配チェック、混合精度シミュレーションを含む`code/numerical.py`
- 学習のNaN/Infや数値問題を診断するための`outputs/prompt-numerical-debugger.md`

これらの安定な実装は、フェーズ3で学習ループを構築するとき、フェーズ4でアテンション機構を実装するときに再登場します。

## 演習

1. **桁落ち。** 単純な公式`E[x^2] - E[x]^2`をfloat32で使って[1000000.0, 1000001.0, 1000002.0]の分散を計算します。次にWelfordのオンラインアルゴリズムを使って計算します。真の分散（0.6667）と比べて誤差を比較します。

2. **精度の探索。** Pythonで`1.0 + x == 1.0`となるような最小の正のfloat32値`x`を見つけます。これが機械イプシロンです。`numpy.finfo(numpy.float32).eps`と一致することを確認します。

3. **log-sum-expのエッジケース。** `logsumexp_stable`関数を以下でテストします：(a) すべての値が等しい場合、(b) 1つの値が他よりはるかに大きい場合、(c) すべての値が非常に負（-1000）。単純版が失敗する場合でも正しい結果が得られることを確認します。

4. **ニューラルネットワーク層の勾配チェック。** 単一の全結合層`y = Wx + b`とその解析的な逆伝播を実装します。`numerical_gradient`を使って3×2の重み行列の正確性を確認します。

5. **損失スケーリングの実験。** float16での学習をシミュレートします：[1e-9, 1e-3]の範囲のランダムな勾配を作成し、float16に変換して、ゼロになる割合を測定します。次に損失スケーリング（1024を掛ける）を適用し、float16に変換して、スケールを戻して、ゼロの割合を再度測定します。

## キー用語

| 用語 | よく言われる説明 | 実際の意味 |
|------|----------------|------------|
| IEEE 754 | 「浮動小数点の標準」 | バイナリ浮動小数点形式、丸め規則、特殊値（inf、nan）を定義する国際規格。すべての現代のCPUとGPUが実装する。 |
| 機械イプシロン | 「精度の限界」 | 与えられた浮動小数点形式で1.0 + e != 1.0となる最小値e。float32では約1.19e-7。 |
| 桁落ち | 「引き算による精度損失」 | ほぼ等しい浮動小数点数を引き算するとき、有効桁が打ち消し合い、丸め誤差が結果を支配する。 |
| オーバーフロー | 「数が大きすぎる」 | 結果が最大表現可能値を超えてinfになる。exp(89)はfloat32でオーバーフロー。 |
| アンダーフロー | 「数が小さすぎる」 | 結果が最小表現可能正数より小さくなって0.0になる。exp(-104)はfloat32でアンダーフロー。 |
| Log-Sum-Expトリック | 「最初に最大値を引く」 | exp(max(x))を括り出してlog(sum(exp(x)))を計算し、オーバーフローとアンダーフローを防ぐ。softmax、クロスエントロピー、対数確率の計算で使用。 |
| 安定なsoftmax | 「爆発しないsoftmax」 | 指数化の前にmax(logits)を引き算。数学的に同一の結果、オーバーフールが不可能。 |
| 勾配チェック | 「誤差逆伝播を検証する」 | 誤差逆伝播からの解析的勾配を有限差分法からの数値的勾配と比較して実装バグを検出する。 |
| 混合精度 | 「float16フォワード、float32バックワード」 | 速度重視の演算に低精度、数値的にデリケートな演算に高精度の浮動小数点を使用。典型的な高速化は2〜3倍。 |
| 損失スケーリング | 「勾配アンダーフローを防ぐ」 | 誤差逆伝播の前に損失に大きな定数を掛け、float16の表現可能範囲内に勾配を収め、重みの更新前に同じ定数で割る。 |
| bfloat16 | 「Brain floating point」 | Googleの16ビット形式で8指数ビット（float32と同じ範囲）と7仮数ビット（float16より精度が低い）。学習に適している。 |
| 勾配クリッピング | 「勾配ノルムを制限する」 | 勾配ベクトルのノルムが閾値を超えないようにスケールする。爆発する勾配が重みを台無しにするのを防ぐ。 |
| NaN | 「非数（Not a Number）」 | 未定義の演算（0/0、inf-inf、sqrt(-1)）からの特殊浮動小数点値。以降のすべての算術演算に伝播する。 |
| Inf | 「無限大（Infinity）」 | オーバーフローまたはゼロ除算からの特殊浮動小数点値。NaN（inf - inf、inf * 0）を生成するために結合される可能性がある。 |
| 数値的勾配 | 「力ずくの微分」 | f(x+h)とf(x-h)を評価して2hで割ることで微分を近似する。遅いが検証には信頼できる。 |

## 参考資料

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) ―決定的なリファレンス、密度は高いが完全
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) ―float16学習の損失スケーリングを導入したNVIDIAの論文
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) ―PyTorchの混合精度の実践ガイド
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) ―GoogleがTPUにこの形式を選んだ理由
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) ―浮動小数点の合計における丸め誤差を低減するアルゴリズム
