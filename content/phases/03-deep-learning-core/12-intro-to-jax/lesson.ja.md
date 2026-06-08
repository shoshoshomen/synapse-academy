# JAX入門

> PyTorchはテンソルを変更する。TensorFlowはグラフを構築する。JAXは純粋関数をコンパイルする。最後のものが深層学習の考え方を変える。


## 学習目標

- JAXの関数型API（jax.numpy、jax.grad、jax.jit、jax.vmap）を使って純粋関数のニューラルネットワークコードを書く
- PyTorchのイーガー変更とJAXの関数型コンパイルモデルの主要な設計の違いを説明する
- jitコンパイルとvmapベクトル化を適用して素直なPythonと比較して訓練ループを加速する
- JAXでシンプルなネットワークを訓練し、明示的な状態管理とPyTorchのオブジェクト指向アプローチを対比する

## 問題設定

PyTorchでニューラルネットワークを構築する方法を知っている。`nn.Module`を定義し、`.backward()`を呼び出し、オプティマイザをステップする。機能する。何百万人もの人々が使う。

しかしPyTorchにはDNAに組み込まれた制約がある: Pythonで1度に1つずつイーガーに操作をトレースする。すべての`tensor + tensor`は別々のカーネル起動だ。すべての訓練ステップが同じPythonコードを再解釈する。2,048台のTPUで5400億パラメータのモデルを訓練する必要がある時まで、これは問題なく機能する。そうなるとオーバーヘッドが問題になる。

Google DeepMindはGeminiをJAXで訓練する。AnthropicはClaudeをJAXで訓練した。これらは小さな操作ではない。地球上で最大のニューラルネットワーク訓練実行だ。JAXを選んだのは、訓練ループをPython呼び出しのシーケンスとしてではなくコンパイル可能なプログラムとして扱うからだ。

JAXは3つのスーパーパワーを持つNumPyだ: 自動微分、XLAへのJITコンパイル、自動ベクトル化。1つの例を処理する関数を書く。JAXはバッチを処理し、勾配を計算し、マシンコードにコンパイルし、複数のデバイスで実行する関数を提供する。元の関数を変更せずに。

## 概念

### JAXの哲学

JAXは関数型フレームワークだ。クラスなし、可変状態なし、`.backward()`メソッドなし。代わりに:

| PyTorch | JAX |
|---------|-----|
| 状態を持つ`nn.Module`クラス | 純粋関数: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| イーガー実行 | XLA経由のJITコンパイル |
| `for x in batch:`の手動ループ | `jax.vmap(f)`による自動ベクトル化 |
| `DataParallel` / `FSDP` | `jax.pmap(f)`による自動並列性 |
| 可変の`model.parameters()` | 配列の不変のpytree |

これはスタイルの好みではない。コンパイラの制約だ。JITコンパイルは純粋関数を必要とする。同じ入力は常に同じ出力を生成し、副作用がない。その制限こそが100倍の高速化を可能にする。

### jax.numpy: 親しみやすいサーフェス

JAXはアクセラレータ上でNumPy APIを再実装する:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

同じ関数名。同じブロードキャストルール。同じスライシングセマンティクス。しかし配列はGPU/TPUに存在し、すべての操作はコンパイラによってトレース可能だ。

1つの重要な違い: JAX配列は不変だ。`a[0] = 5`はできない。代わりに: `a = a.at[0].set(5)`。これは1週間は不自然に感じるが、それからクリックする。不変性こそが`grad`、`jit`、`vmap`のような変換を合成可能にするものだ。

### jax.grad: 関数型自動微分

PyTorchは勾配をテンソルに付与する(`.grad`)。JAXは勾配を関数に付与する。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad`は関数を取り、勾配を計算する新しい関数を返す。`.backward()`呼び出しなし。テンソルに保存された計算グラフなし。勾配は呼び出し、合成、またはJITコンパイルできる別の関数だ。

これは任意に合成できる:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

2次微分。3次微分。ヤコビアン。ヘッセ行列。すべて`grad`を合成することで。PyTorchもこれができる（`torch.autograd.functional.hessian`）が、後付けだ。JAXではそれが基盤だ。

制約: `grad`は純粋関数にしか機能しない。内部にprintステートメントを入れない（実行ではなくトレース中に実行される）。外部状態の変更なし。明示的なキー管理なしの乱数生成なし。

### jit: XLAにコンパイル

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

最初の呼び出しで、JAXは関数をトレースする。実行せずにどの操作が起こるかを記録する。そのトレースをXLA（Accelerated Linear Algebra）に渡す。これはGPUとTPUのためのGoogleのコンパイラだ。XLAは操作を融合し、冗長なメモリコピーを排除し、最適化されたマシンコードを生成する。

その後の呼び出しはPythonを完全にスキップする。コンパイルされたコードがC++の速度でアクセラレータ上で実行される。

JITが役立つ時:
- 訓練ステップ（同じ計算が何千回も繰り返される）
- 推論（同じモデル、異なる入力）
- 同様の形状の入力で複数回呼ばれる任意の関数

JITが害になる時:
- 値に依存するPython制御フローを持つ関数（xがトレースされた配列の場合の`if x > 0`）
- ワンショット計算（コンパイルオーバーヘッドが実行時間を超える）
- デバッグ（トレーシングが実際の実行を隠す）

制御フローの制限は本物だ。`jax.lax.cond`が`if/else`を置き換える。`jax.lax.scan`が`for`ループを置き換える。これらはオプションではない。コンパイルの代償だ。

### vmap: 自動ベクトル化

1つの例を処理する関数を書く:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap`がバッチを処理するために持ち上げる:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)`は: `params`をバッチ処理しない（共有）、`x`の軸0でバッチ処理する、を意味する。手動の`for`ループなし。リシェイプなし。バッチ次元のスレッディングなし。JAXがバッチ次元を把握して全計算をベクトル化する。

これは構文糖ではない。`vmap`はPythonループより10〜100倍速く実行される融合ベクトル化コードを生成する。そして`jit`と`grad`と合成できる:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

例ごとの勾配。1行で。これはPyTorchではハックなしではほぼ不可能だ。

### pmap: デバイスを越えたデータ並列性

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap`はすべての利用可能なデバイス（GPU/TPU）全体に関数を複製し、バッチを分割する。関数内部では、`jax.lax.pmean`と`jax.lax.psum`がデバイス間で勾配を同期させる。

Googleは`pmap`（と後継の`shard_map`）を使って何千ものTPU v5eチップでGeminiを訓練する。プログラミングモデル: シングルデバイス版を書き、`pmap`でラップして完了。

### Pytree: ユニバーサルデータ構造

JAXは「pytree」、つまりリスト、タプル、辞書、配列のネストした組み合わせを操作する。モデルのパラメータはpytreeだ:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

すべてのJAX変換。`grad`、`jit`、`vmap`。pytreeをたどる方法を知っている。`jax.tree.map(f, tree)`は`f`をすべてのリーフに適用する。これが一度にすべてのパラメータを更新する方法だ:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

`.parameters()`メソッドなし。パラメータ登録なし。木構造がモデルだ。

### 関数型 vs オブジェクト指向

PyTorchはオブジェクト内に状態を保存する:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAXは明示的な状態を持つ純粋関数を使う:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

paramsが渡される。何も保存されない。何も変更されない。これはすべての関数をテスト可能、合成可能、コンパイル可能にする。これはまたparamsを自分で管理することを意味する。またはFlaxやEquinoxのようなライブラリを使う。

### JAXエコシステム

JAXはプリミティブを提供する。ライブラリはエルゴノミクスを提供する:

| ライブラリ | 役割 | スタイル |
|---------|------|-------|
| **Flax**（Google） | ニューラルネットワーク層 | 明示的な状態を持つ`nn.Module` |
| **Equinox**（Patrick Kidger） | ニューラルネットワーク層 | Pytreeベース、Pythonicスタイル |
| **Optax**（DeepMind） | オプティマイザ + LRスケジュール | 合成可能な勾配変換 |
| **Orbax**（Google） | チェックポインティング | pytreeの保存/復元 |
| **CLU**（Google） | メトリクス + ロギング | 訓練ループユーティリティ |

Optaxは標準オプティマイザライブラリだ。勾配変換（Adam、SGD、クリッピング）をパラメータ更新から分離し、合成を簡単にする:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### JAX vs PyTorchの使い分け

| 要素 | JAX | PyTorch |
|--------|-----|---------|
| TPUサポート | ファーストクラス（Googleが両方構築） | コミュニティ管理（torch_xla） |
| GPUサポート | 良い（XLA経由のCUDA） | 最高レベル（ネイティブCUDA） |
| デバッグ | 難しい（トレーシング + コンパイル） | 簡単（イーガー、行ごと） |
| エコシステム | 研究重点（Flax、Equinox） | 巨大（HuggingFace、torchvision等） |
| 採用 | ニッチ（Google/DeepMind/Anthropic） | 主流（どこでも） |
| 大規模訓練 | 優れる（XLA、pmap、mesh） | 良い（FSDP、DeepSpeed） |
| プロトタイピング速度 | 遅い（関数型オーバーヘッド） | 速い（変更して進む） |
| プロダクション推論 | TensorFlow Serving、Vertex AI | TorchServe、Triton、ONNX |
| 使用者 | DeepMind（Gemini）、Anthropic（Claude） | Meta（Llama）、OpenAI（GPT）、Stability AI |

正直な答え: JAXを使う特定の理由がない限りPyTorchを使う。それらの理由とは: TPUアクセス、例ごとの勾配の必要性、大規模な多デバイス訓練、またはGoogle/DeepMind/Anthropicでの業務。

### JAXの乱数

JAXはグローバルなランダム状態を持たない。すべてのランダム操作は明示的なPRNGキーを必要とする:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

最初は面倒だ。しかしデバイスとコンパイル間の再現性を保証する。PyTorchの`torch.manual_seed`がマルチGPU設定では保証できないプロパティだ。

## 実装

JAXとOptaxを使ってMNISTで3層MLPを訓練する。784入力、256と128ニューロンの2つの隠れ層、10出力クラス。

### ステップ1: セットアップとデータ

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### ステップ2: パラメータの初期化

クラスなし。pytreeを返す関数だけ:

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He初期化を手動で。1つのシードから分割された3つのPRNGキー。すべての重みはネストした辞書の不変の配列だ。

### ステップ3: 順伝播

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

純粋関数。paramsを入力、予測を出力。`self`なし、保存された状態なし。`loss_fn`はゼロから交差エントロピーを計算する。softmax、log、負の平均。

### ステップ4: JITコンパイルされた訓練ステップ

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad`は1回のパスで損失値と勾配の両方を返す。`@jax.jit`デコレータは両方の関数をXLAにコンパイルする。最初の呼び出しの後、各訓練ステップはPythonに触れずに実行される。

### ステップ5: 訓練ループ

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10エポック。~97%のテスト精度。最初のエポックは遅い（JITコンパイル）。エポック2〜10は速い。

欠けているものに注目: `.zero_grad()`なし、`.backward()`なし、`.step()`なし。更新全体が1つの合成された関数呼び出しだ。勾配が計算され、Adamによって変換され、パラメータに適用される。すべて`train_step`の中で。

## 実用例

### Flax: Google標準

FlaxはJAXで最も一般的なニューラルネットワークライブラリだ。`nn.Module`を戻すが、明示的な状態管理を伴う:

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

PyTorchと同じ構造だが、`params`はモデルとは別だ。`model.init()`がparamsを作成する。`model.apply(params, x)`が順伝播を実行する。モデルオブジェクトには状態がない。

### Equinox: Pythonicな代替

Equinox（Patrick Kidgerによる）はモデルをpytreeとして表現する:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

モデル自体がpytreeだ。`.apply()`は不要。パラメータはモデルのリーフだ。JAXの考え方に近い。

### Optax: 合成可能なオプティマイザ

Optaxは勾配変換を更新から切り離す:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

勾配クリッピング、学習率ウォームアップ、ウェイト減衰。すべて変換のチェーンとして合成される。各変換は勾配を見て変更し次に渡す。モノリシックなオプティマイザクラスなし。

## 成果物

**インストール:**

```bash
pip install jax jaxlib optax flax
```

GPUサポートの場合:

```bash
pip install jax[cuda12]
```

TPU（Google Cloud）の場合:

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**パフォーマンスの注意点:**

- 最初のJIT呼び出しは遅い（コンパイル）。ベンチマーク前にウォームアップする。
- JIT内のJAX配列のPythonループを避ける。`jax.lax.scan`または`jax.lax.fori_loop`を使う。
- `jax.debug.print()`はJIT内で機能する。通常の`print()`は機能しない。
- `jax.profiler`またはTensorBoardでプロファイリングする。XLAコンパイルはボトルネックを隠すことがある。
- JAXはデフォルトでGPUメモリの75%を事前割り当てする。無効にするには`XLA_PYTHON_CLIENT_PREALLOCATE=false`を設定する。

**チェックポインティング:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**このレッスンの成果物:**
- `outputs/prompt-jax-optimizer.md` -- 正しいJAXオプティマイザ設定を選ぶプロンプト
- `outputs/skill-jax-patterns.md` -- JAXの関数型パターンをカバーするスキル

## 演習

1. MLPにドロップアウトを追加する。JAXではドロップアウトはPRNGキーを必要とする。順伝播を通じてキーをスレッドして各ドロップアウト層のために分割する。ありとなしでテスト精度を比較する。

2. `jax.vmap`を使って32枚のMNIST画像のバッチに対して例ごとの勾配を計算する。各例の勾配のノルムを計算する。どの例が最も大きな勾配を持ち、なぜか？

3. 手動のforward関数をどんな数の層でも機能する汎用の`mlp_forward(params, x)`に置き換える。`jax.tree.leaves`を使って深さを自動的に決定する。

4. `@jax.jit`ありとなしで訓練ステップをベンチマークする。それぞれ100ステップの時間を測る。あなたのハードウェアでどのくらいのスピードアップがあるか？最初の呼び出しでのコンパイルオーバーヘッドは何か？

5. `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`を合成することで勾配クリッピングを実装する。クリッピングありとなしで訓練する。訓練を通じた勾配ノルムをプロットして効果を確認する。

## 重要用語

| 用語 | よく言われること | 実際の意味 |
|------|----------------|----------------------|
| XLA | 「JAXを速くするもの」 | Accelerated Linear Algebra。操作を融合し計算グラフから最適化されたGPU/TPUカーネルを生成するコンパイラ |
| JIT | 「ジャストインタイムコンパイル」 | JAXは最初の呼び出しで関数をトレースしてXLAにコンパイルし、その後の呼び出しでコンパイルされたバージョンを実行する |
| 純粋関数 | 「副作用なし」 | 出力が入力だけに依存する関数。グローバル状態なし、変更なし、明示的なキーなしの乱数なし |
| vmap | 「自動バッチ化」 | 1つの例を処理する関数をバッチ全体を処理するように変換する、書き直しなしで |
| pmap | 「自動並列性」 | 複数のデバイスに関数を複製し入力バッチを分割する |
| Pytree | 「配列のネストした辞書」 | JAXがたどって変換できるリスト、タプル、辞書、配列の任意のネストした構造 |
| トレーシング | 「計算を記録する」 | JAXは実際の結果を計算せずに計算グラフを構築するために抽象値で関数を実行する |
| 関数型自動微分 | 「関数のgrad」 | テンソルに勾配ストレージを付与するのではなく関数を変換することで微分を計算する |
| Optax | 「JAXのオプティマイザライブラリ」 | Adam、SGD、クリッピング、スケジューリングなど連鎖する勾配変換の合成可能なライブラリ |
| Flax | 「JAXのnn.Module」 | JAXのためのGoogleのニューラルネットワークライブラリ、明示的な状態を保ちながら層の抽象化を追加する |

## 参考文献

- JAXドキュメント: https://jax.readthedocs.io/ -- 公式ドキュメント、grad、jit、vmapの優れたチュートリアル付き
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) -- 設計思想を説明する元の論文
- Flaxドキュメント: https://flax.readthedocs.io/ -- JAXのためのGoogleのニューラルネットワークライブラリ
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) -- Flaxに対するPythonicな代替
- DeepMind, "Optax: composable gradient transformation and optimisation" -- 標準オプティマイザライブラリ
- "You Don't Know JAX" (Colin Raffel, 2020) -- T5の著者の一人によるJAXの落とし穴とパターンの実践ガイド
