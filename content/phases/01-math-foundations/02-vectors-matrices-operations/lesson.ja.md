# ベクトル、行列と演算

> すべてのニューラルネットワークは、余分なステップを加えた行列積にすぎない。


## 学習目標

- 要素ごとの演算、行列積、転置、行列式、逆行列を持つ Matrix クラスを実装する
- 要素ごとの積と行列積を区別し、それぞれがいつ使われるかを説明する
- スクラッチで作った Matrix クラスのみを使って、1層の全結合ニューラルネットワーク層（`relu(W @ x + b)`）を実装する
- ブロードキャストのルールと、ニューラルネットワークフレームワークでのバイアス加算の動作を説明する

## 問題提起

ニューラルネットワークを作りたいとする。コードを読むと次のような行が見える:

```
output = activation(weights @ input + bias)
```

この `@` は行列積だ。`weights` は行列で、`input` はベクトルだ。これらの演算が何をしているかわからなければ、この行は魔法に見える。わかっていれば、3つの演算で層の順伝播全体だとわかる。

モデルが処理するすべての画像はピクセル値の行列だ。すべての単語埋め込みはベクトルだ。すべてのニューラルネットワークのすべての層は行列変換だ。変数を理解せずにコードが書けないのと同様、行列演算に習熟せずにAIシステムは作れない。

このレッスンでは、スクラッチからその習熟を身につける。

## 概念

### ベクトル: 順序付けられた数のリスト

ベクトルは方向と大きさを持つ数のリストだ。AIでは、ベクトルはデータ点、特徴量、またはパラメータを表す。

```
v = [3, 4]        -- 2Dベクトル
w = [1, 0, -2]    -- 3Dベクトル
```

2Dベクトル `[3, 4]` は平面上の座標 (3, 4) を指す。その長さ（大きさ）は5だ（3-4-5の直角三角形）。

### 行列: 数のグリッド

行列は2次元のグリッドだ。行と列がある。m×n行列はm行n列を持つ。

```
A = | 1  2  3 |     -- 2×3行列（2行、3列）
    | 4  5  6 |
```

ニューラルネットワークでは、重み行列が入力ベクトルを出力ベクトルに変換する。784個の入力と128個の出力を持つ層には、128×784の重み行列が使われる。

### 形状が重要な理由

行列積には厳密なルールがある: `(m×n) @ (n×p) = (m×p)`。内側の次元が一致しなければならない。

```
(128×784) @ (784×1) = (128×1)
  重み        入力       出力

内側の次元: 784 = 784  -- 有効
```

PyTorchで形状の不一致エラーが出たら、これが原因だ。

### 演算マップ

| 演算 | 何をするか | ニューラルネットワークでの使用 |
|-----------|-------------|-------------------|
| 加算 | 要素ごとに結合する | バイアスを出力に加算 |
| スカラー積 | すべての要素をスケールする | 学習率 × 勾配 |
| 行列積 | ベクトルを変換する | 層の順伝播 |
| 転置 | 行と列を入れ替える | 誤差逆伝播 |
| 行列式 | 行列を要約するスカラー | 逆行列の存在確認 |
| 逆行列 | 変換を元に戻す | 線形系の求解 |
| 単位行列 | 何もしない行列 | 初期化、残差接続 |

### 要素ごとの積と行列積

この区別は初学者が頻繁に混乱する。

要素ごとの積: 対応する位置同士を掛ける。両方の行列が同じ形状でなければならない。

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

行列積: 行と列の内積を取る。内側の次元が一致しなければならない。

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

演算が異なれば、結果もルールも異なる。

### ブロードキャスト

出力の行列にバイアスベクトルを加算するとき、形状が一致しない。ブロードキャストは小さい配列を大きい配列に合わせて引き伸ばす。

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

ブロードキャストはベクトルを行方向に引き伸ばす:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

最新のフレームワークはこれを自動的に行う。これを理解しておくと、形状がおかしいのにコードが動く場合の混乱を防げる。

## 実装

### ステップ1: Vector クラス

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### ステップ2: 主要な演算を持つ Matrix クラス

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("行列は特異（逆行列なし）")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### ステップ3: 動作確認

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### ステップ4: ニューラルネットワークとの接続

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"入力形状: {inputs.shape}")
print(f"重み形状: {weights.shape}")
print(f"出力形状: {output.shape}")
print(f"出力: {output.data}")
```

これが1つの全結合層だ: `output = relu(W @ x + b)`。すべてのニューラルネットワークのすべての全結合層はまさにこれをやっている。

## 実際に使う

NumPyを使えば、同じことがより少ない行数で、桁違いの速さで実現できる。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (要素ごと) =\n", A * B)
print("A @ B (行列積) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nニューラルネットワーク層: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"出力:\n{output}")
```

PythonにおけるAP演算子 `@` は `__matmul__` を呼び出す。NumPyはこれをCとFortranで書かれた最適化されたBLASルーチンで実装している。同じ数学、100倍の速さだ。

NumPyのブロードキャスト:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPyは1Dのバイアスを両方の行に自動的にブロードキャストする。これがすべてのニューラルネットワークフレームワークでのバイアス加算の動作だ。

## 納品物

このレッスンでは、幾何学的直感を通して行列演算を教えるためのプロンプトを生成する。`outputs/prompt-matrix-operations.md` を参照。

ここで作った Matrix クラスは、フェーズ3 レッスン10で構築するミニニューラルネットワークフレームワークの基盤となる。

## 演習

1. **逆行列の確認。** `A @ A.inverse_2x2()` を計算し、単位行列が得られることを確認せよ。3つの異なる2×2行列で試せ。行列式がゼロの場合はどうなるか？

2. **3×3逆行列の実装。** 余因子行列を使って3×3行列の逆行列を計算できるように Matrix クラスを拡張せよ。NumPyの `np.linalg.inv` と比較して検証せよ。

3. **2層ネットワークを作る。** Matrix クラスのみ（NumPy不使用）を使って、2層ニューラルネットワーク: 入力(3) → 隠れ層(4) → 出力(2) を作れ。ランダムな重みで初期化し、順伝播を実行し、すべての形状が正しいことを確認せよ。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|----------------|----------------------|
| ベクトル | 「矢印」 | 順序付けられた数のリスト。AIでは高次元空間上の点。 |
| 行列 | 「数の表」 | 線形変換。あるベクトルを別のベクトルに写す。 |
| 行列積 | 「ただ数を掛ける」 | 第1行列のすべての行と第2行列のすべての列の内積。順序が重要。 |
| 転置 | 「ひっくり返す」 | 行と列を入れ替える。m×n行列をn×mに変換する。誤差逆伝播で重要。 |
| 行列式 | 「行列から出てくる何かの数」 | 行列が面積（2D）や体積（3D）をどれだけ拡大するかの指標。ゼロは次元を消すことを意味する。 |
| 逆行列 | 「行列を元に戻す」 | 変換を逆にする行列。行列式がゼロでない場合のみ存在する。 |
| 単位行列 | 「退屈な行列」 | 1を掛けることに相当する行列。残差接続（ResNet）で使用される。 |
| ブロードキャスト | 「魔法の形状修正」 | 欠落している次元に沿って繰り返すことで、小さい配列を大きい配列に合わせて引き伸ばす。 |
| 要素ごとの積 | 「普通の掛け算」 | 対応する位置同士を掛ける。両方の配列が同じ形状（またはブロードキャスト可能）でなければならない。 |

## 参考資料

- [3Blue1Brown: 線形代数の本質](https://www.3blue1brown.com/topics/linear-algebra) - ここで扱うすべての演算の視覚的直感
- [NumPy ブロードキャストのドキュメント](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPyが従う正確なルール
- [Stanford CS229 線形代数レビュー](http://cs229.stanford.edu/section/cs229-linalg.pdf) - ML特有の線形代数の簡潔なリファレンス
