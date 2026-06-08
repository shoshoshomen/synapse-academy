# 確率と分布

> 確率はAIが不確実性を表現するために使う言語だ。


## 学習目標

- ベルヌーイ分布、カテゴリカル分布、ポアソン分布、一様分布、正規分布のPMFとPDFをスクラッチから実装する
- 期待値と分散を計算し、中心極限定理を使ってガウス分布が至る所に現れる理由を説明する
- 数値安定化トリック（最大ロジットの減算）付きのソフトマックスとlog-ソフトマックス関数を構築する
- ロジットからクロスエントロピー損失を計算し、負の対数尤度との関連を示す

## 問題提起

分類器が `[0.03, 0.91, 0.06]` を出力する。言語モデルが50,000候補から次の単語を選ぶ。拡散モデルが学習した分布からサンプリングして画像を生成する。これらはすべて確率の実用例だ。

モデルが行うすべての予測は確率分布だ。すべての損失関数は予測された分布が真の分布からどれだけ離れているかを測る。すべての学習ステップはある分布を別の分布に近づけるようにパラメータを調整する。確率なしには、ML論文を1本も読めず、モデルを1つもデバッグできず、学習損失がNaNになる理由も理解できない。

## 概念

### 事象、標本空間、確率

標本空間Sはすべての可能な結果の集合だ。事象は標本空間の部分集合だ。確率は事象を0から1の間の数値に対応付ける。

```
コイン投げ:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

サイコロを1回振る:
  S = {1, 2, 3, 4, 5, 6}
  P(偶数) = P({2, 4, 6}) = 3/6 = 0.5
```

確率を定義する3つの公理:
1. 任意の事象A について P(A) >= 0
2. P(S) = 1（何かが必ず起きる）
3. AとBが同時に起きない場合、P(A または B) = P(A) + P(B)

すべてのその他の内容（ベイズの定理、期待値、分布）はこの3つのルールから導出される。

### 条件付き確率と独立性

P(A|B) はBが起きたときにAが起きる確率だ。

```
P(A|B) = P(A かつ B) / P(B)

例: トランプのデッキ
  P(King | Face card) = P(King かつ Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

2つの事象は、一方を知っても他方について何もわからないとき独立だ:

```
独立:      P(A|B) = P(A)
等価:      P(A かつ B) = P(A) * P(B)
```

コイン投げは独立だ。カードを戻さずに引くのは独立でない。

### 確率質量関数と確率密度関数

離散確率変数は確率質量関数（PMF）を持つ。各結果には直接読み取れる特定の確率がある。

```
PMF: P(X = k)

公正なサイコロ:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  すべての確率の和 = 1
```

連続確率変数は確率密度関数（PDF）を持つ。1点での密度は確率ではない。確率は区間にわたって密度を積分することで得られる。

```
PDF: f(x)

P(a <= X <= b) = a から b まで f(x) の積分

f(x) は1より大きくなれる（確率ではなく密度）
-inf から +inf まで f(x) dx の積分 = 1
```

この区別はMLで重要だ。分類の出力はPMFだ（離散的な選択）。VAEの潜在空間はPDFを使う（連続的）。

### よく使われる分布

**ベルヌーイ分布:** 1回の試行、2つの結果。二値分類をモデル化する。

```
P(X = 1) = p
P(X = 0) = 1 - p
平均 = p,  分散 = p(1-p)
```

**カテゴリカル分布:** 1回の試行、k個の結果。多クラス分類をモデル化する（ソフトマックス出力）。

```
P(X = i) = p_i,  ここで p_i の和 = 1
例: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**一様分布:** すべての結果が等しく起こりやすい。ランダム初期化に使われる。

```
離散: P(X = k) = 1/n, k ∈ {1, ..., n}
連続: f(x) = 1/(b-a), x ∈ [a, b]
```

**正規分布（ガウス分布）:** ベル曲線。平均（mu）と分散（sigma^2）でパラメータ化される。

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

標準正規分布: mu = 0, sigma = 1
  データの68%は1 sigma以内
  95%は2 sigma以内
  99.7%は3 sigma以内
```

**ポアソン分布:** 一定区間でのまれな事象の数。事象発生率をモデル化する。

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
平均 = lambda,  分散 = lambda
```

### 期待値と分散

期待値は加重平均の結果だ。

```
離散:   E[X] = x_i * P(X = x_i) の和
連続: E[X] = x * f(x) dx の積分
```

分散は平均の周りの広がりを測る。

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
標準偏差 = sqrt(Var(X))
```

MLでは期待値は損失関数（データ分布にわたる平均損失）として現れる。分散はモデルの安定性を示す。勾配の高い分散は学習が不安定なことを意味する。

### 結合分布と周辺分布

結合分布 P(X, Y) は2つの確率変数を一緒に記述する。

結合PMFの例（X = 天気、Y = 傘）:

| | Y=0（傘なし） | Y=1（傘あり） | 周辺 P(X) |
|---|---|---|---|
| X=0（晴れ） | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1（雨） | 0.05 | 0.45 | P(X=1) = 0.50 |
| **周辺 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

周辺分布は他の変数を足し合わせる:

```
P(X = x) = すべてのy についての P(X = x, Y = y) の和
```

上の表の行と列の合計が周辺分布だ。

### 正規分布がいたるところに現れる理由

中心極限定理: 多くの独立した確率変数の和（または平均）は、元の分布に関わらず正規分布に収束する。

```
サイコロを1回振る:  一様分布（平坦）
2つのサイコロの平均:  三角形（ピーク状）
30個のサイコロの平均: ほぼ完全なベル曲線

これはどんな出発分布でも機能する。
```

これが以下の理由だ:
- 測定誤差はほぼ正規分布（多くの小さな独立した原因）
- ニューラルネットワークの重み初期化に正規分布を使う
- SGDの勾配ノイズはほぼ正規分布（多くのサンプル勾配の和）
- 正規分布は与えられた平均と分散に対して最大エントロピーの分布

### 対数確率

生の確率は数値的な問題を引き起こす。多くの小さな確率を掛け合わせると、すぐにアンダーフローしてゼロになる。

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0（~30項の後にアンダーフロー）
```

対数確率がこれを解決する。乗算が加算になる。

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> 有限の数（アンダーフローなし）
```

ルール:
- log(a * b) = log(a) + log(b)
- 対数確率は常に <= 0（0 < P <= 1 なので）
- より負 = より起きにくい
- クロスエントロピー損失は正しいクラスの負の対数確率だ

### 確率分布としてのソフトマックス

ニューラルネットワークは生のスコア（ロジット）を出力する。ソフトマックスはそれらを有効な確率分布に変換する。

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

特性:
  - すべての出力は (0, 1) の範囲
  - すべての出力の和は1
  - 入力の相対的な順序を保持
  - exp() がロジット間の差を増幅する
```

ソフトマックストリック: オーバーフローを防ぐために指数をとる前に最大ロジットを引く。

```
z = [100, 101, 102]
exp(102) = オーバーフロー

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  （安全）

同じ結果、オーバーフローなし。
```

Log-ソフトマックスはソフトマックスと対数を組み合わせて数値安定性を高める。PyTorchはクロスエントロピー損失に内部でこれを使っている。

### サンプリング

サンプリングとは分布からランダムな値を引くことだ。MLでは:
- ドロップアウトはランダムにどのニューロンをゼロにするかをサンプリングする
- データ拡張はランダムな変換をサンプリングする
- 言語モデルは予測された分布から次のトークンをサンプリングする
- 拡散モデルはノイズをサンプリングして徐々にノイズを除去する

任意の分布からのサンプリングには、逆変換サンプリング、棄却サンプリング、再パラメータ化トリック（VAEで使用）などの技法が必要だ。

## 実装

### ステップ1: 確率の基礎

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### ステップ2: スクラッチからのPMFとPDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### ステップ3: 期待値と分散

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"サイコロ: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### ステップ4: 分布からのサンプリング

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### ステップ5: ソフトマックスと対数確率

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### ステップ6: 中心極限定理のデモ

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### ステップ7: 可視化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

すべての可視化を含む完全な実装は `code/probability.py` にある。

## 実際に使う

NumPyとSciPyを使えば、上記のすべてが1行で書ける:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

スクラッチから構築した。これでライブラリが何をしているかがわかる。

## 演習

1. 指数分布の逆変換サンプリングを実装せよ。10,000個の値をサンプリングし、ヒストグラムを真のPDFと比較して検証せよ。

2. 2つのイカサマサイコロの結合分布表を構築せよ。周辺分布を計算し、サイコロが独立かどうかを確認せよ。

3. 正解クラスがインデックス3のとき、ロジット `[2.0, 0.5, -1.0, 3.0, 0.1]` を出力する5クラス分類器のクロスエントロピー損失を計算せよ。次にPyTorchの `nn.CrossEntropyLoss` で答えを検証せよ。

4. 対数確率のリストを受け取り、最も可能性の高い系列、総対数確率、対応する生の確率を返す関数を書け。各単語の確率が0.01の50語の文でテストせよ。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|----------------|----------------------|
| 標本空間 | 「すべての可能性」 | 実験のすべての可能な結果の集合S |
| PMF | 「確率関数」 | 各離散的な結果の正確な確率を与え、和が1になる関数 |
| PDF | 「確率曲線」 | 連続変数の密度関数。区間にわたって積分することで確率が得られる |
| 条件付き確率 | 「何かが与えられたときの確率」 | P(A\|B) = P(A かつ B) / P(B)。ベイズ的思考とベイズの定理の基礎 |
| 独立性 | 「互いに影響しない」 | P(A かつ B) = P(A) * P(B)。一方の事象を知っても他方について何もわからない |
| 期待値 | 「平均」 | すべての結果の確率加重和。損失関数は期待値だ |
| 分散 | 「どれだけ広がっているか」 | 平均からの期待二乗偏差。高い分散 = ノイズが多く不安定な推定 |
| 正規分布 | 「ベル曲線」 | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))。CLTにより至る所に現れる |
| 中心極限定理 | 「平均は正規分布になる」 | 多くの独立したサンプルの平均は、元の分布に関わらず正規分布に収束する |
| 結合分布 | 「2変数を一緒に」 | P(X, Y) はXとYの結果のすべての組み合わせの確率を記述する |
| 周辺分布 | 「他の変数を足し合わせる」 | P(X) = sum_y P(X, Y)。結合分布から1変数の分布を回復する |
| 対数確率 | 「確率の対数」 | log P(x)。積を和に変換し、長い系列での数値アンダーフローを防ぐ |
| ソフトマックス | 「スコアを確率に変換する」 | softmax(z_i) = exp(z_i) / sum(exp(z_j))。実数値のロジットを有効な確率分布に写す |
| クロスエントロピー | 「損失関数」 | -sum(p_true * log(p_predicted))。2つの分布がどれだけ異なるかを測る。低いほど良い |
| ロジット | 「生のモデル出力」 | ソフトマックス前の正規化されていないスコア。ロジスティック関数にちなんで命名 |
| サンプリング | 「ランダムな値を引く」 | 確率分布に従って値を生成する。モデルが出力を生成する方法 |

## 参考資料

- [3Blue1Brown: 中心極限定理とは？](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 平均が正規分布になる理由の視覚的証明
- [Stanford CS229 確率レビュー](https://cs229.stanford.edu/section/cs229-prob.pdf) - ここで扱った内容以上を網羅した簡潔なリファレンス
- [Log-Sum-Expトリック](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 数値安定性がなぜ重要かと達成方法
