# 動的計画法——方策反復と価値反復

> 動的計画法はチート付きのRLだ。遷移関数と報酬関数がすでにわかっている；あとは`V`や`π`が動かなくなるまでベルマン方程式を反復するだけだ。これはサンプリングベースのあらゆる手法が近づこうとする基準値だ。


## 問題

あなたは既知のモデルを持つMDPを持っている：すべての状態-行動ペアに対して`P(s' | s, a)`と`R(s, a, s')`をクエリできる。在庫管理者は需要分布を知っている。ボードゲームは決定的な遷移を持つ。GridWorldはPythonの4行だ。あなたは*モデル*を持っている。

モデルフリーRL（Q学習、PPO、REINFORCE）はモデルがない場合——環境からサンプリングしかできない場合——のために発明された。しかしモデルがある場合は、より速く、より良い方法がある：動的計画法。Bellmanが1957年に設計した。それらは今でも正しさを定義する：人々が「このMDPの最適方策」と言う時、それはDPが返す方策を意味する。

2026年に動的計画法が必要な3つの理由。第一に、RLリサーチのすべての表形式環境（GridWorld、FrozenLake、CliffWalking）はゴールドスタンダード方策を生成するためにDPで解かれる。第二に、厳密な値はサンプリング手法を*デバッグ*できる：Q学習の`V*(s_0)`の推定値がDPの答えと30%ずれているなら、Q学習にバグがある。第三に、現代のオフラインRLと計画手法（MCTS、AlphaZeroの探索、フェーズ9・10のモデルベースRL）はすべて学習済みまたは与えられたモデルに対してベルマンバックアップを反復する。

## コンセプト

![方策反復と価値反復の比較](../assets/dp.svg)

**2つのアルゴリズム、どちらもベルマンの不動点反復。**

**方策反復。** 方策が変化しなくなるまで2つのステップを交互に繰り返す。

1. *評価：* 方策`π`が与えられたとき、`V^π`を`V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`を収束まで繰り返し適用して計算する。
2. *改善：* `V^π`が与えられたとき、`π`を`V^π`に対して貪欲にする：`π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`。

収束は保証される：（a）各改善ステップは`π`を同じままにするか、いくつかの状態で`V^π`を厳密に増加させる、（b）決定的方策の空間は有限だから。通常、大きな状態空間でも約5〜20回の外側反復で収束する。

**価値反復。** 評価と改善を1回のスイープに折りたたむ。ベルマン*最適性*方程式を適用する：

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

`max_s |V_{new}(s) - V(s)| < ε`まで繰り返す。最後に貪欲行動を取って方策を抽出する。反復あたりの速度は厳密に速い——内側の評価ループがない——が収束には通常より多くの反復が必要。

**一般化方策反復（GPI）。** 統一的なフレーミング。価値関数と方策は相互改善のループに閉じ込められている；両者を相互一貫性に向けて駆動するあらゆる手法（非同期価値反復、修正方策反復、Q学習、アクタークリティック、PPO）はGPIのインスタンスだ。

**`γ < 1`が重要な理由。** ベルマン演算子はsup-ノルムで`γ`-縮小だ：`||T V - T V'||_∞ ≤ γ ||V - V'||_∞`。縮小は一意の不動点と幾何収束を意味する。`γ < 1`を外すと保証が失われる——有限ホライズンか吸収終端状態が必要になる。

## 実装する

### ステップ1：GridWorld MDPモデルの構築

レッスン01と同じ4×4 GridWorldを使う。確率的バリアントを追加：確率`0.1`でエージェントがランダムに垂直方向にスリップする。

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)`は`(s', r, p)`のリストを返す。これがモデル全体だ。

### ステップ2：方策評価

方策`π(s) = {action: prob}`が与えられたとき、`V`が動かなくなるまでベルマン方程式を反復する：

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### ステップ3：方策改善

`π`を`V`に対する貪欲方策に置き換える。`π`が変化しなければ、最適点にいるので返す。

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### ステップ4：組み合わせる

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # 任意のスタート
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

4×4での典型的な収束：4〜6回の外側反復。`V*(0,0) ≈ -6`とステップ数を厳密に削減する方策を出力する。

### ステップ5：価値反復（1ループ版）

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

同じ不動点、コードが少ない。

## 落とし穴

- **終端状態の処理を忘れる。** 吸収状態にベルマンを適用すると、何も変えない「最良行動」を選んでしまう。`if s == terminal: V[s] = 0`でガードする。
- **sup-ノルム対L2収束。** `max |V_new - V|`を使い、平均を使わない。理論的保証はsup-ノルムに対するものだ。
- **インプレース対同期更新。** インプレースで`V[s]`を更新（Gauss-Seidel）は別の`V_new`辞書より速く収束する（Jacobi）。本番コードはインプレースを使う。
- **方策の同点。** 2つの行動が同じQ値を持つ場合、`argmax`は各反復で異なるタイブレークをする可能性があり、「方策安定」チェックが振動を引き起こす。安定したタイブレーク（固定順序での最初の行動）を使う。
- **状態空間爆発。** DPはスイープあたり`O(|S| · |A|)`。約10⁷状態まで機能する。それを超えると関数近似が必要（フェーズ9・05以降）。

## 使ってみる

2026年では、DPは正しさのベースラインとプランナーの内部ループだ：

| ユースケース | メソッド |
|------------|---------|
| 小さな表形式MDPを厳密に解く | 価値反復（シンプル）または方策反復（外側ステップが少ない） |
| Q学習/PPO実装を検証する | おもちゃ環境でのDPの最適V*と比較 |
| モデルベースRL（フェーズ9・10） | 学習済み遷移モデルへのベルマンバックアップ |
| AlphaZero/MuZeroでの計画 | モンテカルロ木探索=非同期ベルマンバックアップ |
| オフラインRL（CQL、IQL） | 保守的なQ反復——OOD行動へのペナルティ付きDP |

誰かが「最適価値関数」と言う時、彼らはDPの不動点を意味する。論文で`V*`や`Q*`を見た時、このループを思い浮かべよう。

## 成果物を出す

`outputs/skill-dp-solver.md`として保存：

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## 演習

1. **易。** `γ ∈ {0.9, 0.99}`で4×4 GridWorldに価値反復を実行する。`max |ΔV| < 1e-6`まで何スイープかかるか？`V*`を4×4グリッドとして表示する。
2. **中。** 確率的GridWorld（スリップ確率`0.1`）で方策反復対価値反復を比較する。スイープ数、実行時間、最終`V*(0,0)`を計測する。反復回数と実行時間でどちらが速く収束するか？
3. **難。** 修正方策反復を構築する：評価ステップで収束まで実行する代わりに`k`スイープのみ実行する。`k ∈ {1, 2, 5, 10, 50}`に対して`V*(0,0)`誤差対`k`をプロットする。曲線は評価/改善トレードオフについて何を示しているか？

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|--------------|-----------|
| 方策反復 | 「DPアルゴリズム」 | 方策が変化しなくなるまで評価（`V^π`）と改善（`V^π`に対する貪欲`π`）を交互に行う。 |
| 価値反復 | 「より速いDP」 | 1回のスイープで適用されるベルマン最適バックアップ；`V*`に幾何収束する。 |
| ベルマン演算子 | 「再帰」 | `(T V)(s) = max_a Σ P (r + γ V(s'))`；sup-ノルムで`γ`-縮小。 |
| 縮小 | 「DPが収束する理由」 | `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|`を持つ演算子`T`は一意の不動点を持つ。 |
| GPI | 「すべてはDPだ」 | 一般化方策反復：`V`と`π`を相互一貫性に向けて駆動するあらゆる手法。 |
| 同期更新 | 「Jacobi型」 | スイープを通じて古い`V`を使用；分析はきれいだが遅い。 |
| インプレース更新 | 「Gauss-Seidel型」 | 更新中の`V`を使用；実践ではより速く収束する。 |

## 参考資料

- [Sutton & Barto (2018). 第4章——動的計画法](http://incompleteideas.net/book/RLbook2020.pdf) — 方策反復と価値反復の標準的な解説。
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) — 縮小マッピング論証の厳密な処理。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 修正方策反復とその収束解析。
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — 元の方策反復論文。
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — DPから近似DP/ディープRLへの橋渡しで、以降のすべてのレッスンで使われる。
