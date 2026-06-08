# モンテカルロ法——完全エピソードからの学習

> 動的計画法はモデルを必要とする。モンテカルロはエピソード以外に何も必要としない。方策を実行し、リターンを観察し、平均する。RLで最も単純なアイデア——そして以降のすべてを解き放つもの。


## 問題

動的計画法はエレガントだが、すべての状態-行動ペアに対して`P(s' | s, a)`をクエリできることを前提とする。現実世界のほとんどのものはそのように機能しない。ロボットは関節トルク後のカメラピクセルの分布を解析的に計算できない。価格アルゴリズムはすべての可能な顧客反応に対して積分できない。LLMはトークン後のすべての可能な継続を列挙できない。

環境から*サンプリング*する能力だけを必要とするメソッドが必要だ。方策を実行する。軌跡`s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`を得る。価値を推定するために使う。それがモンテカルロだ。

DPからMCへのシフトは哲学的に重要だ：*既知モデル+厳密バックアップ*から*サンプリングされたロールアウト+平均リターン*に移行する。分散は跳ね上がるが、適用可能性は爆発する。このレッスン以降のすべてのRLアルゴリズム——TD、Q学習、REINFORCE、PPO、GRPO——は本質的にモンテカルロ推定量であり、時にはブートストラップが積み重なっている。

## コンセプト

![モンテカルロ：ロールアウト、リターンの計算、平均；初訪対全訪](../assets/monte-carlo.svg)

**1行でのコアアイデア：** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)` ここで`G^{(i)}(s)`は方策`π`のもとで`s`への訪問後に観測されたリターン。

**初訪対全訪MC。** 状態`s`を複数回訪問するエピソードが与えられた場合、初訪MCは最初の訪問からのリターンのみをカウントする；全訪MCはすべての訪問をカウントする。どちらも極限では不偏だ。初訪はより単純に分析できる（iidサンプル）。全訪はエピソードごとにより多くのデータを使い、実践では通常より速く収束する。

**インクリメンタル平均。** すべてのリターンを保存する代わりに、実行平均を更新する：

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

整理すると：`V_new = V_old + α · (target - V_old)` ここで`α = 1/n`。`1/n`を定数のステップサイズ`α ∈ (0, 1)`に置き換えると、`π`の変化を追跡する非定常MC推定量になる。この動きが、MCからTDへ、そして現代のすべてのRLアルゴリズムへのジャンプ全体だ。

**探索が問題になる。** DPは列挙によってすべての状態に触れた。MCは方策が訪問する状態しか見ない。`π`が決定的な場合、状態空間の全領域が一切サンプリングされず、その価値推定値は永遠にゼロのままになる。歴史的な順序での3つの修正：

1. **探索的スタート。** 各エピソードをランダムな(s, a)ペアから始める。カバレッジを保証する；実践では非現実的（ロボットを任意の状態に「リセット」できない）。
2. **ε-greedy。** 現在のQに対して貪欲に行動するが、確率`ε`でランダムな行動を選ぶ。すべての状態-行動ペアが漸近的にサンプリングされる。
3. **オフポリシーMC。** 行動方策`μ`のもとでデータを収集し、重要度サンプリングを通じてターゲット方策`π`について学ぶ。高い分散があるが、DQNのようなリプレイバッファ手法への橋渡しだ。

**モンテカルロ制御。** 評価→改善→評価、方策反復と同じだが評価はサンプリングベース：

1. `π`を実行し、エピソードを得る。
2. 観測されたリターンから`Q(s, a)`を更新する。
3. `π`を`Q`に対してε-greedyにする。
4. 繰り返す。

穏やかな条件（各ペアが無限回訪問される、`α`がRobbins-Monroを満たす）のもとで確率1で`Q*`と`π*`に収束する。

## 実装する

### ステップ1：ロールアウト → (s, a, r)のリスト

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

モデルなし、`env.reset()`と`env.step(s, a)`のみ。gymと同じインターフェース、ただし最小限に削ぎ落とされている。

### ステップ2：リターンの計算（逆順スイープ）

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

1パス、`O(T)`。後ろ向き再帰`G_t = r_{t+1} + γ G_{t+1}`は再合計を避ける。

### ステップ3：初訪MC評価

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

3行が仕事をする：最初の訪問で状態を既訪としてマーク、カウントをインクリメント、実行平均を更新。

### ステップ4：ε-greedy MCコントロール（オンポリシー）

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### ステップ5：DPのゴールドスタンダードと比較する

`V^π`のMC推定値は、エピソード→∞につれてレッスン02のDP結果と一致するはずだ。実践では：4×4 GridWorldで50,000エピソードでDP答えの`~0.1`以内に収まる。

## 落とし穴

- **無限エピソード。** MCはエピソードが*終了*することを必要とする。方策が永遠にループする可能性がある場合は`max_steps`でキャップし、キャップを暗黙の失敗として扱う。ランダム方策を持つGridWorldは定期的にタイムアウトする——これは正常、ただし正しくカウントすることを確認する。
- **分散。** MCは完全なリターンを使う。長いエピソードでは分散が巨大になる——最後の1つの不運な報酬が同じ量だけ`V(s_0)`をシフトさせる。TDメソッド（レッスン04）はブートストラップによってこれを削減する。
- **状態のカバレッジ。** 新鮮なQに対する貪欲なMCは1つの行動しか試さない。必ず探索が必要（ε-greedy、探索的スタート、UCB）。
- **非定常方策。** MCコントロールのように`π`が変化する場合、古いリターンは異なる方策からのもの。定数-αのMCはこれを処理する；サンプル平均MCはしない。
- **オフポリシー重要度サンプリング。** 重み`π(a|s)/μ(a|s)`は軌跡全体にかけられる。分散はホライズンとともに爆発する。per-decision重み付きISでキャップするかTDに切り替える。

## 使ってみる

2026年におけるモンテカルロ法の役割：

| ユースケース | MCの理由 |
|------------|---------|
| 短ホライズンゲーム（ブラックジャック、ポーカー） | エピソードが自然に終了；リターンがきれい。 |
| ログ済み方策のオフライン評価 | 保存された軌跡上で割引リターンを平均する。 |
| モンテカルロ木探索（AlphaZero） | 木の葉からのMCロールアウトが選択を導く。 |
| LLM RL評価 | 与えられた方策のサンプリングされた完了に対する平均報酬を計算する。 |
| PPOでのベースライン推定 | アドバンテージターゲット`A_t = G_t - V(s_t)`はMCの`G_t`を使う。 |
| RL教育 | 実際に機能する最も単純なアルゴリズム——ブートストラップを取り除いてコアを見る。 |

現代のディープRLアルゴリズム（PPO、SAC）は`n`ステップリターンまたはGAEを介して純粋MC（完全リターン）と純粋TD（1ステップブートストラップ）を補間する。両端は同じ推定量のインスタンスだ。

## 成果物を出す

`outputs/skill-mc-evaluator.md`として保存：

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## 演習

1. **易。** 4×4 GridWorldで一様ランダム方策の初訪MC評価を実装する。10,000エピソードを実行する。DP答えに対してエピソード数の関数として`V(0,0)`をプロットする。
2. **中。** `ε ∈ {0.01, 0.1, 0.3}`でε-greedy MCコントロールを実装する。20,000エピソード後の平均リターンを比較する。曲線はどのように見えるか？バイアス-分散トレードオフはどこにあるか？
3. **難。** 重要度サンプリングを使った*オフポリシー*MCを実装する：一様ランダム方策`μ`のもとでデータを収集し、決定的最適方策`π`の`V^π`を推定する。プレーンIS対per-decision IS対重み付きISを比較する。どれが最も低い分散か？

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|--------------|-----------|
| モンテカルロ | 「ランダムサンプリング」 | 分布からのiidサンプルを平均して期待値を推定する。 |
| リターン`G_t` | 「将来の報酬」 | ステップ`t`からエピソード終了までの割引報酬和：`Σ_{k≥0} γ^k r_{t+k+1}`。 |
| 初訪MC | 「各状態を1回カウント」 | エピソードでの最初の訪問のみが価値推定に寄与する。 |
| 全訪MC | 「すべての訪問を使う」 | すべての訪問が寄与する；わずかにバイアスがあるがサンプル効率が良い。 |
| ε-greedy | 「探索ノイズ」 | 確率`1-ε`で貪欲行動；確率`ε`でランダム行動。 |
| 重要度サンプリング | 「誤った分布からのサンプリングの補正」 | `π(a\|s)/μ(a\|s)`の積でリターンを再重み付けして`μ`データから`V^π`を推定する。 |
| オンポリシー | 「自分のデータから学ぶ」 | ターゲット方策=行動方策。通常のMC、PPO、SARSA。 |
| オフポリシー | 「他者のデータから学ぶ」 | ターゲット方策≠行動方策。重要度サンプリングMC、Q学習、DQN。 |

## 参考資料

- [Sutton & Barto (2018). 第5章——モンテカルロ法](http://incompleteideas.net/book/RLbook2020.pdf) — 標準的な解説。
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — 初訪対全訪の分析。
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — オフポリシーMCと分散制御。
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — 現代の低分散IS推定量。
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — MC/TDの自己対戦が超人的なプレイに収束する最初の大規模実証例；このフェーズの後半のすべてのレッスンの概念的前駆者。
