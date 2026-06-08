# 方策勾配——REINFORCEをゼロから

> 価値の推定をやめる。方策を直接パラメータ化し、期待リターンの勾配を計算し、上り坂を進む。Williams（1992）が1つの定理で書いた。PPO、GRPO、そしてすべてのLLM RLループが存在する理由だ。


## 問題

Q学習とDQNは*価値*関数をパラメータ化する。行動は`argmax Q`で選ぶ。これは離散行動と離散状態では問題ない。行動が連続の場合（10次元トルクに対する`argmax`とは？）や確率的方策が欲しい場合（`argmax`は定義上決定的）は壊れる。

方策勾配は代わりに*方策*をパラメータ化する。`π_θ(a | s)`は行動の分布を出力するニューラルネットだ。行動するためにサンプリングする。`θ`に対する期待リターンの勾配を計算する。上り坂を進む。`argmax`なし。ベルマン再帰なし。ただ`J(θ) = E_{π_θ}[G]`に対する勾配上昇。

REINFORCEの定理（Williams 1992）はこの勾配が計算可能だと教えてくれる：`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。エピソードを実行する。リターンを計算する。各ステップで`∇ log π_θ(a | s)`を掛ける。平均する。勾配上昇する。完了。

2026年のすべてのLLM-RLアルゴリズム——PPO、DPO、GRPO——はREINFORCEの改良だ。指先までそれを理解することがこのフェーズの残りの前提条件であり、フェーズ10・07（RLHF実装）と10・08（DPO）の前提条件だ。

## コンセプト

![方策勾配：ソフトマックス方策、log-π勾配、リターン重み付け更新](../assets/policy-gradient.svg)

**方策勾配定理。** `θ`でパラメータ化された任意の方策`π_θ`に対して：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

ここで`G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}`はステップ`t`からの割引リターンだ。期待値は`π_θ`からサンプリングされた完全な軌跡`τ`に対するもの。

**証明は短い。** 期待値のもとで`J(θ) = Σ_τ P(τ; θ) G(τ)`を微分する。`∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`（log微分トリック）を使う。`log P(τ; θ) = Σ log π_θ(a_t | s_t) + θに依存しない環境項`を因数分解する。環境項は消える。2行の代数で定理が得られる。

**分散削減トリック。** バニラREINFORCEは殺人的な分散を持つ——リターンはノイジー、`∇ log π`はノイジー、その積は非常にノイジー。2つの標準的な修正：

1. **ベースライン減算。** `G_t`を`G_t - b(s_t)`に置き換える。ここで`b(s_t)`は`a_t`に依存しない任意のベースライン。`E[b(s_t) · ∇ log π(a_t | s_t)] = 0`なので不偏。典型的な選択：クリティックが学習した`b(s_t) = V̂(s_t)` → アクタークリティック（レッスン07）。
2. **報酬-to-go。** `Σ_t G_t · ∇ log π_θ(a_t | s_t)`を`Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`に置き換える。与えられた行動には将来のリターンのみが重要——過去の報酬はゼロ平均ノイズを寄与する。

組み合わせると：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

これがベースライン付きREINFORCE——A2C（レッスン07）とPPO（レッスン08）の直接の祖先。

**ソフトマックス方策パラメータ化。** 離散行動の場合、標準的な選択：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

ここで`f_θ`は行動ごとのスコアを出力する任意のニューラルネット。勾配はきれいな形式を持つ：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

つまり、取られた行動のスコアマイナスその方策のもとでの期待値。

**連続行動のためのガウシアン方策。** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)`は閉じた形式を持つ。それがフェーズ9・07のSACが必要とするすべてだ。

## 実装する

### ステップ1：ソフトマックス方策ネットワーク

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

表形式環境には線形方策（行動ごとに1つの重みベクトル）を使う。Atariではこのソフトマックスヘッドを持つCNNに交換する。

### ステップ2：サンプリングと対数確率

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### ステップ3：log確率を記録したロールアウト

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### ステップ4：REINFORCE更新

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

勾配`∇ log π(a|s) = e_a - π(·|s)`（`a`のワンホットマイナス確率）がソフトマックス方策勾配の心臓部だ。筋肉の記憶に焼き付けよう。

### ステップ5：ベースライン

`G`の最近のエピソードにわたる実行平均は4×4 GridWorldを動かすために十分な分散削減；約500エピソードで収束する。ベースラインを学習された`V̂(s)`にアップグレードするとアクタークリティックになる。

## 落とし穴

- **爆発する勾配。** リターンは巨大になる可能性がある。`∇ log π`を掛ける前に必ずバッチ全体で`G`を`~N(0, 1)`に正規化する。
- **エントロピー崩壊。** 方策が早期にほぼ決定的な行動に収束し、探索を停止し、行き詰まる。修正：目的にエントロピーボーナス`β · H(π(·|s))`を追加する。
- **高い分散。** バニラREINFORCEは何千ものエピソードが必要。クリティックベースライン（レッスン07）またはTRPO/PPOの信頼領域（レッスン08）が標準的な修正だ。
- **サンプル非効率性。** オンポリシーとは1回の更新後にすべての遷移を捨てることを意味する。重要度サンプリングを介したオフポリシー補正がデータを戻すが、分散のコストがかかる（PPOの比率はクリップされたIS重みだ）。
- **非定常勾配。** 100エピソード前からの同じ勾配は古い`π`を使う。オンポリシーメソッドがこの理由で数回のロールアウトごとに更新する。
- **クレジット割り当て。** 報酬-to-goなしでは、過去の報酬がノイズを寄与する。常に報酬-to-goを使う。

## 使ってみる

2026年では、REINFORCEはほとんど直接実行されないが、その勾配式はどこにでもある：

| ユースケース | 派生メソッド |
|------------|------------|
| 連続制御 | ガウシアン方策を持つPPO / SAC |
| LLM RLHF | KLペナルティ付きPPO、トークンレベル方策上で実行 |
| LLM推論（DeepSeek） | GRPO——グループ相対ベースライン付きREINFORCE、クリティックなし |
| マルチエージェント | 中央集権型クリティックREINFORCE（MADDPG、COMA） |
| 離散行動ロボティクス | A2C、A3C、PPO |
| 嗜好のみの設定 | DPO——REINFORCE を嗜好-尤度損失として書き直したもの、サンプリングなし |

2026年の訓練スクリプトで`loss = -advantage * log_prob`を見る時、それはベースライン付きREINFORCEだ。論文全体（DPO、GRPO、RLOO）がこの1行の上の分散削減トリックだ。

## 成果物を出す

`outputs/skill-policy-gradient-trainer.md`として保存：

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## 演習

1. **易。** 線形ソフトマックス方策を持つ4×4 GridWorldにREINFORCEを実装する。ベースラインなしで1,000エピソード訓練する。学習曲線をプロットし；分散（リターンの標準偏差）を計測する。
2. **中。** 実行平均ベースラインを追加する。再び訓練する。バニラ実行と比較してサンプル効率と分散を比較する。ベースラインは収束までのステップ数をどれだけ削減するか？
3. **難。** エントロピーボーナス`β · H(π)`を追加する。`β ∈ {0, 0.01, 0.1, 1.0}`をスイープする。最終リターンと方策エントロピーをプロットする。このタスクでのスウィートスポットはどこか？

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|--------------|-----------|
| 方策勾配 | 「方策を直接訓練する」 | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`；log微分トリックから導出。 |
| REINFORCE | 「元のPGアルゴリズム」 | Williams（1992）；モンテカルロリターンにlog方策勾配を掛けたもの。 |
| log微分トリック | 「スコア関数推定量」 | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`；期待値の勾配を扱いやすくする。 |
| ベースライン | 「分散削減」 | `G`から引く任意の`b(s)`；`E[b · ∇ log π] = 0`なので不偏。 |
| 報酬-to-go | 「将来のリターンのみが重要」 | 完全な`G_0`の代わりに`G_t^{from t}`；正確で低分散。 |
| エントロピーボーナス | 「探索を促進する」 | `+β · H(π(·\|s))`項が方策の崩壊を防ぐ。 |
| オンポリシー | 「今見たものから訓練する」 | 勾配期待値は現在の方策に対するもの——古いデータを直接再利用できない。 |
| アドバンテージ | 「平均よりどれだけ良いか」 | `A(s, a) = G(s, a) - V(s)`；ベースライン付きREINFORCEが掛ける符号付き量。 |

## 参考資料

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — 元のREINFORCE論文。
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — 関数近似を持つ現代的な方策勾配定理。
- [Sutton & Barto (2018). 第13章——方策勾配メソッド](http://incompleteideas.net/book/RLbook2020.pdf) — 教科書的解説。
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — PyTorchコードを含む明確な教育的解説。
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — REINFORCEをTRPO、PPOの信頼領域ファミリーに結びつける分散削減と自然勾配の視点。
