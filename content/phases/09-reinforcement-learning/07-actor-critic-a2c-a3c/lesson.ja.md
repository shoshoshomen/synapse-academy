# アクタークリティック——A2CとA3C

> REINFORCEはノイジーだ。`V̂(s)`を学習するクリティックを追加し、リターンから引くと、同じ期待値を持つがはるかに低い分散のアドバンテージが得られる。それがアクタークリティックだ。A2Cは同期的に実行し；A3Cはスレッドで実行する。どちらも現代のすべてのディープRLメソッドのメンタルモデルだ。


## 問題

バニラREINFORCEは機能するが、分散が酷い。モンテカルロリターン`G_t`はエピソード間で10倍を超えて変動する可能性がある。そのノイズを`∇ log π`と掛けて平均すると、はるかに少ないDQN更新で同じ距離だけ方策を移動できるのに、何千ものエピソードが必要な勾配推定量が生まれる。

分散は生のリターンを使うことから来ている。ベースライン`b(s_t)`——状態の任意の関数、学習された価値を含む——を引くと、期待値は変わらずに分散が下がる。最良の扱いやすいベースラインは`V̂(s_t)`だ。`∇ log π`に掛ける量が*アドバンテージ*になる：

`A(s, a) = G - V̂(s)`

行動は平均以上のリターンをもたらした場合は良い；平均以下の場合は悪い。学習したクリティックを持つREINFORCEが*アクタークリティック*だ。クリティックはアクターに低分散の教師を与える。これが2015年以降のすべてのディープ方策メソッド（A2C、A3C、PPO、SAC、IMPALA）だ。

## コンセプト

![アクタークリティック：方策ネットプラス価値ネット、アドバンテージとしてのTD残差](../assets/actor-critic.svg)

**2つのネットワーク、1つの共有損失：**

- **アクター** `π_θ(a | s)`：方策。行動するためにサンプリングされる。方策勾配で訓練される。
- **クリティック** `V_φ(s)`：状態からの期待リターンを推定する。`(V_φ(s) - target)²`を最小化するために訓練される。

**アドバンテージ。** 2つの標準的な形式：

- *MCアドバンテージ：* `A_t = G_t - V_φ(s_t)`。不偏、高い分散。
- *TDアドバンテージ：* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。バイアスあり（`V_φ`を使う）、はるかに低い分散。*TD残差*`δ_t`とも呼ばれる。

**nステップアドバンテージ。** 2つの間を補間する：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1`は純粋なTD。`n = ∞`はMC。ほとんどの実装はAtariで`n = 5`、MuJoCoのPPOで`n = 2048`を使う。

**一般化アドバンテージ推定（GAE）。** Schulman et al.（2016）はすべてのnステップアドバンテージに対する指数重み付き平均を提案した：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

`λ ∈ [0, 1]`で。`λ = 0`はTD（低分散、高バイアス）。`λ = 1`はMC（高分散、不偏）。`λ = 0.95`が2026年のデフォルト——バイアス/分散ダイアルを望む場所まで調整する。

**A2C：同期アドバンテージアクタークリティック。** `N`並列環境で`T`ステップ収集する。各ステップのアドバンテージを計算する。組み合わせたバッチでアクターとクリティックを更新する。繰り返す。A3Cのよりシンプルでよりスケーラブルな兄弟。

**A3C：非同期アドバンテージアクタークリティック。** Mnih et al.（2016）。`N`のワーカースレッドを生成し、それぞれが環境を実行する。各ワーカーは自分のロールアウトに対してローカルで勾配を計算し、非同期的に共有パラメータサーバに適用する。リプレイバッファは不要——ワーカーが異なる軌跡を実行することで非相関化する。A3CはCPUでスケールで訓練できることを証明した。2026年では、GPUが大きなバッチを望むため、GPUベースのA2C（バッチ並列環境）が支配的だ。

**組み合わせた損失。**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

3つの項：方策勾配損失、価値回帰、エントロピーボーナス。`c_v ~ 0.5`、`c_e ~ 0.01`が標準的な出発点。

## 実装する

### ステップ1：クリティック

MSEで更新される線形クリティック`V_φ(s) = w · features(s)`：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

表形式環境では、クリティックは数百エピソードで収束する。Atariでは線形クリティックを共有CNNトランク+価値ヘッドに置き換える。

### ステップ2：nステップアドバンテージ

長さ`T`のロールアウトとブートストラップされた最終`V(s_T)`が与えられた場合：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns`がクリティックターゲットだ。`advantages`が`∇ log π`に掛けられるものだ。

### ステップ3：組み合わせた更新

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # クリティック
    critic_update(w, x, target_v, lr_v)

    # アクター
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

オンポリシー、更新ごとに1つのロールアウト、アクターとクリティックに別々の学習率。

### ステップ4：並列化（A3C対A2C）

- **A3C：** `N`スレッドを起動する。各々が自分の環境と自分のフォワードパスを実行する。定期的に勾配更新を共有マスターにプッシュする。マスターのロックなし——レースはOK、ただノイズを追加するだけ。
- **A2C：** 単一プロセスで`N`の環境インスタンスを実行し、観測を`[N, obs_dim]`バッチにスタックし、バッチフォワードパス、バッチバックワードパス。GPU利用率が高く、決定的で、推論しやすい。2026年のデフォルト。

おもちゃコードは明確さのためにシングルスレッド；バッチA2Cへの書き直しはnumpyの3行だ。

## 落とし穴

- **アクター勾配前のクリティックバイアス。** クリティックがランダムな場合、そのベースラインは情報がなく、純粋なノイズで訓練している。方策勾配をオンにする前にクリティックを数百ステップウォームアップするか、遅いアクター学習率を使う。
- **アドバンテージ正規化。** アドバンテージをバッチごとにゼロ平均/単位標準偏差に正規化する。ほぼゼロのコストで訓練を大幅に安定させる。
- **共有トランク。** 画像入力にはアクターとクリティックに共有特徴抽出器を使う。別々のヘッド。共有特徴は両方の損失に便乗する。
- **オンポリシーコントラクト。** A2Cはデータを正確に1回の更新に再利用する。それ以上使うと勾配がバイアスされる（重要度サンプリング補正がPPOが追加するものだ）。
- **エントロピー崩壊。** `c_e > 0`なしで、方策は数百回の更新でほぼ決定的になり、探索を停止する。
- **報酬スケール。** アドバンテージの大きさは報酬スケールに依存する。一貫した勾配の大きさのために報酬を正規化する（例：実行標準偏差で除算）。

## 使ってみる

A2C/A3Cは2026年では最終選択ではほとんどないが、以降のすべてが改良するアーキテクチャだ：

| メソッド | A2Cとの関係 |
|---------|------------|
| PPO | A2C + マルチエポック更新のためのクリップされた重要度比 |
| IMPALA | A3C + Vトレースオフポリシー補正 |
| SAC（フェーズ9・07） | ソフト価値クリティックを持つオフポリシーA2C（次のレッスン） |
| GRPO（フェーズ9・12） | クリティックなしのA2C——グループ相対アドバンテージ |
| DPO | 嗜好ランキング損失に折りたたまれたA2C、サンプリングなし |
| AlphaStar / OpenAI Five | リーグ訓練+模倣事前訓練を持つA2C |

2026年の論文で「アドバンテージ」を見る時、アクタークリティックを考える。

## 成果物を出す

`outputs/skill-actor-critic-trainer.md`として保存：

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## 演習

1. **易。** 4×4 GridWorldでMCアドバンテージ（`G_t - V(s_t)`）を使ってアクタークリティックを訓練する。レッスン06のREINFORCE-with-running-mean-baselineとサンプル効率を比較する。
2. **中。** TDアドバンテージ（`r + γ V(s') - V(s)`）に切り替える。アドバンテージバッチの分散を計測する。どれだけ下がるか？
3. **難。** GAE(λ)を実装する。`λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`をスイープする。最終リターン対サンプル効率をプロットする。このタスクでバイアス/分散のスウィートスポットはどこか？

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|--------------|-----------|
| アクター | 「方策ネット」 | `π_θ(a\|s)`、方策勾配で更新される。 |
| クリティック | 「価値ネット」 | `V_φ(s)`、リターン/TDターゲットへのMSE回帰で更新される。 |
| アドバンテージ | 「平均よりどれだけ良いか」 | `A(s, a) = Q(s, a) - V(s)`またはその推定量。`∇ log π`の乗数。 |
| TD残差 | 「δ」 | `δ_t = r + γ V(s') - V(s)`；1ステップアドバンテージ推定。 |
| GAE | 「補間ノブ」 | `λ`でパラメータ化されたnステップアドバンテージの指数重み付き和。 |
| A2C | 「同期アクタークリティック」 | 環境間でバッチ化；ロールアウトごとに1勾配ステップ。 |
| A3C | 「非同期アクタークリティック」 | ワーカースレッドが共有パラメータサーバに勾配をプッシュする。元の論文；2026年では一般的でない。 |
| ブートストラップ | 「ホライズンでVを使う」 | ロールアウトを打ち切り、和を閉じるために`γ^n V(s_{t+n})`を追加する。 |

## 参考資料

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C、元の非同期アクタークリティック論文。
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE。
- [Sutton & Barto (2018). 第13章——アクタークリティックメソッド](http://incompleteideas.net/book/RLbook2020.pdf) — 基礎；クリティックがニューラルネットの場合は第9章の関数近似と組み合わせる。
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — Vトレースオフポリシー補正を持つスケーラブルな分散アクタークリティック。
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — 読む価値のある本番A2C/PPO実装。
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — 2タイムスケールアクタークリティック分解に対する基礎的な収束結果。
