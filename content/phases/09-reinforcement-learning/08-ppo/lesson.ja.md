# 近接方策最適化（PPO）

> A2Cは各ロールアウトを1回の更新後に捨てる。PPOは方策勾配をクリップされた重要度比でラップするので、方策が爆発せずに同じデータで10+エポック実行できる。Schulman et al.（2017）。2026年でも依然としてデフォルトの方策勾配アルゴリズム。


## 問題

A2C（レッスン07）はオンポリシーだ：勾配`E_{π_θ}[A · ∇ log π_θ]`は*現在の*`π_θ`からサンプリングされたデータを必要とする。1回更新すると、`π_θ`が変わり；使用したデータはオフポリシーになる。再利用すると勾配がバイアスされる。

ロールアウトは高価だ。Atariでは、8環境×128ステップ=1024遷移で数十秒の環境時間が必要だ。1回の勾配ステップ後にそれを捨てるのは無駄だ。

信頼領域方策最適化（TRPO、Schulman 2015）が最初の修正だった：各更新を新旧方策間のKL発散が`δ`以下になるように制約する。理論的にはきれいだが、更新ごとに共役勾配解が必要。2026年では誰もTRPOを実行しない。

PPO（Schulman et al. 2017）はハードな信頼領域制約をシンプルなクリップされた目的関数に置き換える。コード1行追加。ロールアウトごとに10エポック。共役勾配なし。十分な理論的保証。9年後もMuJoCoからRLHFまでのすべてのデフォルト方策勾配アルゴリズムだ。

## コンセプト

![PPOクリップされたサロゲート目的関数：1 ± εでの比率クリッピング](../assets/ppo.svg)

**重要度比。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

これは新しい方策対データを収集した方策の尤度比だ。`r_t = 1`は変化なし。`r_t = 2`は新しい方策が古い方策より`a_t`を取る確率が2倍高いことを意味する。

**クリップされたサロゲート。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

2つの項：

- アドバンテージ`A_t > 0`で比率が`1 + ε`を超えようとする場合、クリップが勾配を平坦化——良い行動を古い確率より`+ε`以上押し上げない。
- アドバンテージ`A_t < 0`で比率が`1 - ε`を超えようとする場合（クリップされた削減と比べて悪い行動をより起こしやすくするという意味）、クリップが勾配を制限——悪い行動を`-ε`以下に押し下げない。

`min`は他の方向を処理する：比率が*有益な*方向に動いた場合、まだ勾配が得られる（あなたを傷つける側のクリッピングなし）。

典型的な`ε = 0.2`。`r_t`の関数として目的関数をプロット：「良い側」に平坦な屋根と「悪い側」に平坦な床を持つ区分線形関数。

**完全なPPO損失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

A2Cと同じアクタークリティック構造。3つの係数、通常`c_v = 0.5`、`c_e = 0.01`、`ε = 0.2`。

**訓練ループ。**

1. `N`並列環境で`T`ステップずつ`N × T`遷移を収集する。
2. アドバンテージ（GAE）を計算し、定数として凍結する。
3. `π_{θ_old}`を現在の`π_θ`のスナップショットとして凍結する。
4. `K`エポックの間、`(s, a, A, V_target, log π_old(a|s))`の各ミニバッチに対して：
   - `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`を計算する。
   - `L^{CLIP}` + 価値損失 + エントロピーを適用する。
   - 勾配ステップ。
5. ロールアウトを捨てる。ステップ1に戻る。

`K = 10`と64のミニバッチが標準的なハイパーパラメータセットだ。PPOは堅牢：正確な数値は通常±50%の範囲内では重要でない。

**KLペナルティバリアント。** 元の論文は適応的なKLペナルティを使う代替案を提案した：観測されたKLに基づいて`β`を調整した`L = L^{PG} - β · KL(π_θ || π_old)`。クリッピングバージョンが支配的になった；KLバリアントはRLHF（参照方策へのKLが常に必要な別の制約）で生き残る。

## 実装する

### ステップ1：ロールアウト時に`log π_old(a | s)`を記録する

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

スナップショットはロールアウト時に1回取られる。更新エポック中は変わらない。

### ステップ2：GAEアドバンテージを計算する（レッスン07）

A2Cと同じ。バッチ全体で正規化する。

### ステップ3：クリップされたサロゲート更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # クリップされた
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

「クリップ→ゼロ勾配」パターンがPPOの心臓部だ。新しい方策が有益な方向にすでに大きくずれている場合、更新が止まる。

### ステップ4：価値とエントロピー

クリティックターゲットに標準MSEを追加し、アクターにエントロピーボーナスを追加する（A2Cと同じ）。

### ステップ5：診断

各更新で監視する3つのこと：

- **平均KL** `E[log π_old - log π_θ]`。`[0, 0.02]`に収まるべき。`0.1`を超えると`K_EPOCHS`またはLRを下げる。
- **クリップ率**——比率が`[1-ε, 1+ε]`の外にあるサンプルの割合。`~0.1-0.3`になるべき。`~0`なら、クリップが機能していない→LRまたは`K_EPOCHS`を上げる。`~0.5+`なら、ロールアウトにオーバーフィットしている→下げる。
- **説明分散** `1 - Var(V_target - V_pred) / Var(V_target)`。クリティック品質指標。クリティックが学習するにつれて1に向かって上昇するべき。

## 落とし穴

- **クリップ係数の誤調整。** `ε = 0.2`が事実上の標準。`0.1`にすると更新が小心すぎる；`0.3+`は不安定を招く。
- **エポックが多すぎる。** `K > 20`は方策が`π_old`から大きくずれるために定期的に不安定化する。エポックをキャップする、特に大きなネットワークの場合。
- **報酬正規化なし。** 大きな報酬スケールはクリップ範囲を食いつぶす。アドバンテージを計算する前に報酬を正規化する（実行標準偏差）。
- **アドバンテージ正規化を忘れる。** バッチごとのゼロ平均/単位標準偏差正規化が標準。スキップするとほとんどのベンチマークでPPOが台無しになる。
- **学習率が減衰しない。** PPOはゼロへの線形LR減衰の恩恵を受ける。定数LRはしばしばより悪い。
- **重要度比の数学的エラー。** 数値安定性のために常に`exp(log_new - log_old)`であり、`new / old`ではない。
- **勾配の符号が間違っている。** サロゲートを最大化=`-L^{CLIP}`を*最小化*する。符号の反転が最も一般的なPPOバグだ。

## 使ってみる

PPOは2026年のデフォルトRLアルゴリズムとして驚くほど多くのドメインで使われる：

| ユースケース | PPOバリアント |
|------------|-------------|
| MuJoCo/ロボティクス制御 | ガウシアン方策、GAE(0.95)を持つPPO |
| Atari/離散ゲーム | カテゴリカル方策、128ステップのローリングロールアウトを持つPPO |
| LLMのRLHF | 参照モデルへのKLペナルティ付きPPO、応答末尾でRMからの報酬 |
| 大規模ゲームエージェント | IMPALA + PPO（AlphaStar、OpenAI Five） |
| 推論LLM | GRPO（レッスン12）——クリティックなしのPPOバリアント |
| 嗜好のみのデータ | DPO——PPO+KLの閉形式折りたたみ、オンラインサンプリングなし |

PPOの*損失形状*——クリップされたサロゲート+価値+エントロピー——がDPO、GRPO、ほぼすべてのRLHFパイプラインの足場だ。

## 成果物を出す

`outputs/skill-ppo-trainer.md`として保存：

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## 演習

1. **易。** `ε=0.2, K=4`で4×4 GridWorldにPPOを実行する。同じ環境ステップでのA2C（ロールアウトごとに1エポック）とサンプル効率を比較する。
2. **中。** `K ∈ {1, 4, 10, 30}`をスイープする。リターン対環境ステップをプロットし、更新ごとの平均KLを追跡する。このタスクで何`K`でKLが爆発するか？
3. **難。** クリップされたサロゲートを適応的なKLペナルティに置き換える（`KL > 2·target`なら`β`を2倍、`KL < target/2`なら半分に）。最終リターン、安定性、クリップなしと比較する。

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|--------------|-----------|
| 重要度比 | 「r_t(θ)」 | `π_θ(a\|s) / π_old(a\|s)`；データを収集した方策からの偏差。 |
| クリップされたサロゲート | 「PPOの主要トリック」 | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；有益な側のクリップを超えた平坦な勾配。 |
| 信頼領域 | 「TRPOのPPOの意図」 | 単調改善を保証するために各更新のKLを制限する。 |
| KLペナルティ | 「ソフト信頼領域」 | 代替PPO：`L - β · KL(π_θ \|\| π_old)`。適応的な`β`。 |
| クリップ率 | 「クリッピングが発動する頻度」 | 診断——0.1-0.3になるべき；外れると誤調整を意味する。 |
| マルチエポック訓練 | 「データ再利用」 | 各ロールアウトでKエポック；分散コストをサンプル効率と交換。 |
| ほぼオンポリシー | 「ほぼオンポリシー」 | PPOは名目上オンポリシーだが、K>1エポックは若干オフポリシーのデータを安全に使う。 |
| PPO-KL | 「もう1つのPPO」 | KLペナルティバリアント；参照へのKLがすでに制約のRLHFで使用。 |

## 参考資料

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 論文。
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO、PPOの前身。
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — すべてのPPOハイパーパラメータのアブレーション。
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT；RLHFにおけるPPOレシピ。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — PyTorchを使った明確な現代的解説。
- [CleanRL PPO実装](https://github.com/vwxyzjn/cleanrl) — 多くの論文で使用される参照単一ファイルPPO。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — 言語モデルに対するPPOの本番レシピ；レッスン09（RLHF）と合わせて読む。
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — 「37のコードレベルの最適化」論文；どのPPOトリックが重要でどれが俗説か。
