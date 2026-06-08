# MDP・状態・行動・報酬

> マルコフ決定過程は5つの要素で構成される：状態、行動、遷移、報酬、割引。RLにおけるすべて——Q学習、PPO、DPO、GRPO——はこの形式に基づいて最適化される。一度理解すれば、強化学習の残りすべてを無料で読めるようになる。


## 問題

チェスのボットを書いているとしよう。あるいは在庫計画システム、取引エージェント、推論モデルを訓練するためのPPOループ。4つの異なるドメイン、しかし驚くべき事実がある：いずれも同じ数学的オブジェクトに帰結する。

教師あり学習は`(x, y)`のペアを与え、関数を適合させるよう求める。強化学習はラベルを与えない——状態・取った行動・スカラー報酬のストリームのみだ。その手がゲームに勝ったか？補充の決定がコストを削減したか？取引は利益を生んだか？LLMが生成したトークンが審判からより高い報酬をもたらすか？

このストリームから学習するには、まず形式化が必要だ。「見たもの」「したこと」「次に起きたこと」「それがどれほど良かったか」——それぞれが推論できるオブジェクトになる必要がある。その形式化がマルコフ決定過程だ。このフェーズのすべてのRLアルゴリズム——終盤のRLHFとGRPOループも含めて——はこの形式に基づいて最適化される。

## コンセプト

![マルコフ決定過程：状態、行動、遷移、報酬、割引](../assets/mdp.svg)

**5つの要素。**

- **状態** `S`。エージェントが決定するために必要なすべて。GridWorldではマス目。チェスではボード。LLMではコンテキストウィンドウとメモリ。
- **行動** `A`。選択肢。上/下/左/右に移動。手を指す。トークンを出力する。
- **遷移** `P(s' | s, a)`。状態`s`と行動`a`が与えられたとき、次の状態の分布。チェスでは決定的、在庫管理では確率的、LLMデコードではほぼ決定的。
- **報酬** `R(s, a, s')`。スカラー信号。勝ち=+1、負け=-1。売上マイナスコスト。GRPOの対数尤度比項。
- **割引** `γ ∈ [0, 1)`。将来の報酬が現在と比べてどれだけ重要か。`γ = 0.99`は約100ステップの有効ホライズンを持つ；`γ = 0.9`は約10ステップ。

**マルコフ性** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`。将来は現在の状態のみに依存する。そうでない場合、状態表現が不完全——これはメソッドの欠陥ではなく、状態の欠陥だ。

**方策とリターン。** 方策`π(a | s)`は状態を行動の分布にマッピングする。リターン`G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …`は将来の報酬の割引和だ。価値`V^π(s) = E[G_t | s_t = s]`は方策`π`のもとで`s`から始まる期待リターンだ。Q値`Q^π(s, a) = E[G_t | s_t = s, a_t = a]`は特定の行動から始まる期待リターンだ。すべてのRLアルゴリズムはこの2つのどちらかを推定し、それに応じて`π`を改善する。

**ベルマン方程式。** このフェーズのすべてで使われる不動点方程式：

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

これらは期待リターンを「今のステップの報酬」と「着地点の割引価値」に分割する。再帰的だ。フェーズ9のすべてのアルゴリズムは、この方程式を収束まで反復（動的計画法）するか、サンプリングするか（モンテカルロ）、1ステップだけブートストラップする（時間差分）。

## 実装する

### ステップ1：小さな決定的MDP

4×4のGridWorld。エージェントは左上からスタート、右下の終端状態を目指す。ステップごとに-1の報酬、行動は`{up, down, left, right}`。`code/main.py`を参照。

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

5行。これが環境全体だ。決定的な遷移、一定のステップペナルティ、吸収終端状態。

### ステップ2：方策をロールアウトする

方策は状態から行動の分布への関数だ。最も単純なもの：一様ランダム。

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

ランダム方策を1000回実行する。この4×4ボードでの平均リターンは-60から-80程度。最適なリターンは-6（右下への最短経路）。このギャップを埋めることがフェーズ9のすべてだ。

### ステップ3：ベルマン方程式で`V^π`を厳密に計算する

小さなMDPでは、ベルマン方程式は線形システムだ。状態を列挙し、期待値を適用し、値が変化しなくなるまで反復する。

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

これは反復方策評価だ。Sutton & Bartoの最初のアルゴリズムであり、以降のすべてのRLメソッドの理論的基盤だ。

### ステップ4：`γ`は物理的な意味を持つハイパーパラメータ

有効ホライズンはおおよそ`1 / (1 - γ)`。`γ = 0.9` → 10ステップ。`γ = 0.99` → 100ステップ。`γ = 0.999` → 1000ステップ。

低すぎるとエージェントが近視眼的に行動する。高すぎるとクレジット割り当てがノイジーになる——遠い将来の報酬に対して多くの初期ステップが責任を共有するから。LLM RLHFは通常`γ = 1`を使う（エピソードが短く有界だから）。制御タスクでは`0.95–0.99`、長ホライズンの戦略ゲームでは`0.999`。

## 落とし穴

- **非マルコフ的状態。** 決定するために最後の3つの観測が必要な場合、「状態」は現在の観測だけではない。修正：フレームを積み重ねる（DQNはAtariで4フレームを積む）かリカレント状態を使う（観測に対するLSTM/GRU）。
- **疎な報酬。** 勝利のみの報酬は大きな状態空間での学習をほぼ不可能にする。報酬を形成する（中間的な信号）か、模倣でブートストラップする（フェーズ9・09）。
- **報酬ハッキング。** プロキシ報酬を最適化すると病的な行動を生む。OpenAIのボートレーシングエージェントは、ゴールを目指す代わりにパワーアップを集め続けて永遠に回転し続けた。常にターゲットの結果からではなくプロキシから報酬を定義する。
- **割引の誤設定。** `γ = 1`を無限ホライズンタスクに使うとすべての値が無限大になる。必ず有限ホライズンか`γ < 1`を使う。
- **報酬スケール。** {+100, -100}対{+1, -1}の報酬は同一の最適方策をもたらすが、勾配の大きさは大きく異なる。PPO/DQNに入れる前に`[-1, 1]`程度に正規化する。

## 使ってみる

2026年のスタックでは、コードを書く前にすべてのRLパイプラインをMDPに落とし込む：

| 状況 | 状態 | 行動 | 報酬 | γ |
|------|------|------|------|---|
| 制御（ロコモーション、マニピュレーション） | 関節角度＋速度 | 連続トルク | タスク固有の形成済み | 0.99 |
| ゲーム（チェス、囲碁、ポーカー） | ボード＋履歴 | 合法手 | 勝ち=+1/負け=-1 | 1.0（有限） |
| 在庫/価格設定 | 在庫＋需要 | 注文数量 | 売上マイナスコスト | 0.95 |
| LLMのRLHF | コンテキストトークン | 次のトークン | エピソード末尾での報酬モデルスコア | 1.0（エピソード約200トークン） |
| 推論のためのGRPO | プロンプト＋部分応答 | 次のトークン | 末尾での検証器0/1 | 1.0 |

訓練ループを書く前に5つのタプルを書くこと。「RLが機能しない」というバグレポートのほとんどは、紙上で壊れていたMDP定式化にさかのぼる。

## 成果物を出す

`outputs/skill-mdp-modeler.md`として保存：

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## 演習

1. **易。** 4×4 GridWorldとランダム方策のロールアウトを`code/main.py`に実装する。10,000エピソードを実行。リターンの平均と標準偏差を報告する。最適リターン（-6）と比較する。
2. **中。** `γ ∈ {0.5, 0.9, 0.99}`で一様ランダム方策に対して`policy_evaluation`を実行する。各ケースで`V`を4×4グリッドとして表示する。終端近くの状態価値が大きい`γ`でより速く成長する理由を説明する。
3. **難。** GridWorldを確率的にする：各行動が確率`p = 0.1`で隣接する方向にスリップする。一様方策を再評価する。`V[start]`は良くなるか悪くなるか？なぜか？

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|--------------|-----------|
| MDP | 「強化学習のセットアップ」 | マルコフ性を満たすタプル`(S, A, P, R, γ)`。 |
| 状態 | 「エージェントが見るもの」 | 選択した方策クラスのもとでの将来のダイナミクスに対する十分統計量。 |
| 方策 | 「エージェントの行動」 | 条件付き分布`π(a \| s)`または決定的マッピング`s → a`。 |
| リターン | 「総報酬」 | 現在のステップからの割引和`Σ γ^t r_t`。 |
| 価値 | 「状態がどれほど良いか」 | `π`のもとで`s`から始まる期待リターン。 |
| Q値 | 「行動がどれほど良いか」 | `π`のもとで`s`から最初の行動`a`で始まる期待リターン。 |
| ベルマン方程式 | 「動的計画の再帰」 | 価値/Qを1ステップ報酬と割引後継価値に分解する不動点方程式。 |
| 割引`γ` | 「将来対現在」 | 遠い将来の報酬への幾何的重み；有効ホライズン`~1/(1-γ)`。 |

## 参考資料

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 教科書。第3章はMDPとベルマン方程式を扱う；第1章はその後のすべての基礎となる報酬仮説を動機づける。
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — ベルマン方程式の起源。
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — ディープRL視点からの簡潔なMDP入門。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — MDPと厳密な解法に関するオペレーションズ・リサーチのリファレンス。
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — MDPを動的計画の特殊化として最も明確に導出したもの。
