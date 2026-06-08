# 時間差分学習——Q学習とSARSA

> モンテカルロはエピソードが終わるまで待つ。TDは次の価値推定をブートストラップして毎ステップ後に更新する。Q学習はオフポリシーで楽観的；SARSAはオンポリシーで慎重だ。どちらもコード1行。どちらもこのフェーズのすべてのディープRLメソッドの基礎をなす。


## 問題

モンテカルロは機能するが、2つの高コストな要求がある。終了するエピソードが必要で、最終的なリターンが出るまで更新しない。エピソードが1,000ステップなら、MCは何も更新するために1,000ステップ待つ。高分散、低バイアス、そして実践では遅い。

動的計画法は逆のプロファイルを持つ——ゼロ分散のブートストラップバックアップ——しかし既知のモデルを必要とする。

時間差分（TD）学習は中間を取る。単一の遷移`(s, a, r, s')`から1ステップターゲット`r + γ V(s')`を形成し、`V(s)`をそこに向けて少しずらす。モデルなし。完全なエピソードなし。右辺に近似`V`を使うことによるバイアスがあるが、MCよりも劇的に低い分散とステップ1からのオンライン更新。

これが現代のすべてのRL——DQN、A2C、PPO、SAC——が回転する軸だ。フェーズ9の残りは、このレッスンで書く1ステップTD更新の上に積み重ねられた関数近似とトリックのレイヤーだ。

## コンセプト

![Q学習対SARSA：オフポリシーmaxとオンポリシーQ(s', a')](../assets/td.svg)

**Vのために TD(0)更新：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

括弧内の量がTD誤差`δ = r + γ V(s') - V(s)`だ。MCの`G_t - V(s_t)`のオンライン類似だ。収束にはRobbins-Monroを満たす`α`（`Σ α = ∞`、`Σ α² < ∞`）とすべての状態が無限回訪問されることが必要。

**Q学習。** 制御のためのオフポリシーTDメソッド：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max`は、エージェントが実際に取る行動に関わらず、`s'`から先は*貪欲*方策が続くと仮定する。その分離により、Q学習はε-greedyで探索しながら`Q*`を学習できる。Mnih et al.（2015）はこれをAtariのディープQ学習に変換した（レッスン05）。

**SARSA。** オンポリシーTDメソッド：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

名前はタプル`(s, a, r, s', a')`。SARSAは貪欲な`argmax`ではなく、エージェントが次に*実際に*取る行動`a'`を使う。実行中のε-greedy`π`に対して`Q^π`に収束し、極限`ε → 0`では`Q*`になる。

**cliff-walkingの違い。** 古典的なcliff-walkingタスク（崖からの落下=報酬-100）では、Q学習は崖のエッジに沿った最適パスを学習するが探索中に時々ペナルティを受ける。SARSAは崖から1ステップ離れた安全なパスを学習する——Q値に探索ノイズを考慮しているから。訓練すると`ε → 0`で両者とも最適に達する。実践では重要だ：探索が実際にデプロイ中に起きている場合、SARSAの行動はより保守的だ。

**期待SARSA。** `Q(s', a')`をその期待値で置き換える：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

SARSAより低い分散（`a'`のサンプルなし）、同じオンポリシーターゲット。現代の教科書ではデフォルトであることが多い。

**nステップTDとTD(λ)。** ブートストラップ前に`n`ステップ待つことでTD(0)とMCの間を補間する。`n=1`はTD。`n=∞`はMC。TD(λ)は幾何重み`(1-λ)λ^{n-1}`で全`n`を平均する。ほとんどのディープRLは3〜20の間の`n`を使う。

## 実装する

### ステップ1：ε-greedy方策のSARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

8行。Q学習との*唯一*の違いはターゲット行だ。

### ステップ2：Q学習

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max`がターゲットを行動から切り離す。その1つのシンボルがオンポリシーとオフポリシーの違いだ。

### ステップ3：学習曲線

100エピソードごとの平均リターンを追跡する。Q学習は単純な決定的GridWorldで速く収束する；SARSAはcliff-walkingでより保守的だ。`code/main.py`の4×4 GridWorldでは、`α=0.1, ε=0.1`で約2,000エピソード後に両者ともほぼ最適になる。

### ステップ4：DPの真値と比較する

価値反復（レッスン02）を実行して`Q*`を得る。`max_{s,a} |Q_learned(s,a) - Q*(s,a)|`を確認する。健全な表形式TDエージェントは10,000エピソード後に4×4 GridWorldで`~0.5`以内に収まる。

## 落とし穴

- **初期Q値が重要。** 楽観的初期化（負の報酬タスクで`Q = 0`）は探索を促進する。悲観的初期化は貪欲方策を永遠に閉じ込める可能性がある。
- **αスケジュール。** 定数`α`は非定常問題に対して良い。減衰`α_n = 1/n`は理論的には収束を与えるが実践では遅すぎる——`α`を`[0.05, 0.3]`に固定して学習曲線を監視する。
- **εスケジュール。** 高い`ε`からスタート（`ε=1.0`）、`ε=0.05`に減衰する。GLIE（無限探索極限での貪欲）がQ学習の収束条件だ。
- **Q学習の最大化バイアス。** `max`演算子はQがノイジーな時に上向きにバイアスがかかる。過大評価につながる——Hasseltのダブルなぜなら学習（レッスン05のDDQNで使用）は2つのQテーブルでこれを修正する。
- **非終了エピソード。** TDはターミナルなしで学習できるが、ステップをキャップするかキャップでのブートストラップを正しく処理する必要がある。標準：キャップを非ターミナルとして扱い、ブートストラップを続ける。
- **状態ハッシュ。** 状態がタプル/テンソルの場合、ハッシュ可能なキーを使う（リストではなくタプル；生の浮動小数点ではなく丸めた浮動小数点のタプル）。

## 使ってみる

2026年のTDランドスケープ：

| タスク | メソッド | 理由 |
|--------|---------|------|
| 小さな表形式環境 | Q学習 | 最適方策を直接学習する。 |
| オンポリシー安全重視 | SARSA / 期待SARSA | 探索中に保守的。 |
| 高次元状態 | DQN（フェーズ9・05） | リプレイとターゲットネットを持つニューラルネットQ関数。 |
| 連続行動 | SAC / TD3（フェーズ9・07） | QネットワークへのTD更新；方策ネットが行動を出力する。 |
| LLM RL（報酬モデルベース） | PPO / GRPO（フェーズ9・08、12） | GAE経由のTDスタイルアドバンテージを持つアクタークリティック。 |
| オフラインRL | CQL / IQL（フェーズ9・08） | 保守的な正則化を持つQ学習。 |

2026年の論文で読む「RL」の90%はQ学習またはSARSAの何らかの精緻化だ。より深く読む前に指の先まで表形式の更新を理解する。

## 成果物を出す

`outputs/skill-td-agent.md`として保存：

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## 演習

1. **易。** 4×4 GridWorldにQ学習とSARSAを実装する。2,000エピソードの学習曲線（100エピソードごとの平均リターン）をプロットする。どちらが速く収束するか？
2. **中。** cliff-walking環境（4×12、最終行は報酬-100でスタートにリセットの崖）を構築する。Q学習とSARSAの最終方策を比較する。各々が取るパスのスクリーンショットを撮る。どちらが崖に近いか？
3. **難。** ダブルQ学習を実装する。ノイジー報酬GridWorld（ステップごとの報酬にガウシアンノイズσ=5を加算）で、Q学習が`V*(0,0)`を意味のある量で過大評価するが、ダブルQ学習はしないことを示す。

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|--------------|-----------|
| TD誤差 | 「更新信号」 | `δ = r + γ V(s') - V(s)`、ブートストラップされた残差。 |
| TD(0) | 「1ステップTD」 | 次の状態の推定値のみを使って各遷移後に更新する。 |
| Q学習 | 「オフポリシーRL入門」 | 次の状態行動に対して`max`を持つTD更新；行動方策に関わらず`Q*`を学習する。 |
| SARSA | 「オンポリシーQ学習」 | 実際の次の行動を使うTD更新；現在のε-greedyのπに対して`Q^π`を学習する。 |
| 期待SARSA | 「低分散SARSA」 | サンプリングされた`a'`をπのもとでの期待値に置き換える。 |
| GLIE | 「正しい探索スケジュール」 | 無限探索極限での貪欲；Q学習収束に必要。 |
| ブートストラップ | 「ターゲットに現在の推定値を使う」 | TDとMCを区別するもの。バイアスの源だが分散を大幅に削減する。 |
| 最大化バイアス | 「Q学習の過大評価」 | ノイジーな推定値に対する`max`は上向きにバイアスがかかる；ダブルQ学習で修正。 |

## 参考資料

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 元の論文と収束証明。
- [Sutton & Barto (2018). 第6章——時間差分学習](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)、SARSA、Q学習、期待SARSA。
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — 最大化バイアスの修正。
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — 期待SARSAの動機。
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — SARSAを命名した論文（当時は「修正コネクショニストQ学習」と呼ばれた）。
- [Sutton & Barto (2018). 第7章——nステップブートストラップ](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)をTD(n)に一般化；Q学習から適格性トレース、そのちにPPOのGAEへの道筋。
