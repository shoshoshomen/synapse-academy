# ディープQネットワーク（DQN）

> 2013年：Mnihは生のピクセルで1つのQ学習ネットワークを訓練し、7つのAtariゲームすべてで古典的なRLエージェントを上回った。2015年：49ゲームに拡張し、Natureに掲載され、ディープRL時代が始まった。DQNはQ学習プラス関数近似を安定させる3つのトリックだ。


## 問題

表形式のQ学習はすべての（状態、行動）ペアに対して別々のQ値が必要だ。チェス盤は約10⁴³の状態を持つ。Atariフレームは210×160×3=100,800の特徴量。表形式RLは何千という状態で死滅し、数十億は言うまでもない。

修正は後付けで明らかだ：Qテーブルをニューラルネットワーク`Q(s, a; θ)`で置き換える。しかし後付けで明らかなことが数十年かかった。Q学習を使った素朴な関数近似は「致命的三つ組」のもとで発散する——関数近似+ブートストラップ+オフポリシー学習。Mnih et al.（2013、2015）は学習を安定させる3つのエンジニアリングトリックを特定した：

1. **エクスペリエンスリプレイ**が遷移を非相関化する。
2. **ターゲットネットワーク**がブートストラップターゲットを凍結する。
3. **報酬クリッピング**が勾配の大きさを正規化する。

AtariでのDQNは、単一のアーキテクチャと単一のハイパーパラメータセットで何十ものコントロール問題を生のピクセルから解いた最初の例だ。それ以来構築されたすべての「ディープRL」——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——はこの3トリックベースの上に積み重ねられている。

## コンセプト

![DQN訓練ループ：環境、リプレイバッファ、オンラインネット、ターゲットネット、ベルマンTD損失](../assets/dqn.svg)

**目的。** DQNはニューラルQ関数に対して1ステップTD損失を最小化する：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ`=オンラインネットワーク、各ステップで勾配降下で更新される。`θ^-`=ターゲットネットワーク、定期的に`θ`からコピーされる（約10,000ステップごと）。`D`=過去の遷移のリプレイバッファ。

**3つのトリック、重要順に：**

**エクスペリエンスリプレイ。** 約10⁶遷移のリングバッファ。各訓練ステップは均一にランダムにミニバッチをサンプリングする。これは時間的相関を破り（連続フレームはほぼ同一）、ネットワークが希少な報酬遷移から何度も学習できるようにし、連続する勾配更新を非相関化する。これなしでは、ニューラルネットを使ったオンポリシーTDはAtariで発散する。

**ターゲットネットワーク。** ベルマン方程式の両辺に同じネットワーク`Q(·; θ)`を使うと、ターゲットが毎更新で動く——「自分の尻尾を追いかける」。修正：重みが凍結された2番目のネットワーク`Q(·; θ^-)`を保持する。`C`ステップごとに`θ → θ^-`をコピーする。これで数千の勾配ステップにわたって回帰ターゲットが安定する。ソフト更新`θ^- ← τ θ + (1-τ) θ^-`（DDPG、SACで使用）はより滑らかなバリアントだ。

**報酬クリッピング。** Atariの報酬の大きさは1から1000+まで変わる。`{-1, 0, +1}`にクリッピングすることで、どの1ゲームも勾配を支配しない。報酬の大きさが重要な時は誤り；Atariでは符号だけが重要なので問題ない。

**ダブルDQN。** Hasselt（2016）が最大化バイアスを修正：オンラインネットで行動を*選択*し、ターゲットネットで*評価*する。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

ドロップイン置換、一貫して優れる。デフォルトで使う。

**その他の改善（Rainbow、2017年）：** 優先リプレイ（TD誤差の大きさに比例した遷移サンプリング）、dueling アーキテクチャ（別々の`V(s)`とアドバンテージヘッド）、ノイジーネットワーク（学習された探索）、nステップリターン、分布型Q（C51/QR-DQN）、マルチステップブートストラップ。各々がいくつかのパーセントを追加；ゲインはほぼ加法的だ。

## 実装する

ここのコードはstdlibのみのnumpyなし——小さな連続GridWorldで手巻きの単一隠れ層MLPを使い、各訓練ステップがマイクロ秒で実行される。アルゴリズムはスケールでのAtari DQNと同一だ。

### ステップ1：リプレイバッファ

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atariでは約50,000の容量；おもちゃ環境では5,000で十分。

### ステップ2：小さなQネットワーク（手動MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

フォワードパス：線形→ReLU→線形。これがネット全体だ。

### ステップ3：DQN更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

形状はレッスン04のQ学習と2つの違いがある：(a) テーブルのインデックスの代わりに微分可能な`Q(·; θ)`を通じてバックプロップする、(b) ターゲットは`Q(·; θ^-)`を使う。

### ステップ4：外側ループ

各エピソードで、`Q(·; θ)`上でε-greedyに行動し、遷移をバッファにプッシュし、ミニバッチをサンプリングし、勾配ステップを取り、定期的に`θ^- ← θ`を同期する。パターン：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

16次元のワンホット状態を持つ小さなGridWorldでは、エージェントは約500エピソードでほぼ最適な方策を学習する。Atariでは、これを2億フレームにスケールしてCNNの特徴抽出器を追加する。

## 落とし穴

- **致命的三つ組。** 関数近似+オフポリシー+ブートストラップは発散する可能性がある。DQNはターゲットネット+リプレイで軽減する；どちらも外さない。
- **探索。** εは通常訓練の最初の約10%で1.0から0.01まで減衰する必要がある。十分な初期探索なしにQネットは局所的な盆地に収束する。
- **過大評価。** ノイジーなQ上の`max`は上向きにバイアスがかかる。本番では常にダブルDQNを使う。
- **報酬スケール。** 報酬をクリップするか正規化する；勾配の大きさは報酬の大きさに比例する。
- **リプレイバッファのコールドスタート。** バッファに数千の遷移が入るまで訓練しない。約20サンプルでの初期の勾配はオーバーフィットする。
- **ターゲット同期頻度。** 頻繁すぎる≈ターゲットネットなし；頻繁でなさすぎる≈古いターゲット。Atari DQNは10,000環境ステップを使う。経験則：訓練ホライズンの約1/100ごとに同期。
- **観測の前処理。** Atari DQNは4フレームを積み重ねて状態をマルコフにする。速度情報を持つ環境はフレームスタックまたはリカレント状態が必要。

## 使ってみる

2026年では、DQNはほとんどSOTAではないが、参照オフポリシーアルゴリズムとして残る：

| タスク | 選択手法 | DQNでない理由 |
|--------|---------|-------------|
| 離散行動Atariライク | Rainbow DQN または Muesli | 同じフレームワーク、より多くのトリック。 |
| 連続制御 | SAC / TD3（フェーズ9・07） | DQNには方策ネットがない。 |
| オンポリシー/高スループット | PPO（フェーズ9・08） | リプレイバッファなし；スケールが容易。 |
| オフラインRL | CQL / IQL / Decision Transformer | 保守的なQターゲット、ブートストラップの爆発なし。 |
| 大規模離散行動空間（レコメンダー） | 行動埋め込みを使ったDQN、またはIMPALA | 問題ない；装飾が重要。 |
| LLM RL | PPO / GRPO | シーケンスレベル、ステップレベルではない；異なる損失。 |

教訓はまだ旅する。リプレイとターゲットネットワークはSAC、TD3、DDPG、SAC-X、AlphaZeroの自己対戦バッファ、すべてのオフラインRLメソッドに現れる。報酬クリッピングはPPOのアドバンテージ正規化として生き残る。アーキテクチャはブループリントだ。

## 成果物を出す

`outputs/skill-dqn-trainer.md`として保存：

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## 演習

1. **易。** `code/main.py`を実行する。エピソードごとのリターン曲線をプロットする。実行平均が-10を超えるまでに何エピソードかかるか？
2. **中。** ターゲットネットワークを無効にする（ベルマンターゲットの両辺にオンラインネットを使う）。訓練の不安定性を計測する——リターンは振動するか発散するか？
3. **難。** ダブルDQNを追加：オンラインネットで`argmax a'`を選び、ターゲットネットで評価する。ノイジー報酬GridWorldで1,000エピソード後のダブルDQNあり/なしで`Q(s_0, best_a)`対真の`V*(s_0)`のバイアスを比較する。

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|--------------|-----------|
| DQN | 「ディープQ学習」 | ニューラルQ関数、リプレイバッファ、ターゲットネットワークを持つQ学習。 |
| エクスペリエンスリプレイ | 「シャッフルされた遷移」 | 各勾配ステップで均一にサンプリングされたリングバッファ；データを非相関化する。 |
| ターゲットネットワーク | 「凍結ブートストラップ」 | ベルマンターゲットで使用されるQの定期的なコピー；訓練を安定させる。 |
| 致命的三つ組 | 「RLが発散する理由」 | 関数近似+ブートストラップ+オフポリシー=収束保証なし。 |
| ダブルDQN | 「最大化バイアスの修正」 | オンラインネットが行動を選択し、ターゲットネットが評価する。 |
| Dueling DQN | 「VとAのヘッド」 | Q = V + A - mean(A)と分解；同じ出力、より良い勾配フロー。 |
| Rainbow | 「すべてのトリック」 | DDQN+PER+dueling+nステップ+ノイジー+分布型を1つに。 |
| PER | 「優先リプレイ」 | TD誤差の大きさに比例して遷移をサンプリングする。 |

## 参考資料

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — ディープRLを始めた2013年NeurIPSワークショップ論文。
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Nature論文、49ゲームDQN。
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN。
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — Dueling DQN。
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — トリック積み重ね論文。
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — 明確な現代的解説。
- [Sutton & Barto (2018). 第9章——近似を使ったオンポリシー予測](http://incompleteideas.net/book/RLbook2020.pdf) — DQNのターゲットネットワークとリプレイバッファが対処するために設計された「致命的三つ組」（関数近似+ブートストラップ+オフポリシー）の教科書的処理。
- [CleanRL DQN実装](https://docs.cleanrl.dev/rl-algorithms/dqn/) — アブレーション研究で使用される参照単一ファイルDQN；このレッスンのゼロから始めるバージョンと並んで読む価値がある。
