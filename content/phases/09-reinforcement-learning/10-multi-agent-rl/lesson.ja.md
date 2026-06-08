# マルチエージェントRL

> シングルエージェントRLは環境が定常であると仮定する。同じ世界に2つの学習エージェントを置くと、その仮定は崩れる：各エージェントはお互いの環境の一部であり、両者が変化し続ける。マルチエージェントRLは、マルコフ仮定がもはや成立しない時に学習を収束させるためのトリックの集合だ。


## 問題

部屋をナビゲートするロボットを学習させることはシングルエージェントRLの問題だ。サッカーチームはそうではない。AlphaStarとStarCraftの対戦相手もそうではない。入札エージェントのマーケットプレイスもそうではない。四方向の停止で交渉する2台の車もそうではない。多対多の現実世界の問題はそうではない。

あらゆるマルチエージェント設定では、どの1つのエージェントの視点からも、他のエージェントは環境の*一部*だ。彼らが学習し行動を変えるにつれて、環境は非定常になる。マルコフ性——「次の状態は現在の状態と自分の行動のみに依存する」——は次の状態が*他の*エージェントの選択にも依存し、彼らの方策は動く標的だから、違反される。

これは表形式の収束証明を破る（Q学習の保証は定常環境を仮定する）。素朴なディープRLも破る：エージェントがループで互いを追いかけ、安定した方策に収束しない。マルチエージェント固有の手法が必要だ：集中訓練/分散実行、反事実ベースライン、リーグプレイ、自己対戦。

2026年の応用：ロボット群、交通ルーティング、自動運転車フリート、市場シミュレータ、マルチエージェントLLMシステム（フェーズ16）、そして複数の知的なプレイヤーを持つゲーム。

## コンセプト

![4つのMARLレジーム：独立、集中クリティック、自己対戦、リーグ](../assets/marl.svg)

**形式：マルコフゲーム。** MDPの一般化：状態`S`、結合行動`a = (a_1, …, a_n)`、遷移`P(s' | s, a)`、エージェントごとの報酬`R_i(s, a, s')`。各エージェント`i`は自分の方策`π_i`のもとで自分のリターンを最大化する。報酬が同一の場合は**完全協調**。ゼロサムの場合は**競合**。混合の場合は**一般和**。

**コアの課題：**

- **非定常性。** エージェント`i`の視点からの`P(s' | s, a_i)`は変化している`π_{-i}`に依存する。
- **クレジット割り当て。** 共有された報酬があるとき、どのエージェントがそれを引き起こしたか？
- **探索の協調。** エージェントは同じ状態を冗長に探索するのではなく、補完的な戦略を探索する必要がある。
- **スケーラビリティ。** 結合行動空間は`n`について指数的に成長する。
- **部分観測。** 各エージェントは自分の観測しか見ない；グローバル状態は隠されている。

**4つの支配的なレジーム：**

**1. 独立Q学習/独立PPO（IQL、IPPO）。** 各エージェントが他のエージェントを環境の一部として扱い、自分のQまたは方策を学習する。シンプルで、時々機能する（特にリプレイバッファが平滑化エージェントモデルトリックとして機能する疎結合タスク）。理論的収束：なし。実践では：疎結合タスクには問題ない、密結合タスクには悪い。

**2. 集中訓練、分散実行（CTDE）。** 最も一般的な現代的パラダイム。各エージェントはデプロイ時に標準的な分散実行で——ローカル観測`o_i`を条件とする自分の*方策*`π_i`を持つ。*訓練*中は、集中クリティック`Q(s, a_1, …, a_n)`が完全なグローバル状態と結合行動を条件とする。例：
- **MADDPG**（Lowe et al. 2017）：エージェントごとの集中クリティックを持つDDPG。
- **COMA**（Foerster et al. 2017）：反事実ベースライン——「代わりに行動`a'`を取っていたら報酬はどうなっていたか？」と問う——私の貢献を分離する。
- **MAPPO** / **IPPO**（共有クリティック付き）（Yu et al. 2022）：集中価値関数を持つPPO。2026年の協調MARLで支配的。
- **QMIX**（Rashid et al. 2018）：価値分解——`Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`と単調混合。

**3. 自己対戦。** 同じエージェントの2つのコピーが互いに対戦する。対戦相手の方策は過去のスナップショットから私の方策だ。AlphaGo/AlphaZero/MuZero。OpenAI Five。ゼロサムゲームで最もよく機能；訓練信号が対称だ。

**4. リーグプレイ。** 一般和/競合環境への自己対戦の拡張：過去と現在の方策の集団を保持し、リーグから対戦相手をサンプリングし、彼らに対して訓練する。エクスプロイタ（現在のベストを倒すことに特化）とメインエクスプロイタ（エクスプロイタを倒すことに特化）を追加する。AlphaStar（StarCraft II）。ゲームが「じゃんけん」の戦略サイクルを認める時に必要。

**通信。** エージェントが互いに学習されたメッセージ`m_i`を送ることを許可する。協調設定で機能する。Foerster et al.（2016）はエンドツーエンドで訓練できる微分可能なエージェント間通信を示した。今日のLLMベースのマルチエージェントシステム（フェーズ16）は本質的に自然言語でコミュニケーションしている。

## 実装する

このレッスンは2つの協調エージェントを持つ6×6 GridWorldを使う。エージェントは対角のコーナーからスタートし、共有のゴールに達する必要がある。共有報酬：どちらかのエージェントがまだ移動中の間は1ステップごとに`-1`、両方が到着したら`+10`。`code/main.py`を参照。

### ステップ1：マルチエージェント環境

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # 2つのエージェント

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*結合*行動空間は`|A|² = 16`。グローバル状態は2つの位置だ。

### ステップ2：独立Q学習

各エージェントは結合状態をキーとする自分のQテーブルで実行する。各ステップで：両者がε-greedy行動を選び、結合遷移を収集し、各者が共有報酬で自分のQを更新する。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

このタスクでは報酬が密で整合しているから機能する。密結合タスク（例：一方のエージェントが他方を*待つ*必要がある場合）では失敗する。

### ステップ3：分解価値更新を持つ集中Q

結合行動に対する1つのQ `Q(s, a_1, a_2)`を使う。共有報酬から更新する。実行時は周辺化で分散化：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。指数的な結合行動空間と*正確な*グローバルビューを交換する。

### ステップ4：シンプルな自己対戦（競合2エージェント）

同じエージェント、2つの役割。エージェントAをエージェントBに対して訓練；`K`エピソード後、Aの重みをBにコピーする。対称訓練、一貫した進歩。AlphaZeroレシピのミニチュア版。

## 落とし穴

- **非定常リプレイ。** 独立エージェントを使ったエクスペリエンスリプレイは、古い遷移が今では時代遅れの対戦相手によって生成されたため、シングルエージェントより悪い。修正：新しさで再ラベルするか重み付けする。
- **クレジット割り当ての曖昧さ。** 長いエピソード後の共有報酬；どのエージェントが貢献したかを明確に言う方法がない。修正：反事実ベースライン（COMA）、またはエージェントごとの報酬形成。
- **方策ドリフト/チェイシング。** 各エージェントの最良応答が互いの更新ごとに変わる。修正：集中クリティック、遅い学習率、または1つずつ凍結。
- **協調による報酬ハッキング。** エージェントが設計者が予期しない協調的なエクスプロイトを見つける。オークションエージェントがゼロビッドに収束する。修正：慎重な報酬設計、行動制約。
- **探索の冗長性。** 両方のエージェントが同じ状態-行動ペアを探索する。修正：エージェントごとのエントロピー、または役割条件付け。
- **リーグサイクル。** 純粋な自己対戦は支配サイクルに詰まる可能性がある。修正：多様な対戦相手とのリーグプレイ。
- **サンプル爆発。** `n`エージェント×状態空間×結合行動。関数近似で近似；分解された行動空間（エージェントごとに1つの方策出力ヘッド）。

## 使ってみる

2026年のMARLアプリケーションマップ：

| ドメイン | メソッド | 注記 |
|---------|---------|-----|
| 協調ナビゲーション/マニピュレーション | MAPPO / QMIX | CTDE；共有クリティック+分散アクター。 |
| 2プレイヤーゲーム（チェス、囲碁、ポーカー） | MCTSを使った自己対戦（AlphaZero） | ゼロサム；対称訓練。 |
| 複雑なマルチプレイヤー（Dota、StarCraft） | リーグプレイ+模倣事前訓練 | OpenAI Five、AlphaStar。 |
| 自動運転車フリート | CTDEのMAPPO/PPOとアテンション | 部分観測；可変チームサイズ。 |
| オークション市場 | ゲーム理論的均衡+RL | `n` → ∞の時はミーンフィールドRL。 |
| LLMマルチエージェントシステム（フェーズ16） | 自然言語通信+役割条件付け | エージェント計画層でのRLループ。 |

2026年では、MARLの最大の成長領域はLLMベースだ：言語モデルエージェントの群れが交渉し、議論し、ソフトウェアを構築する。RLはトークンレベルではなく*軌跡レベル*の出力に対する好み最適化として現れる（フェーズ16・03）。

## 成果物を出す

`outputs/skill-marl-architect.md`として保存：

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## 演習

1. **易。** 2エージェントの協調GridWorldで独立Q学習を訓練する。平均リターンが0を超えるまでに何エピソードかかるか？結合学習曲線をプロットする。
2. **中。** 「協調」タスクを追加する：ゴールは両エージェントが同じターンにそれに踏み込んだ時のみ達成される。独立Qはまだ収束するか？何が壊れるか？
3. **難。** MAPPOスタイルの訓練のために集中クリティックを実装し、協調タスクでの独立PPOとの収束速度を比較する。

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|--------------|-----------|
| マルコフゲーム | 「マルチエージェントMDP」 | `(S, A_1, …, A_n, P, R_1, …, R_n)`；各エージェントが自分の報酬を持つ。 |
| CTDE | 「集中訓練、分散実行」 | 訓練時の結合クリティック；各エージェントの方策はローカル観測のみを使用。 |
| IPPO | 「独立PPO」 | 各エージェントが別々にPPOを実行する。シンプルなベースライン；しばしば過小評価される。 |
| MAPPO | 「マルチエージェントPPO」 | グローバル状態を条件とした集中価値関数を持つPPO。 |
| QMIX | 「単調価値分解」 | `Q_tot = f_monotone(Q_1, …, Q_n)`で分散型argmaxを可能にする。 |
| COMA | 「反事実マルチエージェント」 | アドバンテージ = 私のQ マイナス 私の行動を周辺化した期待Q。 |
| 自己対戦 | 「エージェント対過去の自分」 | 単一エージェント、2つの役割；ゼロサムゲームの標準。 |
| リーグプレイ | 「集団訓練」 | 過去の方策をキャッシュし、プールから対戦相手をサンプリングする；戦略サイクルを処理。 |

## 参考資料

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 集中クリティックを持つCTDE。
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — クレジット割り当てのための反事実ベースライン。
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — 単調性を持つ価値分解。
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPOはMARLに驚くほど強い。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — スケールでのリーグプレイ。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — ゼロサムゲームでの純粋な自己対戦。
- [Sutton & Barto (2018). 第15章——神経科学と第17章——フロンティア](http://incompleteideas.net/book/RLbook2020.pdf) — マルチエージェント設定とCTDEが対処するために設計された非定常性問題の教科書的処理を含む。
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — 収束結果を持つ協調、競合、混合MARLを網羅したサーベイ。
