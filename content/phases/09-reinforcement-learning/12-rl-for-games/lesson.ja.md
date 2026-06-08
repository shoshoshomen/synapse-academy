# ゲームのためのRL——AlphaZero、MuZero、そしてLLM推論の時代

> 1992年：TD-Gammonが純粋なTDでバックギャモンの人間チャンピオンを倒した。2016年：AlphaGoがイ・セドルを倒した。2017年：AlphaZeroがチェス、将棋、囲碁をゼロから制覇した。2024年：DeepSeek-R1はGRPOがPPOに置き換わって同じレシピが推論に機能することを証明した。ゲームはこのフェーズのすべてのブレークスルーを駆動するベンチマークだ。


## 問題

ゲームはRLが欲しいものすべてを持っている。明確な報酬（勝ち/負け）。無限のエピソード（自己対戦がリセット）。完全なシミュレーション（ゲームが*シミュレータそのもの*）。離散または小さな連続行動空間。対立的な堅牢性を強制するマルチエージェント構造。

そしてゲームはすべての主要なRLブレークスルーがテストされた方法だ。TD-Gammon（バックギャモン、1992）。Atari-DQN（2013）。AlphaGo（2016）。AlphaZero（2017）。OpenAI Five（Dota 2、2019）。AlphaStar（StarCraft II、2019）。MuZero（学習済みモデル、2019）。AlphaTensor（行列乗算、2022）。AlphaDev（ソートアルゴリズム、2023）。DeepSeek-R1（数学的推論、2025）——ゲームRLのテクニックがテキストで機能することの最新のデモンストレーション。

このカプストーンは3つのランドマークアーキテクチャ——AlphaZero、MuZero、GRPO——を**自己対戦+探索+方策改善**という単一の統一レンズを通じて調査する。各々が前のものを一般化する；GRPOは特にAlphaZeroのレシピをLLM推論に適用したもので、トークンを行動として、数学的検証を勝利信号として使う。

## コンセプト

![AlphaZero ↔ MuZero ↔ GRPO：同じループ、異なる環境](../assets/rl-games.svg)

**統一ループ。**

```
while True:
    trajectory = self_play(current_policy, search)     # 自分に対してゲームをプレイ
    policy_target = search.improved_policy(trajectory) # 探索が生の方策を改善
    policy_net.update(policy_target, value_target)     # 探索出力で教師あり
```

**AlphaZero（2017）。** Silver et al. 既知のルールを持つゲーム（チェス、将棋、囲碁）が与えられた場合：

- 方策-価値ネットワーク：1つのタワー`f_θ(s) → (p, v)`。`p`は合法手の事前確率。`v`は期待されるゲーム結果。
- モンテカルロ木探索（MCTS）：各手で可能な継続の木を展開する。`(p, v)`を事前確率+ブートストラップとして使用。UCB（PUCT）でノードを選択：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自己対戦：エージェント対エージェントでゲームをプレイ。手`t`で、MCTSの訪問分布`π_t`が方策訓練ターゲットになる。
- 損失：`L = (v - z)² - π · log p + c · ||θ||²`。`z`はゲーム結果（+1/0/-1）。

人間の知識ゼロ。手作りのヒューリスティックゼロ。チェス、将棋、囲碁をそれぞれ数千万の自己対戦ゲーム後に制覇した単一のレシピ。

**MuZero（2019）。** Schrittwieser et al. ルールが既知であることの要件を取り除く。

- 固定された環境の代わりに、*潜在ダイナミクスモデル*`(h, g, f)`を学習する：
  - `h(s)`：観測を潜在状態にエンコードする。
  - `g(s_latent, a)`：次の潜在状態+報酬を予測する。
  - `f(s_latent)`：方策事前確率+価値を予測する。
- MCTSは*学習された潜在空間*で実行される。同じ探索、同じ訓練ループ。
- 囲碁、チェス、将棋*およびAtari*で機能——1つのアルゴリズム、ルール知識なし。

**確率的MuZero（2022）。** 確率的ダイナミクスとチャンスノードを追加；バックギャモンクラスのゲームに拡張。

**Muesli、Gumbel MuZero（2022〜2024）。** サンプル効率と決定的探索の改善。

**GRPO（2024〜2025）。** DeepSeek-R1レシピ。同じAlphaZero形状のループ、言語モデル推論に適用：

- 「ゲーム」：数学/コーディング/推論問題に答える。「勝ち」=検証器（テストケースが通る、数値的な答えが一致する）が1を返す。
- 方策：LLM。行動：トークン。状態：プロンプト+これまでの返答。
- クリティックなし（PPOスタイルのV_φ）。代わりに、各プロンプトに対して方策から`G`個の完了をサンプリングする。各々の報酬を計算する。**グループ相対アドバンテージ**`A_i = (r_i - mean_r) / std_r`をREINFORCEスタイルの更新のための信号として使用する。
- ドリフトを防ぐための参照方策へのKLペナルティ（RLHFと同様）。
- 完全な損失：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

報酬モデルなし、クリティックなし、MCTSなし。グループ相対ベースラインが3つすべてを置き換える。推論ベンチマークでのコンピュートの何分の1かでPPO-RLHFと同等またはそれを上回る品質。

**R1レシピの全貌。** DeepSeek-R1（DeepSeek 2025）は1つの論文に2つのモデルが含まれる：

- **R1-Zero。** DeepSeek-V3ベースモデルから始める。SFTなし。2つの報酬コンポーネントでGRPOを直接適用する：*精度報酬*（ルールベース——最終答えが正しい数値に解析されるか/コードがユニットテストを通るか）と*フォーマット報酬*（完了が`<think>…</think>`タグで思考の連鎖をラップするか）。何千ものステップで、平均応答長は約100から約10,000トークンに成長し、数学ベンチマークスコアはo1-previewに近いレベルまで上昇する。モデルはゼロから推論することを学習する。欠点：思考の連鎖がしばしば読みにくく、言語を混ぜ、スタイルの磨きが欠けている。
- **R1。** R1-Zeroの読みやすさの問題を4段階のパイプラインで修正する：
  1. **コールドスタートSFT。** クリーンなフォーマットを持つ長CoTデモンストレーション数千件を収集する。それらについてベースモデルを教師あり微調整する。読みやすい出発点を与える。
  2. **推論指向のGRPO。** 精度+フォーマット報酬に加えてコードスイッチを防ぐための*言語一貫性*報酬でGRPOを適用する。
  3. **リジェクションサンプリング+SFTラウンド2。** RLチェックポイントから約60万の推論軌跡をサンプリングし、正しい最終答えと読みやすいCoTを持つものだけを残し、約20万の非推論SFT例（執筆、QA、自己認識）と組み合わせる。再度ベースをファインチューニングする。
  4. **フルスペクトルGRPO。** 推論（ルールベース報酬）と一般的整合（有用性/無害性嗜好ベース報酬）の両方をカバーするもう1回のRLラウンド。

結果はオープンウェイトでAIMEとMATH-500でo1に匹敵し、蒸留に十分小さい。同じ論文では、R1の推論トレースでのSFTによって6つの蒸留された密モデル（Qwen-1.5BからLlama-70B）も公開している——学生でのRLなし。強いRLの教師からの蒸留は一貫して学生のスケールでのRLよりゼロから優れる。

**推論モデルへのGRPOとPPOの違い。** DeepSeekMath論文（2024年2月）の3つの理由：(1) 訓練する価値ネットワークなし、メモリを半分に；(2) グループベースラインが推論タスクの生成する疎な軌跡末尾報酬を自然に処理する；(3) プロンプトごとの正規化がアドバンテージを異なる難易度の問題間で比較可能にする、PPOの単一クリティックにはできないこと。

**探索なし対探索ベース。** ゲームは分岐した：

- *長いホライズンを持つ完全情報ゲーム*（囲碁、チェス）：依然として探索ベース。AlphaZero/MuZeroが支配的。
- *LLM推論*：本番ではまだMCTSなし；完全なロールアウトに対するGRPO、推論コンピュートにBest-of-N。プロセス報酬モデル（PRM）がステップレベルの探索の追加を示唆している。

## 実装する

`code/main.py`のコードは複数のサンプルグループを持つバンディットで**GRPOをミニチュアで**実装する。アルゴリズムはLLMと同じ；方策と環境だけがよりシンプルだ。*損失*と*グループ相対アドバンテージ*を教える——これが2025年のイノベーションだ。

### ステップ1：小さな検証器環境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

本物のGRPOでは検証器はユニットテストを実行するか数学的等価性をチェックする。

### ステップ2：方策：プロンプトごとのK個の答えトークンに対するソフトマックス

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

プロンプトを条件としたLLMの最終層出力に相当する。

### ステップ3：グループサンプリングとグループ相対アドバンテージ

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KLペナルティ：thetaを参照に引き寄せる
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

グループ相対アドバンテージが2024年のDeepSeekのトリックだ。クリティックは不要。「ベースライン」がグループ平均であり、正規化がグループ標準偏差を使う。

### ステップ4：REINFORCEベースラインとの比較（価値なし）

同じセットアップ、同じコンピュート、プレーンなREINFORCE。GRPOはより速く、より安定して収束する。

### ステップ5：エントロピーとKLを観察する

RLHFと同じ診断：参照へのKLの平均、方策エントロピー、時間経過での報酬。これらが安定したら訓練は終了だ。

## 落とし穴

- **検証器ゲーミングによる報酬ハッキング。** GRPOはRLHFのリスクを継承する：検証器が間違っているかエクスプロイトできる場合、LLMはエクスプロイトを見つける。堅牢な検証器（複数のテストケース、形式的証明）が重要だ。
- **グループサイズが小さすぎる。** グループベースラインの分散は`1/√G`で変化する。`G = 4`以下では、アドバンテージ信号がノイジー；標準的な選択は`G = 8`から`64`。
- **長さバイアス。** 異なる長さのLLM完了は異なる対数確率を持つ。トークン数で正規化するか、シーケンスレベルの対数確率を使うか、最大長に切り詰める。
- **純粋な自己対戦サイクル。** AlphaZeroスタイルの訓練は一般和ゲームで支配ループに詰まる可能性がある。多様な対戦相手プール（リーグプレイ、レッスン10）で軽減される。
- **探索-方策不一致。** AlphaZeroは方策が探索の出力を模倣するように訓練する。方策ネットが探索の分布を表現するには小さすぎる場合、訓練が行き詰まる。
- **コンピュートの最低ライン。** MuZero/AlphaZeroは大規模なコンピュートが必要。単一のアブレーションが数百GPU時間になることが多い。学習用のミニチュアデモ（例：Connect FourのAlphaZero）が存在する。
- **検証器のカバレッジ。** バグのあるソリューションに通るユニットテストはバグを強化する。エッジケースをキャッチする検証器を設計する。

## 使ってみる

2026年のゲームRLランドスケープ、ドメイン別：

| ドメイン | 支配的な手法 |
|---------|------------|
| 2プレイヤーゼロサムボードゲーム（囲碁、チェス、将棋） | AlphaZero / MuZero / KataGo |
| 不完全情報カードゲーム（ポーカー） | CFR + ディープラーニング（DeepStack、Libratus、Pluribus） |
| Atari / ピクセルゲーム | Muesli / MuZero / IMPALA-PPO |
| 大規模マルチプレイヤー戦略（Dota、StarCraft） | PPO + 自己対戦 + リーグ（OpenAI Five、AlphaStar） |
| LLM数学/コード推論 | GRPO（DeepSeek-R1、Qwen-RL、オープン複製） |
| LLM整合 | DPO / RLHF-PPO（GRPOではない；検証器は検証可能ではなく嗜好） |
| ロボティクス | PPO + DR（ゲームRLではないが同じ方策勾配ツールを使う） |
| 組み合わせ問題 | AlphaZeroバリアント（AlphaTensor、AlphaDev） |

*レシピ*——自己対戦、探索拡張改善、方策蒸留——はテキスト、ピクセル、物理的制御にまたがる。GRPOは最も新しいインスタンス；より多くが来ている。

## 成果物を出す

`outputs/skill-game-rl-designer.md`として保存：

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## 演習

1. **易。** `code/main.py`のGRPOバンディットを実装する。2プロンプト×各4答えトークンで訓練する。`G=8`で1,000回未満の更新で収束する。
2. **中。** PPO（クリップ）とバニラREINFORCEをプラグインする。同じバンディットでのGRPOとのサンプル効率と報酬分散を比較する。
3. **難。** 長さ2の「推論チェーン」に拡張する：エージェントが2つのトークンを出力し、検証器がペアに報酬を与える。GRPOが2ステップのシーケンスにわたってクレジット割り当てをどのように処理するかを計測する。（ヒント：*完全シーケンス*ごとにグループアドバンテージを計算し、両方のトークン位置に伝播する。）

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|--------------|-----------|
| MCTS | 「学習ネットを使った木探索」 | モンテカルロ木探索；学習された`(p, v)`事前確率を持つUCB1/PUCT選択。 |
| AlphaZero | 「自己対戦+MCTS」 | MCTSの訪問とゲーム結果に一致するように訓練された方策-価値ネット。 |
| MuZero | 「学習モデルのAlphaZero」 | 学習されたダイナミクスを介して潜在空間で同じループ。 |
| GRPO | 「クリティックなしのPPO」 | グループ相対方策最適化；グループ平均ベースライン+KLを持つREINFORCE。 |
| PUCT | 「AlphaZeroのUCB」 | `Q + c · p · √N / (1 + N_a)`——価値推定と事前確率のバランスを取る。 |
| 自己対戦 | 「エージェント対過去の自分」 | ゼロサムの標準；対称訓練信号。 |
| リーグプレイ | 「集団ベースの自己対戦」 | 過去+現在+エクスプロイタが対戦相手としてサンプリングされる。 |
| 検証器報酬 | 「検証可能なRL」 | 報酬が決定的チェッカー（テストが通る、答えが一致する）から来る。 |
| プロセス報酬 | 「PRM」 | 最終答えだけでなく各推論ステップにスコアをつける。 |

## 参考資料

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)。
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404)。
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4)。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z)。
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — GRPOとグループ相対ベースラインを紹介した論文。
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — 完全な4段階R1レシピとR1-Zeroアブレーション。
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — スケールでのCFR+ディープラーニング。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — すべてを始めた論文。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — カスタム報酬関数でGRPOを適用するための本番リファレンス。
- [Qwen Team (2024). Qwen2.5-Math — GRPO複製](https://github.com/QwenLM/Qwen2.5-Math) — 複数のスケールでのR1レシピのオープン複製。
- [Sutton & Barto (2018). 第17章——強化学習のフロンティア](http://incompleteideas.net/book/RLbook2020.pdf) — R1がLLMスケールでインスタンス化する自己対戦、探索、「設計された報酬」の教科書的フレーミング。
