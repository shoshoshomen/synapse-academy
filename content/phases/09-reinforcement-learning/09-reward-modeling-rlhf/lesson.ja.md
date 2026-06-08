# 報酬モデリングとRLHF

> 人間は「良いアシスタントの返答」のための報酬関数を書けないが、2つの返答を比較してより良いものを選ぶことはできる。その比較を報酬モデルに適合させ、それに対してRLで言語モデルを訓練する。Christiano 2017。InstructGPT 2022。GPT-3をChatGPTに変えたレシピ。2026年ではDPOに大部分が置き換えられつつある——しかしメンタルモデルは残る。


## 問題

次トークン予測の目的で言語モデルを訓練した。文法的な英語を書く。しかし嘘をつき、話が長引き、断ることを断る。より多くの事前訓練ではこれを修正できない——ウェブのテキストが問題であり、解決策ではない。

「インストラクションXに対してレスポンスAはレスポンスBより良い」と言う*スカラー報酬*が欲しい。その報酬関数を手書きすることは不可能だ。「有用性」はトークンに対する閉形式の表現ではない。しかし人間は2つの出力を比較して好みをマークできる。それは大規模に安価に収集できる。

RLHF（Christiano et al. 2017；Ouyang et al. 2022）は好みを報酬モデルに変換し、その報酬に対してPPOでLMを最適化する。3つのステップ：SFT→RM→PPO。ChatGPT、Claude、Gemini、2023〜2025年のすべての整合済みLLMを出荷したレシピだ。

2026年ではPPOステップはほとんどDPO（フェーズ10・08）に置き換えられる——整合チューニングには安くてほぼ同等に良いから。しかし*報酬モデル*の部分はすべてのBest-of-Nサンプラー、検証可能な報酬からのすべてのRLパイプライン、プロセス報酬モデルを使うすべての推論モデルの基盤として残る。RLHFを理解することで整合スタック全体を理解できる。

## コンセプト

![3段階RLHF：SFT、ペアワイズ好みによるRM訓練、KLペナルティ付きPPO](../assets/rlhf.svg)

**ステージ1：教師あり微調整（SFT）。** 事前訓練済みベースモデルから始める。ターゲット行動のデモンストレーション（インストラクション追従の返答、有用な返答等）をファインチューンする。結果：良い行動に*バイアスがかかった*モデル`π_SFT`だが、まだ無限の行動空間を持つ。

**ステージ2：報酬モデル訓練。**

- プロンプト`x`に対する返答のペア`(y_+, y_-)`を収集し、人間が「y_+がy_-より好まれる」とラベリングする。
- 報酬モデル`R_φ(x, y)`を`y_+`により高いスコアを割り当てるように訓練する。
- 損失：**Bradley-Terryペアワイズロジスティック**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σはシグモイド。報酬の差が好みの対数オッズを意味する。BTは1952年から標準（Bradley-Terry）で現代のRLHFでも支配的な選択だ。

- `R_φ`は通常SFTモデルからスカラーヘッドを上に付けて初期化する。同じトランスフォーマーバックボーン；1つの線形レイヤーが報酬を出力する。

**ステージ3：KLペナルティ付きRMに対するPPO。**

- 訓練可能な方策`π_θ`を`π_SFT`から初期化する。凍結した*参照*`π_ref = π_SFT`を保持する。
- 返答`y`の末尾での報酬：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KLペナルティは`π_θ`が`π_SFT`から任意にずれるのを防ぐ——ハードな信頼領域ではなく*正則化*だ。`β`は通常`0.01`-`0.05`。
- この報酬でPPO（レッスン08）を実行する。アドバンテージはトークンレベルの軌跡で計算されるが、RMは完全な返答のみにスコアをつける。

**KLが重要な理由？** これなしでは、PPOは喜んで報酬ハッキング戦略を見つける——RMはインディストリビューションの完了でのみ訓練された。アウトオブディストリビューションの返答はどの人間が書いたものより高くスコアされる可能性がある。KLは`π_θ`をRMが訓練されたマニフォールドの近くに保つ。これがRLHFで最も重要なノブだ。

**2026年の状況：**

- **DPO**（Rafailov 2023）：閉形式代数がステージ2+3を好みデータに対する単一の教師あり損失に折りたたむ。RMなし、PPOなし。整合ベンチマークでわずかのコンピュートで同等の品質。フェーズ10・08で扱う。
- **GRPO**（DeepSeek 2024〜2025）：クリティックの代わりにグループ相対ベースラインを持つPPO、人間が訓練したRMの代わりに*検証器*（コードが実行される/数学の答えが一致する）からの報酬。推論モデルで支配的。フェーズ9・12で扱う。
- **プロセス報酬モデル（PRM）：** 部分的な解（各推論ステップ）にスコアをつける；RLHFとGRPOの推論バリアントの両方で使用。
- **Constitutional AI / RLAIF：** 整合済みLLMを使って人間の代わりに好みを生成する。好みのバジェットをスケールする。

## 実装する

このレッスンは文字列として表現された小さな合成「プロンプト」と「返答」を使う。RMはトークンのバッグ表現に対する線形スコアラーだ。本物のLLMなし——パイプラインの*形状*が重要で、スケールではない。`code/main.py`を参照。

### ステップ1：合成好みデータ

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

本物のRLHFではこれは人間のラベラーに置き換えられる。形状——`(プロンプト、好まれる返答、拒否された返答)`——は同一だ。

### ステップ2：Bradley-Terry報酬モデル

線形スコア：`R(x, y) = w · bag(y)`。BTのペアワイズlog損失を最小化するように訓練：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

数百回の更新後、`w`は良いワードトークンに正の重みを割り当て、悪いワードには負の重みを割り当てる。

### ステップ3：RMの上のPPOライク方策

おもちゃの方策は語彙から1つのトークンを生成する。RMのもとでトークンをスコアし、`log π_θ(token | prompt)`を計算し、参照へのKLペナルティを追加し、クリップされたPPOサロゲートを適用する。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### ステップ4：KLを監視する

各更新で平均`KL(π_θ || π_ref)`を追跡する。`~5-10`を超えて忍び寄ると方策が`π_SFT`から大きくずれている——`β`の上昇または報酬ハッキングが始まっている。これが本物のRLHFのトップ診断だ。

### ステップ5：TRLを使った本番レシピ

おもちゃのパイプラインを理解したら、本物のライブラリユーザーが書く同じループがここにある。Hugging Faceの[TRL](https://huggingface.co/docs/trl)が参照実装——ステージ2に`RewardTrainer`、ステージ3に（KL-to-referenceが組み込まれた）`PPOTrainer`。

```python
# ステージ2：ペアワイズ好みからの報酬モデル
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# データセット行：{"prompt", "chosen", "rejected"}——Bradley-Terry形式
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# ステージ3：KLペナルティ付きでSFT参照に対するPPO
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # 凍結

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — 3つのPPO診断
```

ライブラリが3つのことをしてくれる。`adap_kl_ctrl=True`は適応的β スケジュールを実装する：観測されたKLが`target_kl`を超えるとβを2倍に；半分以下になるとβを半分に。参照モデルは慣例で凍結——`policy`とパラメータを誤って共有しないこと。そして価値ヘッドは方策と同じバックボーン上に存在する（`AutoModelForCausalLMWithValueHead`がスカラーMLPヘッドを付ける）、これがTRLが`policy/kl`と`value/loss`を別々に報告する理由だ。

## 落とし穴

- **過剰最適化/報酬ハッキング。** RMは不完全；`π_θ`は高スコアだが悪い完了を見つける。症状：報酬が際限なく上がるが人間の評価スコアが横ばいまたは下がる。修正：早期停止、`β`の引き上げ、RMの訓練データの拡大。
- **長さハッキング。** 有用な返答で訓練されたRMはしばしば暗黙的に長さを報酬化する。方策は返答を水増しするように学習する。対処：長さ正規化された報酬、または長さを意識したRMを持つRLAIF。
- **小さすぎるRM。** RMは少なくとも方策と同じくらい大きい必要がある。小さなRMは方策の出力を忠実にスコアできない。
- **KLチューニング。** βが低すぎる→ずれと報酬ハッキング。高すぎる→方策がほとんど変わらない。標準的なトリックは固定のKLをターゲットにする*適応的*βだ。
- **好みデータのノイズ。** 人間のラベルの約30%はノイジーまたは曖昧だ。同意フィルターされたデータでRMを訓練するか、BTにtemperatureを使うことでキャリブレーションする。
- **オフポリシーの問題。** PPOデータは最初のエポック後にやや オフポリシーだ。レッスン08と同様にクリップ率を監視する。

## 使ってみる

2026年のRLHFは階層化されている：

| 層 | ターゲット | メソッド |
|----|---------|---------|
| インストラクション追従、有用性、無害性 | 整合 | DPO（フェーズ10・08）がRLHF-PPOより好まれる。 |
| 推論の正確性（数学、コード） | 能力 | 検証器報酬を持つGRPO（フェーズ9・12）。 |
| 長ホライズンマルチステップタスク | エージェント | ステップに対するプロセス報酬モデルを持つPPO/GRPO。 |
| 安全/拒否行動 | 安全性 | 別の安全RMを持つRLHF-PPO、またはConstitutional AI。 |
| 推論時のBest-of-N | 高速整合 | デコード時にRMを使用；方策訓練は不要。 |
| 報酬蒸留 | 推論コンピュート | 凍結LMの上に小さな「報酬ヘッド」を訓練する。 |

RLHFは2022〜2024年の*ザ*メソッドだった。2026年では、本番の整合パイプラインはDPOファースト、PPOはRMを多用するステップや安全性に重要なステップのみだ。

## 成果物を出す

`outputs/skill-rlhf-architect.md`として保存：

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## 演習

1. **易。** `code/main.py`のBradley-Terry報酬モデルを500の合成好みペアで訓練する。保留した100ペアでペアワイズ精度を計測する。90%を超えるべき。
2. **中。** `β ∈ {0.0, 0.1, 1.0}`でおもちゃのPPO-RLHFループを実行する。各々について、更新にわたるRMスコア対KL-to-referenceをプロットする。どの実行が報酬ハックするか？
3. **難。** DPO（閉形式好み-尤度損失）を同じ好みデータで実装し、RLHF-PPOパイプラインと使用コンピュートおよび達成された最終RMスコアを比較する。

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|--------------|-----------|
| RLHF | 「整合RL」 | 3段階SFT+RM+PPOパイプライン（Christiano 2017、Ouyang 2022）。 |
| 報酬モデル（RM） | 「スコアリングネット」 | Bradley-Terryを通じてペアワイズ好みに適合された学習スカラー関数。 |
| Bradley-Terry | 「ペアワイズロジスティック損失」 | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；標準的なRM目的関数。 |
| KLペナルティ | 「参照の近くに留まる」 | 報酬の`β · KL(π_θ \|\| π_ref)`；報酬ハッキング防止の正則化。 |
| 報酬ハッキング | 「グッドハートの法則」 | 方策がRMの欠陥を悪用する；症状：報酬上昇、人間評価横ばい。 |
| RLAIF | 「AIラベリングされた好み」 | 人間の代わりに別のLMからラベルが来るRLHF。 |
| PRM | 「プロセス報酬モデル」 | 部分的な推論ステップにスコアをつける；推論パイプラインで使用。 |
| Constitutional AI | 「Anthropicのメソッド」 | 明示的なルールに導かれたAI生成の好み。 |

## 参考資料

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — RLHFを始めた論文。
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — ChatGPTの背後にあるレシピ。
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — 要約のための初期RLHF。
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO；2026年のRLHF後のデフォルト。
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIFと自己批評ループ。
- [Anthropic RLHF論文 (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — HH論文。
- [Hugging Face TRLライブラリ](https://huggingface.co/docs/trl) — 本番`RewardTrainer`と`PPOTrainer`。適応KLと価値ヘッドの詳細についてはトレーナーソースを読む。
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) by Lambert, Castricato, von Werra, Havrilla — 3段階パイプラインの標準的な解説（図付き）。
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — ライブラリ；`examples/`にはLlama、Mistral、Qwenのエンドツーエンドのスクリプトがある。
- [Sutton & Barto (2018). 第17.4章——報酬信号の設計](http://incompleteideas.net/book/RLbook2020.pdf) — 報酬仮説の視点；報酬ハッキングについて考えるための必須前提条件。
