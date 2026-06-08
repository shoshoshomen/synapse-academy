# GPT — Causal Language Modeling

> BERTは両側を見る。GPTは過去だけを見る。三角形マスクはモダンAIにおける最も重要なコード1行だ。


## 問題

言語モデルは1つの問いに答える：最初の`t-1`トークンが与えられたとき、トークン`t`に対する確率分布は何か？そのシグナルで学習すると——次トークン予測——任意のテキストを1トークンずつ生成できるモデルが得られる。

シーケンス全体を並列にエンドツーエンドで学習するには、各位置の予測が以前の位置のみに依存する必要がある。そうしないとモデルは答えを見ることでチートする。

Causal Maskがこれを行う。softmax前のアテンションスコアに追加される`-inf`値の上三角行列が1つだ。softmax後、それらの位置は0になる。各位置は自分自身と以前の位置にのみアテンションを向けられる。そして1回のフォワードパスでシーケンス全体に適用するため、1つのフォワードパスでN個の並列な次トークン予測が得られる。

GPT-1（2018年）、GPT-2（2019年）、GPT-3（2020年）、GPT-4（2023年）、GPT-5（2024年）、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi——これらはすべて同じコアループを持つデコーダのみのCausal Transformerだ。ただより大きく、より良いデータ、より良いRLHFだ。

## 概念

![Causal Maskが三角形のアテンション行列を作る](../assets/causal-attention.svg)

### マスク

長さ`N`のシーケンスが与えられたとき、`N × N`行列を構築する：

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

softmax前の生のアテンションスコアに`M`を加算する。`exp(-inf) = 0`なので、マスクされた位置の重みはゼロになる。アテンション行列の各行は以前の位置のみに対する確率分布だ。

実装コスト：1回の`torch.tril()`呼び出し。計算時間：ナノ秒。この分野への影響：すべて。

### 並列学習、逐次推論

学習：シーケンス全体`(N, d_model)`を一度フォワードし、N個のクロスエントロピー損失（位置ごとに1つ）を計算し、合計し、バックプロップ。シーケンスに沿って並列。これがGPTの学習スケールの理由だ——1回のGPUパスで1バッチ100万トークンを処理できる。

推論：トークンごとに生成する。`[t1, t2, t3]`を渡し、`t4`を得る。`[t1, t2, t3, t4]`を渡し、`t5`を得る。以下同様。KVキャッシュ（レッスン12）は`t1…tn`の隠れ状態を保存して各ステップで再計算しないようにする。しかし推論での逐次深さ = 出力長。それが自己回帰税であり、すべてのLLMのレイテンシボトルネックがデコーディングである理由だ。

### 損失——1つずれ

トークン`[t1, t2, t3, t4]`が与えられたとき：

- 入力：`[t1, t2, t3]`
- ターゲット：`[t2, t3, t4]`

すべての位置`i`に対して、`-log P(target_i | inputs[:i+1])`を計算する。合計する。これがシーケンス全体のクロスエントロピーだ。

聞いたことのあるすべてのTransformer LMはこの損失で学習する。事前学習、ファインチューニング、SFT——同じ損失、異なるデータ。

### デコーディング戦略

学習後、サンプリングの選択は人々が思っている以上に重要だ。

| 手法 | 何をするか | いつ使うか |
|--------|--------------|-------------|
| Greedy | 各ステップでargmax | 決定論的タスク、コード補完 |
| Temperature | ロジットをTで割ってサンプル | 創造的タスク、高いT = より多様性 |
| Top-k | 上位kトークンのみからサンプル | 低確率の末尾を排除 |
| Top-p（nucleus） | 累積確率≥pの最小セットからサンプル | 2020年以降のデフォルト；分布の形に適応 |
| Min-p | `p > min_p * max_p`のトークンを保持 | 2024年以降；top-pより長い末尾の除去が優れる |
| 投機的デコーディング | ドラフトモデルがNトークンを提案、大きなモデルが検証 | 同じ品質で2〜3倍のレイテンシ削減 |

2026年、min-p + 温度0.7はオープンウェイトモデルの合理的なデフォルトだ。投機的デコーディングはすべての本番推論スタックの必須要素だ。

### 「GPTレシピ」が機能した理由

1. **デコーダのみ。** エンコーダのオーバーヘッドなし。層ごとにアテンション + FFNの1パス。
2. **スケーリング。** 124M → 1.5B → 175B → 兆。Chinchillaスケーリング則（レッスン13）が計算の使い方を伝える。
3. **インコンテキスト学習。** 6B〜13Bあたりで出現した。モデルはファインチューニングなしでフューショット例に従える。
4. **RLHF。** 人間の好みに基づく事後学習が生の事前学習済みテキストをチャットアシスタントに変換した。
5. **Pre-norm + RoPE + SwiGLU。** スケールでの安定した学習。

コアアーキテクチャはGPT-2以来あまり変わっていない。興味深いことはすべてデータ、スケール、事後学習で起きた。

## 実装する

### ステップ1：Causal Mask

`code/main.py`を参照。1行：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

softmax前のアテンションスコアに加算する。それがメカニズム全体だ。

### ステップ2：2層GPT風モデル

2つのデコーダブロック（マスク済みSelf-Attention + FFN、クロスアテンションなし）を積み重ねる。トークン埋め込み、位置エンコーディング、アンエンベッディング（トークン埋め込み行列と結び付け——GPT-2以来の標準トリック）を追加する。

### ステップ3：次トークン予測、エンドツーエンド

20トークンのトイ語彙で、すべての位置でロジットを生成する。1つずれたターゲットに対してクロスエントロピー損失を計算する。勾配なし——これはフォワードパスのサニティチェックだ。

### ステップ4：サンプリング

Greedy、Temperature、Top-k、Top-p、Min-pを実装する。固定プロンプトで各々を実行して出力を比較する。サンプリング関数は10行だ。

## 使ってみる

PyTorch、2026年のイディオム：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

内部では、`generate()`はフォワードパスを実行し、最後の位置のロジットを取り、次のトークンをサンプルし、それを追加して繰り返す。すべての本番LLM推論スタック（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）は重い最適化——バッチ化プレフィル、継続的バッチング、KVキャッシュページング、投機的デコーディング——で同じループを実装する。

**GPT対BERT、各1行：** GPTは`P(x_t | x_{<t})`を予測する。BERTは`P(x_masked | x_unmasked)`を予測する。損失がモデルが生成できるかどうかを決める。

## 成果物を出す

`outputs/skill-sampling-tuner.md`を参照。このスキルは新しい生成タスクのサンプリングパラメータを選択し、決定論的デコーディングが必要な場合をフラグする。

## 演習

1. **易.** `code/main.py`を実行し、softmax後のCausal Attentionの行列が下三角形であることを確認せよ。スポットチェック：行3は列0〜3のみに重みを持つはずだ。
2. **中.** 幅4のビームサーチを実装せよ。10の短いプロンプトでビーム4とGreedyのパープレキシティを比較せよ。ビームは常に勝つか？（ヒント：通常翻訳では勝つが、オープンエンドのチャットでは勝たない。）
3. **難.** 投機的デコーディングを実装せよ：ドラフトとして2層モデルを、検証器として6層モデルを使う。長さ64の100回の補完での実時間のスピードアップを測定せよ。出力が検証器のGreedyと一致することを確認せよ。

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| Causal Mask | 「三角形」 | アテンションスコアに加算される上三角`-inf`行列。位置`i`が位置`≤ i`のみを見るようにする。 |
| 次トークン予測（Next-token prediction） | 「損失」 | すべての位置での真の次トークンに対するモデルの分布のクロスエントロピー。 |
| 自己回帰（Autoregressive） | 「一度に1つずつ生成」 | 出力を入力として戻す；学習中は並列性あり、生成中はなし。 |
| ロジット（Logits） | 「pre-softmaxスコア」 | softmax前のLMヘッドの生の出力；サンプリングはこれに対して行われる。 |
| Temperature | 「創造性ノブ」 | ロジットをTで割る；T→0 = Greedy、T→∞ = 一様。 |
| Top-p | 「Nucleusサンプリング」 | 分布を≥pに合計する最小セットに切り詰める；残りからサンプル。 |
| Min-p | 「Top-pより優れる」 | `p ≥ min_p × max_p`のトークンを保持；分布の鋭さにカットオフを適応させる。 |
| 投機的デコーディング（Speculative decoding） | 「ドラフト + 検証」 | 安価なモデルがNトークンを提案；大きなモデルが並列に検証する。 |
| Teacher forcing | 「学習トリック」 | 学習中、モデルの予測ではなく真の前のトークンを供給する。すべてのseq2seq LMの標準。 |

## 参考資料

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1。
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2。
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3とインコンテキスト学習。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 投機的デコーディング論文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 標準Causal LMリファレンスコード。
