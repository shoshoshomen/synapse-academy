# トランスフォーマー以前のテキスト生成 — N-gram言語モデル

> ある単語が予測しにくければ、モデルの精度が低い。パープレキシティは「驚き」を数値化する。スムージングはその値を有限に保つ。


## 問題設定

トランスフォーマーの登場以前、RNNの登場以前、単語埋め込みの登場以前、言語モデルは直前の`n-1`単語に続く単語を、その出現頻度を数えることで予測していた。「the cat」→「sat」が47回、「the cat」→「jumped」が12回、「the cat」→「refrigerator」が0回、のように。これを正規化して確率分布を得る。

これがN-gram言語モデルである。1980年代から2015年まで、あらゆる音声認識、スペルチェック、フレーズベースの機械翻訳システムで使われてきた。安価なオンデバイス言語モデリングが必要な場合は今でも使われている。

未知のN-gramに対してどう対処するかが重要な問題となる。単純なカウントベースのモデルは、学習データに存在しないものに確率ゼロを割り当てる。これは壊滅的な問題となる。文は長く、ほぼすべての長い文には少なくとも1つの未知のシーケンスが含まれるからだ。50年のスムージング研究がこれを解決した。その成果がKneser-Neyスムージングであり、現代のディープラーニングはその実験的な伝統を継承している。

## 概念

![N-gramモデル: カウント、スムージング、生成](../assets/ngram.svg)

**N-gram確率:** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。`n`を固定して（一般的にtrigramは3、4-gramは4）、カウントから計算する：

```text
P(w | context) = count(context, w) / count(context)
```

**ゼロカウント問題。** 学習データに存在しないN-gramには確率ゼロが割り当てられる。2007年のBrownコーパスを対象とした研究では、4-gramモデルでも保留した4-gramの30%が学習データに存在しないことが判明した。スムージングなしでは実際のテキストを評価することができない。

**スムージングの手法（洗練度の順）：**

1. **ラプラス（add-one）。** すべてのカウントに1を加える。シンプルだが、稀なイベントに対しては性能が悪い。
2. **Good-Turing。** 出現頻度の頻度に基づき、高頻度イベントから未観測イベントへ確率質量を再分配する。
3. **補間（Interpolation）。** N-gram、(n-1)-gramなどの推定値を調整可能な重みで組み合わせる。
4. **バックオフ（Backoff）。** N-gramのカウントがゼロなら(n-1)-gramにフォールバックする。Katzバックオフがこれを定式化している。
5. **絶対割引（Absolute discounting）。** 全カウントから固定の割引値`D`を引き、未観測イベントに再分配する。
6. **Kneser-Ney。** 絶対割引に加え、低次モデルのより巧妙な選択として、生の出現頻度ではなく*継続確率*（ある単語が出現するコンテキスト数）を使用する。

Kneser-Neyの洞察は深い。「San Francisco」は一般的なbigram。単語「Francisco」はほとんど「San」の後に現れる。単純な絶対割引では「Francisco」のunigramの確率を高く見積もる（カウントが高いため）。Kneser-Neyは「Francisco」が1つのコンテキストにしか現れないことに気づき、その継続確率を低く設定する。結果として、「Francisco」で終わる未知のbigramに対して適切に低い確率が割り当てられる。

**評価：パープレキシティ。** 保留テストセットにおける単語あたりの平均負の対数尤度の指数。低いほど良い。パープレキシティが100とは、100単語の中から一様に選択するのと同程度に混乱していることを意味する。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

## 実装する

### ステップ1: trigramカウント

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

入力はトークン化された文のリスト。出力はN-gramカウントとコンテキストカウント。`<s>`と`</s>`は文の境界を示す。

### ステップ2: ラプラススムージング

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

全カウントに1を加える。スムージングはされるが、未観測イベントへの確率質量過剰割当により、稀な既知イベントの精度も低下する。

### ステップ3: Kneser-Ney（bigram、補間）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

3つの構成要素がある。`continuation_prob`は「この単語はいくつの異なるコンテキストで現れるか？」を捉える（Kneser-Neyのイノベーション）。`lambda_prev`は割引によって解放された確率質量であり、バックオフの重みに使用される。最終的な確率は、割引された主要項と重み付けされた継続項の和になる。

### ステップ4: サンプリングによるテキスト生成

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

確率に比例したサンプリング。シードごとに異なる出力を生成する。ビームサーチ風の出力を得るには、各ステップで確率最大のものを選択（greedy）し、温度パラメータで若干のランダム性を加える。

### ステップ5: パープレキシティ

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

低いほど良い。Brownコーパスでは、チューニング済みの4-gram KNモデルがパープレキシティ約140を達成する。同じテストセットでトランスフォーマーLMは15〜30を達成する。この差は約10倍。この差がフィールドが進化した理由である。

## 使ってみる

- **古典的なNLP教材。** スムージング、MLE、パープレキシティに触れる最良の機会。
- **KenLM。** 本番向けN-gramライブラリ。低レイテンシが重要な音声・MTシステムのリスコアラとして使用される。
- **オンデバイス自動補完。** キーボードのtrigramモデル。今でも使われている。
- **ベースライン。** ニューラルLMを「良い」と判断する前に、必ずN-gram LMのパープレキシティを計算すること。トランスフォーマーがKNを大幅に上回らない場合は、何か問題がある。

## 成果物を出す

`outputs/prompt-lm-baseline.md`として保存：

```markdown
---
name: lm-baseline
description: ニューラルLMを学習する前に再現可能なN-gram言語モデルベースラインを構築する。
phase: 5
lesson: 16
---

コーパスと目的（次単語予測、リスコアリング、パープレキシティベースライン）を与えられた場合、以下を出力する：

1. N-gramの次数。一般的な英語ではtrigram、コーパスが大きければ4-gram、音声リスコアリングには5-gram。
2. スムージング。デフォルトはmodified Kneser-Ney；ラプラスは教育目的のみ。
3. ライブラリ。本番向けは`kenlm`、教育目的は`nltk.lm`、自作は学習目的のみ。
4. 評価。学習セットとテストセット間で一貫したトークン化による保留パープレキシティ。

比較するシステム間でトークン化が異なる場合のパープレキシティの報告は拒否すること — パープレキシティの数値は同一のトークン化の下でのみ比較可能。テストセットのOOV率も報告すること；KNはUNKトークンを学習時に予約しない限りOOVを適切に処理できない。
```

## 演習

1. **易。** 1,000文のシェイクスピアコーパスでtrigram LMを学習する。20文を生成する。局所的には妥当だが全体的に支離滅裂になるはずである。これが定番のデモである。
2. **中。** 保留したシェイクスピアの分割データでKNモデルのパープレキシティを実装する。ラプラスと比較する。KNがパープレキシティを30〜50%低下させるはずである。
3. **難。** Trigramスペルチェッカーを構築する：スペルミスのある単語とそのコンテキストを与えられた場合、修正候補を生成しLMのコンテキスト確率でランク付けする。Birkbeckスペリングコーパス（公開）で評価する。

## キーワード

| 用語 | 一般的な説明 | 実際の意味 |
|------|------------|----------|
| N-gram | 単語シーケンス | `n`個の連続したトークンのシーケンス。 |
| スムージング | ゼロ回避 | 未観測イベントが非ゼロの確率を持つよう確率質量を再分配すること。 |
| パープレキシティ | LM品質指標 | 保留データでの`exp(-平均対数確率)`。低いほど良い。 |
| バックオフ | 短いコンテキストへのフォールバック | trigramのカウントがゼロなら、bigramを使用する。Katzバックオフがこれを定式化している。 |
| Kneser-Ney | N-gramの最良スムージング | 絶対割引＋低次モデルの継続確率。 |
| 継続確率 | KN固有 | 生のカウントではなく、単語が出現するコンテキスト数で重み付けされた`P(w)`。 |

## 参考資料

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — N-gram LMとスムージングの定番解説。
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — Kneser-Neyが最良のN-gramスムーザーであると確立した論文。
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — オリジナルのKN論文。
- [KenLM](https://kheafield.com/code/kenlm/) — 高速な本番向けN-gram LM。2026年でもレイテンシ重視アプリケーションで使用されている。
