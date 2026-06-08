# テキスト要約

> 抽出型システムはドキュメントが何を言ったかを伝える。抽象型システムは著者が何を意味したかを伝える。異なるタスク、異なる落とし穴。


## 問題の背景

2,000ワードのニュース記事がフィードに届く。120ワードでそれを捉える必要がある。記事から3つの最も重要な文を選ぶか（抽出型）、自分の言葉でコンテンツを書き直すか（抽象型）。両方とも要約と呼ばれる。それらは完全に異なる問題だ。

抽出型要約はランキング問題だ。各文にスコアを付けて上位`k`個を返す。出力は逐語的に持ち上げられるため常に文法的だ。リスクは記事全体に分散したコンテンツを見逃すことだ。

抽象型要約は生成問題だ。Transformerが入力を条件に新しいテキストを生成する。出力は流暢で圧縮的だが、ソースになかった事実を幻覚する可能性がある。リスクは確信を持った作り話だ。

このレッスンでは両方を構築し、それぞれが持つ失敗モードを示す。

## 概念

**抽出型。** 記事をノードが文でエッジが類似度のグラフとして扱う。グラフに対してPageRank（または類似のもの）を実行して、他すべてとどれだけ連結しているかで文にスコアを付ける。最高スコアの文が要約だ。標準的な実装は**TextRank**（Mihalcea and Tarau、2004）だ。

**抽象型。** Transformerエンコーダー-デコーダー（BART、T5、Pegasus）を文書-要約ペアでファインチューニングする。推論時、モデルはドキュメントを読み、クロスAttentionを使ってトークンごとに要約を生成する。Pegasusは特に、ギャップセンテンス事前訓練目標を使用していて、多くのファインチューニングなしで要約に優れている。

**ROUGE**（Recall-Oriented Understudy for Gisting Evaluation）による評価。ROUGE-1とROUGE-2はユニグラムとバイグラムの重複をスコアリングする。ROUGE-Lは最長共通部分列をスコアリングする。高いほど良いが、40 ROUGE-Lは「良い」、50は「例外的」だ。すべての論文が3つすべてを報告する。`rouge-score`パッケージを使う。

## 実装する

### ステップ1: TextRank（抽出型）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

名前を付ける価値のある2つのこと。類似度関数はログ正規化された単語オーバーラップを使い、これがオリジナルのTextRankバリアントだ。TF-IDFベクトルのコサインも機能する。減衰係数0.85と反復回数はPageRankのデフォルトだ。

### ステップ2: BARTによる抽象型

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(長いニュース記事テキスト)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNNはCNN/DailyMailコーパスでファインチューニングされている。すぐに使えるニューススタイルの要約を生成する。他のドメイン（科学論文、対話、法律）では対応するPegasusチェックポイントを使うか、ターゲットデータでファインチューニングする。

### ステップ3: ROUGE評価

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

常にステミングを使う。これなしでは「running」と「run」が異なる単語としてカウントされ、ROUGEが過少カウントする。

### ROUGEを超える（2026年の要約評価）

ROUGEは20年間支配的な要約指標だったが、2026年には単独では不十分だ。NLG論文の大規模なメタ分析が示したこと：

- **BERTScore**（文脈的埋め込み類似度）は2023年まで普及し、現在はほとんどの要約論文でROUGEと並んで報告されている。
- **BARTScore**は評価を生成として扱う：事前訓練済みBARTがソースを与えられた時に要約をどれほど確率的に割り当てるかでスコアを付ける。
- **MoverScore**（文脈的埋め込みに対するアースムーバー距離）は2025年の要約ベンチマークでトップに達した。ROUGEより意味的オーバーラップをよく捉えるからだ。
- **FactCC**と**QAベースの事実性**は2021〜2023年に一般的だったが、今では**G-Eval**（一貫性、一致性、流暢さ、関連性を連鎖思考推論でスコアリングするGPT-4プロンプトチェーン）に置き換えられることが多い。
- **G-Eval**と類似のLLMジャッジアプローチは、ルーブリックがよく設計されていれば人間の判断に約80%の確率でマッチする。

本番の推奨：レガシー比較のためにROUGE-L、意味的オーバーラップのためにBERTScore、一貫性と事実性のためにG-Evalを報告する。本番データを信頼する前に50〜100個の人間ラベル付き要約に対してキャリブレートする。

### ステップ4: 事実性の問題

抽象型要約は幻覚に傾向がある。抽出型要約はソースから逐語的に持ち上げられるためはるかに低い幻覚リスクを持つが、ソース文が脈絡なく使われた、古くなった、または順序がずれて引用された場合には誤解を招く可能性がある。これがコンプライアンス関連のコンテンツで本番システムがまだ抽出型を好む最大の理由だ。

名前を付けるべき幻覚タイプ：

- **エンティティスワップ。** ソースが「John Smith」と言う。要約が「John Brown」と言う。
- **数字のドリフト。** ソースが「25,000」と言う。要約が「2,500万」と言う。
- **極性の反転。** ソースが「提案を却下した」と言う。要約が「提案を受け入れた」と言う。
- **事実の発明。** ソースはCEOに言及しない。要約がCEOが承認したと言う。

機能する評価アプローチ：

- **FactCC。** ソース文と要約文の含意に対して訓練された二値分類器。事実/非事実を予測する。
- **QAベースの事実性。** 答えがソースにある質問をQAモデルに聞く。要約が異なる答えを支持する場合はフラグを立てる。
- **エンティティレベルF1。** ソースと要約の固有表現を比較する。要約だけに存在するエンティティは疑わしい。

事実性が重要なユーザー向けのコンテンツ（ニュース、医療、法律、金融）では、抽出型がより安全なデフォルトだ。抽象型はループ内に事実性チェックが必要だ。

## 使ってみる

2026年のスタック：

| ユースケース | 推奨 |
|-------------|------|
| ニュース、3〜5文の要約、英語 | `facebook/bart-large-cnn` |
| 科学論文 | `google/pegasus-pubmed`またはチューニングされたT5 |
| 複数文書、長文 | 32k以上のコンテキストを持つLLM（プロンプト） |
| 対話要約 | `philschmid/bart-large-cnn-samsum` |
| 構造的に幻覚リスクが低い抽出型 | TextRankまたは`sumy`のLSA/LexRank |

2026年では、計算量が制約でない場合、長いコンテキストを持つLLMが専門的なモデルを上回ることが多い。トレードオフはコストと再現性だ。専門的なモデルはより一貫した出力を提供する。

## 成果物を出す

`outputs/skill-summary-picker.md`として保存する：

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## 演習

1. **易しい。** 5つのニュース記事でTextRankを実行する。上位3文を参照要約と比較する。ROUGE-Lを測定する。CNN/DailyMailスタイルの記事で30〜45 ROUGE-Lが見られるはずだ。
2. **普通。** エンティティレベルの事実性を実装する：ソースと要約から固有表現を抽出（spaCy）し、要約でのソースエンティティの再現率と要約エンティティのソースに対する適合率を計算する。高い適合率と低い再現率は安全だが簡潔すぎる。低い適合率は幻覚されたエンティティを意味する。
3. **難しい。** BART-large-CNNをLLM（ClaudeまたはGPT-4）と50個のCNN/DailyMail記事で比較する。ROUGE-L、事実性（エンティティF1による）、要約あたりのコストを報告する。それぞれが勝つ場所を文書化する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------------------|
| 抽出型 | 文を選ぶ | ソースから文を逐語的に返す。幻覚しない。 |
| 抽象型 | 書き直す | ソースを条件に新しいテキストを生成する。幻覚の可能性あり。 |
| ROUGE | 要約指標 | システム出力と参照のN-gram/LCSオーバーラップ。 |
| TextRank | グラフベース抽出型 | 文類似度グラフに対するPageRank。 |
| 事実性 | 正しいか | 要約の主張がソースによって支持されているか。 |
| 幻覚 | 作られたコンテンツ | ソースが支持しない要約内のコンテンツ。 |

## 参考資料

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — 抽出型の標準的な論文。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART論文。
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasusとギャップセンテンス目標。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE論文。
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — 事実性の全貌を扱った論文。
