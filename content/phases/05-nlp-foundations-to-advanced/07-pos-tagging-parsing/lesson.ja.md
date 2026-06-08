# 品詞タグ付けと構文解析

> 文法はしばらく流行らなかった。そしてすべてのLLMパイプラインが構造化抽出を検証する必要が生じ、戻ってきた。


## 問題の背景

レッスン01ではレンマ化に品詞タグが必要だと約束した。`running`が動詞だとわからなければ、レンマタイザーはそれを`run`に戻せない。`better`が形容詞だとわからなければ、`good`に戻せない。

その約束はサブフィールド全体を隠していた。品詞タグ付けは文法カテゴリを割り当てる。構文解析は文の木構造を復元する：どの単語がどれを修飾するか、どの動詞がどの引数を支配するか。古典的NLPは20年間この両方を磨き続けた。そしてディープラーニングが事前訓練済みTransformerの上のトークン分類タスクにまとめ上げ、研究コミュニティは先へ進んだ。

応用コミュニティはそうではない。すべての構造化抽出パイプラインが今もPOSと依存木を使っている。LLMが生成したJSONは文法上の制約に対して検証される。質問応答システムは依存解析を使ってクエリを分解する。機械翻訳の品質評価者は解析木のアライメントを確認する。

知っておく価値がある。このレッスンではタグセット、ベースライン、そしてゼロから実装を止めてspaCyを呼ぶべきポイントを紹介する。

## 概念

**品詞タグ付け**は各トークンに文法カテゴリのラベルを付ける。**Penn Treebank（PTB）**タグセットが英語のデフォルトだ。36個のタグがあり、気難しさを感じる区別がある：`NN`単数名詞、`NNS`複数名詞、`NNP`固有名詞単数、`VBD`動詞過去形、`VBZ`動詞三人称単数現在形など。**Universal Dependencies（UD）**タグセットはより粗く（17タグ）、言語非依存だ。言語横断の研究のデフォルトになった。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**構文解析**は木を生成する。2つの主要なスタイル：

- **句構造解析。** 名詞句、動詞句、前置詞句が互いの中にネストする。出力は葉として単語を持つ非終端カテゴリ（NP、VP、PP）の木だ。
- **依存解析。** 各単語は依存する一つの主要語を持ち、文法関係でラベル付けされる。出力はすべてのエッジが（主要語、従属語、関係）の三つ組みである木だ。

依存解析は2010年代に勝った。特に自由語順言語に対してより言語横断的に機能するからだ。

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## 実装する

### ステップ1: 最頻出タグベースライン

うまく機能する最もシンプルな品詞タガー。各単語について、訓練時に最もよく見られたタグを予測する。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

Brownコーパスでこのベースラインは約85%の精度に達する。良くはないが、真剣なモデルがこれを下回るべきでない床だ。

### ステップ2: バイグラムHMMタガー

シーケンスの結合確率をモデル化する：

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

2つのテーブル：遷移確率（前のタグが与えられたときのタグ）、出力確率（タグが与えられたときの単語）。両方をLaplaceスムージングを使ったカウントから推定する。Viterbi（タグラティス上の動的プログラミング）でデコードする。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Brownでのバイグラムは約93%の精度に達する。85%から93%へのジャンプは主に遷移確率によるものだ——モデルは`DET NOUN`が一般的で`NOUN DET`はまれだと学習する。

### ステップ3: 現代のタガーがこれを超える理由

遷移確率と出力確率はローカルだ。「I bought a saw（ノコギリを買った）」の`saw`が名詞で「I saw the movie（映画を見た）」の`saw`が動詞という点を捉えられない。任意の特徴（接尾辞、単語の形状、前後の単語、単語自体）を持つCRFは約97%に達する。BiLSTM-CRFまたはTransformerは98%以上に達する。

このタスクの上限はアノテーターの不一致によって設定される。人間のアノテーターはPenn Treebankで約97%の確率で一致する。98%を超えたモデルはおそらくテストセットに過学習している。

### ステップ4: 依存解析のスケッチ

ゼロからの完全な依存解析はスコープ外だ。正典的な教科書の扱いはJurafsky and Martinにある。知っておくべき2つの古典的な流派：

- **遷移ベース**パーサー（arc-eager、arc-standard）はシフト-リデュースパーサーのように動作する：トークンを読み込み、スタックにシフトして、アークを作成するリデュースアクションを適用する。貪欲デコードは速い。古典的な実装はMaltParser。現代のニューラル版はChen and Manningの遷移ベースパーサー。
- **グラフベース**パーサー（EisnerのアルゴリズムやDozat-Manningバイアファイン）は可能なすべての主要語-従属語エッジをスコアリングして最大全域木を選ぶ。遅いがより正確。

ほとんどの応用作業ではspaCyを呼ぶ：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

`dep`列を下から上に読むと、文の文法構造が見えてくる。

## 使ってみる

すべての本番NLPライブラリは品詞と依存解析器を標準パイプラインの一部として提供している。

- **spaCy**（`en_core_web_sm`/`md`/`lg`/`trf`）。速くて正確で、トークン化+NER+レンマ化と統合されている。`token.tag_`（Penn）、`token.pos_`（UD）、`token.dep_`（依存関係）。
- **Stanford NLP（stanza）**。StanfordのCorNLPの後継。60以上の言語で最先端。
- **trankit**。Transformerベースで、UD精度が良い。
- **NLTK**。`pos_tag`。使えるが遅くて古い。教育用に適している。

### 2026年でもこれが重要な場面

- **レンマ化。** レッスン01では正しくレンマ化するためにPOSが必要だ。常に。
- **LLM出力からの構造化抽出。** 生成された文が文法上の制約（例：主語-動詞の一致、必要な修飾語）を守っているか検証する。
- **アスペクトベース感情分析。** 依存解析は形容詞がどの名詞を修飾するかを教えてくれる。
- **クエリ理解。** 「Wes Andersonが監督した映画でBill Murrayが主演したもの」は解析を通じて構造化された制約に分解できる。
- **言語横断転用。** UDタグと依存関係は言語非依存で、新しい言語のゼロショット構造化分析ができる。
- **低計算量パイプライン。** Transformerを使えない場合、POS＋依存解析＋ガゼッターは驚くほど遠くまで届く。

## 成果物を出す

`outputs/skill-grammar-pipeline.md`として保存する：

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## 演習

1. **易しい。** 小さいタグ付きコーパス（例：NLTKのBrownサブセット）で最頻出タグベースラインを使い、ホールドアウト文で精度を測定する。約85%の結果を確認する。
2. **普通。** 上記のバイグラムHMMを訓練し、タグごとの適合率/再現率を報告する。HMMはどのタグを最も混乱させるか？
3. **難しい。** spaCyの依存解析を使って1000文のサンプルから主語-動詞-目的語の三つ組みを抽出する。手動でラベル付けした50の三つ組みで評価する。抽出が失敗する場所（多くの場合、受動態、等位接続、省略された主語）を文書化する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------------------|
| 品詞タグ | 単語のタイプ | 文法カテゴリ。PTBは36個、UDは17個。 |
| Penn Treebank | 標準タグセット | 英語専用。細かい動詞の時制と名詞の数。 |
| Universal Dependencies | 多言語タグセット | PTBより粗い。言語中立、言語横断作業のデフォルト。 |
| 依存解析 | 文の木 | 各単語は一つの主要語を持ち、各エッジは文法関係を持つ。 |
| Viterbi | 動的プログラミング | 出力確率と遷移確率が与えられたときの最高確率タグシーケンスを見つける。 |

## 参考資料

- [Jurafsky and Martin — Speech and Language Processing、第8章と第18章](https://web.stanford.edu/~jurafsky/slp3/) — POSと解析の正典的な教科書の扱い。
- [Universal Dependenciesプロジェクト](https://universaldependencies.org/) — すべての多言語パーサーで使われる言語横断タグセットとツリーバンクコレクション。
- [spaCy言語機能ガイド](https://spacy.io/usage/linguistic-features) — `Token`で公開されるすべての属性の実用的なリファレンス。
- [Chen and Manning（2014）. ニューラルネットワークを使った高速で正確な依存解析](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — ニューラルパーサーを主流に押し上げた論文。
