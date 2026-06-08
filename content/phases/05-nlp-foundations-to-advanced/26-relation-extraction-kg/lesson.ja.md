# 関係抽出と知識グラフ構築

> NERはエンティティを見つけた。エンティティリンキングがアンカーした。関係抽出はエンティティ間のエッジを見つける。知識グラフはノード、エッジ、そしてその出典の集合だ。


## 問題設定

アナリストが読む：「Tim CookはAppleのCEOに2011年になった。」4つの事実：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

関係抽出（RE）はフリーテキストを構造化されたトリプル`(主語, 関係, 目的語)`に変換する。コーパス全体で集計すると知識グラフになる。集計してクエリすると、RAG、分析、またはコンプライアンス監査のための推論基盤になる。

2026年の問題：LLMは熱心に関係を抽出する。熱心すぎる。ソーステキストがサポートしないトリプルを幻覚する。出典なしでは、本物のトリプルともっともらしいフィクションを区別できない。2026年の答えはAEVSスタイルのアンカーと検証パイプラインだ。

## 概念

![テキスト → トリプル → 知識グラフ](../assets/relation-extraction.svg)

**トリプル形式。** `(主語エンティティ, 関係タイプ, 目的語エンティティ)`。関係はクローズドオントロジー（Wikidataプロパティ、FIBO、UMLS）またはオープンセット（OpenIEスタイル、何でもあり）から来る。

**3つの抽出アプローチ。**

1. **ルール/パターンベース。** Hearstパターン：「X such as Y」→ `(Y, isA, X)`。プラス手作りの正規表現。脆く、精密で、説明可能。
2. **教師ありの分類器。** 文中の2つのエンティティ言及が与えられた場合、固定セットから関係を予測する。TACRED、ACE、KBPで学習。2015〜2022年の標準。
3. **生成的LLM。** トリプルを出力するようにモデルにプロンプトを送る。すぐに機能する。出典が必要、そうでないともっともらしいジャンクを幻覚する。

**AEVS（アンカー-抽出-検証-補足、2026年）。** 現在の幻覚軽減フレームワーク：

- **アンカー。** すべてのエンティティスパンと関係フレーズスパンを正確な位置で識別する。
- **抽出。** アンカースパンにリンクされたトリプルを生成する。
- **検証。** 各トリプル要素をソーステキストに照合；サポートされていないものを拒否する。
- **補足。** カバレッジパスにより、アンカーされたスパンが削除されないようにする。

幻覚が急激に減少する。より多くの計算が必要だが、監査可能。

**オープン対クローズドのトレードオフ。**

- **クローズドオントロジー。** 固定されたプロパティリスト（例：Wikidataの11,000以上のプロパティ）。予測可能。クエリ可能。発明しにくい。
- **オープンIE。** 動詞フレーズが関係になる。高再現率。低精度。クエリが煩雑。

本番KGは通常混在させる：発見のためにオープンIE、その後メイングラフにマージする前に関係をクローズドオントロジーに正規化する。

## 実装する

### ステップ1: パターンベースの抽出

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

フルのトイエクストラクターは`code/main.py`を参照。Hearstパターンはデバッグ可能なためドメイン固有パイプラインで依然として使われる。

### ステップ2: 教師ありの関係分類

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBELはseq2seqの関係エクストラクター：テキスト入力、トリプル出力、すでにWikidataのプロパティIDで。遠距離教師データで微調整。標準的なオープンウェイトのベースライン。

### ステップ3: アンカー付きのLLMプロンプト抽出

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

返されたすべてのスパンをソースに照合して検証する。`text[start:end] != triple_entity`のものはすべて拒否する。これが最小形式のAEVSの「検証」ステップだ。

### ステップ4: クローズドオントロジーへの正規化

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (subject/objectを逆転)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # マッピングされていないオープン関係を削除またはマニュアルレビューへ
```

正規化はエンジニアリング作業の60〜80%を占めることが多い。それに見合う予算を組む。

### ステップ5: 小さなグラフを構築してクエリする

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

これがすべてのRAG-over-KGシステムの原子。RDFトリプルストア（Blazegraph、Virtuoso）、プロパティグラフ（Neo4j）、またはベクトル強化グラフストアでスケールする。

## ピットフォール

- **RE前の共参照。** 「彼がAppleを設立した」— REは「彼」が誰かを知る必要がある。まず共参照を実行する（レッスン24）。
- **エンティティ正規化。** 「Apple Inc」と「Apple」は同じノードに解決されなければならない。まずエンティティリンキング（レッスン25）。
- **幻覚トリプル。** LLMはテキストがサポートしないトリプルを出力する。スパン検証を強制する。
- **関係正規化のドリフト。** オープンIEの関係は不整合（「was born in」、「came from」、「is a native of」）。正規化IDに集約しないとグラフがクエリ不可能になる。
- **時間的エラー。** 「Tim CookはAppleのCEOだ」— 今は真実、2005年には偽。多くの関係は時間境界がある。修飾子を使用する（WikidataのP580開始時刻、P582終了時刻）。
- **ドメインの不一致。** REBELはWikipediaで学習。法律、医療、科学的テキストはしばしばドメイン微調整のREモデルが必要。

## 使ってみる

2026年のスタック：

| 状況 | 選択 |
|------|------|
| 高速な本番、汎用ドメイン | REBELまたはLlamaPredとWikidata正規化 |
| ドメイン固有（バイオ医学、法律） | SciREXスタイルのドメイン微調整 + カスタムオントロジー |
| LLMプロンプト、監査付き出力 | AEVSパイプライン：アンカー → 抽出 → 検証 → 補足 |
| 大量のニュース情報抽出 | パターンベース + 教師ありのハイブリッド |
| スクラッチからKGを構築する | オープンIE + 手動正規化パス |
| 時系列KG | 修飾子付きの抽出（開始/終了時刻、時点） |

統合パターン：NER → coref → エンティティリンキング → 関係抽出 → オントロジーマッピング → グラフロード。すべてのステージが潜在的な品質ゲートだ。

## 成果物を出す

`outputs/skill-re-designer.md`として保存：

```markdown
---
name: re-designer
description: 出典と正規化を持つ関係抽出パイプラインを設計する。
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

コーパス（ドメイン、言語、量）と下流の用途（KG-RAG、分析、コンプライアンス）を与えられた場合、以下を出力する：

1. エクストラクター。パターンベース/教師あり/LLM/AEVSハイブリッド。精度対再現率の目標に関連した理由。
2. オントロジー。クローズドプロパティリスト（Wikidata/ドメイン）または正規化パス付きのオープンIE。
3. 出典。すべてのトリプルがソース文字スパン + 文書IDを持つ。監査に欠かせない。
4. マージ戦略。正規化エンティティID + 関係ID + 時間修飾子；重複排除ポリシー。
5. 評価。200件の手動ラベル付きトリプルの精度/再現率 + LLM抽出サンプルの幻覚率。

スパン検証（ソースの出典）なしのLLMベースのREパイプラインを拒否する。正規化なしで本番グラフに流れるオープンIE出力を拒否する。時間境界のある関係（雇用者、配偶者、役職）に時間修飾子のないパイプラインをフラグする。
```

## 演習

1. **易。** `code/main.py`のパターンエクストラクターを5つのニュース記事の文で実行する。精度を手動確認する。
2. **中。** REBEL（または小型LLM）を同じ文で使用する。トリプルを比較する。どちらのエクストラクターが精度が高いか？再現率が高いか？
3. **難。** AEVSパイプラインを構築する：LLMで抽出 + ソースに対してスパンを検証する。50件のWikipediaスタイルの文で検証ステップの前後の幻覚率を測定する。

## キーワード

| 用語 | 一般的な説明 | 実際の意味 |
|------|------------|----------|
| トリプル | 主語-関係-目的語 | KGの原子単位である`(s, r, o)`タプル。 |
| オープンIE | 何でも抽出 | オープン語彙の関係フレーズ；高再現率、低精度。 |
| クローズドオントロジー | 固定スキーマ | 限られた関係タイプのセット（Wikidata、UMLS、FIBO）。 |
| 正規化 | すべてを標準化 | 表面名/関係を正規化IDにマッピングする。 |
| AEVS | グラウンドされた抽出 | アンカー-抽出-検証-補足パイプライン（2026年）。 |
| 出典 | 情報源リンク | すべてのトリプルがソースへの文書ID + 文字スパンを持つ。 |
| 遠距離教師 | 安価なラベル | テキストを既存のKGと整合させて学習データを作成する。 |

## 参考資料

- [Mintz et al. (2009). ラベルなしデータを使った関係抽出のための遠距離教師](https://www.aclweb.org/anthology/P09-1113.pdf) — 遠距離教師の論文。
- [Huguet Cabot, Navigli (2021). REBEL: エンドツーエンドの言語生成による関係抽出](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq REの主力。
- [Wadden et al. (2019). 文脈化スパン表現によるエンティティ、関係、イベント抽出（DyGIE++）](https://arxiv.org/abs/1909.03546) — 共同IE。
- [AEVS — アンカー-抽出-検証-補足フレームワーク](https://www.mdpi.com/2073-431X/15/3/178) — 2026年の幻覚軽減設計。
- [Wikidata SPARQLチュートリアル](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — 正規化グラフクエリ。
