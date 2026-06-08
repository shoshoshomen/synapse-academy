# エンティティリンキングと曖昧性解消

> NERは「パリ」を見つけた。エンティティリンキングが判断する：フランスのパリか？パリス・ヒルトンか？テキサス州パリスか？パリ（トロイの王子）か？リンキングなしでは知識グラフは曖昧なままだ。


## 問題設定

「Jordan beat the press.」という文がある。NERは「Jordan」をPERSONとしてタグ付けする。良い。しかし*どの* Jordanなのか？

- マイケル・ジョーダン（バスケットボール）？
- マイケル・B・ジョーダン（俳優）？
- マイケル・I・ジョーダン（バークレーのMLの教授 — はい、この混同はML論文で実際に起きている）？
- ヨルダン（国）？
- ジョーダン（ヘブライ語の名前）？

エンティティリンキング（EL）は各言及を知識ベースの一意のエントリに解決する：Wikidata、Wikipedia、DBpedia、またはドメイン固有のKB。2つのサブタスク：

1. **候補生成。** 「Jordan」を与えられた場合、どのKBエントリが妥当か？
2. **曖昧性解消。** コンテキストを与えられた場合、どの候補が正しいか？

両ステップは学習可能。両方がベンチマークされている。組み合わせたパイプラインは10年間安定している — 変わるのは曖昧性解消の品質だ。

## 概念

![エンティティリンキングパイプライン：言及 → 候補 → 曖昧性解消されたエンティティ](../assets/entity-linking.svg)

**候補生成。** 言及の表面形（「Jordan」）を与えられた場合、エイリアスインデックスで候補を検索する。WikipediaのエイリアスはほとんどのNEをカバーする：「JFK」→ ジョン・F・ケネディ、ジャクリーン・ケネディ、JFK空港、JFK（映画）。典型的なインデックスは言及ごとに10〜30の候補を返す。

**曖昧性解消：3つのアプローチ。**

1. **事前確率 + コンテキスト（Milne & Witten、2008年）。** `P(エンティティ | 言及) × context-similarity(エンティティ, テキスト)`。機能し、高速で、学習不要。
2. **埋め込みベース（ESS / REL / Blink）。** 言及 + コンテキストをエンコード。各候補の説明をエンコード。最大コサインを選ぶ。2020〜2024年のデフォルト。
3. **生成的（GENRE、2021年；LLMベース、2023年以降）。** エンティティの正規化された名前をトークンごとにデコードする。有効なエンティティ名のトライに制約されているため、出力は有効なKB IDが保証される。

**エンドツーエンド対パイプライン。** 現代のモデル（ELQ、BLINK、ExtEnD、GENRE）はNER + 候補生成 + 曖昧性解消を一回で実行する。コンポーネントを交換できるため、パイプラインシステムが本番では依然として主流。

### 2つの測定基準

- **言及再現率（候補生成）。** 正しいKBエントリが候補リストに現れるゴールド言及の割合。パイプライン全体の下限。
- **曖昧性解消精度/F1。** 正しい候補が与えられた場合、トップ1が正しい頻度。

常に両方を報告する。80%の候補再現率で99%の曖昧性解消精度のシステムは80%のパイプラインだ。

## 実装する

### ステップ1: Wikipediaのリダイレクトからエイリアスインデックスを構築する

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipediaのエイリアスデータ：約1,800万（エイリアス、エンティティ）ペア。Wikidataのダンプからダウンロード。転置インデックスとして保存。

### ステップ2: コンテキストベースの曖昧性解消

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard重複はトイの例。埋め込みのコサイン類似度に置き換える（トランスフォーマーバージョンは`code/main.py`のステップ2を参照）。

### ステップ3: 埋め込みベース（BLINKスタイル）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

インデックス作成時に、すべてのKBエンティティを一度埋め込む。クエリ時に、言及 + コンテキストを一度埋め込み、候補プールに対してドット積を取り、最大値を選ぶ。

### ステップ4: 生成的エンティティリンキング（概念）

GENREはエンティティのWikipediaタイトルを文字ごとにデコードする。制約付きデコーディング（レッスン20参照）により、有効なタイトルのみが出力される。KBバックのトライとの緊密な統合。現代の後継はREL-GENおよび構造化出力付きのLLMプロンプトELだ。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

ホワイトリスト（Outlines `choice`）と組み合わせると、2026年に出荷する最もシンプルなELパイプラインになる。

### ステップ5: AIDA-CoNLLでの評価

AIDA-CoNLLは標準的なELベンチマーク：1,393件のReutersの記事、3.4万件の言及、Wikipediaエンティティ。KB内精度（`P@1`）とKB外NIL検出率を報告する。

## ピットフォール

- **NILの処理。** 一部の言及はKBに存在しない（新興エンティティ、無名人物）。システムは間違ったエンティティを推測する代わりにNILを予測しなければならない。別途測定。
- **言及境界エラー。** 上流のNERが部分的なスパンを見逃す（「Bank of America」が「Bank」とだけタグ付けされる）。ELの再現率が下がる。
- **人気バイアス。** 学習済みシステムは頻繁なエンティティを過剰に予測する。ML論文での「Michael I. Jordan」の言及はバスケットボールのジョーダンにリンクされることが多い。
- **クロス言語EL。** 中国語テキストの言及を英語のWikipediaエンティティにマッピングする。多言語エンコーダーまたは翻訳ステップが必要。
- **KB陳腐化。** 新しい会社、イベント、人物は昨年のWikipediaダンプにない。本番パイプラインには更新ループが必要。

## 使ってみる

2026年のスタック：

| 状況 | 選択 |
|------|------|
| 汎用英語 + Wikipedia | BLINKまたはREL |
| クロス言語、KB = Wikipedia | mGENRE |
| LLMフレンドリー、1日数件の言及 | 候補リスト + 制約付きJSONでClaude/GPT-4にプロンプト |
| ドメイン固有KB（医療、法律） | KBアウェアリトリーバル付きカスタムBERT + ドメインAIDAスタイルセットで微調整 |
| 超低レイテンシ | 完全一致事前確率のみ（Milne-Wittenベースライン） |
| リサーチSOTA | GENRE / ExtEnD / 生成的LLM-EL |

2026年に出荷される本番パターン：NER → coref → 各言及のEL → クラスターをクラスターごとに1つの正規化エンティティに集約する。出力：文書内の言及ごとに1つのKB IDではなく、エンティティごとに1つのKB ID。

## 成果物を出す

`outputs/skill-entity-linker.md`として保存：

```markdown
---
name: entity-linker
description: エンティティリンキングパイプライン — KB、候補生成、曖昧性解消、評価を設計する。
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

ユースケース（ドメインKB、言語、量、レイテンシ予算）を与えられた場合、以下を出力する：

1. 知識ベース。Wikidata / Wikipedia / カスタムKB。バージョン日付。更新頻度。
2. 候補生成器。エイリアスインデックス、埋め込み、またはハイブリッド。ターゲット言及再現率 @ K。
3. 曖昧性解消器。事前確率+コンテキスト、埋め込みベース、生成的、またはLLMプロンプト。
4. NIL戦略。トップスコアの閾値、分類器、または明示的なNIL候補。
5. 評価。言及再現率 @ 30、トップ1精度、保留セットでのNIL検出F1。

言及再現率ベースライン（候補生成が正しいエンティティを見つけたことを確認せずに曖昧性解消器を評価できない）なしのELパイプラインを拒否する。有効なKB IDへの制約付き出力なしのLLMプロンプトELを使用するパイプラインを拒否する。ドメイン微調整なしで人気バイアスがマイノリティエンティティ（名前の衝突など）に影響するシステムをフラグする。
```

## 演習

1. **易。** `code/main.py`の事前確率+コンテキスト曖昧性解消器を10件の曖昧な言及（パリ、ジョーダン、Apple）で実装する。正しいエンティティを手作業でラベル付けする。精度を測定する。
2. **中。** センテンストランスフォーマーで50件の曖昧な言及をエンコードする。各候補の説明を埋め込む。埋め込みベースの曖昧性解消とJaccardコンテキスト重複を比較する。
3. **難。** 1,000エンティティのドメイン固有KB（例：会社の従業員 + 製品）を構築する。NER + ELをエンドツーエンドで実装する。100件の保留文でPrecisionとRecallを測定する。

## キーワード

| 用語 | 一般的な説明 | 実際の意味 |
|------|------------|----------|
| エンティティリンキング（EL） | Wikipediaにリンク | 言及を一意のKBエントリにマッピングする。 |
| 候補生成 | 誰の可能性があるか？ | 言及のための妥当なKBエントリのショートリストを返す。 |
| 曖昧性解消 | 正しいものを選ぶ | コンテキストを使って候補をスコアリングし、勝者を選ぶ。 |
| エイリアスインデックス | 検索テーブル | 表面形 → 候補エンティティへのマッピング。 |
| NIL | KBにない | KBエントリが一致しないという明示的な予測。 |
| KB | 知識ベース | Wikidata、Wikipedia、DBpedia、またはドメイン固有KB。 |
| AIDA-CoNLL | ベンチマーク | ゴールドエンティティリンクを持つ1,393件のReuters記事。 |

## 参考資料

- [Milne, Witten (2008). Wikipediaとリンクする方法を学ぶ](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — 基礎的な事前確率+コンテキストアプローチ。
- [Wu et al. (2020). 密エンティティ検索によるゼロショットエンティティリンキング（BLINK）](https://arxiv.org/abs/1911.03814) — 埋め込みベースの主力。
- [De Cao et al. (2021). 自己回帰エンティティ検索（GENRE）](https://arxiv.org/abs/2010.00904) — 制約付きデコーディングによる生成的EL。
- [Hoffart et al. (2011). テキスト中の名前付きエンティティの堅牢な曖昧性解消（AIDA）](https://www.aclweb.org/anthology/D11-1072.pdf) — ベンチマーク論文。
- [REL: 巨人の肩に立つエンティティリンカー（2020年）](https://arxiv.org/abs/2006.01969) — オープンな本番スタック。
