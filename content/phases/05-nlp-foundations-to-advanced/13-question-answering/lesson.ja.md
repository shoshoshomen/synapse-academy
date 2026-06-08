# 質問応答システム

> 3つのシステムが現代のQAを形成した。抽出型はスパンを見つけた。検索拡張型はそれをドキュメントに根拠付けた。生成型は回答を生成した。すべての現代のAIアシスタントは3つの混合だ。


## 問題の背景

ユーザーが「最初のiPhoneはいつ発売されたか？」と入力して「2007年6月29日」を期待する。「Appleの歴史は長く多様だ」ではない。孤立した「2007」でもない。直接的で根拠のある正しい答えだ。

3つのアーキテクチャがここ10年のQAを支配した。

- **抽出型QA。** 質問と回答を含むことが分かっているパッセージが与えられ、パッセージ内の回答スパンの開始と終了インデックスを見つける。SQuADが標準的なベンチマークだ。
- **オープンドメインQA。** パッセージは与えられない。最初に関連するパッセージを検索してから回答を抽出または生成する。これが今日のすべてのRAGパイプラインの基盤だ。
- **生成型/クローズドブックQA。** 大規模言語モデルがパラメトリックメモリから答える。検索なし。推論時最速、事実に最も信頼できない。

2026年のトレンドはハイブリッドだ：最良の少数のパッセージを検索してから、それらのパッセージに根拠付けて回答するよう生成モデルにプロンプトする。それがRAGで、レッスン14では検索の半分を深く扱う。このレッスンではQAの半分を構築する。

## 概念

**抽出型。** TransformerでBERTファミリーで質問とパッセージを一緒にエンコードする。回答の開始と終了トークンインデックスを予測する2つのヘッドを訓練する。損失は有効な位置に対する交差エントロピー。出力はパッセージからのスパン。（構造的に）幻覚しない。（構造的に）パッセージが答えられない質問を扱えない。

**検索拡張（RAG）。** 2つのステージ。まず検索器がコーパスから上位`k`個のパッセージを見つける。次に読み取り器（抽出型または生成型）がそれらのパッセージを使って回答を生成する。検索器-読み取り器の分割でそれぞれを独立して訓練・評価できる。現代のRAGはしばしばその間にリランカーを追加する。

**生成型。** デコーダーオンリーLLM（GPT、Claude、Llama）が学習済みの重みから答える。検索ステップなし。一般的な知識には優れているが、まれまたは最近の事実には壊滅的。幻覚率は事前訓練データでの事実の頻度と逆相関する。

## 実装する

### ステップ1: 事前訓練済みモデルによる抽出型QA

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2`は回答不能な質問を含むSQuAD 2.0で訓練されている。デフォルトでは、`question-answering`パイプラインはモデルのヌルスコアが勝つ場合でも最高スコアのスパンを返す——自動的に空の答えを返す*わけではない*。明示的な「回答なし」の動作を得るには、パイプライン呼び出しに`handle_impossible_answer=True`を渡す：その後、パイプラインはヌルスコアがすべてのスパンスコアを超えた場合のみ空の答えを返す。いずれの場合も`score`フィールドを必ずチェックすること。

### ステップ2: 検索拡張パイプライン（スケッチ）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

2段階パイプライン。高密度検索器（Sentence-BERT）が意味的類似度によって関連するパッセージを見つける。抽出型読み取り器（RoBERTa-SQuAD）が組み合わされた上位パッセージから回答スパンを取り出す。小さなコーパスで機能する。百万文書のコーパスにはFAISSまたはベクターデータベースを使う。

### ステップ3: RAGによる生成型

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

プロンプトのパターンが重要だ。モデルにコンテキストに根拠を置いて、コンテキストが不十分な場合は「わかりません」と返すよう明示的に伝えることで、ナイーブなプロンプティングと比較して幻覚率が40〜60%削減される。より精巧なパターンは引用、信頼スコア、構造化抽出を追加する。

### ステップ4: 現実世界を反映した評価

SQuADは**完全一致（EM）**と**トークンレベルF1**を使う。EMは正規化後（小文字化、句読点の除去、冠詞の除去）の厳密なマッチだ——予測が正確にマッチするかスコア0のどちらかだ。F1は予測と参照のトークンオーバーラップで計算され、部分的なクレジットを与える。両方とも言い換えを過小評価する：「June 29, 2007」と「June 29th, 2007」は通常0 EM（序数が正規化を壊す）だが、重複するトークンから相当なF1を得る。

本番QAのために：

- **回答精度**（指標が意味的等価性を捉えないため、LLMによるまたは人間による判定）。
- **引用精度。** 引用されたパッセージは実際に回答を支持するか？生成された引用と取得されたパッセージの間の文字列マッチで自動的にチェックするのは簡単だ。
- **拒否キャリブレーション。** 取得されたパッセージに答えがない場合、システムは正しく「わかりません」と言うか？偽の信頼率を測定する。
- **検索再現率。** 読み取り器を評価する前に、検索器が上位`k`に正しいパッセージを入れるかどうかを測定する。読み取り器は欠落したパッセージを修正できない。

### RAGAS：2026年の本番評価フレームワーク

`RAGAS`はRAGシステムのために特別に構築されており、2026年の出荷デフォルトだ。ゴールド参照なしで4つの次元をスコアリングする：

- **事実性（Faithfulness）。** 回答の各主張は取得されたコンテキストから来ているか？NLIベースの含意で測定。主要な幻覚指標。
- **回答関連性（Answer relevance）。** 回答は質問に答えているか？回答から仮想の質問を生成して実際の質問と比較することで測定。
- **コンテキスト精度（Context precision）。** 取得されたチャンクのうち、実際に関連していた割合は？低精度 = プロンプトのノイズ。
- **コンテキスト再現率（Context recall）。** 取得されたセットは必要なすべての情報を含んでいたか？低再現率 = 読み取り器が成功できない。

参照なしスコアリングにより、厳選されたゴールド回答なしで本番のライブトラフィックを評価できる。完全一致指標が役に立たないオープンエンドの質問に対してはその上にLLM-as-judgeを重ねる。

`pip install ragas`。検索器と読み取り器を接続する。クエリごとに4つのスカラーを得る。回帰に対してアラートを出す。

## 使ってみる

2026年のスタック。

| ユースケース | 推奨 |
|-------------|------|
| パッセージが与えられ、回答スパンを見つける | `deepset/roberta-base-squad2` |
| 固定コーパス上、クローズドブックは許容できない | RAG：高密度検索器 + LLM読み取り器 |
| ドキュメントストアに対するリアルタイム | ハイブリッド（BM25 + 高密度）検索器 + リランカー（レッスン14）を使ったRAG |
| 会話型QA（フォローアップの質問） | 会話履歴を持つLLM + 各ターンでのRAG |
| 高度な事実性、規制ドメイン | 権威あるコーパスに対する抽出型。生成型単独は絶対NG |

抽出型QAは2026年ではLLMを使ったRAGがより多くのケースを扱うため流行らない。逐語的な引用が必要な文脈ではまだ出荷される：法律調査、規制コンプライアンス、監査ツール。

## 成果物を出す

`outputs/skill-qa-architect.md`として保存する：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 演習

1. **易しい。** 10個のWikipediaパッセージで上記のSQuAD抽出パイプラインをセットアップする。10個の質問を手作りする。回答が何回正しいかを測定する。パッセージと質問がきれいなら7〜9正解が見られるはずだ。
2. **普通。** 拒否分類器を追加する。上位の検索スコアがしきい値（例：コサイン0.3）を下回る場合、読み取り器を呼ぶ代わりに「わかりません」を返す。ホールドアウトセットでしきい値を調整する。
3. **難しい。** 自分で選んだ10,000文書のコーパスに対してRAGパイプラインを構築する。RRFフュージョンを使ったハイブリッド検索（BM25 + 高密度）を実装する（レッスン14参照）。ハイブリッドステップの有無で回答精度を測定する。どの質問タイプが最も恩恵を受けるかを文書化する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------------------|
| 抽出型QA | 回答スパンを見つける | 与えられたパッセージ内の回答の開始と終了インデックスを予測する。 |
| オープンドメインQA | コーパスに対するQA | パッセージは与えられない。検索してから回答しなければならない。 |
| RAG | 検索して生成 | 検索拡張生成。検索器と読み取り器のパイプライン。 |
| SQuAD | 標準的なベンチマーク | Stanford Question Answering Dataset。EMとF1指標。 |
| 幻覚 | 作り上げられた回答 | 取得されたコンテキストによって支持されない読み取り器の出力。 |
| 拒否キャリブレーション | いつ黙るかを知る | 回答できない時にシステムが正しく「わかりません」と言う。 |

## 参考資料

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — ベンチマーク論文。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR、QAの標準的な高密度検索器。
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — RAGと名付けた論文。
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — 包括的なRAGサーベイ。
