# RAGのチャンキング戦略

> チャンキングの設定は埋め込みモデルの選択と同じくらいRAGの検索品質に影響する（Vectara NAACL 2025）。チャンキングを間違えると、どれだけリランキングしても救えない。


## 問題設定

50ページの契約書をRAGシステムに入れた。ユーザーが「解約条項は何か？」と質問した。検索器が表紙を返した。なぜ？モデルは512トークンのチャンクで学習されており、解約条項は20ページ先に、改ページをまたいで、クエリと結びつけるローカルキーワードなしで存在しているからだ。

解決策は「より良い埋め込みモデルを買う」ことではない。解決策はチャンキングだ。どれくらいの大きさか？オーバーラップは？どこで分割するか？前後のコンテキストを含めるか？

2026年2月のベンチマークは驚くべき結果を示している：

- Vectaraの2026年研究：再帰的な512トークンのチャンキングがセマンティックチャンキングを69% → 54%の精度で上回った。
- Natural QuestionsでのSPLADE + Mistral-8B：オーバーラップは測定可能な利益をゼロ提供した。
- コンテキストクリフ：回答品質は約2,500トークンのコンテキストで急激に低下する。

「明らかな」答え（セマンティックチャンキング、20%オーバーラップ、1000トークン）はしばしば間違っている。このレッスンでは6つの戦略の直感を構築し、どの戦略をいつ使うかを伝える。

## 概念

![一つのパッセージで可視化された6つのチャンキング戦略](../assets/chunking.svg)

**固定チャンキング。** N文字またはトークンごとに分割する。最もシンプルなベースライン。文の途中で分割する。圧縮は良いが一貫性は悪い。

**再帰的。** LangChainの`RecursiveCharacterTextSplitter`。まず`\n\n`で分割を試み、次に`\n`、`.`、スペースと続く。クリーンにフォールバックする。2026年のデフォルト。

**セマンティック。** 各文を埋め込む。隣接する文のコサイン類似度を計算する。類似度が閾値以下に下がった場所で分割する。トピックの一貫性を保持する。より遅い；時に検索を害する40トークンの断片を生成することがある。

**文。** 文の境界で分割する。一文一チャンクまたはN文のウィンドウ。コストのごく一部でセマンティックチャンキングと同等（最大約5kトークン）。

**親ドキュメント。** 検索用に小さな子チャンクを、コンテキスト用に大きな親チャンクを*両方*保存する。子チャンクで検索し、親チャンクを返す。劣化が緩やか：悪い子チャンクでも適切な親を返す。

**遅延チャンキング（2024年）。** 最初にトークンレベルで文書全体を埋め込み、その後トークン埋め込みをチャンク埋め込みにプールする。チャンク横断のコンテキストを保持する。長コンテキストの埋め込み器（BGE-M3、Jina v3）で機能する。計算コストが高い。

**コンテキスト検索（Anthropic、2024年）。** 各チャンクの前にLLMが生成した文書内での位置のサマリーを付加する（「このチャンクは解約条項のセクション3.2です...」）。Anthropic自身のベンチマークで35〜50%の検索改善。インデックス作成が高価。

### すべてのデフォルトを上回るルール

クエリタイプに合わせてチャンクサイズを調整する：

| クエリタイプ | チャンクサイズ |
|----------|-------------|
| 事実型（「CEOの名前は？」） | 256〜512トークン |
| 分析的/マルチホップ | 512〜1024トークン |
| セクション全体の理解 | 1024〜2048トークン |

NVIDIAの2026年ベンチマーク。チャンクは回答とローカルコンテキストを含むのに十分なほど大きく、検索器のトップKが答えではなくコンテキストのノイズに集中しないよう十分に小さくなければならない。

## 実装する

### ステップ1: 固定チャンキングと再帰的チャンキング

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### ステップ2: セマンティックチャンキング

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

自分のドメインで`threshold`を調整する。高すぎると断片が生まれる。低すぎると巨大なチャンクになる。

### ステップ3: 親ドキュメント

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

重要な洞察：親の重複を除く。複数の子チャンクが同じ親にマッピングされることがある；すべてを返すとコンテキストが無駄になる。

### ステップ4: コンテキスト検索（Anthropicパターン）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

コンテキスト化されたチャンクをインデックスする。クエリ時には追加の周辺シグナルから検索が恩恵を受ける。

### ステップ5: 評価

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

常にベンチマークを取ること。自分のコーパスで「最良」の戦略はどのブログ記事とも一致しないかもしれない。

## ピットフォール

- **事実型クエリのみでチャンキングを評価する。** マルチホップクエリは全く異なる勝者を明らかにする。クエリタイプで層化した評価セットを使用する。
- **最小サイズなしのセマンティックチャンキング。** 検索を害する40トークンの断片を生成する。常に`min_tokens`を強制する。
- **カーゴカルトとしてのオーバーラップ。** 2026年の研究ではオーバーラップは利益をゼロ提供し、インデックスコストを2倍にすることがよくある。仮定せず測定する。
- **最小/最大の強制なし。** 5トークンや5000トークンのチャンクはどちらも検索を壊す。クランプする。
- **クロスドキュメントチャンキング。** チャンクが2つの文書をまたがないようにする。常に文書ごとにチャンキングし、その後マージする。

## 使ってみる

2026年のスタック：

| 状況 | 戦略 |
|------|------|
| 最初のビルド、未知のコーパス | 再帰的、512トークン、オーバーラップなし |
| 事実型QA | 再帰的、256〜512トークン |
| 分析的/マルチホップ | 再帰的、512〜1024トークン + 親ドキュメント |
| 重いクロスリファレンス（契約書、論文） | 遅延チャンキングまたはコンテキスト検索 |
| 会話/ダイアログコーパス | ターンレベルのチャンク + スピーカーメタデータ |
| 短い発話（ツイート、レビュー） | ドキュメント1件 = チャンク1件 |

再帰的512から始める。50クエリの評価セットでrecall@5を測定する。そこから調整する。

## 成果物を出す

`outputs/skill-chunker.md`として保存：

```markdown
---
name: chunker
description: 与えられたコーパスとクエリ分布に合わせたチャンキング戦略、サイズ、オーバーラップを選ぶ。
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

コーパス（文書タイプ、平均長さ、ドメイン）とクエリ分布（事実型/分析的/マルチホップ）を与えられた場合、以下を出力する：

1. 戦略。再帰的/文/セマンティック/親ドキュメント/遅延/コンテキスト検索。理由。
2. チャンクサイズ。トークン数。クエリタイプに関連した理由。
3. オーバーラップ。デフォルト0；>0の場合は理由を示す。
4. 最小/最大強制。`min_tokens`、`max_tokens`のガード。
5. 評価計画。50クエリの層化評価セット（事実型、分析的、マルチホップ）でのrecall@5。

最小/最大チャンクサイズの強制なしのチャンキング戦略の推奨を拒否する。アブレーションで効果が示されていない場合、20%超のオーバーラップを拒否する。最小トークンフロアなしのセマンティックチャンキングの推奨をフラグする。
```

## 演習

1. **易。** 一つの20ページ文書をfixed(512, 0)、recursive(512, 0)、recursive(512, 100)でチャンキングする。チャンク数と境界の品質を比較する。
2. **中。** 5文書で30クエリの評価セットを構築する。再帰的、セマンティック、親ドキュメントでrecall@5を測定する。どれが勝つか？ブログ記事と一致するか？
3. **難。** コンテキスト検索を実装する。ベースラインの再帰的に対してMRRの改善を測定する。インデックスコスト（LLMの呼び出し）対精度の向上を報告する。

## キーワード

| 用語 | 一般的な説明 | 実際の意味 |
|------|------------|----------|
| チャンク | 文書の断片 | 埋め込まれ、インデックスされ、検索されるサブドキュメント単位。 |
| オーバーラップ | 安全マージン | 隣接するチャンク間で共有されるNトークン；2026年のベンチマークでは多くの場合無用。 |
| セマンティックチャンキング | スマートチャンキング | 隣接する文の埋め込み類似度が下がる場所で分割する。 |
| 親ドキュメント | 2レベルの検索 | 小さな子チャンクで検索し、大きな親チャンクを返す。 |
| 遅延チャンキング | 埋め込み後にチャンク | トークンレベルで文書全体を埋め込み、チャンクベクトルにプールする。 |
| コンテキスト検索 | Anthropicのトリック | インデックス作成前に各チャンクにLLMが生成したサマリーを付加する。 |
| コンテキストクリフ | 2500トークンの壁 | RAGで約2.5kのコンテキストトークン付近で観察された品質低下（2026年1月）。 |

## 参考資料

- [Yepes et al. / LangChain — 再帰的文字分割ドキュメント](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 本番でのデフォルト。
- [Vectara (2024, NAACL 2025). チャンキング設定の分析](https://arxiv.org/abs/2410.13070) — チャンキングは埋め込みモデルの選択と同じくらい重要。
- [Jina AI — 長コンテキスト埋め込みモデルでの遅延チャンキング (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — 遅延チャンキングの論文。
- [Anthropic — コンテキスト検索](https://www.anthropic.com/news/contextual-retrieval) — LLMが生成したコンテキストプレフィックスで35〜50%の検索改善。
- [NVIDIA 2026年チャンクサイズベンチマーク — Premaiサマリー](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — クエリタイプ別のチャンクサイズ。
