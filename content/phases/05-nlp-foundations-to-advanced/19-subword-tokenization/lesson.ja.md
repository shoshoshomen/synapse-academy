# サブワードトークン化 — BPE、WordPiece、Unigram、SentencePiece

> 単語トークナイザーは未知の単語で詰まる。文字トークナイザーはシーケンス長を爆発させる。サブワードトークナイザーは中間を取る。2026年のすべての最先端LLMはこれを採用している。


## 問題設定

語彙に50,000単語がある。ユーザーが「untokenizable」と入力する。トークナイザーは`[UNK]`を返す。モデルはその単語についての信号を持たない。さらに悪いことに：コーパスの90パーセンタイルの文書は40個の稀な単語を含み、これは文書あたり40ビットの情報が失われることを意味する。

サブワードトークン化はこれを解決する。一般的な単語は単一のトークンとして残る。稀な単語は意味のある部分に分解される：`untokenizable` → `un`、`token`、`izable`。学習データはすべてをカバーする。なぜなら任意の文字列は最終的にバイト列だからだ。

2026年のすべてのフロンティアLLMは3つのアルゴリズム（BPE、Unigram、WordPiece）のいずれかに基づき、3つのライブラリ（tiktoken、SentencePiece、HF Tokenizers）のいずれかでラップされている。言語モデルを出荷するにはどれかを選ばなければならない。

## 概念

![BPE vs Unigram vs WordPiece、文字ごとの比較](../assets/subword-tokenization.svg)

**BPE（Byte-Pair Encoding）。** 文字レベルの語彙から始める。すべての隣接ペアをカウントする。最も頻繁なペアを新しいトークンにマージする。目標の語彙サイズに達するまで繰り返す。主要アルゴリズム：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**バイトレベルBPE。** Unicode文字ではなく生バイト（256の基本トークン）で同じアルゴリズム。任意のバイト列がエンコードできるため`[UNK]`トークンがゼロになる。GPT-2は50,257トークン（256バイト＋50,000マージ＋1特殊トークン）を使用する。

**Unigram。** 大きな語彙から始める。各トークンにユニグラム確率を割り当てる。コーパスの対数尤度の低下が最小となるトークンを反復的に刈り込む。推論時に確率的：トークン化をサンプリングできる（サブワード正則化によるデータ拡張に有用）。T5、mBART、ALBERT、XLNet、Gemmaで使用される。

**WordPiece。** 生の頻度ではなく学習コーパスの尤度を最大化するペアをマージする。BERT、DistilBERT、ELECTRAで使用される。

**SentencePiece対tiktoken。** SentencePieceはUnicodeテキストから直接語彙を*学習する*ライブラリ（BPEまたはUnigram）で、空白を`▁`としてエンコードする。tiktokenはOpenAIの事前構築済み語彙に対する高速*エンコーダー*；学習はしない。

経験則：

- **新しい語彙の学習：** SentencePiece（多言語、前処理不要）またはHF Tokenizers。
- **GPT語彙に対する高速推論：** tiktoken（GPT-4以降にはcl100k_base、o200k_base）。
- **両方：** HF Tokenizers — 一つのライブラリで学習と配信。

## 実装する

### ステップ1: スクラッチからBPE

`code/main.py`を参照。ループ：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

アルゴリズムがエンコードする3つの事実。`</w>`は単語末尾をマークし、「low」（接尾辞）と「lower」（接頭辞）を区別する。頻度重み付けにより高頻度ペアが早期に勝つ。マージリストは順序付けされている — 推論では学習順にマージを適用する。

### ステップ2: 学習済みマージでエンコード

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

単純なO(n·|merges|)。本番実装（tiktoken、HF Tokenizers）はマージランクルックアップと優先度キューを使用してほぼ線形時間で実行する。

### ステップ3: 実践でのSentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # または "unigram"
    character_coverage=0.9995, # CJKには低めに（例：英語は0.9995、日本語は0.995）
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注：前処理不要、空白を`▁`としてエンコード、`character_coverage`は稀な文字をどの程度積極的に保持するか対`<unk>`にマップするかを制御する。

### ステップ4: OpenAI互換語彙のためのtiktoken

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

エンコードのみ。高速（Rustバックエンド）。バイト数カウント、コスト推定、コンテキストウィンドウ予算計画でのGPT-4/5トークン化との完全一致。

## 2026年でも起きるピットフォール

- **トークナイザードリフト。** 語彙Aで学習し、語彙Bに対してデプロイ。トークンIDが異なる；モデルはゴミを出力する。CIで`tokenizer.json`のハッシュを確認する。
- **空白の曖昧さ。** BPEの「hello」と「 hello」は異なるトークンを生成する。常に`add_special_tokens`と`add_prefix_space`を明示的に指定する。
- **多言語の学習不足。** 英語偏重のコーパスは非ラテンスクリプトを5〜10倍多くのトークンに分割する語彙を生成する。同じプロンプトがGPT-3.5では日本語/アラビア語で5〜10倍のコストになる。o200k_baseがこれを部分的に修正した。
- **絵文字の分割。** 単一の絵文字が5トークンになることがある。コンテキストの予算を立てる際に絵文字の処理を確認する。

## 使ってみる

2026年のスタック：

| 状況 | 選択 |
|-----------|------|
| 単言語モデルをスクラッチで学習 | HF Tokenizers（BPE） |
| 多言語モデルを学習 | SentencePiece（Unigram、`character_coverage=0.9995`） |
| OpenAI互換APIを配信 | tiktoken（GPT-4以降には`o200k_base`） |
| ドメイン固有の語彙（コード、数学、タンパク質） | ドメインコーパスでカスタムBPEを学習してベース語彙とマージ |
| エッジ推論、小型モデル | Unigram（より小さな語彙の方が良く機能する） |

語彙サイズはスケーリングの決定であり定数ではない。大まかなヒューリスティック：1B未満のパラメータには32k、1〜10Bには50〜100k、多言語/フロンティアには200k以上。

## 成果物を出す

`outputs/skill-bpe-vs-wordpiece.md`として保存：

```markdown
---
name: tokenizer-picker
description: 特定のコーパスとデプロイターゲットに合わせたトークナイザーアルゴリズム、語彙サイズ、ライブラリを選ぶ。
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

コーパス（サイズ、言語、ドメイン）とデプロイターゲット（スクラッチから学習 / 微調整 / API互換推論）を与えられた場合、以下を出力する：

1. アルゴリズム。BPE、Unigram、またはWordPiece。一文の理由。
2. ライブラリ。SentencePiece、HF Tokenizers、またはtiktoken。理由。
3. 語彙サイズ。最も近い1kに丸める。モデルサイズと言語カバレッジに関連した理由。
4. カバレッジ設定。`character_coverage`、`byte_fallback`、特殊トークンリスト。
5. 検証計画。保留セットでの単語あたりの平均トークン数、OOV率、圧縮率、ラウンドトリップデコード等価性。

稀なスクリプトのコンテンツを含むコーパスでcharacter_coverage < 0.995のトークナイザーを学習することを拒否する。CIで凍結された`tokenizer.json`ハッシュチェックなしに語彙を出荷することを拒否する。16k未満の語彙の単言語トークナイザーをスペック不足としてフラグする。
```

## 演習

1. **易。** `code/main.py`の小さなコーパスで500マージのBPEを学習する。3つの保留された単語をエンコードする。ちょうど1トークンになったものと1トークン超になったものはいくつか？
2. **中。** 100の英語Wikipediaの文で`cl100k_base`、`o200k_base`、語彙数32kで学習したSentencePiece BPEのトークン数を比較する。各圧縮率を報告する。
3. **難。** 同じコーパスをBPE、Unigram、WordPieceで学習する。それぞれを小さな感情分類器に使用した際の下流精度を測定する。選択はF1を1ポイント以上動かすか？

## キーワード

| 用語 | 一般的な説明 | 実際の意味 |
|------|------------|----------|
| BPE | Byte-Pair Encoding | 目標語彙サイズに達するまで最頻出文字ペアを貪欲にマージ。 |
| バイトレベルBPE | 未知トークンなし | 生の256バイトに対するBPE；GPT-2 / Llamaが使用。 |
| Unigram | 確率的トークナイザー | 対数尤度を使用して大きな候補セットから刈り込む；T5、Gemmaで使用。 |
| SentencePiece | 空白を扱うもの | 生テキストでBPE/Unigramを学習するライブラリ；空白を`▁`としてエンコード。 |
| tiktoken | 高速なもの | 事前構築済み語彙のためのOpenAIのRustバックエンドBPEエンコーダー。学習なし。 |
| マージリスト | 魔法の数字 | 順序付けされた`(a, b) → ab`マージのリスト；推論は順番に適用する。 |
| 文字カバレッジ | どのくらい稀なら稀すぎるか | トークナイザーがカバーしなければならない学習コーパス内の文字の割合；約0.9995が一般的。 |

## 参考資料

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPEの論文。
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — Unigramの論文。
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — ライブラリ。
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — 簡潔なリファレンス。
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — クックブック＋エンコーディングリスト。
