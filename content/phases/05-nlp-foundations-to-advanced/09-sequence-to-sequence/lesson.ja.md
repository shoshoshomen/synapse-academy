# Sequence-to-Sequenceモデル

> 翻訳者のふりをする2つのRNN。それらが当たるボトルネックがAttentionが存在する理由だ。


## 問題の背景

分類は可変長シーケンスを単一のラベルにマッピングする。翻訳は可変長シーケンスを別の可変長シーケンスにマッピングする。入力と出力は異なる語彙に存在し、おそらく異なる言語で、長さの一致の保証はない。

seq2seqアーキテクチャ（Sutskever, Vinyals, Le, 2014）は意図的にシンプルなレシピでこれを解決した。2つのRNN。一つはソース文を読んで固定サイズのコンテキストベクトルを生成する。もう一つはそのベクトルを読んでターゲット文をトークンごとに生成する。レッスン08で書いたのと同じコードを、異なる方法で組み合わせたものだ。

これを学ぶ価値がある2つの理由がある。第一に、コンテキストベクトルのボトルネックはNLPで最も教育的に有用な失敗だ。AttentionとTransformerが得意とすることすべてを動機付けている。第二に、訓練レシピ（teacher forcing、スケジュールされたサンプリング、推論時のビームサーチ）はLLMを含むすべての現代の生成システムに今も適用される。

## 概念

**エンコーダー。** ソース文を読むRNN。その最終隠れ状態が**コンテキストベクトル**だ——入力全体の固定サイズの要約。ソース以外は何も失わない、とされる。

**デコーダー。** コンテキストベクトルから初期化された別のRNN。各ステップで前に生成されたトークンを入力として受け取り、ターゲット語彙の分布を生成する。次のトークンを選択するためにサンプリングまたはargmaxを使う。フィードバックする。`<EOS>`トークンが生成されるか最大長に達するまで繰り返す。

**訓練：** デコーダーの各ステップでの交差エントロピー損失をシーケンスにわたって合計する。両方のネットワークを通じた標準的な時間逆伝播。

**Teacher forcing。** 訓練中、ステップ`t`でのデコーダーの入力はデコーダー自身の前の予測ではなく、位置`t-1`の*正解*トークンだ。これは訓練を安定させる。これなしでは初期の間違いが連鎖し、モデルは決して学習しない。推論時にはモデル自身の予測を使わなければならないため、常に訓練/推論の分布ギャップがある。そのギャップを**exposure bias**と呼ぶ。

**ボトルネック。** エンコーダーがソースについて学んだすべてがその一つのコンテキストベクトルに押し込まれなければならない。長い文は詳細を失う。まれな単語がぼやける。語順の変換（chat noir vs. black cat）は計算ではなく記憶しなければならない。

Attention（レッスン10）はデコーダーが最後の状態だけでなく*すべての*エンコーダー隠れ状態を見られるようにすることでこれを修正する。それがすべての売り文句だ。

## 実装する

### ステップ1: エンコーダー

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs`は形状`[batch, seq_len, hidden_dim]`だ——入力位置ごとに一つの隠れ状態。`hidden`は形状`[1, batch, hidden_dim]`だ——最後のステップ。レッスン08では「分類のために出力にプール」と言った。ここでは最後の隠れ状態をコンテキストベクトルとして保持し、ステップごとの出力は無視する。

### ステップ2: デコーダー

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

デコーダーは一度に一ステップ呼ばれる。入力：単一トークンのバッチと現在の隠れ状態。出力：次のトークンの語彙ロジットと更新された隠れ状態。

### ステップ3: teacher forcingを使った訓練ループ

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

名前を付ける価値のある2つのノブ。`ignore_index=0`はパディングトークンの損失をスキップする。`teacher_forcing_ratio`は各ステップで真のトークンとモデルの予測のどちらを使うかの確率だ。1.0（完全なteacher forcing）から始めて、exposure biasのギャップを埋めるために訓練中に約0.5までアニールする。

### ステップ4: 推論ループ（貪欲）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

貪欲デコードは各ステップで最高確率のトークンを選択する。彷徨うことがある：一つのトークンにコミットしたら、取り消せない。**ビームサーチ**は上位`k`個の部分シーケンスを保持し続け、最後に最高スコアの完全なものを選ぶ。ビーム幅3〜5が標準だ。

### ステップ5: ボトルネックの実証

モデルをトイコピータスクで訓練する：ソース`[a, b, c, d, e]`、ターゲット`[a, b, c, d, e]`。シーケンス長を増やす。精度を観察する。

```
seq_len=5   コピー精度: 98%
seq_len=10  コピー精度: 91%
seq_len=20  コピー精度: 62%
seq_len=40  コピー精度: 23%
```

単一のGRU隠れ状態は40トークンの入力を損失なく記憶できない。情報は各エンコーダーステップに存在しているが、デコーダーは最後の状態しか見ない。Attentionはこれを直接修正する。

## 使ってみる

PyTorchには`nn.Transformer`と`nn.LSTM`ベースのseq2seqテンプレートがある。Hugging Faceの`transformers`ライブラリは数十億トークンで訓練された完全なエンコーダー-デコーダーモデル（BART、T5、mBART、NLLB）を提供する。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

現代のエンコーダー-デコーダーはRNNからTransformerに移行した。高レベルの形（エンコーダー、デコーダー、トークンごとの生成）は2014年のseq2seq論文と同一だ。各ブロック内の仕組みが異なる。

### RNNベースのseq2seqをまだ使う場面

新しいプロジェクトではほぼない。具体的な例外：

- 制限されたメモリで一度に1トークン入力を消費するストリーミング翻訳。
- Transformerのメモリコストが大きすぎるデバイス上のテキスト生成。
- 教育目的。エンコーダー-デコーダーのボトルネックを理解することがTransformerが勝った理由を理解する最速の道だ。

### Exposure biasとその対策

- **スケジュールされたサンプリング。** モデルが自分の間違いから回復することを学習するように、訓練中にteacher forcingの比率をアニールする。
- **最小リスク訓練。** トークンレベルの交差エントロピーの代わりに文レベルのBLEUスコアで訓練する。実際に求めるものにより近い。
- **強化学習ファインチューニング。** シーケンス生成器を指標で報酬付けする。現代のLLMのRLHFで使われる。

3つともTransformerベースの生成にまだ適用される。

## 成果物を出す

`outputs/prompt-seq2seq-design.md`として保存する：

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## 演習

1. **易しい。** トイコピータスクを実装する。ターゲットがソースと等しい入出力ペアでGRU seq2seqを訓練する。長さ5、10、20での精度を測定する。ボトルネックを再現する。
2. **普通。** ビーム幅3のビームサーチデコードを追加する。小さな並列コーパスで貪欲デコードに対してBLEUを測定する。ビームサーチが勝つ場所（通常最後のトークン）と差がない場所を文書化する。
3. **難しい。** `facebook/bart-base`を1万ペアの言い換えデータセットでファインチューニングする。ファインチューニングされたモデルのビーム4出力をベースモデルのホールドアウト入力での出力と比較する。BLEUを報告し、10個の定性的な例を選ぶ。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------------------|
| エンコーダー | 入力RNN | ソースを読む。ステップごとの隠れ状態と最終コンテキストベクトルを生成する。 |
| デコーダー | 出力RNN | コンテキストベクトルから初期化。ターゲットトークンを一度に一つ生成する。 |
| コンテキストベクトル | 要約 | エンコーダーの最終隠れ状態。固定サイズ。Attentionが解決するボトルネック。 |
| Teacher forcing | 真のトークンを使う | 訓練時に前の正解トークンをフィードする。学習を安定させる。 |
| Exposure bias | 訓練/テストギャップ | 真のトークンで訓練されたモデルは自分の間違いから回復する練習をしていない。 |
| ビームサーチ | より良いデコード | 貪欲にコミットする代わりに各ステップで上位kの部分シーケンスを保持する。 |

## 参考資料

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — オリジナルのseq2seq論文。4ページ。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — GRUとエンコーダー-デコーダーフレームワークを導入。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — Attention論文。このレッスンの直後に読む。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — 構築可能なseq2seq+attentionコード。
