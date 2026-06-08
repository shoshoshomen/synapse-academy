# テキスト向けCNNとRNN

> 畳み込みはn-gramを学習する。再帰は記憶する。両方ともAttentionに取って代わられた。どちらも制約のあるハードウェアでまだ重要だ。


## 問題の背景

TF-IDFとWord2Vecは語順を無視したフラットなベクトルを生成した。それらの上に構築された分類器は「dog bites man（犬が人を噛む）」と「man bites dog（人が犬を噛む）」を区別できなかった。語順が時々シグナルを運ぶ。

2つのアーキテクチャファミリーがTransformerが登場する前にそのギャップを埋めた。

**テキスト向け畳み込みネット（TextCNN）。** 単語埋め込みのシーケンスに1D畳み込みを適用する。幅3のフィルターは学習可能なトリグラム検出器だ：3つの単語にまたがってスコアを出力する。複数の幅（2、3、4、5）を積み重ねてマルチスケールパターンを検出する。固定サイズ表現にMax-poolingする。フラットで並列、速い。

**再帰ネット（RNN、LSTM、GRU）。** トークンを一つずつ処理し、情報を前に運ぶ隠れ状態を維持する。逐次的でメモリを持ち、入力長が柔軟だ。2014年から2017年まで系列モデリングを支配し、そしてAttentionが来た。

このレッスンでは両方を構築し、次にAttentionを動機付けた失敗を示す。

## 概念

**TextCNN**（Kim, 2014）。トークンが埋め込まれる。幅`k`の1D畳み込みが埋め込みの連続する`k`-gramにフィルターをスライドさせ、特徴マップを生成する。そのマップのグローバルMax-poolingが最強の活性化を選ぶ。複数のフィルター幅からMax-poolingされた出力を連結する。分類ヘッドに送る。

なぜ機能するか。フィルターは学習可能なn-gramだ。Max-poolingは位置不変なので、「not good」はレビューの最初でも途中でも同じ特徴を発火する。100フィルターを持つ3つのフィルター幅で300の学習済みn-gram検出器が得られる。訓練は並列だ。逐次依存関係はない。

**RNN。** 各タイムステップ`t`で、隠れ状態`h_t = f(W * x_t + U * h_{t-1} + b)`。`W`、`U`、`b`を時間を通じて共有する。タイムステップ`T`の隠れ状態は全プレフィックスの要約だ。分類のためには`h_1 ... h_T`にプールする（最大、平均、または最後）。

プレーンRNNは勾配消失に苦しむ。**LSTM**はゲートを追加して何を忘れ、何を格納し、何を出力するかを決定し、長いシーケンスを通じて勾配を安定させる。**GRU**はLSTMを2つのゲートに簡略化する。より少ないパラメーターで同様のパフォーマンスを発揮する。

**双方向RNN**は1つのRNNを前向きに、もう1つを後ろ向きに実行し、隠れ状態を連結する。全トークンの表現が左右両方のコンテキストを見る。タグ付けタスクに不可欠だ。

## 実装する

### ステップ1: PyTorchでのTextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)`は`[batch, seq_len, embed_dim]`を`[batch, embed_dim, seq_len]`に変形する。`nn.Conv1d`が中間軸をチャンネルとして扱うからだ。Poolingされた出力は入力長にかかわらず固定サイズだ。

### ステップ2: LSTM分類器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

最後の状態プールではなく、シーケンスにMax-poolingする。分類のためには、Max-poolingは通常最後の隠れ状態を取るよりも優れている。長いシーケンスの最後の情報が最後の状態を支配する傾向があるからだ。

### ステップ3: 勾配消失のデモ（直感）

ゲートのないプレーンRNNは長距離依存関係を学習できない。トイタスクを考える：シーケンスのどこかにトークン`A`が現れたかどうかを予測する。`A`が位置1にあってシーケンスが100トークン長の場合、損失からの勾配は再帰重みの99回の乗算を通じて逆流しなければならない。重みが1未満なら勾配は消失する。1より大きければ爆発する。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# weight=0.9で100ステップの場合:
#   0.9 ^ 100 ≈ 2.7e-5
# ステップ100からステップ1への勾配は実質的にゼロだ。
```

LSTMはネットワークを加算的な相互作用だけで流れる**セル状態**でこれを修正する（忘却ゲートが乗法的にスケールするが、勾配は「ハイウェイ」に沿って流れ続ける）。GRUはより少ないパラメーターで同様のことをする。両方とも100以上のステップのシーケンスで安定した訓練を提供する。

### ステップ4: それでもまだ十分でなかった理由

LSTMを使っても3つの問題が残った。

1. **逐次ボトルネック。** 長さ1000のシーケンスでRNNを訓練するには1000の逐次前方/後方ステップが必要だ。時間を跨いで並列化できない。
2. **エンコーダー-デコーダー設定での固定サイズコンテキストベクトル。** デコーダーはエンコーダーの最終隠れ状態だけを見て、入力全体を圧縮している。長い入力は詳細を失う。レッスン09でこれを直接扱う。
3. **遠距離依存関係の精度上限。** LSTMはプレーンRNNより優れているが、200以上のステップを跨いで特定の情報を伝播させるのに苦労する。

Attentionは3つすべてを解決した。TransformerはRecurrenceを完全に廃棄した。レッスン10が転換点だ。

## 使ってみる

PyTorchの`nn.LSTM`、`nn.GRU`、`nn.Conv1d`は本番対応済みだ。訓練コードは標準的だ。

Hugging Faceは入力レイヤーとして組み込める事前訓練済み埋め込みを提供する：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

制約に合う場合のチェックリスト。

- **エッジ/デバイス上推論。** GloVe埋め込みを使ったTextCNNはTransformerより10〜100倍小さい。デプロイ先がスマートフォンなら、これがスタックだ。
- **ストリーミング/オンライン分類。** RNNは一度に1トークン処理する。Transformerはシーケンス全体が必要だ。リアルタイムの着信テキストにはLSTMがまだ勝つ。
- **ベースライン用タイニーモデル。** 新しいタスクでの高速イテレーション。TextCNNをCPUで5分で訓練する。
- **限られたデータでのシーケンスラベリング。** BiLSTM-CRF（レッスン06）は1k〜10kのラベル付き文に対する本番グレードのNERアーキテクチャだ。

それ以外はすべてTransformerに行く。

## 成果物を出す

`outputs/prompt-text-encoder-picker.md`として保存する：

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## 演習

1. **易しい。** 3クラスのトイデータセット（データは自分で考案）でTextCNNを訓練する。フィルター幅（2、3、4）が平均F1で単一幅（3）より優れていることを確認する。
2. **普通。** LSTM分類器でMax-pooling、平均pooling、最後の状態poolingを実装する。小さなデータセットで比較する。どのpoolingが勝つかを文書化し、なぜかを仮説を立てる。
3. **難しい。** BiLSTM-CRF NERタガーを構築する（レッスン06とこのレッスンを組み合わせる）。CoNLL-2003で訓練する。レッスン06のCRF単体ベースラインとBERTファインチューニングと比較する。訓練時間、メモリ、F1を報告する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------------------|
| TextCNN | テキスト向けCNN | 単語埋め込みにグローバルMax-poolingを付けた1D畳み込みのスタック。Kim（2014）。 |
| RNN | 再帰ネット | 各タイムステップで更新される隠れ状態：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | ゲート付きRNN | 入力/忘却/出力ゲート+セル状態を追加。長いシーケンスを通じて安定して訓練する。 |
| GRU | シンプルなLSTM | 3つでなく2つのゲート。同様の精度、より少ないパラメーター。 |
| 双方向 | 両方向 | 前向き+後ろ向きRNNを連結。各トークンがコンテキストの両側を見る。 |
| 勾配消失 | 訓練シグナルが消える | プレーンRNNの<1重みの繰り返し乗算により、初期ステップの勾配が実質的にゼロになる。 |

## 参考資料

- [Kim, Y. (2014). 文分類のための畳み込みニューラルネットワーク](https://arxiv.org/abs/1408.5882) — TextCNN論文。8ページ。読みやすい。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM論文。意外と明快。
- [Olah, C. (2015). LSTMネットワークを理解する](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — LSTMを誰にでもアクセス可能にしたダイアグラム。
