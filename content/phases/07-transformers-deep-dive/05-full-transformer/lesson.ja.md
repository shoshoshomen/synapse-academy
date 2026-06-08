# 完全なTransformer — エンコーダ + デコーダ

> アテンションはスターだ。その他のすべて——残差接続、正規化、フィードフォワード、クロスアテンション——は深く積み重ねられるようにする足場だ。


## 問題

単一のアテンション層は特徴抽出器であってモデルではない。1層あたり1つの行列積では言語に対して十分な容量がない。深さが必要だ——そして深さは適切な配管なしでは壊れる。

2017年のVaswani論文は、1つのアテンション層をスタック可能なブロックに変えた6つの設計決定をパッケージ化した。それ以来のすべてのTransformer——エンコーダのみ（BERT）、デコーダのみ（GPT）、エンコーダ-デコーダ（T5）——は同じ骨格を継承している。2026年にはブロックが改良されている（RMSNorm、SwiGLU、Pre-norm、RoPE）が、骨格は同一だ。

このレッスンは骨格だ。次のレッスンではそれを専門化する——06はエンコーダ用、07はデコーダ用、08はエンコーダ-デコーダ用。

## 概念

![エンコーダとデコーダのブロック内部、配線済み](../assets/full-transformer.svg)

### 6つの部品

1. **埋め込み + 位置信号。** トークン → ベクトル。位置はRoPE（現代）またはSinusoidal（古典）で注入。
2. **Self-Attention。** すべての位置が他のすべての位置にアテンション。デコーダではマスク済み。
3. **フィードフォワードネットワーク（FFN）。** 位置ごとの2層MLP：`W_2 · activation(W_1 · x)`。デフォルトの拡張比は4倍。
4. **残差接続（Residual connection）。** `x + sublayer(x)`。これなしでは〜6層を超えると勾配が消失する。
5. **Layer normalization。** `LayerNorm`または`RMSNorm`（現代）。残差ストリームを安定させる。
6. **クロスアテンション（デコーダのみ）。** クエリはデコーダから、キーとバリューはエンコーダ出力から来る。

あるベクトルが1つのブロックをどう流れるか観察する：アテンションは位置をまたいで混合し、残差接続がそれを前に運び、FFNが変換し、normがストリームを安定に保つ。

```figure
transformer-block
```

### エンコーダブロック（BERT、T5エンコーダで使用）

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

エンコーダは双方向だ。マスクなし。すべての位置がすべての位置を見る。

### デコーダブロック（GPT、T5デコーダで使用）

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

デコーダはブロックごとに3つのサブ層を持つ。中間のもの——クロスアテンション——だけがエンコーダからデコーダへ情報が流れる唯一の場所だ。純粋なデコーダのみのアーキテクチャ（GPT）では、クロスアテンションは省略され、マスク済みSelf-Attention + FFNだけになる。

### Pre-normとPost-norm

元の論文：`x + sublayer(LN(x))`対`LN(x + sublayer(x))`。Post-normは2019年頃に不人気になった——慎重なウォームアップなしに深く学習するのが難しい。Pre-norm（`LN`はサブ層の*前*）が2026年のデフォルト：Llama、Qwen、GPT-3+、Mistralはすべてそれを使用する。

### 2026年のモダン化されたブロック

Vaswani 2017はLayerNorm + ReLUを搭載していた。現代のスタックは両方を置き換えた。本番ブロックの実際の姿：

| コンポーネント | 2017 | 2026 |
|-----------|------|------|
| 正規化 | LayerNorm | RMSNorm |
| FFN活性化 | ReLU | SwiGLU |
| FFN拡張 | 4倍 | 2.6倍（SwiGLUは3つの行列を使用、合計パラメータは一致） |
| 位置 | 絶対Sinusoidal | RoPE |
| アテンション | 完全MHA | GQA（またはMLA） |
| バイアス項 | あり | なし |

RMSNormはLayerNormの平均センタリングを除く（演算が1つ少ない）。これにより計算が節約され、経験的には少なくとも同程度に安定している。SwiGLU（`Swish(W1 x) ⊙ W3 x`）はLlama、PaLM、Qwenの論文でReLU/GELU FFNを一貫して〜0.5ポイントのpplで上回る。

### パラメータ数

`d_model = d`、FFN拡張`r`の1ブロックに対して：

- MHA：`4 · d²`（Q、K、V、O射影）
- FFN（SwiGLU）：`3 · d · (r · d)` ≈ `3rd²`
- Norm：無視できる

`d = 4096、r = 2.6、layers = 32`（大まかにLlama 3 8B）で、合計：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B パラメータ/層 × 32 ≈ 7B`（埋め込みとヘッドを除く）。公開されている数値と一致する。

## 実装する

### ステップ1：構築ブロック

レッスン03のミニ`Matrix`クラスを使用（このファイルに独立してコピー）：

- `layer_norm(x, eps=1e-5)` — 平均を引き、標準偏差で割る。
- `rms_norm(x, eps=1e-6)` — RMSで割る。平均を引かない。
- `gelu(x)`と`silu(x) * W3 x`（SwiGLU）。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)`と`decoder_block(x, enc_out, params)`。

完全な配線は`code/main.py`を参照。

### ステップ2：2層エンコーダと2層デコーダを配線する

積み重ねる。エンコーダ出力をすべてのデコーダのクロスアテンションに渡す。出力射影の前に最後のLNを追加する。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### ステップ3：トイ例でフォワードを実行する

6トークンのソースと5トークンのターゲットを通じて供給する。出力の形状が`(5, vocab)`であることを確認する。学習なし——このレッスンはアーキテクチャについてで、損失についてではない。

### ステップ4：RMSNorm + SwiGLUに入れ替える

LayerNormとReLU-FFNをRMSNormとSwiGLUに置き換える。形状がまだ一致することを確認する。これが関数の置き換え1つでの2026年のモダン化だ。

## 使ってみる

PyTorch/TFのリファレンス実装：`nn.TransformerEncoderLayer`、`nn.TransformerDecoderLayer`。しかし2026年の本番コードのほとんどは独自のブロックを作っている：

- Flash AttentionはアテンションOの内部で呼び出され、`nn.MultiheadAttention`経由ではない。
- GQA / MLAはstdlibリファレンスにない。
- RoPE、RMSNorm、SwiGLUはPyTorchのデフォルトではない。

HFの`transformers`には読む価値のあるクリーンなリファレンスブロックがある：`modeling_llama.py`は2026年のデコーダのみブロックの標準だ。約500行で、一度通読する価値がある。

**エンコーダ対デコーダ対エンコーダ-デコーダ——いつ何を選ぶか：**

| ニーズ | 選択 | 例 |
|------|------|---------|
| 分類、埋め込み、テキストに対するQA | エンコーダのみ | BERT、DeBERTa、ModernBERT |
| テキスト生成、チャット、コード、推論 | デコーダのみ | GPT、Llama、Claude、Qwen |
| 構造化入力 → 構造化出力（翻訳、要約） | エンコーダ-デコーダ | T5、BART、Whisper |

デコーダのみが言語で勝ったのは、最もきれいにスケールし、理解と生成の両方を処理するからだ。エンコーダ-デコーダは入力に明確な「ソースシーケンス」のアイデンティティがある場合（翻訳、音声認識、構造化タスク）に依然として最善だ。

## 成果物を出す

`outputs/skill-transformer-block-reviewer.md`を参照。このスキルは新しいTransformerブロックの実装を2026年のデフォルトと照らし合わせてレビューし、欠けている部分（Pre-norm、RoPE、RMSNorm、GQA、FFN拡張比）をフラグする。

## 演習

1. **易.** `d_model=512、n_heads=8、ffn_expansion=4、swiglu=True`でエンコーダブロックのパラメータ数を計算せよ。ブロックを実装して`sum(p.numel() for p in block.parameters())`で検証せよ。
2. **中.** Post-normからPre-normに切り替えよ。両方を初期化し、ランダム入力で12層積み重ねた後の活性化normを測定せよ。Post-normの活性化は爆発するはずで、Pre-normのものは境界内に収まるはずだ。
3. **難.** トイコピータスク（`x`を逆順にコピー）で4層エンコーダ-デコーダを実装せよ。100ステップ学習せよ。損失を報告せよ。RMSNorm + SwiGLU + RoPEに入れ替える——損失は下がるか？

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| ブロック（Block） | 「1つのTransformer層」 | 残差接続でラップされたnorm + アテンション + norm + FFNのスタック。 |
| 残差接続（Residual） | 「スキップ接続」 | `x + f(x)`出力；深いスタックを通じた勾配フローを可能にする。 |
| Pre-norm | 「後じゃなく前に正規化」 | 現代：`x + sublayer(LN(x))`。ウォームアップなしで深く学習できる。 |
| RMSNorm | 「平均なしのLayerNorm」 | RMSで割る；演算1つ少ない、同じ経験的安定性。 |
| SwiGLU | 「みんなが移行したFFN」 | `Swish(W1 x) ⊙ W3 x → W2`。LMのpplでReLU/GELUを上回る。 |
| クロスアテンション（Cross-attention） | 「デコーダがエンコーダを見る方法」 | デコーダからQ、エンコーダ出力からK/VのMHA。 |
| FFN拡張（FFN expansion） | 「中間MLPの幅」 | 隠れサイズとd_modelの比、通常4（LayerNorm）または2.6（SwiGLU）。 |
| バイアスなし（Bias-free） | 「+b項を削除」 | 現代のスタックは線形層のバイアスを省略；わずかなppl改善、小さいモデル。 |

## 参考資料

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 元のブロック仕様。
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — Pre-normが深くPost-normを上回る理由。
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU論文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 2026年の標準デコーダのみブロック。
