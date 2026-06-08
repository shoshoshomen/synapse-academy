# Transformerをゼロから作る — カプストーン

> 13のレッスン。1つのモデル。ショートカットなし。


## 問題

すべての論文を読んだ。Attention、Multi-Head分割、位置エンコーディング、エンコーダとデコーダブロック、BERTとGPTの損失、MoE、KVキャッシュを実装した。今度はそれらを本物のタスクで一緒に機能させる。

カプストーン：文字レベルの言語モデリングタスクに小さなデコーダのみのTransformerをエンドツーエンドで学習する。シェイクスピアを読む。新しいシェイクスピアを生成する。ラップトップで10分以内に学習できる程度に小さい。より大きなデータセットと長い学習に差し替えれば本物のLMになる程度には正確だ。

これはコースの「nanoGPT」だ。オリジナルではない——Karpathyの2023年nanoGPTチュートリアルが、すべての学生が少なくとも一度は書く参考実装だ。その形を借りて、これまでカバーした内容を中心に作り直す。

## 概念

![Transformerをゼロから作るブロック図](../assets/capstone.svg)

アノテーション付きのアーキテクチャ：

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### 成果物

- `GPTConfig` — すべてのハイパーパラメータを設定する1か所。
- `MultiHeadAttention` — 因果的、バッチ処理済み、オプションのFlash風パスウェイ（PyTorchの`scaled_dot_product_attention`）付き。
- `SwiGLUFFN` — モダンなFFN。
- `Block` — Pre-norm、残差ラップされたAttention + FFN。
- `GPT` — 埋め込み、スタックされたブロック、LMヘッド、generate()。
- AdamW、コサインLR、勾配クリッピング付き学習ループ。
- シェイクスピアテキスト上の文字レベルトークナイザー。

### 含まないもの

- RoPE — レッスン04で概念的に実装済み。ここでは簡略化のために学習済み位置埋め込みを使う。演習でRoPEへの差し替えを求める。
- 生成中のKVキャッシュ — 各生成ステップはフルプレフィックスに対してAttentionを再計算する。遅いが単純だ。演習でKVキャッシュの追加を求める。
- Flash Attention — PyTorch 2.0+は入力が一致する場合に自動的にディスパッチする；`F.scaled_dot_product_attention`を使う。
- MoE — ブロックごとに単一のFFN。レッスン11でMoEを見た。

### 目標メトリクス

Mac M2ラップトップで、4レイヤー、4ヘッド、d_model=128のGPTを`tinyshakespeare.txt`で2,000ステップ学習：

- 学習損失は約6分で〜4.2（ランダム）から〜1.5に収束する。
- サンプリング出力はシェイクスピア風に見える：古風な言葉、改行、"ROMEO:"のような固有名詞が出現する。
- バリデーション損失（テキストの最後の10%のホールドアウト）は学習損失を密接に追跡する；このサイズ/予算ではオーバーフィットなし。

## 実装する

このレッスンはPyTorchを使う。`torch`をインストール（CPUビルドで問題ない）。`code/main.py`を参照。スクリプトが処理するもの：

- `tinyshakespeare.txt`が欠落している場合のダウンロード（またはローカルコピーの読み取り）。
- バイトレベルの文字トークナイザー。
- 90/10での学習/バリデーション分割。
- サポートされるハードウェアでのbf16オートキャスト付き学習ループ。
- 学習完了後のサンプリング。

### ステップ1：データ

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65個のユニークな文字。小さな語彙。4バイトのvocab_sizeに収まる。BPEなし、トークナイザーの手間なし。

### ステップ2：モデル

`code/main.py`を参照。ブロックはレッスン05のテキストブック通り——Pre-norm、RMSNorm、SwiGLU、因果的MHA。4/4/128でのパラメータ数：約800K。

### ステップ3：学習ループ

長さ256のトークンウィンドウのランダムバッチを取得する。フォワード。シフト-バイ-ワンのクロスエントロピー。バックワード。AdamWステップ。ログ。繰り返す。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### ステップ4：サンプリング

プロンプトが与えられ、繰り返しフォワードし、top-pロジットからサンプリングし、追加し、継続する。500トークン後に停止する。

### ステップ5：出力を読む

2,000ステップ後：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

シェイクスピアではない。しかしシェイクスピア風だ。約800Kパラメータとラップトップ6分での明確な成果だ。

## 使ってみる

このカプストーンは参照アーキテクチャだ。本物のものに発展させる3つの拡張：

1. **トークナイザーを差し替える。** BPEを使う（例：`tiktoken.get_encoding("cl100k_base")`）。語彙サイズは65から〜50,000に跳ね上がる。補償のためにモデル容量をスケールアップする必要がある。
2. **より大きいコーパスで学習する。** `OpenWebText`や`fineweb-edu`（HuggingFace）を使う。A100シングルで10Bトークンは125Mパラメータのために約24時間かかる。
3. **RoPE + KVキャッシュ + Flash Attentionを追加する。** 以下の演習が各手順を案内する。

これは125MパラメータのGPTになり、流暢な英語を生成する。フロンティアモデルではない。しかし同じコードパス——ただより大きく——が、KarpathyとEleutherAIとAllen Instituteが2026年に研究チェックポイントを学習するために使うものだ。

## 成果物を出す

`outputs/skill-transformer-review.md`を参照。このスキルは、過去13のレッスンすべてにわたって正確性についてゼロから作ったTransformer実装をレビューする。

## 演習

1. **易.** `code/main.py`を実行せよ。学習済みモデルの最終ステップのバリデーション損失が2.0未満であることを確認せよ。`max_steps`を2,000から5,000に変更する——バリデーション損失は改善し続けるか？
2. **中.** 学習済み位置埋め込みをRoPEに置き換えよ。`MultiHeadAttention`内でQとKに回転を適用せよ。学習してバリデーション損失が少なくとも同程度であることを確認せよ。
3. **中.** サンプリングループにKVキャッシュを実装せよ。キャッシュありとなしで500トークンを生成せよ。ラップトップでの実時間は5〜20倍向上するはずだ。
4. **難.** 次の次のトークンを予測する2番目のヘッドをモデルに追加せよ（MTP — DeepSeek-V3のMulti-Token Prediction）。合同学習せよ。役立つか？
5. **難.** ブロックごとの単一FFNを4エキスパートMoEに置き換えよ。ルーター + top-2ルーティング。マッチしたアクティブパラメータでバリデーション損失がどう変わるかを確認せよ。

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| nanoGPT | 「Karpathyのチュートリアルリポ」 | 最小限のデコーダのみのTransformer学習コード、約300 LOC；標準参考実装。 |
| tinyshakespeare | 「標準的なおもちゃコーパス」 | 約1.1 MBのテキスト；2015年以来すべての文字-LMチュートリアルで使われている。 |
| 結合埋め込み（Tied embeddings） | 「入出力行列を共有」 | LMヘッドの重み = トークン埋め込み行列の転置；パラメータを節約し、品質を向上させる。 |
| bf16オートキャスト（bf16 autocast） | 「学習精度のトリック」 | bf16でフォワード/バックワードを実行し、オプティマイザーの状態をfp32に保つ；2021年以来の標準。 |
| 勾配クリッピング（Gradient clipping） | 「スパイクを止める」 | グローバルな勾配ノルムを1.0でキャップ；学習の爆発を防ぐ。 |
| コサインLRスケジュール（Cosine LR schedule） | 「2020年以降のデフォルト」 | LRが線形に増加（ウォームアップ）し、ピークの10%までコサイン形状に減衰する。 |
| MFU | 「モデルFLOP利用率」 | 達成FLOPs / 理論ピーク；2026年ではdenseで40%、MoEで30%が良好。 |
| バリデーション損失（Val loss） | 「ホールドアウト損失」 | モデルが見たことのないデータのクロスエントロピー；オーバーフィット検出器。 |

## 参考資料

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — クラシックなアノテーション付き実装。
