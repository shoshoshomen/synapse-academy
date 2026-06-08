# BERT — Masked Language Modeling

> GPTは次の単語を予測する。BERTは欠けた単語を予測する。一文の違い——そして埋め込みに関するあらゆるものの半十年。


## 問題

2018年、あらゆるNLPタスク——感情分析、固有表現認識、QA、含意推論——が独自のラベル付きデータで一からモデルを学習していた。「英語を理解する」ファインチューニング可能な事前学習済みチェックポイントはなかった。ELMo（2018年）は双方向LSTMによる文脈的埋め込みを事前学習できることを示した；助けにはなったが汎化しなかった。

BERT（Devlin et al. 2018年）は問いかけた：Transformerエンコーダを取り、インターネット上のすべての文で学習し、両側のコンテキストから欠けた単語を予測することを強制したらどうか？そして下流タスクで1つのヘッドをファインチューニングする。パラメータ効率の高さは啓示だった。

結果：18ヶ月以内にBERTとその変形（RoBERTa、ALBERT、ELECTRA）が存在したすべてのNLPリーダーボードを制した。2020年までに地球上のすべての検索エンジン、コンテンツモデレーションパイプライン、セマンティック検索システムの内部にBERTがあった。

2026年においても、エンコーダのみのモデルは分類、検索、構造化抽出の適切なツールだ——デコーダよりもトークンあたり5〜10倍速く動作し、その埋め込みはすべての現代の検索スタックの骨格だ。ModernBERT（2024年12月）はFlash Attention + RoPE + GeGLUで8Kコンテキストにアーキテクチャを押し上げた。

## 概念

![Masked Language Modeling：トークンを選び、マスクし、元のトークンを予測する](../assets/bert-mlm.svg)

### 学習信号

文を取る：`the quick brown fox jumps over the lazy dog`。

トークンの15%をランダムにマスクする：

```
入力:  the [MASK] brown fox jumps [MASK] the lazy dog
ターゲット: the  quick brown fox jumps  over  the lazy dog
```

マスクされた位置の元のトークンを予測するよう学習する。エンコーダが双方向のため、位置1の`[MASK]`を予測するのに位置2+の`brown fox jumps`を使える。それはGPTにできないことだ。

### BERTのマスクルール

予測のために選ばれた15%のトークンのうち：

- 80%は`[MASK]`に置き換えられる。
- 10%はランダムなトークンに置き換えられる。
- 10%はそのまま残る。

なぜ常に`[MASK]`にしないのか？`[MASK]`は推論時に決して現れないからだ。マスクされた位置の100%で`[MASK]`を期待するようにモデルを学習すると、事前学習とファインチューニングの間に分布シフトが生じる。10%のランダム + 10%の変更なしがモデルを正直に保つ。

### Next Sentence Prediction（NSP）——そしてそれが削除された理由

元のBERTはNSPでも学習した：2文AとBが与えられたとき、BがAの後に続くかを予測する。RoBERTa（2019年）はそれをアブレーションして、NSPは助けにならず、むしろ悪化させることを示した。現代のエンコーダはそれをスキップする。

### 2026年に変わったこと：ModernBERT

2024年のModernBERT論文は2026年のプリミティブでブロックを再構築した：

| コンポーネント | 元のBERT（2018年） | ModernBERT（2024年） |
|-----------|----------------------|-------------------|
| 位置 | 学習済み絶対 | RoPE |
| 活性化 | GELU | GeGLU |
| 正規化 | LayerNorm | Pre-norm RMSNorm |
| アテンション | 完全密 | 交互局所（128）+ グローバル |
| コンテキスト長 | 512 | 8192 |
| トークナイザー | WordPiece | BPE |

そして2018年のスタックとは異なり、Flash-Attention-ネイティブだ。シーケンス長8KでのDeBERTa-v3より2〜3倍速く、より高いGLUEスコアを持つ。

### 2026年においてもエンコーダを選ぶユースケース

| タスク | エンコーダがデコーダを上回る理由 |
|------|---------------------------|
| 検索/セマンティック検索の埋め込み | 双方向コンテキスト = トークンあたりの埋め込み品質が高い |
| 分類（感情、意図、毒性） | 1回のフォワードパス；生成オーバーヘッドなし |
| 固有表現認識/トークンラベリング | 位置ごとの出力、ネイティブに双方向 |
| ゼロショット含意推論（NLI） | エンコーダの上の分類ヘッド |
| RAGのリランカー | クロスエンコーダスコアリング、LLMリランカーより10倍速い |

## 実装する

### ステップ1：マスキングロジック

`code/main.py`を参照。関数`create_mlm_batch`はトークンIDのリスト、語彙サイズ、マスク確率を受け取る。入力ID（マスク適用済み）とラベル（マスクされた位置のみ、それ以外は-100——PyTorchの無視インデックス規則）を返す。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: 元のまま残す
    return input_ids, labels
```

### ステップ2：ミニコーパスでMLM予測を実行する

20単語の語彙、200文のコーパスで2層エンコーダ + MLMヘッドを学習する。勾配なし——フォワードパスのサニティチェックを行う。完全な学習にはPyTorchが必要。

### ステップ3：マスクタイプを比較する

3方向ルールがどのようにモデルを`[MASK]`なしでも使えるようにするかを示す。マスクされていない文とマスクされた文で予測する。モデルは学習中に両パターンを見たので、どちらも合理的なトークン分布を生成するはずだ。

### ステップ4：ファインチューニングヘッド

MLMヘッドをトイ感情データセットの分類ヘッドに置き換える。ヘッドのみが学習し；エンコーダは凍結される。これはすべてのBERTアプリケーションが従うパターンだ。

## 使ってみる

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**埋め込みモデルはファインチューニングされたBERTだ。** `sentence-transformers`の`all-MiniLM-L6-v2`のようなモデルは対照損失で学習されたBERTだ。エンコーダは同じ。損失が変わっただけ。

**クロスエンコーダリランカーもファインチューニングされたBERTだ。** `[CLS] query [SEP] doc [SEP]`に対するペア分類。クエリとドキュメント間の双方向アテンションがクロスエンコーダのbiencoderに対する品質アドバンテージをもたらすものだ。

**2026年でBERTを選ばない場合。** 生成的なもの全般。エンコーダにはトークンを自己回帰的に生成する合理的な方法がない。また：小さなデコーダが品質を合わせられ、より柔軟性のある1B未満パラメータのもの（Phi-3-Mini、Qwen2-1.5B）。

## 成果物を出す

`outputs/skill-bert-finetuner.md`を参照。このスキルは、新しい分類または抽出タスクのためのBERTファインチューニング（バックボーン選択、ヘッド仕様、データ、評価、停止）をスコープする。

## 演習

1. **易.** `code/main.py`を実行し、10,000トークンにわたるマスク分布を出力せよ。〜15%が選択され、そのうち〜80%が`[MASK]`になることを確認せよ。
2. **中.** 単語全体のマスキングを実装せよ：単語がサブワードに分割される場合、すべてのサブワードをまとめてマスクするかしないかにせよ。500文のコーパスでMLM精度が改善するかどうかを測定せよ。
3. **難.** パブリックデータセットの10,000文で小さな（2層、d=64）BERTを学習せよ。SST-2感情のための`[CLS]`トークンをファインチューニングせよ。同じパラメータ数のデコーダのみベースラインと比較せよ——どちらが勝つか？

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| MLM | 「Masked Language Modeling」 | 学習信号：トークンの15%をランダムに`[MASK]`に置き換え、元のトークンを予測する。 |
| 双方向（Bidirectional） | 「両方向を見る」 | エンコーダアテンションにCausal Maskなし——すべての位置が他のすべての位置を見る。 |
| `[CLS]` | 「プーラートークン」 | すべてのシーケンスの先頭に追加される特別なトークン；その最終埋め込みが文レベルの表現として使われる。 |
| `[SEP]` | 「セグメント区切り」 | ペアのシーケンス（例：クエリ/ドキュメント、文A/B）を区切る。 |
| NSP | 「Next Sentence Prediction」 | BERTの第二の事前学習タスク；RoBERTaで無用と判明、2019年以降削除。 |
| ファインチューニング（Fine-tuning） | 「タスクへの適応」 | エンコーダをほぼ凍結したまま、下流タスクの小さなヘッドを学習する。 |
| クロスエンコーダ（Cross-encoder） | 「リランカー」 | クエリとドキュメントの両方を入力として受け取り、関連性スコアを出力するBERT。 |
| ModernBERT | 「2024年のリフレッシュ」 | RoPE、RMSNorm、GeGLU、交互ローカル/グローバルアテンション、8Kコンテキストで再構築されたエンコーダ。 |

## 参考資料

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 元の論文。
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — BERTを正しく学習する方法；NSPを廃止。
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — 置き換えトークン検出がマッチした計算でMLMを上回る。
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — ModernBERT論文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — 標準エンコーダリファレンス。
