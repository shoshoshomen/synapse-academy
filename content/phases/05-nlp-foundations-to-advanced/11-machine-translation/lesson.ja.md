# 機械翻訳

> 翻訳は30年間NLP研究に資金を提供してきたタスクで、今も提供し続けている。


## 問題の背景

モデルはある言語の文を読み、別の言語の文を生成する。長さは変わる。語順は変わる。一部のソース単語は複数のターゲット単語にマッピングされ、その逆もある。慣用句は一対一のマッピングを拒む。フランス語で「I miss you」は「tu me manques」で、文字通りには「あなたが私に欠けている」だ。単語レベルのアライメントはそこでは成り立たない。

機械翻訳は、NLPがエンコーダー-デコーダー、Attention、Transformer、そして最終的にLLMパラダイム全体を発明することを強いたタスクだ。翻訳品質が測定可能で、人間とマシンのギャップが頑固だったため、すべての前進が達成された。

このレッスンでは歴史の授業をスキップして2026年の動作パイプラインを教える：事前訓練済み多言語エンコーダー-デコーダー（NLLB-200またはmBART）、サブワードトークン化、ビームサーチ、BLEUとchrFの評価、そして本番に未発見のまま出荷される少数の失敗モード。

## 概念

現代のMTは並列テキストで訓練されたTransformerエンコーダー-デコーダーだ。エンコーダーはその言語のトークン化でソースを読む。デコーダーはエンコーダーの出力をクロスAttention（レッスン10）を通じて使いながら、サブワードごとにターゲットを生成する。デコードはビームサーチを使って貪欲デコードの罠を避ける。出力は逆トークン化、逆真のケース化されて参照に対してスコアリングされる。

3つの運用上の選択が実世界のMT品質を左右する。

- **トークナイザー。** 混合言語コーパスで訓練されたSentencePiece BPE。言語を跨いだ共有語彙がNLLBでのゼロショットペアを可能にする。
- **モデルサイズ。** NLLB-200 distilled 600Mはラップトップに収まる。NLLB-200 3.3Bが公開されている本番デフォルト。54.5Bが研究の上限。
- **デコード。** 一般的なコンテンツにはビーム幅4〜5。短すぎる出力を避けるための長さペナルティ。用語の一貫性が必要な場合は制約付きデコード。

## 実装する

### ステップ1: 事前訓練済みMTの呼び出し

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

3つのことが重要だ。`src_lang`はトークナイザーにどのスクリプトとセグメンテーションを適用するかを伝える。`forced_bos_token_id`はデコーダーにどの言語を生成するかを伝える。両方ともNLLB固有のトリックで、mBARTとM2M-100は独自の規約を持ち互換性はない。

### ステップ2: BLEUとchrF

BLEUは出力と参照のn-gramオーバーラップを測定する。4つの参照n-gramサイズ（1〜4）、精度の幾何平均、短すぎる出力のための簡潔さペナルティ。スコアは[0, 100]の範囲。一般的に使われる。解釈が難しい：30 BLEUは「使用可能」、40は「良い」、50は「例外的」、1 BLEU未満の差はノイズだ。

chrFは文字レベルのFスコアを測定する。BLEUがマッチを過少カウントする形態的に豊かな言語により敏感だ。BLEUと並んで報告されることが多い。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

常に`sacrebleu`を使う。トークン化を正規化してスコアが論文間で比較可能になる。自分でBLEU計算をすることが誤解を招くベンチマークが生まれる方法だ。

### 3層の評価階層（2026年）

現代のMT評価は3つの相補的な指標ファミリーを使う。少なくとも2つで出荷する。

- **ヒューリスティック**（BLEU、chrF）。速く、参照ベースで、解釈可能で、言い換えに鈍感。レガシーな比較と回帰検出に使う。
- **学習済み**（COMET、BLEURT、BERTScore）。人間の判断で訓練されたニューラルモデル。翻訳とソースおよび参照の意味的類似度を比較する。COMETは2023年以降のMT研究との最高の相関を持ち、品質が重要な場合の2026年の本番デフォルトだ。
- **LLM-as-judge**（参照なし）。大規模モデルに流暢さ、妥当性、トーン、文化的適切さについて翻訳をスコアリングするよう促す。ルーブリックがよく設計されていれば、GPT-4-as-judgeは人間の一致に約80%の確率でマッチする。参照が存在しないオープンエンドのコンテンツに使う。

実用的な2026年スタック：BLEUとchrFには`sacrebleu`、COMETには`unbabel-comet`、最終的な人間向けシグナルにはプロンプトされたLLM。本番データを信頼する前に50〜100個の人間ラベル付きの例に対してすべての指標をキャリブレートする。

参照なし指標（COMET-QE、BLEURT-QE、LLM-as-judge）は参照なしで翻訳を評価でき、参照翻訳が存在しないロングテール言語ペアで重要だ。

### ステップ3: 本番で壊れること

上記の動作パイプラインは80%の時間で流暢に翻訳し、残りの20%はサイレントに失敗する。名前の付いた失敗モード：

- **幻覚。** モデルがソースになかったコンテンツを作り上げる。馴染みのないドメイン語彙で一般的。症状：出力は流暢だがソースが述べていない事実を主張する。対策：ドメイン用語の制約付きデコード、規制コンテンツの人間レビュー、入力よりはるかに長い出力の監視。
- **オフターゲット生成。** モデルが間違った言語に翻訳する。NLLBはまれな言語ペアでこれに驚くほど傾向がある。対策：`forced_bos_token_id`を検証して、出力に常に言語IDモデルのチェックを実行する。
- **用語ドリフト。** 「Sign up」がドキュメント1では「s'inscrire」、ドキュメント2では「créer un compte」になる。UIテキストとユーザー向け文字列では、生の品質より一貫性の方が重要だ。対策：用語集制約デコードまたは後編集辞書。
- **文体のミスマッチ。** フランス語の「tu」と「vous」、日本語の丁寧さのレベル。モデルは訓練でより一般的だった形式を選ぶ。顧客向けのコンテンツではこれは通常間違いだ。対策：モデルがサポートしていれば文体トークンで接頭辞を付けるか、公式コーパスのみで小さいモデルをファインチューニングする。
- **短い入力での長さ爆発。** 非常に短い入力文は、ソーストークンが約5未満で長さペナルティが急落するため、過度に長い翻訳を生成することが多い。対策：ソース長に比例したハードな最大長キャップ。

### ステップ4: ドメインのファインチューニング

事前訓練済みモデルはジェネラリストだ。法律、医療、またはゲームダイアログの翻訳はドメインの並列データでのファインチューニングから測定可能な恩恵を受ける。レシピは特別ではない：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

数千の高品質な並列ペアは、数十万のノイジーなウェブスクレイプされたものより優れている。訓練データの品質が最大の本番レバーだ。

## 使ってみる

MTの2026年の本番スタック：

| ユースケース | 推奨する出発点 |
|-------------|--------------|
| 200言語間の任意翻訳 | `facebook/nllb-200-distilled-600M`（ラップトップ）または`nllb-200-3.3B`（本番） |
| 英語中心、高品質、50言語 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短い実行、安価な推論、英語-フランス語/ドイツ語/スペイン語 | Helsinki-NLP / Marianモデル |
| レイテンシクリティカルなブラウザ側 | ONNX量子化Marian（〜50 MB） |
| 最高品質、コストを払う意思あり | 翻訳プロンプトを使ったGPT-4 / Claude / Gemini |

LLMは2026年時点でいくつかの言語ペアで専門的なMTモデルを上回るようになり、特に慣用的なコンテンツと長いコンテキストで優れている。トレードオフはトークンあたりのコストとレイテンシだ。コンテキスト長、スタイルの一貫性、またはプロンプティングによるドメイン適応がスループットより重要な場合はLLMを選ぶ。

## 成果物を出す

`outputs/skill-mt-evaluator.md`として保存する：

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## 演習

1. **易しい。** `nllb-200-distilled-600M`を使って5文の英語段落をフランス語に翻訳し、英語に戻す。ラウンドトリップがオリジナルにどれだけ近いかを測定する。単語選択のドリフトを伴う意味保存が見られるはずだ。
2. **普通。** `fasttext lid.176`または`langdetect`を使って翻訳出力に言語IDチェックを実装する。オフターゲット生成を返す前に捕捉するようにMT呼び出しに組み込む。
3. **難しい。** 自分で選んだ5,000ペアのドメインコーパスで`nllb-200-distilled-600M`をファインチューニングする。ファインチューニング前後のホールドアウトセットでBLEUを測定する。どの種類の文が改善し、どれが退行したかを報告する。

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|-----------------|-----------------------|
| BLEU | 翻訳スコア | 簡潔さペナルティ付きのN-gram精度。[0, 100]。 |
| chrF | 文字Fスコア | 文字レベルのFスコア。形態的に豊かな言語により敏感。 |
| NMT | ニューラルMT | 並列テキストで訓練されたTransformerエンコーダー-デコーダー。2017年以降のデフォルト。 |
| NLLB | No Language Left Behind | Metaの200言語MTモデルファミリー。 |
| 制約付きデコード | 制御された出力 | 特定のトークンまたはn-gramを出力に現れる/現れないように強制する。 |
| 幻覚 | 作り上げられたコンテンツ | ソースによって支持されないモデル出力。 |

## 参考資料

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — NLLB論文。
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — `sacrebleu`がBLEUを報告する唯一の正しい方法である理由。
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — chrF論文。
- [Hugging Face MTガイド](https://huggingface.co/docs/transformers/tasks/translation) — 実用的なファインチューニングのウォークスルー。
