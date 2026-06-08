# ゼロからトークナイザーを構築する

> レッスン01ではおもちゃを手に入れた。このレッスンでは武器を手に入れる。


## 学習目標

- Unicode、空白の正規化、特殊トークンを処理する本番グレードのBPEトークナイザーを構築する
- バイトレベルのフォールバックを実装して、トークナイザーがどんな入力（絵文字、CJK文字、コードを含む）も未知トークンなしにエンコードできるようにする
- BPEマージを適用する前に単語境界でテキストを分割する前処理用の正規表現パターンを追加する
- コーパスでカスタムトークナイザーを訓練し、多言語テキストでtiktokenと圧縮比を比較する

## 問題

レッスン01のBPEトークナイザーは英語テキストで動作する。今度は日本語を投げてみよう。または絵文字。タブとスペースが混在するPythonコード。

壊れる。

BPEが間違っているからではない――実装が不完全だからだ。本番トークナイザーは、あらゆるエンコーディングの生バイトを処理し、分割前にUnicodeを正規化し、マージされることのない特殊トークンを管理し、前処理とサブワード分割を連鎖させ、そしてこれをすべて15兆トークンを処理する訓練パイプラインのボトルネックにならないほど速く実行する。

GPT-2のトークナイザーは50,257トークンを持つ。Llama 3は128,256。GPT-4は約10万。これはおもちゃの数字ではない。これらの語彙を支えるマージテーブルは数百ギガバイトのテキストで訓練されており、その周辺の仕組み――正規化、前処理、特殊トークンの注入、チャットテンプレートのフォーマット――が「hello world」を処理するトークナイザーとインターネット全体を処理するトークナイザーを分けている。

その仕組みを構築しよう。

## 概念

### 完全なパイプライン

本番トークナイザーは1つのアルゴリズムではない。5つのステージからなるパイプラインであり、それぞれが異なる問題を解決する。

```mermaid
graph LR
    A[生テキスト] --> B[正規化]
    B --> C[前処理]
    C --> D[BPEマージ]
    D --> E[特殊トークン]
    E --> F[トークンID]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

各ステージには特定の役割がある：

| ステージ | 何をするか | なぜ重要か |
|---------|-----------|-----------|
| 正規化 | NFKC Unicode、オプションで小文字化、アクセント除去 | 「fi」合字（U+FB01）が「fi」（2文字）になる。これなしでは同じ単語が異なるトークンになる |
| 前処理 | BPEの前にテキストをチャンクに分割 | BPEが単語境界をまたいでマージするのを防ぐ。「the cat」は決して「e c」というトークンを生み出してはならない |
| BPEマージ | バイトシーケンスに学習済みのマージルールを適用 | 核心的な圧縮処理。生バイトをサブワードトークンに変換する |
| 特殊トークン | [BOS]、[EOS]、[PAD]、チャットテンプレートマーカーを注入 | これらのトークンは固定IDを持つ。BPEマージには参加しない。モデルは構造のためにこれらを必要とする |
| IDマッピング | トークン文字列を整数IDに変換 | モデルは文字列ではなく整数を見る |

### バイトレベルBPE

レッスン01のトークナイザーはUTF-8バイトで動作した。それは正しい選択だった。しかし重要なことをスキップしていた：それらのバイトが有効なUTF-8でない場合はどうなるか？

バイトレベルBPEはすべての可能なバイト値（0〜255）を有効なトークンとして扱うことでこれを解決する。基底語彙はちょうど256エントリだ。テキスト、バイナリ、破損したデータ――どんなファイルも未知トークンを生み出すことなくトークナイズできる。

GPT-2はトリックを加えた：語彙を人間が読みやすいよう保つために、各バイトを印字可能なUnicode文字にマッピングする。バイト0x20（スペース）はそのマッピングで「G」という文字になる。これは純粋に見た目の問題だ。アルゴリズムはそれを気にしない。

本当の力はここにある：バイトレベルBPEは地球上のすべての言語を処理する。漢字はUTF-8バイトで各3バイトだ。日本語は3〜4バイトになる。アラビア語、デーヴァナーガリー文字、絵文字――すべてただのバイトシーケンスだ。BPEアルゴリズムはこれらのバイトシーケンスのパターンを、英語のASCIIバイトのパターンを見つけるのとまったく同じ方法で見つける。

### 前処理

BPEがテキストに触れる前に、テキストをチャンクに分割する必要がある。これにより、マージアルゴリズムが単語境界をまたいだトークンを作ることを防ぐ。

GPT-2はテキストを分割するために正規表現パターンを使用する：

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

このパターンは短縮形（「don't」は「don」と「't」になる）、オプションの先行スペースを持つ単語、数字、句読点、空白で分割する。先行スペースは単語に付いたまま保持される――「the cat」は [" the", " cat"] になり、["the", " ", "cat"] にはならない。

LlamaはSentencePieceを使い、正規表現を完全にスキップする。生バイトストリームを1つの長いシーケンスとして扱い、BPEアルゴリズムに境界を判断させる。これはシンプルだが、BPEが単語をまたいだトークンを作る自由度が高くなる。

この選択は重要だ。GPT-2の正規表現は、ある単語の末尾の「the」と次の単語の先頭の「the」がマージされるのを防ぐ。SentencePieceはそれを許可する。これにより時として効率的な圧縮が生まれるが、解釈しにくいトークンになることもある。

### 特殊トークン

すべての本番トークナイザーは構造的なマーカーのためにトークンIDを予約する：

| トークン | 目的 | 使用先 |
|---------|------|-------|
| `[BOS]` / `<s>` | シーケンスの始まり | Llama 3、GPT |
| `[EOS]` / `</s>` | シーケンスの終わり | すべてのモデル |
| `[PAD]` | バッチ整列のためのパディング | BERT、T5 |
| `[UNK]` | 未知トークン（バイトレベルBPEがこれを排除） | BERT、WordPiece |
| `<\|im_start\|>` | チャットメッセージ境界の開始 | ChatGPT、Qwen |
| `<\|im_end\|>` | チャットメッセージ境界の終了 | ChatGPT、Qwen |
| `<\|user\|>` | ユーザーターンマーカー | Llama 3 |
| `<\|assistant\|>` | アシスタントターンマーカー | Llama 3 |

特殊トークンはBPEで分割されることはない。マージアルゴリズムが実行される前に正確にマッチし、固定IDに置き換えられ、周囲のテキストは通常通りトークナイズされる。

### チャットテンプレート

ここがほとんどの人が混乱し、ほとんどの実装が壊れるところだ。

チャットモデルにメッセージを送ると、APIはメッセージのリストを受け取る：

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

モデルはJSONを見ない。フラットなトークンシーケンスを見る。チャットテンプレートは特殊トークンを使ってメッセージをそのフラットなシーケンスに変換する。すべてのモデルがこれを異なる方法で行う：

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

テンプレートを間違えるとモデルはゴミを出力する。モデルは1つの正確なフォーマットで訓練されている。どんなずれでも――改行の欠如、トークンの入れ替え、余分なスペース――入力を訓練分布の外に置いてしまう。

### 速度

Pythonは本番のトークナイゼーションには遅すぎる。

tiktoken（OpenAI）はRustで書かれPythonバインディングを持つ。HuggingFaceのtokenizersもRustだ。SentencePieceはC++だ。これらは純粋なPythonより10〜100倍の高速化を達成している。

参考として：Llama 3の事前学習のために15兆トークンを毎秒100万トークン（高速なPython）でトークナイズすると174日かかる。毎秒1億トークン（Rust）なら1.7日だ。

アルゴリズムを理解するためにPythonで構築している。本番では、コンパイル済みの実装を使い、Pythonラッパーにのみ触れることになる。

## 実装する

### ステップ1: バイトレベルのエンコーディング

基盤。任意の文字列をバイトのシーケンスに変換し、各バイトを表示用の印字可能な文字にマッピングし、逆のプロセスを行う。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

多言語テキストでテストしてバイト数を見てみよう：

```python
texts = [
    ("英語", "hello"),
    ("中国語", "你好"),
    ("絵文字", "🔥"),
    ("混在", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)}文字 -> {len(b)}バイト -> {b}")
```

「hello」は5バイトだ。「你好」は6バイト（文字あたり3バイト）。炎の絵文字は4バイトだ。バイトレベルトークナイザーはどの言語かを気にしない。バイトはバイトだ。

### ステップ2: 正規表現による前処理

GPT-2の正規表現パターンを使ってテキストをチャンクに分割する。各チャンクがBPEによって独立してトークナイズされる。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex`モジュールはUnicodeプロパティエスケープ（文字は`\p{L}`、数字は`\p{N}`）をサポートする。標準ライブラリの`re`モジュールはサポートしないため、ASCIIの文字クラスにフォールバックする。本番の多言語トークナイザーには`regex`をインストールする。

試してみよう：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

先行スペースは単語に付いたまま保持される。短縮形はアポストロフィで分割される。句読点は独自のチャンクになる。BPEはこれらの境界をまたいでトークンをマージすることはない。

### ステップ3: バイトシーケンスへのBPE

レッスン01の核心アルゴリズム。ただし今回は前処理されたチャンクに対して独立して動作する。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### ステップ4: 特殊トークンの処理

特殊トークンには正確なマッチングと固定IDが必要だ。BPEを完全にバイパスする。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### ステップ5: 完全なトークナイザークラス

すべてを連鎖させる：正規化、特殊トークンで分割、前処理、BPEマージ、IDへのマッピング。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### ステップ6: 多言語テスト

本物のテスト。英語、中国語、絵文字、コードを投げてみよう。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"入力:   {text}")
    print(f"トークン数: {len(ids)}個")
    print(f"デコード後: {decoded}")
    print()
```

漢字は各3バイトを生み出す。絵文字は4バイトだ。これらはどれもトークナイザーをクラッシュさせない。未知トークンも生み出さない。これがバイトレベルBPEの力だ。

## 使ってみる

### 実際のトークナイザーを比較する

Llama 3、GPT-4、Mistralの実際のトークナイザーを読み込む。各々が同じ多言語段落をどのように処理するかを見る。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)}トークン): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)}トークン): {pieces[:20]}...")
```

同じテキストで異なるトークン数が出る。128K語彙のLlama 3は一般的なパターンのマージが積極的だ。100Kを持つGPT-4はその中間だ。32KのMistralはより多くのトークンを生み出すが、埋め込み層は小さい。

トレードオフは常に同じだ：語彙が大きいほどシーケンスは短いがパラメータは多くなる。

## 成果物を出す

このレッスンでは本番トークナイザーを構築・デバッグするためのプロンプトを作成する。`outputs/prompt-tokenizer-builder.md`を参照。

## 演習

1. **易:** `get_token_bytes(id)` メソッドを追加して、任意のトークンIDの生バイトを表示できるようにする。最も一般的なマージ済みトークンが実際に何を表しているかを調べるのに使う。
2. **中:** 空白と数字で分割しつつ先行スペースを保持するLlamaスタイルの前処理を実装する。GPT-2の正規表現アプローチと同じコーパスで語彙を比較する。
3. **難:** `{"role": ..., "content": ...}` メッセージのリストを受け取り、Llama 3チャットフォーマットの正しいトークンシーケンスを生成するチャットテンプレートメソッドを追加する。HuggingFaceの実装と照らし合わせてテストする。

## キーワード

| 用語 | 一般的な説明 | 実際の意味 |
|------|-------------|-----------|
| バイトレベルBPE | 「バイトで動作するトークナイザー」 | 256バイト値を基底語彙とするBPE――どんな入力も未知トークンなしに処理できる |
| 前処理 | 「BPEの前の分割」 | BPEが単語境界をまたいでマージするのを防ぐ正規表現またはルールベースの分割 |
| NFKC正規化 | 「Unicodeクリーンアップ」 | 標準分解後の互換合成――「fi」合字が「fi」になり、全角「A」が「A」になる |
| チャットテンプレート | 「メッセージがどのようにトークンになるか」 | ロール/コンテンツメッセージのリストをフラットなトークンシーケンスに変換する正確なフォーマット――モデル固有で訓練フォーマットに一致しなければならない |
| 特殊トークン | 「制御トークン」 | BPEをバイパスする予約済みトークンID――[BOS]、[EOS]、[PAD]、チャットマーカー――マージの前に正確にマッチされる |
| Fertility | 「単語あたりのトークン数」 | 出力トークン数と入力単語数の比率――GPT-4の英語では1.3、韓国語では2〜3、高いほどコンテキストが無駄になる |
| tiktoken | 「OpenAIトークナイザー」 | PythonバインディングつきのRust BPE実装――純粋なPythonより10〜100倍速い |
| マージテーブル | 「語彙」 | 訓練中に学習されたバイトペアマージの順序付きリスト――これがトークナイザーの学習済み知識そのものだ |

## 参考資料

- [OpenAI tiktokenソース](https://github.com/openai/tiktoken) -- GPT-3.5/4で使用されるRust BPE実装
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) -- BPE、WordPiece、Unigramをサポートするトークナイザーライブラリ（Rust）
- [Llama 3論文（Meta、2024）](https://arxiv.org/abs/2407.21783) -- 128K語彙とトークナイザー訓練の詳細
- [SentencePiece（Kudo & Richardson、2018）](https://arxiv.org/abs/1808.06226) -- 言語非依存のトークナイゼーション
- [GPT-2トークナイザーソース](https://github.com/openai/gpt-2/blob/master/src/encoder.py) -- 元のバイト→Unicode変換マッピング
