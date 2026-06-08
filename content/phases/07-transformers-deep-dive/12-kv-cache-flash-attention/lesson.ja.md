# KVキャッシュ、Flash Attention & 推論最適化

> 学習は並列でFLOP律速だ。推論は逐次でメモリ律速だ。ボトルネックが違えば、トリックも違う。


## 問題

ナイーブな自己回帰デコーダは`N`トークンを生成するのに`O(N²)`の作業を行う：各ステップで完全なプレフィックスに対するアテンションを再計算する。4Kトークンの応答では1600万のアテンション操作で、そのほとんどが冗長だ。プレフィックストークンのすべての隠れ状態は一度計算されたら決定論的だ——キャッシュされたキーとバリューに対して新しいトークンのクエリのみを実行する必要がある。

それに加えて、アテンション自体は大量のデータを移動する。標準アテンションはN×Nスコア行列、N×d softmax出力、N×d最終出力をマテリアライズする——HBMへの読み書きが多すぎる。N≥2KではアテンションはFLOP律速になる前にメモリ律速になる。クラシックなアテンションカーネルは現代のGPUを4〜10倍も低使用する。

Dao et al.による2つの最適化がフロンティア推論を「遅い」から「速い」に変えた：

1. **KVキャッシュ。** すべてのプレフィックストークンのKとVベクトルを保存する。各新しいトークンのアテンションはキャッシュされたキーに対する1つのクエリだ。推論は生成ステップあたり`O(N²)`から`O(N)`に削減される。
2. **Flash Attention。** 完全なN×N行列がHBMに当たることがないようにアテンション計算をタイル化する。softmax + 行列積のすべてがSRAMで起きる。A100で2〜4倍の実時間スピードアップ；FP8のH100で5〜10倍。

2026年までに両方ともユニバーサルだ。すべての本番推論スタック（vLLM、TensorRT-LLM、SGLang、llama.cpp）がそれらを前提とする。すべてのフロンティアモデルはFlash Attentionを有効にして出荷される。

## 概念

![KVキャッシュの成長とFlash Attentionタイリング](../assets/kv-cache-flash-attn.svg)

### KVキャッシュの計算

デコーダ層ごと、トークンごと、ヘッドごと：

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          KとV
```

32層、32ヘッド、d_head=128、fp16の7Bモデルの場合：

```
層ごとトークンごと = 2 * 128 * 2 = 512バイト
トークンごと（32層） = 16 KB
32Kコンテキストごと = 512 MB
```

Llama 3 70B（80層、d_head=128、8 KVヘッドのGQA）の場合：

```
層ごとトークンごと = 2 * 8 * 128 * 2 = 4096バイト（4 KB）
32Kコンテキストごと = 10.4 GB
```

その10 GBが、128KコンテキストのLlama 3 70Bがバッチサイズ1でKVキャッシュだけに40 GB A100のほとんどを必要とする理由だ。

**GQAがKVキャッシュの勝者だ。** 64ヘッドのMHAは32 GBになる。MLAはさらに圧縮する。

次元を調整してキャッシュサイズが動く様子を観察する。シーケンス長またはバッチを増やすと、単一のGPUをいかに速く超えるかを見る：

```figure
kv-cache-sizer
```

### Flash Attention — タイリングトリック

標準アテンション：

```
S = Q @ K^T          (HBM読み取り、N×N、HBM書き込み)
P = softmax(S)       (HBM読み取り、HBM書き込み)
O = P @ V            (HBM読み取り、HBM書き込み)
```

3回のHBMラウンドトリップ。H100ではHBM帯域幅は3 TB/s；SRAMは30 TB/s。すべてのHBMトリップはオンチップに対して10倍の低下だ。

Flash Attention：

```
Qの各ブロック（タイルサイズ〜128 × 128）:
    Q_tileをSRAMにロード
    K、Vの各ブロック:
        K_tile、V_tileをSRAMにロード
        S_tile = Q_tile @ K_tile^T を計算    （SRAM）
        ランニングsoftmax集計              （SRAM）
        O_tileに蓄積                      （SRAM）
    O_tileをHBMに書き込む
```

タイルごとに1回のHBMトリップ。合計メモリフットプリントは`O(N²)`から`O(N)`に低下する。バックワードパスはそれらを保存する代わりにフォワードパスから一部の値を再計算する——もう1つのメモリの勝利だ。

**数値トリック。** ランニングsoftmaxはタイル間で`(max, sum)`を維持するため最終正規化が正確だ。近似ではない——Flash Attentionは標準アテンションと同一の出力を計算する（fp16の非結合性を除く）。

**バージョンの進化：**

| バージョン | 年 | 主な変更 | リファレンスハードウェアでのスピードアップ |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | タイルSRAMカーネル | A100で2倍 |
| Flash 2 | 2023 | より良い並列性、Causal優先順序 | A100で3倍 |
| Flash 3 | 2024 | Hopper非同期性、FP8 | H100で1.5〜2倍（〜740 TFLOPs FP16） |
| Flash 4 | 2026 | Blackwell 5ステージパイプライン、software exp2 | 推論優先（最初はフォワードのみ） |

Flash 4はリリース時フォワードパスのみだ。学習はFlash 3を引き続き使用する。Flash 4のGQAとvarlenサポートは保留中（2026年中頃）。

### 投機的デコーディング — もう1つのレイテンシ改善

安価なモデルがNトークンを提案する。大きなモデルがすべてのNを並列に検証する。検証がkトークンを受け入れれば、1つの大モデルのフォワードパスでk回の生成に支払った。コードと文章での典型的なk=3〜5。

2026年のデフォルト：
- **EAGLE 2 / Medusa。** 検証器の隠れ状態を共有する統合されたドラフトヘッド。品質損失なしで2〜3倍のスピードアップ。
- **ドラフトモデルを使った投機的デコーディング。** コンシューマーハードウェアで2〜4倍のスピードアップ。
- **ルックアヘッドデコーディング。** Jacobiイテレーション；ドラフトモデル不要。ニッチだが無料。

### 継続的バッチング

クラシックなバッチ推論：最も遅いシーケンスが完了するまで待ち、新しいバッチを開始する。短い応答が早く完了したときにGPUを無駄にする。

継続的バッチング（最初はOrcaで出荷、現在はvLLM、TensorRT-LLM、SGLangに）：古いものが完了したら新しいリクエストをバッチに即座にスワップする。典型的なチャットワークロードで5〜10倍のスループット向上。

### PagedAttention — 仮想メモリとしてのKVキャッシュ

vLLMの目玉機能。KVキャッシュは16トークンブロックで割り当てられ；ページテーブルが論理位置を物理ブロックにマップする。並列サンプル（ビームサーチ、並列サンプリング）でKVを共有し、プロンプトキャッシングのためのプレフィックスをホットスワップし、メモリを最適化できる。ナイーブな連続割り当てより4倍のスループット改善。

## 実装する

`code/main.py`を参照。以下を実装する：

1. ナイーブな`O(N²)`インクリメンタルデコーダ。
2. `O(N)` KVキャッシュデコーダ。
3. Flash AttentionのランニングMaxアルゴリズムをシミュレートするタイル化されたsoftmax。

### ステップ1：KVキャッシュ

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

シンプル：層ごと、ヘッドごとのリストでトークンごとにK、Vベクトルを成長させ続ける。

### ステップ2：タイル化されたsoftmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention風のsoftmax(qK^T)V、ランニングmax/sumで。"""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

1ショットの`softmax(qK) V`と同一の出力だが、作業セットは常に`tile × d_head`ブロックで、完全な`N × d_head`ではない。

### ステップ3：100トークン生成でナイーブとキャッシュを比較する

アテンション操作を数える。ナイーブ：`O(N²)` = 5050。キャッシュ：`O(N)` = 100。コードが両方を出力する。

## 使ってみる

```python
# HuggingFace transformersはデコーダのみのgenerate()でKVキャッシュを自動有効化する。
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # HopperならFA3を使う
    torch_dtype="bfloat16",
)
# generate()はKVキャッシュを自動的に使用する
```

vLLM本番：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

リクエスト間のプレフィックスキャッシングは2026年の大きな勝利だ——同じシステムプロンプト、フューショット例、または長いコンテキスト文書がコール間でKVを再利用する。繰り返されるツールプロンプトを持つエージェントワークロードでは、プレフィックスキャッシングは日常的に5倍のスループット向上をもたらす。

## 成果物を出す

`outputs/skill-inference-optimizer.md`を参照。このスキルは、新しい推論デプロイメントのためのアテンション実装・KVキャッシュ戦略・量子化・投機的デコーディングを選択する。

## 演習

1. **易.** `code/main.py`を実行せよ。ナイーブとキャッシュデコーダが同じ出力を生成することを確認し；操作数の差異に注意せよ。
2. **中.** プレフィックスキャッシングを実装せよ：プロンプトPといくつかの補完が与えられたとき、KVキャッシュを埋めるためにP上で1つのフォワードパスを実行し、補完ごとに分岐する。各補完のためにPを再エンコードするのと比べてスピードアップを測定せよ。
3. **難.** トイPagedAttentionを実装せよ：フリーリストを持つ固定16トークンブロックのKVキャッシュ。シーケンスが完了したら、ブロックをプールに返す。様々な長さの1,000チャット補完をシミュレートする。連続割り当てとのメモリ断片化を比較せよ。

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| KVキャッシュ（KV cache） | 「デコーディングを速くするトリック」 | すべてのプレフィックストークンのKとVを保存；新しいクエリはそれらにアテンションを向けて再計算しない。 |
| HBM | 「GPUメインメモリ」 | High Bandwidth Memory；H100で80 GB、B200で192 GB。〜3 TB/s帯域幅。 |
| SRAM | 「オンチップメモリ」 | H100でSMあたり〜256 KBの高速メモリ。〜30 TB/s帯域幅。 |
| Flash Attention | 「タイル化アテンションカーネル」 | HBMにN×Nをマテリアライズせずにアテンションを計算する。 |
| 継続的バッチング（Continuous batching） | 「待たないバッチング」 | 終了したシーケンスを外に、新しいものを内にスワップ、バッチを空にしないで。 |
| PagedAttention | 「vLLMの目玉」 | ページテーブルで固定ブロックに割り当てられたKVキャッシュ；断片化を除去する。 |
| プレフィックスキャッシング（Prefix caching） | 「長いプロンプトを再利用」 | リクエスト間で共有プレフィックスのKVをキャッシュ；エージェントの大きなコスト削減。 |
| 投機的デコーディング（Speculative decoding） | 「ドラフト + 検証」 | 安価なドラフトモデルがトークンを提案；大きなモデルが1パスでkを検証する。 |

## 参考資料

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1。
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2。
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3。
- [FlashAttention-4リリースノート (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 5ステージパイプラインとsoftware-exp2トリック；このレッスンで言及されるフォワードのみのリリース注意事項についてはリポジトリREADMEを読む。
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM論文。
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 投機的デコーディング。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — レッスンで引用される統合ドラフトアプローチのEAGLE-1/2論文。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — EAGLEと並んで参照されるMedusaアプローチ。
- [vLLMドキュメント — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 16トークンブロックとページテーブル設計の標準的な深掘り。
