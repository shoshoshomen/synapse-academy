# 投機的デコーディング — ドラフト、検証、繰り返し

> 自己回帰デコーディングはシリアルだ。各トークンは前のトークンを待つ。投機的デコーディングはその連鎖を断ち切る：安価なモデルがNトークンをドラフトし、高価なモデルが1回のフォワードパスですべてのNを検証する。ドラフトが正しければ、N回の生成に対して1回の大きなフォワードで済む。


## 問題

H100でサンプリングする70B LLMは1トークンあたり約30msかかる。3Bドラフトモデルは約3msかかる。3Bに5トークム先行ドラフトさせ、その後70Bを*1回*実行してすべての5を検証すると、合計は最大5個の受け入れられたトークンに対して`5×3 + 30 = 45 ms`だ——直線的な生成の`5×30 = 150 ms`に対して。これが投機的デコーディングのピッチ全体だ：小さな追加GPUメモリ（ドラフトモデル）を取引して2〜4倍の低いデコードレイテンシを得る。

このトリックは分布を保持しなければならない。Leviathan et al.（2023年）とChen et al.が同時期に導入した投機的サンプリングは、出力シーケンスが大きいモデルが単独で生成したものと**同一の分布**であることを保証する。品質のトレードオフなし。単に速い。

4つのドラフト-検証器ペアファミリーが2026年の推論を支配する：

1. **バニラ投機（Leviathan 2023年）。** 別個のドラフトモデル（例：Llama 3 1B）+ 検証器（例：Llama 3 70B）。
2. **Medusa（Cai 2024年）。** 検証器上の複数のデコーディングヘッドが`t+1..t+k`の位置を並列に予測する。別個のドラフトモデルなし。
3. **EAGLEファミリー（Li 2024年、2025年）。** 検証器の隠れ状態を再利用する軽量ドラフト；バニラより高い受け入れ率；典型的に3〜4倍。
4. **ルックアヘッドデコーディング（Fu 2024年）。** Jacobi反復；ドラフトモデル不要。自己投機。ニッチだが依存関係なし。

2026年のすべての本番推論スタックがデフォルトで投機的デコーディングを出荷する。vLLM、TensorRT-LLM、SGLang、llama.cppはすべてバニラ + EAGLE-2以上をサポートする。

## 概念

### コアアルゴリズム

検証器`M_q`と安価なドラフト`M_p`が与えられた時：

1. `x_1..x_k`をすでにデコードされたプレフィックスとする。
2. **ドラフト**：`M_p`を使ってドラフト確率`p_1..p_N`で`d_{k+1}, d_{k+2}, ..., d_{k+N}`を自己回帰的に提案する。
3. **並列検証**：`M_q`を`x_1..x_k, d_{k+1}, ..., d_{k+N}`で1回実行し、位置`k+1..k+N+1`の検証器確率`q_1..q_{N+1}`を取得する。
4. **各ドラフトトークンを左から右に受け入れ/拒否**：各`i`について、確率`min(1, q_i(d_i) / p_i(d_i))`で受け入れる。
5. 位置`j`での最初の拒否時：正規化された「残差」分布`(q_j - p_j)_+`から`t_j`をサンプリングする。`j`以降のすべてのドラフトは破棄される。
6. すべての`N`を受け入れた時：`q_{N+1}`から追加トークン`t_{N+1}`をサンプリングする（無料ボーナストークン）。

残差分布のトリックは、出力が`M_q`が一からサンプリングしたかのように正確に分布するようにする数学的洞察だ。

### 何がスピードアップを決定するか

`α`= ドラフトトークンごとの期待受け入れ率。`c` = ドラフト対検証器のコスト比。ステップごとに：

- ナイーブ生成はトークンごとに1回の大きいモデル呼び出し。
- 投機的は`α`が高い時、`(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)`トークンごとに1回の大きいモデル呼び出し。

`α = 0.75`と`N = 5`での典型的な経験則：3倍少ない大きいモデル呼び出し。ドラフトコストは5倍安い。総実時間は約2.5倍下がる。

**αの依存要素：**

- ドラフトが検証器をどれだけ近似するか。同じファミリー / 同じ学習データはαを大幅に高める。
- デコーディング戦略。貪欲ドラフトに対して貪欲検証器：高いα。温度サンプリング：マッチが難しい；受け入れが下がる。
- タスクの種類。コードと構造化出力の受け入れが多い（予測可能）；自由形式の創造的な文章は少ない。

### Medusa — ドラフトモデルなしのドラフト

Medusaはドラフトモデルを検証器の追加出力ヘッドに置き換える。位置`t`で：

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

各ヘッドが独自のロジットを出力する。推論時、各ヘッドからサンプリングして候補シーケンスを取得し、すべての候補継続を一度に考慮するツリーAttentionスキームで1回のフォワードパスで検証する。

利点：2番目のモデルなし。欠点：学習可能なパラメータを追加する；SFTステージが必要（〜10億トークン）；良いドラフトを使ったバニラ投機より受け入れ率がやや低い。

### EAGLE — 隠れ状態を再利用したより良いドラフト

EAGLE-1/2/3（Li et al.、2024〜2025年）はドラフトモデルを検証器の最終レイヤーの隠れ状態を取り込む小さなTransformer（通常1レイヤー）にする。ドラフトが検証器の特徴表現を見るため、その予測は検証器の出力分布と強く相関する。受け入れ率は〜0.6（バニラ）から0.85+に上昇する。

EAGLE-3（2025年）は候補継続に対するツリーサーチを追加した。vLLMとSGLangはLlama 3/4とQwen 3のデフォルト投機パスウェイとしてEAGLE-2/3を出荷する。

### KVキャッシュダンス

検証は1回のフォワードパスで`N`個のドラフトトークンを検証器に供給する。これにより検証器のKVキャッシュが`N`エントリ分拡張する。いくつかのドラフトが拒否された場合、受け入れられたプレフィックス長まで キャッシュをロールバックしなければならない。

本番実装（vLLMの`--speculative-model`、TensorRT-LLMのLookaheadDecoder）はスクラッチKVバッファでこれを処理する。まず書き込み、受け入れ時にコミット。概念的に難しくないが、細かい作業だ。

## 実装する

`code/main.py`を参照。コア投機的サンプリングアルゴリズム（拒否ステップ + 残差分布）を実装する：

- 手作りの分布に対する決定論的softmax（受け入れ数学を解析的に検証できるように）。
- 大きいモデルの摂動であるドラフトモデル。
- 検証器から直接サンプリングするのと同じ周辺分布を生成する受け入れ / 拒否ループ。

### ステップ1：拒否ステップ

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u`は一様乱数。`q_prob`はドラフトされたトークンに対する検証器の確率。`p_prob`はドラフトモデルの確率。Leviathanの定理は、このベルヌーイ判定に続く拒否時の残差からのサンプリングが検証器の分布を正確に保持するというものだ。

### ステップ2：残差分布

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

要素ごとに`p`を`q`から引き、負の値をゼロにクランプし、再正規化する。拒否時にこれからサンプリングする。

### ステップ3：1回の投機的ステップ

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

5個受け入れ → 1個ボーナス → 1回の検証器パスで6トークン生成。

### ステップ4：受け入れ率を測定する

様々なドラフト品質レベルで10,000回の投機的ステップを実行せよ。ドラフトと検証器分布間のKLダイバージェンスに対する受け入れ率をプロットせよ。きれいな単調関係が見えるはずだ。

### ステップ5：分布の等価性を検証する

経験的に：投機的ループが生成するトークンのヒストグラムは検証器から直接サンプリングして生成するヒストグラムと一致するはずだ。これが実践でのLeviathanの定理だ。カイ二乗検定はサンプリング誤差の範囲内で確認する。

## 使ってみる

本番：

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

TensorRT-LLMは2026年中頃時点で最速のMedusaパスを持つ。`faster-whisper`は小さなドラフトでWhisper-largeの投機的デコーディングをラップする。

**ドラフトの選択：**

| 戦略 | 選ぶ時 | スピードアップ |
|----------|--------------|---------|
| バニラドラフト（1B/3B Llamaファミリー） | 高速プロトタイプ、学習なし | 1.8〜2.3倍 |
| Medusaヘッド | 検証器をファインチューニングできる | 2〜3倍 |
| EAGLE-2 / 3 | 本番、最大速度 | 3〜4倍 |
| ルックアヘッド | ドラフトなし、学習なし、追加パラメータなし | 1.3〜1.6倍 |

**投機的デコーディングをすべき**でない**時：**

- 1〜5トークンの単一シーケンス生成。オーバーヘッドが支配する。
- 激しく創造的 / 高温サンプリング（αが下がる）。
- メモリ制約のデプロイメント（ドラフトモデルがVRAMを追加する）。

## 成果物を出す

`outputs/skill-spec-decode-picker.md`を参照。このスキルは、新しい推論ワークロードの投機的デコーディング戦略（バニラ / Medusa / EAGLE / ルックアヘッド）とチューニングパラメータ（N、ドラフト温度）を選択する。

## 演習

1. **易.** `code/main.py`を実行せよ。投機的トークン分布が50,000トークン上でカイ二乗p > 0.05の範囲内で検証器の直接サンプル分布と一致することを確認せよ。
2. **中.** `α = 0.5, 0.7, 0.85`の`N`の関数としてスピードアップ（大きいモデルのフォワードあたりのトークン）をプロットせよ。各αの最適な`N`を特定せよ。（ヒント：検証呼び出しあたりの期待トークン = `(1 - α^{N+1}) / (1 - α)`。）
3. **難.** 小さなMedusaを実装せよ：レッスン14のカプストーンGPTを取り、位置t+2、t+3、t+4を予測する3つの追加LMヘッドを追加せよ。合同マルチヘッド損失でtinyshakespeareを学習せよ。同じモデルを切り詰めて作ったバニラドラフトに対して受け入れ率を比較せよ。
4. **難.** ロールバックを実装せよ：10トークンプレフィックスKVキャッシュから始め、5個のドラフトトークンを供給し、位置3での拒否をシミュレートせよ。次の反復で「プレフィックス + 最初の2個の受け入れられたドラフト」と正しく一致することをキャッシュ読み取りで確認せよ。

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| ドラフトモデル（Draft model） | 「安価な方」 | 候補トークンを提案する小さいモデル；通常検証器より10〜50倍安い。 |
| 検証器（Verifier） | 「大きい方」 | 分布を保持するターゲットモデル；投機的ステップごとに1回実行する。 |
| 受け入れ率（Acceptance rate, α） | 「ドラフトがどれだけ正しいか」 | 検証器がドラフトを受け入れるトークンごとの確率。0.7〜0.9が典型的。 |
| 残差分布（Residual distribution） | 「拒否時のフォールバック」 | 正規化された`(q - p)_+`；拒否時にこれからサンプリングすることで検証器の分布を保持。 |
| ボーナストークン（Bonus token） | 「無料トークン」 | すべてのNドラフトが受け入れられた時、検証器の次ステップ分布からもう1つサンプリング。 |
| Medusa | 「ドラフトレス投機」 | 検証器上の複数のLMヘッドが位置t+1..t+kを並列に予測。 |
| EAGLE | 「隠れ状態ドラフト」 | 検証器の最終レイヤーの隠れ状態で条件付けされた小さなTransformerドラフト。 |
| ルックアヘッドデコーディング（Lookahead decoding） | 「Jacobi反復」 | 固定点反復を使った自己投機；ドラフトモデルなし。 |
| ツリーAttention（Tree attention） | 「多くの候補を一度に検証」 | 複数のドラフト継続を同時に考慮する分岐検証。 |
| KVロールバック（KV rollback） | 「拒否されたドラフトの取り消し」 | スクラッチKVバッファ；受け入れ時にコミット、拒否時に破棄。 |

## 参考資料

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — コアアルゴリズムと等価性定理。
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — 並列導入；きれいなベルヌーイ拒否証明。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa論文；ツリーAttention検証。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1；隠れ状態条件付きドラフト。
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2；動的ツリー深度。
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3。
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — ルックアヘッド、ドラフトなしアプローチ。
- [vLLMドキュメント — 投機的デコーディング](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 4戦略すべてが組み込まれた標準的な本番リファレンス。
- [SafeAILab / EAGLE参照実装](https://github.com/SafeAILab/EAGLE) — EAGLE-1/2/3の参照コード。
