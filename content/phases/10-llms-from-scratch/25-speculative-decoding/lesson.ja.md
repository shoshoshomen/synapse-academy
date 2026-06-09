# 投機的デコーディングとEAGLE

> フロンティアLLMが1トークンを生成するには、数十億パラメータ全体にわたる完全なフォワードパスが必要だ。そのフォワードパスは圧倒的にオーバープロビジョニングされている。ほとんどの場合、はるかに小さなモデルが次の3〜5トークンを正しく推測でき、大きなモデルはその推測を*検証*するだけでよい。推測が正しければ、1回の処理で5トークンを得られる。投機的デコーディング（Leviathan et al. 2023）はこれを厳密に定式化し、EAGLE-3（2025）は受容率を検証1回あたり約4.5トークンまで引き上げた——一致した出力分布のまま4〜5倍の高速化を実現する。


## 問題

H100上の70Bクラスのモデルのデコードスループットは典型的に毎秒40〜80トークンだ。各トークンはHBMからすべてのモデルの重みを読み込む完全なフォワードパスを必要とする。出力を変えずにモデルを小さくすることはできない。メモリの限界を超えてバッチサイズを増やすこともできない。行き詰まりだ——モデルが1回のフォワードパスで複数のトークンを出力できない限り。

自己回帰生成は本質的にシリアルに見える: `x_{t+1} = sample(p(· | x_{1:t}))`。しかし、並行性の機会がある。「次の4トークンはおそらく [a, b, c, d]」と言う安価な予測器があれば、**大きなモデルの単一フォワードパス**で5つの位置をすべて検証し、最長の一致プレフィックスを受け入れることができる。

Leviathan, Kalai, Matias（2023、"Fast Inference from Transformers via Speculative Decoding"）は、ターゲットモデルのサンプリング分布を保持する巧妙な受容/棄却ルールによってこれを厳密に実現した。同じ出力分布のまま、2〜4倍高速。

## 概念

### 2モデル構成

- **ターゲットモデル** `M_p`: 実際にサンプルが欲しい大きく遅い高品質のモデル。分布: `p(x)`。
- **ドラフトモデル** `M_q`: 小さく速い低品質のモデル。分布: `q(x)`。5〜30倍小さい。

各ステップ:

1. ドラフトモデルが `K` トークンを自己回帰的に提案する: `x_1, x_2, ..., x_K ~ q`。
2. ターゲットモデルが `K+1` 個の位置すべてを並列に1回のフォワードパスで処理し、各提案トークンの `p(x_k)` を生成する。
3. 以下の修正棄却サンプリングルールにより、左から右へ各トークンを受容/棄却する。最長の一致プレフィックスを受け入れる。
4. あるトークンが棄却された場合、補正分布からリプレースメントをサンプリングして停止する。それ以外の場合は `p(· | x_1...x_K)` からボーナストークンをサンプリングする。

ドラフトがターゲットに完全に一致すれば、ターゲットフォワード1回につきK+1トークンを得られる。ドラフトが位置1で間違えれば1トークンしか得られない。

### 厳密性ルール

投機的デコーディングは**`p`からのサンプリングと分布的に等価であることが証明されている**。棄却ルール:

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

ここで `(p - q)+` は点差の正の部分を表す。ドラフトとターゲットが一致する場合（`p ≈ q`）、受容はほぼ1。不一致の場合、残差分布は全体のサンプルが依然として正確に `p` となるよう構築される。

**貪欲法の場合。** 温度=0のサンプリングでは単に `argmax(p) == x_t` を確認するだけでよい。一致すれば受容、不一致なら `argmax(p)` を出力して停止。

### 期待される高速化

ドラフトモデルのトークンレベル受容率を `α` とすると、ターゲットフォワードパス1回あたりの期待生成トークン数は:

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = ドラフト長, α in [0, 1]
```

`α = 0.8, K = 4` の場合: `(1 - 0.8^5)/(1 - 0.8) = 3.36` トークン/フォワード。単一のターゲットフォワードのコストは大まかに `cost_q * K + cost_p`（Kドラフトステップ＋1ターゲット検証）。`cost_p >> cost_q * K` ならスループットの高速化比は `3.36× / 1 = 3.36×`。

唯一の実質的なパラメータは `α` であり、これは完全にドラフト-ターゲット間の一致度に依存する。良いドラフトがすべてだ。

### ドラフトの学習: 蒸留

ランダムな小さなモデルは貧弱なドラフトになる。標準的なレシピはターゲットから蒸留することだ:

1. 小さいアーキテクチャを選ぶ（70Bターゲットには約1B、7Bターゲットには約500M）。
2. 大規模テキストコーパスでターゲットモデルを実行し、次トークン分布を保存する。
3. 正解トークンではなく、ターゲットの分布に対してKLダイバージェンスでドラフトを学習する。

結果: `α` はコーディングで典型的に0.6〜0.8、自然言語チャットで0.7〜0.85。本番では2〜3倍の高速化。

### EAGLE: ツリードラフティング＋特徴量再利用

Li, Wei, Zhang, Zhang（2024、"EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"）は、標準的な投機的デコーディングにおける2つの非効率性を観察した:

1. ドラフトはK個のシリアルなステップを踏み、各ステップは完全なスタック処理。しかし、ドラフトは直近の検証からターゲットの特徴量（隠れ状態）を再利用できるはずだ——ターゲットはドラフトが一から再導出しているリッチな表現をすでに計算している。
2. ドラフトは線形チェーンを出力する。ドラフトが候補の*ツリー*を出力できれば（各ノードに複数の推測）、ターゲットの単一フォワードパスがツリーアテンションマスクを用いて複数の候補パスを並列に検証し、最長の受容ブランチを選ぶことができる。

EAGLE-1の変更点:
- ドラフト入力 = 位置tでのターゲットの最終隠れ状態（生のトークンではない）。
- ドラフトアーキテクチャ = 1つのTransformerデコーダ層（別の小さなモデルではない）。
- 出力 = 深さごとにK = 4〜8候補、深さ4〜6のツリー。

EAGLE-2（2024）は動的ツリートポロジーを追加: ドラフトが不確かな場所でツリーを広げ、確信がある場所では狭くする。検証コストを増やさずに `α_effective` を引き上げる。

EAGLE-3（Li et al. 2025、"EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"）は固定された上位レイヤー特徴依存性を取り除き、新しい「テスト時シミュレーション」損失でドラフトを学習する——ドラフトはティーチャーフォース学習分布ではなく、ターゲットのテスト時分布に一致する出力で学習される。受容率はEAGLE-2の0.75からEAGLE-3の0.82に上がり、検証あたりの平均トークン数は3.0から4.5に向上した。

### ツリーアテンション検証

ドラフトがツリーを出力する場合、ターゲットモデルは**ツリーアテンションマスク**——純粋な線ではなくツリートポロジーをエンコードする因果マスク——を用いて単一のフォワードパスで検証する。各トークンはツリー内の祖先のみにアテンションを当てる。検証パスは依然として1フォワード、1行列積; トポロジカルマスクは追加のKVエントリをわずかに増やすだけだ。

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

`a, b` が競合する第1トークン候補で、`c, d, e, f` が第2トークン候補の場合、6つの位置すべてが1回のフォワードパスで検証される。出力は受容されたパスに沿った最長プレフィックスだ。

### 有利な場合、不利な場合

**有利:**
- 予測可能なテキストを含むチャット/補完（コード、一般的な英語、構造化出力）。`α` が高い。
- デコード中に未使用のGPU計算がある設定（メモリバウンドフェーズ）。ツリードラフティングは利用可能なFLOPSを活用する。

**不利/効果なし:**
- 高度に確率的な出力（高温度でのクリエイティブライティング）。`α` が `1/|vocab|` に向かって低下する。
- 非常に高い同時実行でのバッチサービング——バッチングがすでにFLOPSを埋めており、ツリー検証の余地が少ない。
- ドラフトがあまり小さくない非常に小さいターゲットモデル。

本番環境では、チャットで2〜3倍、コード生成で3〜5倍、クリエイティブライティングではほぼゼロの実クロック高速化が典型的に報告されている。

## 実装する

`code/main.py`:

- 厳密な棄却ルールを実装し、ターゲットの分布を保持することを検証する参照実装 `speculative_decode(target, draft, prompt, K, temperature)`（プレーンターゲットサンプリングに対して経験的KL < 0.01）。
- 深さKのツリーをtop-pブランチングで構築するEAGLEスタイルのツリードラフター。
- 検証器に適切な因果パターンを生成するツリーアテンションマスクビルダー。
- 小さなLM（GPT-2-mediumターゲットからGPT-2-smallを1つ蒸留）で両方を実行する受容率ハーネス。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 使ってみる

- **vLLM** と **SGLang** はファーストクラスの投機的デコーディングを搭載している。フラグ: `--speculative_model`, `--num_speculative_tokens`。EAGLE-2/3のサポートは `--spec_decoding_algorithm eagle` フラグ経由。
- **NVIDIA TensorRT-LLM** はMedusaとEAGLEツリーをネイティブにサポートする。
- **参照ドラフトモデル**: `Qwen/Qwen3-0.6B-spec`（Qwen3-32Bのドラフト）、`meta-llama/Llama-3.2-1B-Instruct-spec`（70Bのドラフト）。
- **Medusaヘッド**（Cai et al. 2024、"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"）: ドラフトモデルの代わりに、ターゲット自体にK個の並列予測ヘッドを追加する。デプロイが簡単で、EAGLEより受容率がわずかに低い。

## 成果物を出す

このレッスンは `outputs/skill-speculative-tuning.md` を生成する——ターゲットモデルのワークロードをプロファイリングし、ドラフトモデル、K（ドラフト長）、ツリー幅、温度、プレーンデコードへのフォールバック時期を選択するスキルだ。

## 演習

1. 厳密な棄却ルールを実装し、経験的に検証する。`speculative_decode` 経由と素のターゲットサンプリング経由でそれぞれ10K個のサンプルを実行し、2つの出力分布間のTV距離を計算する。0.01未満であるべきだ。

2. 高速化の式を計算する。固定の `α` と `K` に対して、ターゲットフォワードあたりの期待トークン数をプロットする。α ∈ {0.5, 0.7, 0.9} に対して最適なKを求める。

3. 小さなドラフトを学習する。124MのGPT-2ターゲットを取り、KL損失で100Mトークンに対して30MのGPT-2ドラフトを蒸留する。保留テキストで `α` を測定する。期待値: 0.6〜0.7。

4. EAGLEスタイルのツリードラフティングを実装する。チェーンの代わりに、ドラフトが各深さでtop-3ブランチを出力する。ツリーアテンションマスクを構築する。ターゲットが最長の正しいブランチを受け入れることを確認する。

5. 失敗モードを測定する。高確率性（温度=1.5）で投機的デコーディングを実行する。αが崩壊し、ドラフトのオーバーヘッドによってアルゴリズムがプレーンデコードより遅くなることを示す。

## キーワード

| 用語 | よく言われる表現 | 実際の意味 |
|------|-----------------|-----------|
| ターゲットモデル | 「大きなモデル」 | サンプルが欲しい遅い高品質モデル（p分布） |
| ドラフトモデル | 「投機者」 | 小さく速い予測器（q分布）; 5〜30倍小さい |
| K / ドラフト長 | 「先読み」 | 検証パスごとの投機トークン数 |
| α / 受容率 | 「ヒット率」 | ドラフトの提案が受け入れられるトークンレベルの確率 |
| 厳密棄却ルール | 「受容テスト」 | ターゲットの分布を保持するr < p/q比較 |
| 残差分布 | 「補正p-q」 | (p - q)+ / ||(p - q)+||_1、棄却時にサンプリングする分布 |
| ツリードラフティング | 「分岐投機」 | ドラフトが候補のツリーを出力し、ツリー構造アテンションマスクで1パスで検証 |
| ツリーアテンションマスク | 「トポロジカルマスク」 | ツリートポロジーをエンコードする因果マスク（各ノードは祖先のみにアテンション） |
| Medusaヘッド | 「並列ヘッド」 | ターゲット自体にK個の追加予測ヘッド; 別のドラフトモデルなし |
| EAGLE特徴量再利用 | 「隠れ状態ドラフト」 | ドラフト入力は生トークンではなくターゲットの最終隠れ状態（ドラフトを縮小） |
| テスト時シミュレーション損失 | 「EAGLE-3学習」 | ティーチャーフォースではなくターゲットのテスト時分布に一致する出力でドラフトを学習 |

## 参考資料

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 厳密な棄却ルールと理論的高速化分析
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMindによる同時期の投機的サンプリング論文
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — ドラフトモデルの代替となる並列ヘッド
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — 特徴量再利用とツリードラフティング
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — 動的ツリートポロジー
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — 学習時テスト時一致
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — Jacobi/先読みデコーディング、投機器なしの代替手法
