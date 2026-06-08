# Mixture of Experts（MoE）

> 密な70B Transformerはすべてのトークンですべてのパラメータをアクティブにする。671B MoEはトークンあたり37Bのみをアクティブにし、すべてのベンチマークで上回る。スパース性は十年で最も重要なスケーリングのアイデアだ。


## 問題

密なTransformerの推論FLOPsはパラメータ数に等しい（フォワードパスに2倍）。密なモデルをスケールアップするとすべてのトークンが全額を支払う。2024年までにフロンティアは計算の壁に当たっていた：有意に賢くなるためにはトークンあたりの指数的に多くのFLOPsが必要だった。

Mixture of Expertsはこのリンクを断ち切る。各FFNを`E`個の独立したエキスパート + トークンごとに`k`個のエキスパートを選択するルーターで置き換える。総パラメータ = `E × FFN_size`。トークンあたりのアクティブパラメータ = `k × FFN_size`。典型的な2026年の設定：`E=256`、`k=8`。ストレージは`E`でスケール、計算は`k`でスケール。

2026年のフロンティアはほぼすべてMoEだ：DeepSeek-V3（671B総計/37Bアクティブ）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。Artificial Analysisの独立したリーダーボードで、上位10のオープンソースモデルはすべてMoEだ。

## 概念

![MoEレイヤー：ルーターがトークンごとにE個のエキスパートからk個を選択する](../assets/moe.svg)

### FFNの交換

密なTransformerブロック：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoEブロック：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # トークンごとにE個からk個を選ぶ
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

すべてのエキスパートは独立したFFN（通常はSwiGLU）だ。ルーターは単一の線形層だ。各トークンは独自の`k`個のエキスパートを選び、それらの出力のゲート付きミックスを取得する。

### 負荷分散問題

ルーターがトークンの90%をエキスパート3に送ると、他のエキスパートは飢える。3つの解決策が試みられた：

1. **補助負荷分散損失**（Switch Transformer、Mixtral）。エキスパート使用の分散に比例したペナルティを追加する。機能するが、ハイパーパラメータと2番目の勾配信号を追加する。
2. **エキスパート容量 + トークンドロップ**（初期Switch）。各エキスパートは最大`C × N/E`トークンを処理する；オーバーフロートークンは層をスキップする。品質に影響する。
3. **補助損失フリー分散**（DeepSeek-V3）。ルーターのtop-k選択をシフトする学習済みのエキスパートごとのバイアスを追加する。バイアスは学習損失の外で更新される。メイン目標にペナルティなし。2024年の大きな突破口。

DeepSeek-V3のアプローチ：各学習ステップ後、すべてのエキスパートについて使用量が目標より上か下かを確認する。バイアスを`±γ`調整する。選択は`scores + bias`を使う。ゲーティングに使われるエキスパート確率は変更されていない元の`scores`だ。ルーティングと表現を切り離す。

### 共有エキスパート

DeepSeek-V2/V3は*共有*エキスパートと*ルーティング済み*エキスパートにも分割する。すべてのトークンはすべての共有エキスパートを通過する。ルーティング済みエキスパートはtop-k経由で選ばれる。共有エキスパートは共通知識を捉え；ルーティング済みエキスパートは専門化する。V3は1つの共有エキスパート + 256個のルーティング済みのtop-8を実行する。

### 細粒度エキスパート

クラシックMoE（GShard、Switch）：各エキスパートは完全なFFNと同じ幅。`E`は小さい（8〜64）、`k`は小さい（1〜2）。

現代の細粒度MoE（DeepSeek-V3、Qwen-MoE）：各エキスパートは狭い（1/8 FFNサイズ）。`E`は大きい（256以上）、`k`はより大きい（8以上）。同じ総パラメータで、組み合わせははるかに速くスケールする。トークンあたり`C(256, 8) = 400兆`の可能な「エキスパート」。品質は上がり、レイテンシは一定のままだ。

### コストプロファイル

トークンごと、層ごと：

| 設定 | トークン/アクティブパラメータ | 総パラメータ |
|--------|-----------------------|--------------|
| Mixtral 8×22B | 〜39B | 141B |
| Llama 3 70B（密） | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2（MoE） | 〜32B | 1T |

DeepSeek-V3はトークンあたりの**アクティブFLOPs数がより少ない**にもかかわらず、ほぼすべてのベンチマークでLlama 3 70B（密）を上回る。より多くのパラメータ = より多くの知識。より多くのアクティブFLOPs = トークンあたりより多くの計算。MoEはそれらを切り離す。

### 落とし穴：メモリ

すべてのエキスパートはどれがファイアするかに関わらずGPU上に存在する。671Bモデルはfp16重みに約1.3 TBのVRAMが必要だ。フロンティアMoEデプロイメントはエキスパート並列性が必要——GPU間でエキスパートをシャード、ネットワーク間でトークンをルーティング。レイテンシは行列積ではなくall-to-all通信に支配される。

## 実装する

`code/main.py`を参照。純粋なstdlibのコンパクトなMoE層：

- `n_experts=8`のSwiGLU風エキスパート（説明のため各1線形）
- top-k=2ルーティング
- softmax正規化されたゲーティング重み
- エキスパートごとのバイアスによる補助損失フリー分散

### ステップ1：ルーター

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # 選ばれたエキスパートの元のスコアでsoftmax
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

バイアスは選択に影響するが、ゲート重みには影響しない。それがDeepSeek-V3のトリックだ——バイアスはモデルの予測を誘導せずに負荷不均衡を修正する。

### ステップ2：100トークンをルーターに通す

どのエキスパートがどれだけ頻繁にファイアするかを追跡する。バイアスなしでは使用量が偏る。バイアス更新ループ（過使用エキスパートには`-γ`、低使用エキスパートには`+γ`）で、使用量は数反復で均一分布に収束する。

### ステップ3：パラメータ数の比較

MoE設定の「密な等価物」を出力する。DeepSeek-V3形状：256ルーティング済み + 1共有、8アクティブ、d_model=7168。総パラメータ数は目が飛び出るほどだ。アクティブ数は密なLlama 3 70Bの7分の1だ。

## 使ってみる

HuggingFaceローディング：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026年本番推論：vLLMはMoEルーティングをネイティブにサポートする。SGLangが最速のエキスパート並列パスを持つ。両方ともtop-k選択とエキスパート並列性を自動的に処理する。

**MoEを選ぶ場合：**
- トークンあたりの低い推論コストでフロンティア品質が欲しい。
- VRAMとエキスパート並列インフラがある。
- ワークロードはコンテキスト重視（長文書）よりトークン重視（チャット、コード）だ。

**MoEを選ばない場合：**
- エッジデプロイメント——アクティブFLOPに対してフルストレージを支払う。
- レイテンシクリティカルなシングルユーザーサービング——エキスパートルーティングがオーバーヘッドを追加する。
- 小さなモデル（<7B）——MoEの品質アドバンテージは計算閾値（〜6Bアクティブパラメータ）を超えた場合にのみ現れる。

## 成果物を出す

`outputs/skill-moe-configurator.md`を参照。このスキルは、パラメータ予算・学習トークン・デプロイメントターゲットを考慮して、新しいMoEのE、k、共有エキスパートレイアウトを選択する。

## 演習

1. **易.** `code/main.py`を実行せよ。補助損失フリーバイアス更新が50反復でエキスパート使用量を均等にする様子を観察せよ。
2. **中.** 学習済みルーターをハッシュベースのルーター（決定論的、学習なし）に置き換えよ。品質と分散を比較せよ。学習済みルーターがなぜ優れているのか？
3. **難.** GRPO風「ロールアウトマッチルーティング」（DeepSeek-V3.2トリック）を実装せよ：推論中にどのエキスパートがファイアするかをログし、勾配計算中に同じルーティングを強制する。トイポリシー勾配設定への影響を測定せよ。

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| エキスパート（Expert） | 「多くの中の1つのFFN」 | 独立したフィードフォワードネットワーク；FFN計算のスパースなスライスに専用のパラメータ。 |
| ルーター（Router） | 「ゲート」 | 各トークンを各エキスパートに対してスコアリングする小さな線形層；top-k選択。 |
| Top-kルーティング（Top-k routing） | 「トークンあたりk個のアクティブエキスパート」 | 各トークンのFFN計算はゲートで重み付けされた正確にk個のエキスパートを通過する。 |
| 補助損失（Auxiliary loss） | 「負荷分散ペナルティ」 | 偏ったエキスパート使用にペナルティを課す追加損失項。 |
| 補助損失フリー（Auxiliary-loss-free） | 「DeepSeek-V3のトリック」 | エキスパートのバイアスのみに基づく選択によるバランス；追加勾配なし。 |
| 共有エキスパート（Shared expert） | 「常時オン」 | すべてのトークンが通過する追加エキスパート；共通知識を捉える。 |
| エキスパート並列性（Expert parallelism） | 「エキスパートでシャード」 | 異なるエキスパートを異なるGPUに分散；ネットワーク間でトークンをルーティング。 |
| スパース性（Sparsity） | 「アクティブパラメータ < 総パラメータ」 | 比率`k × expert_size / (E × expert_size)`；DeepSeek-V3では37/671 ≈ 5.5%。 |

## 参考資料

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — アイデア。
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch、クラシックMoE。
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B。
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + 補助損失フリーMoE + MTP。
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — バイアスベースのバランシング論文。
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — このレッスンのルーターが使う細粒度 + 共有エキスパート分割。
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — 元の共有エキスパート論文。
