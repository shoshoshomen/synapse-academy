# ControlNet、LoRAと条件付け

> テキストだけでは制御シグナルとして不器用だ。ControlNetは事前学習済みの拡散モデルをクローンし、深度マップ、ポーズスケルトン、スクリブル、エッジ画像でそれを操舵できる。LoRAは2Bパラメータのモデルを1,000万パラメータの訓練でファインチューンできる。両者を合わせることで、Stable Diffusionはおもちゃから2026年のすべての代理店で出荷される画像パイプラインに変わった。


## 問題

「混雑した通りで犬を散歩させている赤いドレスの女性」というプロンプトは、犬が*どこに*いるか、女性が*どんなポーズ*をしているか、通りの*視点*がどうかについてモデルに何も伝えない。テキストは画像を指定するために必要なものの約10%を固定するだけだ。残りは視覚的で、言葉で効率的に説明できない。

各シグナル（ポーズ、深度、カニー、セグメンテーション）のための新しい条件付きモデルをゼロから訓練するのは禁止的にコストが高い。2.6BパラメータのSDXLバックボーンを凍結したまま、条件付けを読み取る小さなサイドネットワークを取り付け、バックボーンの中間特徴を少し調整させたい。これがControlNetだ。

また、新しいコンセプト（自分の顔、自分の製品、自分のスタイル）をモデルに教えたい。フルモデルを再訓練せずに。100倍小さいデルタが欲しい。これがLoRA——既存のアテンションウェイトに差し込む低ランクアダプタだ。

ControlNet + LoRA + テキスト = 2026年の実践者のツールキット。ほとんどのプロダクション画像パイプラインはSDXL / SD3 / Fluxベースに2〜5つのLoRA、1〜3つのControlNet、IP-Adapterを重ねている。

## 概念

![ControlNetはエンコーダをクローンし；LoRAは低ランクデルタを追加する](../assets/controlnet-lora.svg)

### ControlNet（Zhang et al.、2023）

事前学習済みSDを取る。U-Netのエンコーダ半分を*クローン*する。元を凍結する。クローンを追加の条件付き入力（エッジ、深度、ポーズ）を受け付けるよう訓練する。クローンを元のデコーダ半分に*ゼロ畳み込み*スキップ接続（ゼロに初期化された1×1畳み込み——ノーオペから始め、デルタを学習する）で接続する。

```
SD U-Netデコーダ:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

ゼロ畳み込み初期化は、ControlNetが恒等から始まることを意味する——訓練前でも害がない。標準拡散損失で100万件の（プロンプト、条件、画像）トリプルで訓練する。

モダリティごとのControlNetは小さなサイドモデルとして出荷される（SDXLには約360M、SD 1.5には約70M）。推論時に合成できる：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA（Hu et al.、2021）

モデル内の任意の線形層 `W ∈ R^{d×d}` に対して、`W` を凍結して低ランクデルタを追加する：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

`r << d`。アテンションには4〜16、重いファインチューンには64〜128のランクが標準。新しいパラメータ数：`d²` の代わりに `2 · d · r`。`d=640` のSDXLアテンションで `r=16`：アダプタあたり20kパラメータ（410kの代わりに——20倍削減）。モデル全体では：LoRAは通常ベースの5GBに対して20〜200MBだ。

推論時にLoRAをスケーリングできる：`W' = W + α · B @ A`。`α = 0.5-1.5` が正常だ。複数のLoRAは加算的にスタックされる（ただし非線形な相互作用があるという通常の注意事項付き）。

### IP-Adapter（Ye et al.、2023）

*画像*を条件付け（テキストと並んで）として受け付ける小さなアダプタ。CLIP画像エンコーダを使って画像トークンを生成し、テキストトークンと並んでクロスアテンションに注入する。ベースモデルあたり約20MB。LoRAなしで「このリファレンスのスタイルで画像を生成する」ことができる。

## 合成性マトリクス

| ツール | 制御するもの | サイズ | 使うべき時 |
|------|------------------|------|-------------|
| ControlNet | 空間的構造（ポーズ、深度、エッジ） | 70〜360MB | 正確なレイアウト、構図 |
| LoRA | スタイル、題材、コンセプト | 20〜200MB | パーソナライゼーション、スタイル |
| IP-Adapter | リファレンス画像からのスタイルまたは題材 | 20MB | テキストで外見を説明できない時 |
| Textual Inversion | 新しいトークンとしての単一コンセプト | 10KB | レガシー、主にLoRAに置き換えられた |
| DreamBooth | 題材のフルファインチューン | 2〜5GB | 強いアイデンティティ、高い計算コスト |
| T2I-Adapter | より軽量なControlNetの代替 | 70MB | エッジデバイス、推論予算 |

ControlNet ≈ 空間的。LoRA ≈ 意味的。両方使う。

## 実装する

`code/main.py` は1次元で2つのメカニズムをシミュレートする：

1. **LoRA。** 事前学習済み線形層 `W`。凍結する。`W + BA` が目標線形層にマッチするよう低ランク `B @ A` を訓練する。`r = 1` でランク1の補正を完璧に学習するのに十分なことを示す。

2. **ControlNet-lite。** 「凍結ベース」予測器と追加シグナルを読み取る「サイドネットワーク」。サイドネットワークの出力はゼロに初期化された学習可能なスカラーでゲートされる（ゼロ畳み込みのバージョン）。訓練し、ゲートがランプアップするのを観察する。

### ステップ1：LoRAの数学

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### ステップ2：ゼロ初期化サイドネットワーク

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

ステップ0では出力はベースと同一だ。訓練初期は `gate` がゆっくり更新される——壊滅的なドリフトなし。

## 落とし穴

- **LoRAのオーバースケーリング。** `α = 2` や `α = 3` は一般的な「強くしたい」ハックだが、過スタイライズされた/壊れた出力を生成する。`α ≤ 1.5` を維持する。
- **ControlNetの重みの衝突。** ポーズControlNetを重み1.0、深度ControlNetを重み1.0で使うと通常オーバーシュートする。重みの合計≈1.0が安全なデフォルト。
- **間違ったベースのLoRA。** SDXLのLoRAはアテンション次元が一致しないためSD 1.5で静かにno-opになる。Diffusers 0.30+で警告が出る。
- **Textual Inversionのドリフト。** あるチェックポイントで訓練されたトークンは別のチェックポイントで大きくドリフトする。LoRAの方が移植性が高い。
- **LoRAのウェイトマージと保存。** LoRAをベースモデルウェイトにベイク（より速い推論のため、ランタイムでの加算不要）できるが、実行時に `α` をスケーリングする能力を失う。両バージョンを保持する。

## 使ってみる

| 目標 | 2026年のパイプライン |
|------|---------------|
| ブランドのアートスタイルを再現する | ランク32で約30枚のキュレーションされた画像で訓練したLoRA |
| 生成画像に自分の顔を入れる | DreamBoothまたはLoRA + IP-Adapter-FaceID |
| 特定のポーズ＋プロンプト | ControlNet-Openpose + SDXL + テキスト |
| 深度対応の構図 | ControlNet-Depth + SD3 |
| リファレンス＋プロンプト | IP-Adapter + テキスト |
| 正確なレイアウト | ControlNet-ScribbleまたはControlNet-Canny |
| 背景の置き換え | ControlNet-Seg + インペインティング（レッスン09） |
| 高速1ステップスタイル | SDXL-TurboのLCM-LoRA |

## 成果物を出す

`outputs/skill-sd-toolkit-composer.md` を保存する。スキルはタスク（入力アセット：プロンプト、オプションのリファレンス画像、オプションのポーズ、オプションの深度、オプションのスクリブル）を受け取り、ツールスタック、ウェイト、再現可能なシードプロトコルを出力する。

## 演習

1. **易。** `code/main.py` でLoRAのランク `r` を1から4まで変える。どのランクでLoRAがランク2の目標デルタを正確にマッチするか？
2. **中。** 2つの目標変換で別々のLoRAを訓練する。それらを一緒に読み込み、加算的な相互作用を示す。どこで相互作用が線形性を破るか？
3. **難。** diffusersを使ってスタックする：SDXL-base + Canny-ControlNet（重み0.8）+ スタイルLoRA（α 0.8）+ IP-Adapter（重み0.6）。スタックの重みが変化するにつれFID対プロンプト順守のトレードオフを測定する。

## キーワード

| 用語 | 一般的な言い方 | 実際の意味 |
|------|-----------------|-----------------------|
| ControlNet | 「空間的制御」 | クローンされたエンコーダ＋ゼロ畳み込みスキップ；条件付け画像を読み取る。 |
| ゼロ畳み込み | 「恒等から始まる」 | ゼロに初期化された1×1畳み込み；ControlNetはノーオペから始まる。 |
| LoRA | 「低ランクアダプタ」 | `W + B @ A`、`r << d`；フルファインチューンより100倍少ないパラメータ。 |
| ランクr | 「ノブ」 | LoRAの圧縮；4〜16が典型、重いパーソナライゼーションには64以上。 |
| α | 「LoRAの強度」 | LoRAデルタのランタイムスケーリング。 |
| IP-Adapter | 「リファレンス画像」 | CLIPイメージトークンによる小さな画像条件付けアダプタ。 |
| DreamBooth | 「フルな題材ファインチューン」 | 題材の約30枚の画像でフルモデルを訓練する。 |
| Textual Inversion | 「新しいトークン」 | 新しい単語埋め込みのみを学習；レガシー、主に置き換えられた。 |

## プロダクションノート：LoRAのスワップ、ControlNetレーン、マルチテナントサービング

実際のテキスト→画像SaaSは同じベースチェックポイントで数百のLoRAと十数個のControlNetをサービスする。サービングの問題はLLMのマルチテナンシー（プロダクション文献は連続バッチングとLoRAX / S-LoRAの下でLLMのケースをカバーする）によく似ている：

- **LoRAをホットスワップし、マージしない。** `W' = W + α·B·A` をベースにマージするとステップごとの推論が約3〜5%速くなるが、`α` とベースが凍結される。LoRAをVRAMにランクrのデルタとしてホットに保持する；diffusersはリクエストごとの有効化のために `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` を公開している。スワップコストは `2 · d · r · num_layers` ウェイト——MBスケール、1秒未満。
- **ControlNetを第2のアテンションレーンとして。** クローンされたエンコーダはベースと並行して実行される。重み1.0のControlNetを2つ = ステップごとに2つの追加フォワードパス、1つのマージされたパスでなく。バッチサイズのヘッドルームが二次的に低下する。アクティブなControlNetごとに約1.5倍のステップコストを予算する。
- **量子化されたLoRAも。** ベースを量子化した場合（レッスン07の8GBでのFluxを参照）、LoRAデルタも8ビットまたは4ビットにきれいに量子化される。QLoRAスタイルの読み込みにより、4ビットのFluxベースに5〜10個のLoRAをメモリを爆発させずにスタックできる。

Flux固有：NielsのFlux-on-8GBノートブックはベースを4ビットに量子化する；その量子化ベースにスタイルLoRA（`pipe.load_lora_weights("user/style-lora")`）を `weight_name="pytorch_lora_weights.safetensors"` でスタックすることは依然として機能する。これが2026年のほとんどのSaaS代理店が出荷するレシピだ。

## 参考資料

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet。
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA（元はLLM用；拡散に移植）。
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter。
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — ControlNetのより軽量な代替。
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth。
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapterドキュメント](https://huggingface.co/docs/diffusers/training/controlnet) — リファレンスパイプライン。
