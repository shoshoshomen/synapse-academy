# Vision Transformer（ViT）

> 画像はパッチのグリッドだ。文はトークンのグリッドだ。同じTransformerが両方を食べる。


## 問題

2020年以前、コンピュータービジョンは畳み込みを意味していた。ImageNet、COCO、検出ベンチマークのすべてのSOTAがCNNバックボーンを使っていた。TransformerはNLP用だった。

Dosovitskiy et al.（2020年）——「An Image is Worth 16x16 Words」——は畳み込みを完全に廃棄できることを示した。画像を固定サイズのパッチにスライスし、各パッチを線形に埋め込みに射影し、シーケンスをバニラのTransformerエンコーダに供給する。十分なスケール（ImageNet-21k事前学習またはそれ以上）でViTはResNetベースのモデルに匹敵するか上回る。

ViTは2026年のより広いパターンの始まりだった：1つのアーキテクチャ、多くのモダリティ。Whisperは音声をトークン化する。ViTは画像をトークン化する。ロボティクス用のアクショントークン。映像用のピクセルトークン。Transformerは気にしない——シーケンスを与えれば学習する。

2026年までに、ViTとその子孫（DeiT、Swin、DINOv2、ViT-22B、SAM 3）がほとんどのビジョンを所有している。CNNはエッジデバイスとレイテンシ重視のタスクで勝ち続ける。それ以外のすべてはスタックのどこかにViTがある。

## 概念

![画像 → パッチ → トークン → Transformer](../assets/vit.svg)

### ステップ1 — パッチ化

`H × W × C`の画像を`N × (P·P·C)`の平坦なパッチのシーケンスに分割する。典型的な設定：`224 × 224`画像、`16 × 16`パッチ → 768値の196パッチ。

```
画像 (224, 224, 3) → 16x16x3パッチの14 × 14グリッド → 長さ768の196ベクトル
```

パッチサイズがレバーだ。小さいパッチ = より多くのトークン、より高い解像度、二次のアテンションコスト。大きいパッチ = より粗い、より安価。

### ステップ2 — 線形埋め込み

単一の学習済み行列が各平坦パッチを`d_model`に射影する。カーネルサイズ`P`、ストライド`P`の畳み込みと等価だ。PyTorchではこれは文字通り`nn.Conv2d(C, d_model, kernel_size=P, stride=P)`——2行の実装だ。

### ステップ3 — `[CLS]`トークンを先頭に追加し、位置埋め込みを追加する

- 学習可能な`[CLS]`トークンを先頭に追加する。その最終隠れ状態が分類に使われる画像表現だ。
- 学習可能な位置埋め込み（ViT元祖）またはSinusoidal 2D（後のバリアント）を追加する。
- 2024年以降、位置に明示的な埋め込みなしで2DのRoPEを使う場合もある。

### ステップ4 — 標準のTransformerエンコーダ

`LayerNorm → Self-Attention → + → LayerNorm → MLP → +`のLブロックを積み重ねる。BERTと同一。ビジョン特有の層なし。これが論文の教育的なパンチラインだ。

### ステップ5 — ヘッド

分類では：`[CLS]`隠れ状態 → 線形 → softmax。DINOv2やSAMでは、`[CLS]`を廃棄し、パッチ埋め込みを直接使う。

### 重要なバリアント

| モデル | 年 | 変更 |
|-------|------|--------|
| ViT | 2020 | 元祖。固定パッチサイズ、完全グローバルアテンション。 |
| DeiT | 2021 | 蒸留；ImageNet-1kのみで学習可能。 |
| Swin | 2021 | シフトウィンドウによる階層型。固定二次未満のコスト。 |
| DINOv2 | 2023 | 自己教師（ラベルなし）。最も優れた一般的なビジョン特徴。 |
| ViT-22B | 2023 | 220億パラメータ；スケーリング則が適用される。 |
| SigLIP | 2023 | ViT + 言語ペア、シグモイド対照損失。 |
| SAM 3 | 2025 | 何でもセグメント；ViT-Large + プロンプタブルマスクデコーダ。 |

### 時間がかかった理由

ViTはCNNの帰納的バイアス（平行移動不変性、局所性）を何も持っていないため、CNNに匹敵するのに*大量*のデータが必要だ。>1億のラベル付き画像または強力な自己教師付き事前学習なしでは、マッチした計算でCNNが勝つ。DeiTは2021年に蒸留トリックでこれを解決した；DINOv2は2023年に自己教師付き学習で永続的に解決した。

## 実装する

`code/main.py`を参照。純粋なstdlibのパッチ化 + 線形埋め込み + サニティチェック。学習なし——現実的なスケールのViTはPyTorchとGPUでの時間が必要だ。

### ステップ1：偽の画像

リストの行として24 × 24 RGB画像。`(R, G, B)`タプル。6×6パッチを使用 → 16パッチ、各108次元の埋め込みベクトル。

### ステップ2：パッチ化

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

ラスタ順：グリッドを横断する行優先。すべてのViTがこの順序を使う。

### ステップ3：線形埋め込み

各平坦パッチをランダムな`(patch_flat_size, d_model)`行列で乗算する。`[CLS]`を先頭に追加した後、出力形状が`(N_patches + 1, d_model)`であることを確認する。

### ステップ4：現実的なViTのパラメータ数を数える

ViT-Baseのパラメータ数を出力する：12層、12ヘッド、d=768、patch=16。ResNet-50（〜25M）と比較する。ViT-Baseは〜86Mになる。ViT-Large〜307M。ViT-Huge〜632M。

## 使ってみる

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196パッチ
cls_emb = out[:, 0]                       # 画像表現
```

**DINOv2埋め込みは2026年の画像特徴のデフォルトだ。** バックボーンを凍結し、小さなヘッドを学習する。分類、検索、検出、キャプションに機能する。MetaのDINOv2チェックポイントはすべてのテキスト以外のビジョンタスクでCLIPを上回る。

**パッチサイズの選択。** 小さなモデルは16×16（ViT-B/16）を使う。密な予測（セグメンテーション）は8×8または14×14（SAM、DINOv2）を使う。非常に大きなモデルは14×14を使う。

## 成果物を出す

`outputs/skill-vit-configurator.md`を参照。このスキルは、データセットサイズ・解像度・計算予算を考慮して、新しいビジョンタスクのViTバリアントとパッチサイズを選択する。

## 演習

1. **易.** `code/main.py`を実行せよ。パッチ数が`(H/P) * (W/P)`に等しく、平坦パッチ次元が`P*P*C`に等しいことを確認せよ。
2. **中.** 2D Sinusoidal位置埋め込みを実装せよ——各パッチの`row`と`col`のための2つの独立したSinusoidalコード、連結。小さなPyTorch ViTに供給し、CIFAR-10で学習可能な位置埋め込みと精度を比較せよ。
3. **難.** 3層ViT（PyTorch）を構築し、4×4パッチで1,000 MNIST画像で学習せよ。テスト精度を測定せよ。次に同じ1,000画像でDINOv2事前学習を追加せよ（簡略化：エンコーダをマスクされたパッチ埋め込みを予測するよう学習するだけ）。精度は改善するか？

## キーワード

| 用語 | 人々が言うこと | 実際の意味 |
|------|-----------------|-----------------------|
| パッチ（Patch） | 「ビジョントランスフォーマーのトークン」 | 画像の`P × P × C`領域のピクセル値の平坦ベクトル。 |
| パッチ化（Patchify） | 「切り刻んで平坦化」 | 画像を重複しないパッチにスライスし、各々をベクトルに平坦化。 |
| `[CLS]`トークン | 「画像の要約」 | 先頭に追加された学習可能なトークン；その最終埋め込みが画像表現。 |
| 帰納的バイアス（Inductive bias） | 「モデルが仮定すること」 | ViTはCNNより事前分布が少ない；ギャップを埋めるためにより多くのデータが必要。 |
| DINOv2 | 「自己教師ViT」 | ラベルなしで画像増強 + モメンタム教師を使って学習。2026年の最良の一般的な画像特徴。 |
| SigLIP | 「CLIPの後継者」 | シグモイド対照損失で学習されたViT + テキストエンコーダ；マッチした計算でCLIPを上回る。 |
| Swin | 「ウィンドウViT」 | ローカルアテンション + シフトウィンドウの階層型ViT；二次未満。 |
| レジスタトークン（Register tokens） | 「2023年のトリック」 | アテンションシンクを吸収するいくつかの追加学習可能トークン；DINOv2特徴を改善する。 |

## 参考資料

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT論文。
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT。
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin。
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2。
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — DINOv2のレジスタトークン修正。
