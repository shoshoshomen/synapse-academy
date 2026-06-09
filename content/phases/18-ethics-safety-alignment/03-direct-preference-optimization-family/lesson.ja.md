# 直接選好最適化ファミリー

> Rafailov ら (2023) は、RLHF の最適解が選好データで閉形式になることを示した。つまり明示的な報酬モデルを省略してポリシーを直接最適化できる。この洞察が一つのファミリーを生み出した——IPO、KTO、SimPO、ORPO、BPO——それぞれが DPO の失敗モードを修正している。2026 年においては、直接アライメントアルゴリズム (DAA) がフロンティアのポストトレーニング実行で PPO を上回る数を占める。しかしレッスン 2 の過剰最適化曲線は依然として当てはまる：DAA は Goodhart から逃れるのではなく、どこで噛まれるかを変えるだけだ。


## 学習目標

- RLHF-with-KL の最適解から DPO の閉形式を導出できる。
- IPO、KTO、SimPO、ORPO、BPO がそれぞれ DPO のどの失敗モードを修正しているか述べられる。
- 「暗示的報酬ギャップ」と「選好強度」を区別し、IPO の恒等写像がなぜ重要かを説明できる。
- 明示的な RM を持たないにもかかわらず、Rafailov ら (NeurIPS 2024) が DAA が過剰最適化することを証明している理由を説明できる。

## 問題の背景

RLHF の目的関数（レッスン 1）：

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

には既知の最適解がある：

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

したがって報酬は最適ポリシーと参照の比で暗示的に定義される：

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

これを Bradley-Terry 選好尤度に代入すると、分配関数 `Z(x)` は `x` のみに依存するため相殺される。残るのはポリシーパラメータだけの損失——報酬モデルは不要だ。これが DPO だ。

問題点：この導出は最適解が到達可能であること、選好データが分布内であること、参照ポリシーが真のモードアンカーであることを仮定している。これらはいずれも正確には成り立たない。ファミリーの各メンバーは異なる違反された仮定を修正する。

## 概念

### DPO (Rafailov ら, 2023)

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

起こりうる問題：

- 暗示的報酬ギャップ `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` が非有界。わずかな選好が任意に大きなギャップを生み出す。
- 損失は選ばれた応答と棄却された応答の対数確率を逆方向に駆動する。棄却された応答がより速く下落する限り、選ばれた応答の絶対対数確率を下げることができる。これが「選ばれた応答の劣化」現象だ。
- 分布外の選好（まれ対まれなペア）が任意の暗示的報酬を生み出す。

### IPO (Azar ら, 2024)

恒等選好最適化（Identity Preference Optimization）は、log-sigmoid を選好確率の恒等写像に置き換える。損失は有界なターゲット上の二乗誤差になる：

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

マージンは `1/(2 beta)` で有界になる。選好強度と暗示的報酬ギャップが比例する。爆発しない。

### KTO (Ethayarajh ら, 2024)

Kahneman-Tversky 最適化はペアワイズ構造を完全に廃棄する。単一のラベル付き出力と「望ましい」または「望ましくない」の二値シグナルが与えられると、プロスペクト理論のユーティリティにマッピングする：

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

利得と損失に異なる重みをつける（損失回避）。利点：ペアリングされていないデータを使えるため、はるかに豊富なデータが利用できる。

### SimPO (Meng ら, 2024)

シンプル選好最適化（Simple Preference Optimization）は、訓練シグナルを生成と一致させる。参照ポリシーを完全に除去し、対数尤度を長さで正規化する：

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

マージン `gamma` で安定化する。長さ正規化により、DPO の長さバイアス失敗モードを利用するインセンティブを除去する（長い `y_w` は構造上より大きな対数確率ギャップを生む）。

### ORPO (Hong ら, 2024)

オッズ比選好最適化（Odds-Ratio Preference Optimization）は、標準的な SFT の負の対数尤度に選好項を加える：

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

参照ポリシー不要——SFT 項が正則化項になる。ベースモデルからアライメント済みモデルまでをワンステージで訓練できる。別途の SFT チェックポイントは不要。

### BPO (ICLR 2026 submission, OpenReview id=b97EwMUWu7)

「選ばれた応答の劣化」問題を特定する：DPO はランキング `y_w > y_l` を保つが、`y_w` の絶対対数確率は下がる可能性がある。BPO は選ばれた応答の下方移動にペナルティを与える一行の修正を加える。Llama-3.1-8B-Instruct の数学推論において DPO より +10.1% の精度向上を報告している。

### 普遍的な結果：DAA もやはり過剰最適化する

Rafailov ら「Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms」（NeurIPS 2024）は、DPO、IPO、SLiC のポリシーを複数のデータセットと KL バジェットで訓練した。ゴールド報酬対 KL の曲線は Gao らのピーク後下落という同じ形状を示している。暗示的報酬は訓練中に分布外サンプルを問い合わせる；KL 正則化はこれを安定させない。

DAA は Goodhart から逃れない。噛まれる面を「報酬モデルの過剰最適化」から「参照ポリシー比の過剰最適化」に変えるだけだ。普遍的な修正——より良いデータ、アンサンブル、アーリーストッピング——は両方に適用される。

### 2026 年の選び方

- 大きなペアリングされた選好データがある場合：保守的な beta の DPO。長さバイアスが明らかな場合は SimPO。
- ペアリングされていない二値フィードバックがある場合：KTO。
- ベースモデルからワンステージパイプラインを望む場合：ORPO。
- DPO ログで選ばれた対数確率の劣化が見られる場合：BPO。
- 選好強度が広く変動し DPO が飽和している場合：IPO。

どのラボもすべての五つをバッテリーで実行し、タスクごとに勝者を選ぶ。数学推論と安全性で最適が同じである理由はない。

## 使ってみる

`code/main.py` は、真の選好強度がペアによって異なるおもちゃの選好データセットで六つの損失（DPO、IPO、KTO、SimPO、ORPO、BPO）を比較する。各損失は小さなソフトマックスポリシーを用いて同じ 500 ペアのサンプルに対して最適化される。最終的な勝率、選ばれた対数確率のドリフト、手法ごとの暗示的報酬の分散をプロットする。

## 成果物を出す

このレッスンでは `outputs/skill-preference-loss-selector.md` を作成する。データセット統計（ペアリング済み対未ペアリング、可変対均一の選好強度、長さ分布）とターゲット（ワンステージまたは SFT-then-preference）が与えられた際に、選好損失を推奨してそれが防ぐ失敗モードを報告する。

## 演習

1. `code/main.py` を実行する。DPO と BPO の最終的な選ばれた対数確率の下落を報告する。BPO の方が選ばれた絶対確率が高いことを検証する。

2. すべてのペアが等しい強度を持つよう選好データを修正する。六つの手法のうち最も頑健なのはどれか？どれが劣化するか？IPO の優位性を説明する。

3. 棄却された応答を平均的に選ばれた応答の 2 倍の長さにする。他を変えずに、DPO の長さ利用を数値的に示し、SimPO の修正を確認する。

4. Rafailov ら（NeurIPS 2024）は DAA が過剰最適化すると主張している。単一ポイント版を再現する：選ばれた差し引かれた棄却された KL 発散をプロットし、大きな beta で DPO の過剰最適化を観察する。

5. BPO 論文のアブストラクト（OpenReview b97EwMUWu7）を読む。BPO が DPO に加える一行の修正を書き下す。`code/main.py` の実装と照合して確認する。

## キーワード

| 用語 | 一般的な呼び方 | 実際の意味 |
|------|--------------|-----------|
| DPO | 「報酬モデルなしの RLHF」 | RLHF 最適解の閉形式から導出される損失；ポリシーパラメータのみ |
| 暗示的報酬 | 「対数比」 | `beta * log(pi(y\|x) / pi_ref(y\|x))` — DPO が暗示する報酬 |
| IPO | 「有界 DPO」 | log-sigmoid を恒等写像に置き換える；暗示的報酬ギャップを `1/(2 beta)` で上限 |
| KTO | 「未ペアリング DPO」 | 損失回避を持つ単一ラベルのプロスペクト理論ユーティリティ |
| SimPO | 「参照なし DPO」 | 長さ正規化対数尤度＋マージン；参照ポリシー不要 |
| ORPO | 「ワンステージ DPO」 | NLL＋オッズ比選好項；ベースモデルからワンパスで訓練 |
| BPO | 「選ばれた応答保護 DPO」 | DPO に選ばれた応答の絶対対数確率を下げることへのペナルティを追加 |
| 選ばれた応答の劣化 | 「選ばれた応答が下がる」 | DPO は棄却されたものがより速く下落する限り、選ばれた対数確率を下げる |
| DAA | 「直接アライメントアルゴリズム」 | 明示的な RM を省略する任意の選好損失手法 |

## 参考資料

- [Rafailov ら — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar ら — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh ら — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov ら — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
