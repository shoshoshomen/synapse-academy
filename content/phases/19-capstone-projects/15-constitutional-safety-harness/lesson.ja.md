# キャップストーン 15 — 憲法的安全ハーネス + レッドチームレンジ

> Anthropicの憲法的分類器、MetaのLlama Guard 4、GoogleのShieldGemma-2、NVIDIAのNemotron 3コンテンツ安全機能、そして多言語カバレッジ用のX-Guardが2026年の安全分類器スタックを定義した。garak、PyRIT、NVIDIA Aegis、promptfooが標準的な敵対的評価ツールとなった。NeMo Guardrails v0.12がこれらをプロダクションパイプラインに統合する。このキャップストーンでは、ターゲットアプリへの多層安全ハーネス、6種類以上の攻撃ファミリーを実行する自律型レッドチームエージェント、そして測定可能な無害性デルタを生成する憲法的自己批評実行をすべて組み合わせる。

**演習フェーズ:** P10 · P11 · P13 · P14 · P18

## 問題

2026年のLLM安全性の最前線は、分類器が機能するかどうか（おおむね機能する）ではなく、過剰拒否や明らかな穴を残すことなく、プロダクションアプリの周囲にそれらを正しく組み合わせる方法にある。Llama Guard 4は英語のポリシー違反を処理する。X-Guard（132言語）は多言語ジェイルブレイクを処理する。ShieldGemma-2は画像ベースのプロンプトインジェクションを検出する。NVIDIA Nemotron 3コンテンツ安全機能はエンタープライズカテゴリをカバーする。AnthropicのConstitutional Classifiersは、推論時ではなく学習時に使用される別のアプローチだ。

攻撃の進化も重要だ。PAIRとTAPはジェイルブレイク発見を自動化する。GCGは勾配ベースのサフィックス攻撃を実行する。マルチターンとコードスイッチ攻撃はエージェントのメモリを悪用する。デプロイされたLLMはいずれも、レッドチームレンジ（garakとPyRITが標準ドライバー）と、文書化されたミティゲーションおよびCVSSスコア付きの調査結果が必要だ。

ターゲットアプリケーション（8Bの命令チューニング済みモデル、または他のキャップストーンのRAGチャットボットのいずれか）を強化し、6種類以上の攻撃ファミリーを実行し、ビフォー/アフターの無害性測定を生成する。

## コンセプト

安全パイプラインは5層で構成される。**入力サニタイズ**: ゼロ幅文字を削除し、base64/rot13をデコードし、Unicodeを正規化する。**ポリシー層**: NeMo Guardrails v0.12のレール（オフドメイン、毒性、PII抽出）。**分類器ゲート**: 入力のLlama Guard 4、非英語のX-Guard、画像入力のShieldGemma-2。**モデル**: ターゲットLLM。**出力フィルター**: 出力のLlama Guard 4、Presidio PIIスクラブ、該当箇所での引用強制。**HITLティア**: 高リスクとフラグされた出力はSlackキューに送られる。

レッドチームレンジはスケジューラーで動作する。PAIRとTAPがジェイルブレイクを自律的に発見する。GCGが勾配ベースのサフィックス攻撃を実行する。ASCII/base64/rot13エンコーディング攻撃。マルチターン攻撃（ペルソナ採用、メモリ悪用）。コードスイッチ攻撃（英語とスワヒリ語またはタイ語の混在）。各実行はCVSSスコアリングと開示タイムラインを含む構造化された調査結果ファイルを生成する。

憲法的自己批評実行はトレーニング時の介入だ。有害試行プロンプト1kを取得し、モデルに応答の草稿を作成させ、成文化された憲法（害を与えないルール）に対して批評し、批評ループでの再学習を行う。保留評価での前後の無害性デルタを測定する。

## アーキテクチャ

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## スタック

- 安全分類器: Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 Content Safety、X-Guard
- ガードレールフレームワーク: NeMo Guardrails v0.12 + OPA
- レッドチームドライバー: garak (NVIDIA)、PyRIT (Microsoft Azure)、NVIDIA Aegis、promptfoo
- ジェイルブレイクエージェント: PAIR (Chao et al., 2023)、Tree-of-Attacks (TAP)、GCGサフィックス
- 憲法的トレーニング: Anthropicスタイルの自己批評ループ + 批評でのSFT
- PIIスクラブ: Presidio
- ターゲット: 8Bの命令チューニング済みモデル、または他のキャップストーンのRAGチャットボット

## 実装する

1. **ターゲットセットアップ。** vLLM上に8Bの命令チューニング済みモデルを立ち上げる（または別のキャップストーンのRAGチャットボットを再利用する）。これがテスト対象のアプリだ。

2. **安全パイプラインのラップ。** ターゲットの周囲に5層のパイプラインを接続する。各層が個別に観測可能であることを確認する（Langfuseで層ごとのスパン）。

3. **分類器カバレッジ。** Llama Guard 4、X-Guard（多言語）、ShieldGemma-2（画像）を読み込む。各分類器を小さなラベル付きセットで実行してベースラインを確立する。

4. **レッドチームスケジューラー。** garak、PyRIT、PAIRエージェント、TAPエージェント、GCGランナー、マルチターン攻撃者、コードスイッチ攻撃者をスケジュールする。各々が別のキューで動作する。

5. **攻撃スイート。** 6種類の攻撃ファミリー: (1) PAIR自動ジェイルブレイク、(2) TAPツリーオブアタックス、(3) GCG勾配サフィックス、(4) ASCII/base64/rot13エンコーディング、(5) マルチターンペルソナ、(6) 多言語コードスイッチ。ファミリーごとの成功率を報告する。

6. **憲法的自己批評。** 有害試行プロンプト1kをキュレートする。各プロンプトについて、ターゲットが応答を草稿する。批評LLMが成文化された憲法（「害を与えない」「証拠を引用する」「違法な要求を拒否する」）に対してスコアリングする。批評者が異議を申し立てるプロンプトは書き直され、ターゲットは批評改善ペアで微調整する。保留評価での前後の無害性を測定する。

7. **過剰拒否の測定。** 良性プロンプトスイート（例: XSTest）での偽陽性率を追跡する。ターゲットは良性の質問に対して役立つままでなければならない。

8. **CVSSスコアリング。** 成功した各ジェイルブレイクについて、CVSS 4.0（攻撃ベクター、複雑さ、影響）でスコアリングする。開示タイムラインとミティゲーション計画を作成する。

9. **レンジ自動化。** 上記すべてがcronで動作し、調査結果がキューに書き込まれ、過剰拒否のリグレッションアラートがSlackに送信される。

## 使ってみる

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## 成果物を出す

`outputs/skill-safety-harness.md`が成果物だ。プロダクション品質の多層安全パイプラインと、前後の無害性デルタを持つ再現可能なレッドチームレンジ。

| 重み | 基準 | 測定方法 |
|:-:|---|---|
| 25 | 攻撃サーフェスカバレッジ | 6種類以上の攻撃ファミリー、2言語以上 |
| 20 | 真陽性/偽陽性のトレードオフ | 攻撃ブロック率 vs XSTest良性通過率 |
| 20 | 自己批評デルタ | 保留評価での前後の無害性 |
| 20 | ドキュメントと開示 | タイムライン付きのCVSSスコア調査結果 |
| 15 | 自動化と再現性 | アラート付きのcronですべて動作 |
| **100** | | |

## 演習

1. RAGチャットボットでgarakのプロンプトインジェクションプラグインを実行し、出力フィルター層の有無での攻撃成功率を比較する。

2. 7番目の攻撃ファミリーを追加する: 取得文書経由の間接プロンプトインジェクション。必要な追加防御を測定する。

3. 「ヘルプ付き拒否」モードを実装する: ガードレールがブロックした場合、ターゲットは単純な拒否の代わりに、より安全な関連回答を提供する。XSTestデルタを測定する。

4. 多言語カバレッジのギャップ: X-Guardのパフォーマンスが低い言語を見つける。それを対象とした微調整データセットを提案する。

5. 30Bモデルで憲法的自己批評を実行し、デルタがスケールするかどうかを測定する。

## キーワード

| 用語 | 一般的な言い方 | 実際の意味 |
|------|-----------------|------------------------|
| 多層安全 | "Defense in depth" | 入力、ゲート、出力、HITLでの複数ガードレール |
| Llama Guard 4 | "Metaの安全分類器" | 2026年の入力/出力コンテンツ分類の参照実装 |
| PAIR | "ジェイルブレイクエージェント" | LLM駆動ジェイルブレイク発見に関する論文（Chao et al.） |
| TAP | "Tree-of-Attacks" | PAIRのツリーサーチ変形 |
| GCG | "Greedy coordinate gradient" | 勾配ベースの敵対的サフィックス攻撃 |
| 憲法的自己批評 | "Anthropicスタイルのトレーニング" | ターゲットが草稿→批評者がスコア→書き直し→再学習 |
| XSTest | "良性プローブセット" | 過剰拒否リグレッションのベンチマーク |
| CVSS 4.0 | "深刻度スコア" | 安全調査結果の標準脆弱性スコアリング |

## 参考資料

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — トレーニング時の参照
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026年の入力/出力分類器
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 画像+マルチモーダル安全
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — エンタープライズ参照
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 132言語の多言語安全
- [garak](https://github.com/NVIDIA/garak) — NVIDIAレッドチームツールキット
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoftレッドチームフレームワーク
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — レールフレームワーク
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — ジェイルブレイクエージェント論文
