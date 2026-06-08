# 対話状態追跡

> 「北の安いレストランが欲しい...やっぱり普通価格で...それとイタリア料理を追加して。」3ターン、3回の状態更新。DSTはスロット-値の辞書を同期させて予約が機能するようにする。


## 問題設定

タスク指向の対話システムでは、ユーザーの目標はスロット-値のペアのセットとしてエンコードされる：`{cuisine: italian, area: north, price: moderate}`。すべてのユーザーターンがスロットを追加、変更、または削除できる。システムは会話全体を読み、現在の状態を正確に出力しなければならない。

一つのスロットを間違えると、システムは間違ったレストランを予約し、間違ったフライトをスケジュールし、間違ったカードに課金する。DSTはユーザーが言ったこととバックエンドが実行することの間の蝶番だ。

2026年にLLMにもかかわらずまだ重要な理由：

- コンプライアンス敏感なドメイン（銀行、医療、航空予約）は自由形式の生成ではなく、決定論的なスロット値を必要とする。
- ツール使用エージェントはAPIを呼び出す前にまだスロット解決が必要だ。
- マルチターン訂正は見た目より難しい：「やっぱり木曜にしてください。」

現代のパイプライン：古典的DST概念 + LLMエクストラクター + 構造化出力ガードレール。

## 概念

![DST: ダイアログ履歴 → スロット-値の状態](../assets/dst.svg)

**タスク構造。** スキーマはドメイン（レストラン、ホテル、タクシー）とそのスロット（料理、エリア、価格、人数）を定義する。各スロットは空、クローズドセットの値（price: {cheap, moderate, expensive}）、または自由形式の値（name: "The Copper Kettle"）で埋めることができる。

**2つのDST定式化。**

- **分類。** 各（スロット、候補値）ペアに対してyes/noを予測する。クローズド語彙スロットで機能する。2020年以前の標準。
- **生成。** ダイアログを与えられた場合、スロット値をフリーテキストとして生成する。オープン語彙スロットで機能する。現代のデフォルト。

**指標。** 共同目標精度（JGA）— *すべての*スロットが正しいターンの割合。全か無か。MultiWOZ 2.4のリーダーボードは2026年に約83%がトップ。

**アーキテクチャ。**

1. **ルールベース（スロット正規表現 + キーワード）。** 狭いドメインの強力なベースライン。デバッグ可能。
2. **TripPy / BERT-DST。** BERTエンコーディングを持つコピーベースの生成。LLM以前の標準。
3. **LDST（LLaMA + LoRA）。** ドメイン-スロットプロンプト付きの命令チューニングLLM。MultiWOZ 2.4でChatGPTレベルの品質に達する。
4. **オントロジーフリー（2024〜26年）。** スキーマをスキップ；スロット名と値を直接生成する。オープンドメインを処理する。
5. **プロンプト + 構造化出力（2024〜26年）。** Pydanticスキーマ + 制約付きデコーディングを持つLLM。コード5行、本番対応。

### 古典的な失敗モード

- **ターンをまたいだ共参照。** 「最初のオプションにしましょう。」どのオプションかを解決する必要がある。
- **上書き対追記。** ユーザーが「イタリア料理を追加して」と言う。料理を置き換えるか追記するか？
- **暗黙的な確認。** 「はい、いいですよ」— それは提案された予約を承諾したか？
- **訂正。** 「やっぱり午後7時にしてください。」他のスロットをクリアせずに時間を更新しなければならない。
- **以前のシステム発話への共参照。** 「はい、それです。」「それ」とは何か？

## 実装する

### ステップ1: ルールベースのスロットエクストラクター

`code/main.py`を参照。正規表現 + 同義語辞書は狭いドメインの正規発話の70%をカバーする：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

正規語彙の外では脆い。決定論的なスロット確認に機能する。

### ステップ2: 状態更新ループ

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

3つの不変条件：

- ユーザーが触れなかったスロットをリセットしない。
- 明示的な否定（「料理はいらない」）はクリアしなければならない。
- ユーザーの訂正（「やっぱり...」）は追記ではなく上書きしなければならない。

### ステップ3: 構造化出力を持つLLM駆動DST

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydanticは有効な状態オブジェクトを保証する。正規表現なし、スキーマの不一致なし、幻覚スロットなし。

### ステップ4: JGA評価

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

キャリブレーション：システムがすべてのスロットを正確に取得するターンの割合は？MultiWOZ 2.4では、2026年のトップシステム：80〜83%。狭い語彙でドメイン内システムがそれを超えなければ、LLMベースラインが上回る。

### ステップ5: 訂正の処理

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

訂正が検出された場合、追記ではなく最後に更新されたスロットを上書きする。LLMの助けなしに正しく行うのは難しい。現代のパターン：常にLLMが履歴全体から状態を再生成させる — これで訂正が自然に処理される。

## ピットフォール

- **全履歴再生成のコスト。** LLMに各ターンで状態を再生成させると総トークンがO(n²)かかる。履歴を上限設定するか古いターンを要約する。
- **スキーマドリフト。** 事後に新しいスロットを追加すると古い学習データが壊れる。スキーマのバージョン管理をする。
- **大文字小文字の区別。** 「Italian」対「italian」対「ITALIAN」— どこでも正規化する。
- **暗黙的な継承。** ユーザーが「4人で」を以前に指定した場合、別の時間の新しいリクエストで人数をクリアしてはいけない。常に全履歴を渡す。
- **自由形式対クローズドセット。** 名前、時間、住所は自由形式のスロットが必要；料理とエリアはクローズド。スキーマで両方を混在させる。

## 使ってみる

2026年のスタック：

| 状況 | アプローチ |
|------|----------|
| 狭いドメイン（1〜2つのインテント） | ルールベース + 正規表現 |
| 広いドメイン、ラベル付きデータあり | LDST（LLaMA + MultiWOZスタイルデータでのLoRA） |
| 広いドメイン、ラベルなし、本番対応 | LLM + Instructor + Pydanticスキーマ |
| 音声/ボイス | ASR + ノーマライザー + LLM-DST |
| マルチドメイン予約フロー | ドメインごとのPydanticモデルを持つスキーマガイドLLM |
| コンプライアンス敏感 | ルールベースプライマリ、確認フロー付きLLMフォールバック |

## 成果物を出す

`outputs/skill-dst-designer.md`として保存：

```markdown
---
name: dst-designer
description: 対話状態追跡器を設計する — スキーマ、エクストラクター、更新ポリシー、評価。
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

ユースケース（ドメイン、言語、語彙の開放性、コンプライアンス要件）を与えられた場合、以下を出力する：

1. スキーマ。ドメインリスト、ドメインごとのスロット、スロットごとのオープン対クローズド語彙。
2. エクストラクター。ルールベース/seq2seq/LLM+Pydantic。理由。
3. 更新ポリシー。全状態再生成/インクリメンタル；訂正処理；否定処理。
4. 評価。保留ダイアログセットでの共同目標精度、スロットレベルの精度/再現率、最も難しいスロットでの混同。
5. 確認フロー。いつユーザーに明示的に確認を求めるか（破壊的アクション、低信頼度の抽出）。

コンプライアンス敏感なスロットのルールベースのセカンダリチェックなしのLLM専用DSTを拒否する。ユーザーの訂正でスロットをロールバックできないDSTを拒否する。バージョンタグのないスキーマをフラグする。
```

## 演習

1. **易。** `code/main.py`で3つのスロット（料理、エリア、価格）のルールベースの状態追跡器を構築する。10個の手作りのダイアログでテストする。JGAを測定する。
2. **中。** Instructor + Pydantic + 小型LLMで同じデータセット。JGAを比較する。最も難しいターンを検査する。
3. **難。** 両方を実装してルーティングする：ルールベースプライマリ、ルールベースが2未満のスロットを信頼度で出力する場合にLLMフォールバック。組み合わせたJGAとターンあたりの推論コストを測定する。

## キーワード

| 用語 | 一般的な説明 | 実際の意味 |
|------|------------|----------|
| DST | 対話状態追跡 | ダイアログのターンをまたいでスロット-値の辞書を維持する。 |
| スロット | ユーザーインテントの単位 | バックエンドが必要とする名前付きパラメータ（料理、日付）。 |
| ドメイン | タスク領域 | レストラン、ホテル、タクシー — スロットのセット。 |
| JGA | 共同目標精度 | すべてのスロットが正確なターンの割合。全か無か。 |
| MultiWOZ | ベンチマーク | マルチドメインWOZデータセット；標準的なDST評価。 |
| オントロジーフリーDST | スキーマなし | スロット名と値を直接生成する、固定リストなし。 |
| 訂正 | 「やっぱり...」 | 以前に埋められたスロットを上書きするターン。 |

## 参考資料

- [Budzianowski et al. (2018). MultiWOZ — 大規模マルチドメインウィザード・オブ・オズ](https://arxiv.org/abs/1810.00278) — 標準的なベンチマーク。
- [Feng et al. (2023). LLM駆動の対話状態追跡に向けて（LDST）](https://arxiv.org/abs/2310.14970) — DST用のLLaMA + LoRA命令チューニング。
- [Heck et al. (2020). TripPy — 値独立ニューラル対話状態追跡のためのトリプルコピー戦略](https://arxiv.org/abs/2005.02877) — コピーベースDSTの主力。
- [King, Flanigan (2024). LLMを使った教師なしエンドツーエンドタスク指向ダイアログ](https://arxiv.org/abs/2404.10753) — EMベースの教師なしTOD。
- [MultiWOZリーダーボード](https://github.com/budzianowski/multiwoz) — 正規化DST結果。
