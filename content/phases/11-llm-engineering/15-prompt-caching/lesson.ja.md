# プロンプトキャッシングとコンテキストキャッシング

> あなたのシステムプロンプトは4,000トークンだ。RAGコンテキストは20,000トークンだ。両方をすべてのリクエストで送信する。そして両方に課金される——毎回。プロンプトキャッシングはプロバイダーがプレフィックスを維持し、再利用時に通常レートの10%で課金するようにする。正しく使えばコストを50〜90%削減し、最初のトークンまでのレイテンシを40〜85%削減できる。


## 問題

コーディングエージェントがすべての会話ターンで同じ15,000トークンのシステムプロンプトをClaudeに送信する。$3/M入力トークンで20ターンは$0.90の入力コストだけ——ユーザーの実際のメッセージがある前の話だ。1日10,000件の会話に掛け算すると、決して変わらないテキストの請求書が$9,000/日に達する。

品質を下げずにプロンプトを縮小することはできない。送信を避けることもできない——モデルはすべてのターンでそれが必要だ。唯一の手段はプロバイダーがすでに見たプレフィックスに全額を支払うのをやめることだ。

その手段がプロンプトキャッシングだ。Anthropicは2024年8月にそれを出荷し（2025年に1時間の拡張TTL変種付き）、OpenAIはその年後半に自動化し、GoogleはGemini 1.5とともに明示的なコンテキストキャッシングを出荷した。3社すべてが今やフロンティアモデルのファーストクラス機能として提供している。

## 概念

![Prompt caching: write once, read cheap](../assets/prompt-caching.svg)

**仕組み。** リクエストのプレフィックスが最近のリクエストのものと一致する場合、プロバイダーはトークンを再エンコードする代わりに前回の実行からKVキャッシュを提供する。最初の時は小さな書き込みプレミアムを支払い、それ以降のたびに大きな読み取り割引を受ける。

**2026年の3つのプロバイダータイプ。**

| プロバイダー | APIスタイル | ヒット割引 | 書き込みプレミアム | デフォルトTTL | 最小キャッシュ可能 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | コンテンツブロックへの明示的 `cache_control` マーカー | 入力90%オフ | 25%割増 | 5分（1時間まで延長可） | 1,024トークン（Sonnet/Opus）、2,048（Haiku） |
| OpenAI | 自動プレフィックス検出 | 入力50%オフ | なし | 最大1時間（ベストエフォート） | 1,024トークン |
| Google（Gemini） | 明示的 `CachedContent` API | ストレージ課金; 読み取りは通常の約25% | トークン・時間ごとのストレージ料金 | ユーザー設定（デフォルト1時間） | 4,096トークン（Flash）、32,768（Pro） |

**不変条件。** 3社すべてがプレフィックスのみをキャッシュする。リクエスト間で任意のトークンが異なれば、最初に異なるトークン以降のすべてはミスになる。*安定した*部分を上に置き、*可変*部分を下に置く。

### キャッシュフレンドリーなレイアウト

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

順序に違反する——システムプロンプトの上にユーザーメッセージを置く、few-shotの間に動的取得を入れ込む——とキャッシュは決してヒットしない。

### 損益分岐の計算

Anthropicの25%書き込みプレミアムは、キャッシュされたブロックが正味でコストを節約するには少なくとも2回読み込まれる必要があることを意味する。1回書き込み + 1回読み込みは平均してリクエストあたり0.675×のコスト（32%節約）; 1回書き込み + 10回読み込みは平均0.205×（80%節約）。経験則: TTL内に少なくとも3回再利用することが予想されるものはキャッシュする。

## 実装する

### ステップ1: 明示的マーカーを使用したAnthropicプロンプトキャッシング

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` マーカーはAnthropicにブロックを5分間保存するよう指示する。その時間内の再利用はヒットし、期限切れ後は再び書き込む。

**レスポンスのusageフィールド:**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```

CIで両方のフィールドを確認する——リクエスト間で `cache_read_input_tokens` がゼロのままなら、キャッシュキーがドリフトしている。

### ステップ2: 1時間の拡張TTL

長時間実行されるバッチジョブでは、5分のデフォルトがジョブ間で期限切れになる。`ttl` を設定する:

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1時間のTTLは書き込みプレミアムの2倍（ベースラインより50%増）かかるが、プレフィックスを5回以上再利用するバッチでは素早くペイバックする。

### ステップ3: OpenAI自動キャッシング

OpenAIは設定するものを何も提供しない。最近のリクエストと一致する1,024トークン以上のプレフィックスが自動的に50%の割引を受ける。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

同じキャッシュフレンドリーなレイアウトルールが適用される。Anthropicのキャッシュを破壊しないがOpenAIのキャッシュを破壊する2つのこと: `user` フィールドの変更（キャッシュキーのコンポーネントとして使用される）とツールの並べ替え。

### ステップ4: Gemini明示的コンテキストキャッシング

Geminiはキャッシュをあなたが作成して名前を付けるファーストクラスオブジェクトとして扱う:

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Geminiはキャッシュが生存している限りトークン・時間ごとにストレージを課金し、通常の入力レートの約25%で読み込む。これは何日にもわたって多くのセッション間で同じ巨大なプロンプトを再利用するときに正しい形だ。

### ステップ5: 本番でのヒット率の測定

シミュレートされた3プロバイダーアカウンタントのコードは `code/main.py` を参照。書き込み/読み取り/ミスカウントを追跡し、1Kリクエストあたりの混合コストを計算する。目標ヒット率でデプロイをゲートする——ほとんどの本番Anthropicセットアップはウォームアップ後に80%以上の読み込み比率を見るべきだ。

## 2026年にもまだ出荷されるピットフォール

- **上部の動的タイムスタンプ。** システムプロンプトの上部に `"Current time: 2026-04-22 15:30:02"` がある。すべてのリクエストがミスする。タイムスタンプをキャッシュブレークポイントの下に移動する。
- **ツールの並べ替え。** ツールを安定した順序でシリアライズする——デプロイ間のdict再シャッフルがすべてのヒットを破壊する。
- **フリーテキストの近似重複。** 「You are helpful.」対「You are a helpful assistant.」——1バイトの違いで完全ミス。
- **小さすぎるブロック。** Anthropicは1,024トークンのフロアを強制する（Haikuには2,048）。小さいブロックは静かにキャッシュされない。
- **盲目のコストダッシュボード。** 「入力トークン」をキャッシュ済みと非キャッシュに分割する。そうしないとトラフィックの減少がキャッシュの勝利のように見える。

## 使ってみる

2026年のキャッシングスタック:

| 状況 | 選択 |
|-----------|------|
| 安定した10k+システムプロンプトを持つエージェント、多くのターン | Anthropic `cache_control`（5分TTL） |
| 30分以上プレフィックスを再利用するバッチジョブ | Anthropic（`ttl: "1h"` 付き） |
| カスタムインフラなしのGPT-5上のサーバーレスエンドポイント | OpenAI自動（プレフィックスを安定して長くするだけ） |
| 巨大なコード/ドキュメントコーパスの複数日再利用 | Gemini明示的 `CachedContent` |
| クロスプロバイダーフォールバック | プロバイダー間でキャッシュ可能なプレフィックスレイアウトを同一に保つのでどのヒットも機能する |

ユーザーメッセージレイヤーのセマンティックキャッシング（フェーズ11・11）と組み合わせる: プロンプトキャッシングは*トークン同一*の再利用を処理し、セマンティックキャッシングは*意味同一*の再利用を処理する。

## 成果物を出す

`outputs/skill-prompt-caching-planner.md` を保存する:

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## 演習

1. **Easy（簡単）。** 5,000トークンのシステムプロンプトを持つClaudeとの10ターンの会話を取る。`cache_control` なしとありで実行する。それぞれの入力トークン請求書を報告する。
2. **Medium（中級）。** プロンプトテンプレートとリクエストログが与えられた場合、プロバイダーごとの期待ヒット率とドルの節約額を計算するテストハーネスを書く（Anthropic 5分、Anthropic 1時間、OpenAI自動、Gemini明示的）。
3. **Hard（上級）。** レイアウトオプティマイザーを構築する: プロンプトと `stable=True/False` でマークされたフィールドのリストが与えられた場合、情報を失うことなく最大のキャッシュフレンドリーな位置に単一のキャッシュブレークポイントを置くようプロンプトを書き直す。実際のAnthropicエンドポイントで検証する。

## キーワード

| 用語 | よく言われる表現 | 実際の意味 |
|------|-----------------|-----------------------|
| プロンプトキャッシング | 「長いプロンプトを安くする」 | マッチするプレフィックスのプロバイダー側KVキャッシュの再利用; 繰り返し入力トークンに50〜90%の割引 |
| `cache_control` | 「Anthropicのマーカー」 | 「ここまでのすべてはキャッシュ可能」を宣言するコンテンツブロック属性; `{"type": "ephemeral"}` |
| キャッシュ書き込み | 「プレミアムを支払う」 | キャッシュを作成する最初のリクエスト; AnthropicではLM入力レートの約1.25倍で課金、OpenAIでは無料 |
| キャッシュ読み込み | 「割引」 | プレフィックスに一致する後続のリクエスト; 10%（Anthropic）、50%（OpenAI）、約25%（Gemini）で課金 |
| TTL | 「どれくらい保持されるか」 | キャッシュがウォームを維持する秒数; Anthropicデフォルト5分（1時間まで延長可）、OpenAIベストエフォートで最大1時間、Geminiユーザー設定 |
| 拡張TTL | 「1時間AnthropicキャッシュREM」 | `{"type": "ephemeral", "ttl": "1h"}`; 書き込みプレミアムが2倍だがバッチ再利用では価値がある |
| プレフィックスマッチ | 「なぜキャッシュがミスしたか」 | キャッシュはブレークポイントまでのすべてのトークンがバイト同一の場合のみヒットする |
| コンテキストキャッシング（Gemini） | 「明示的なもの」 | Googleの名前付きストレージ課金キャッシュオブジェクト; 大きなコーパスの複数日再利用に最適 |

## 参考資料

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`、1時間TTL、損益分岐テーブル
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) — 自動プレフィックスマッチング
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` APIとストレージ価格
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) — レイテンシ数値を含むオリジナルローンチポスト
- フェーズ11・05（コンテキストエンジニアリング）— キャッシュが位置できるようプロンプトをどこでスライスするか
- フェーズ11・11（キャッシングとコスト）— ユーザーメッセージでのセマンティックキャッシュとプロンプトキャッシングを組み合わせる
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — プロンプトキャッシングがユーザーに公開するKVキャッシュメモリモデル
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — プリフィルはプロンプトキャッシングがショートカットするフェーズ
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — プロンプトキャッシングは投機的デコーディング、Flash Attention、MQA/GQAとともに推論コスト曲線を曲げるレバー
