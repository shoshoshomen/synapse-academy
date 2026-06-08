# チャットボット — ルールベースからニューラル、LLMエージェントへ

> ELIZAはパターンマッチングで応答した。DialogFlowはインテントをマッピングした。GPTは重みから回答した。Claudeはツールを実行し結果を検証する。各時代は前の時代の最も深刻な失敗を解決してきた。


## 問題設定

ユーザーが「フライトを変更したい」と言う。システムは何を望んでいるか把握し、どの情報が不足しているか確認し、それを取得し、アクションを完了する必要がある。次にユーザーが「待って、キャンセルにしたら？」と言えば、システムはコンテキストを記憶し、タスクを切り替え、状態を保持しなければならない。

会話はMLシステムにとって難しい。入力はオープンエンドである。出力は多くのターンにわたって一貫している必要がある。システムが現実世界に作用する（フライトを変更する、カードに課金する）必要がある場合もある。間違いのたびにユーザーに見えてしまう。

チャットボットのアーキテクチャは4つのパラダイムを経てきた。各パラダイムは前のものがあまりにも目立つ失敗をしたために導入された。このレッスンではそれらを順に辿る。2026年の本番環境は最後の2つのハイブリッドである。

## 概念

![チャットボットの進化: ルールベース → 検索 → ニューラル → エージェント](../assets/chatbot.svg)

**ルールベース（ELIZA、AIML、DialogFlow）。** 手作業で作成されたパターンがユーザー入力にマッチし、応答を生成する。インテント分類器が事前定義されたフローにルーティングする。スロット充填ステートマシンが必要な情報を収集する。設計されたスコープ内では見事に機能する。その外ではすぐに失敗する。幻覚が許容されないセーフティクリティカルなドメイン（銀行認証、航空券予約）では今でも使われている。

**検索ベース。** FAQスタイルのシステム。（発言、回答）ペアをすべてエンコードする。実行時にユーザーのメッセージをエンコードし、最も近い保存済み応答を検索する。Zendeskの古典的な「類似記事」機能を想像してほしい。言い換えに対してルールよりも良く対応する。生成がないため幻覚もない。

**ニューラル（seq2seq）。** 会話ログで学習されたエンコーダ・デコーダ。ゼロから応答を生成する。流暢だが一般的な出力（「わかりません」）や事実のずれが起きやすい。一貫してトピックに沿うことができない。2016〜2019年にGoogle、Facebook、Microsoftがすべて失望させるチャットボットを持っていた理由である。

**LLMエージェント。** 計画を立て、ツールを呼び、結果を確認するループに包まれた言語モデル。長いプロンプトを持つチャットボットではなく、エージェントループである：計画 → ツール呼び出し → 結果観察 → 次のステップ決定。検索優先のグラウンディング（RAG）が幻覚を防ぐ。ツール呼び出しが実際に何かを実行させる。これが2026年のアーキテクチャである。

4つのパラダイムは順次置き換えではない。2026年の本番チャットボットはすべての4つをルーティングする：認証と破壊的アクションにはルールベース、FAQには検索、自然な言い回しにはニューラル生成、曖昧なオープンエンドなクエリにはLLMエージェント。

## 実装する

### ステップ1: ルールベースのパターンマッチング

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20行でELIZA。「私は悲しい」→「なぜ悲しいのですか？」という反転トリックは、Weizenbaum 1966の定番の心理療法士デモである。今でも教訓的だ。

### ステップ2: 検索ベース（FAQ）

この説明的なスニペットは`pip install sentence-transformers`（torchを含む）が必要。このレッスンの実行可能な`code/main.py`は外部依存関係なしに動作させるためにstdlibのJaccard類似度を使用している。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

閾値ベースの拒否が重要な設計上の選択である。最良のマッチが十分に近くない場合、`None`を返してシステムをエスカレートさせる。

### ステップ3: ニューラル生成（ベースライン）

小さな命令調整済みエンコーダ・デコーダ（FLAN-T5）や微調整済みの会話モデルを使用する。2026年において単独では本番不適合（矛盾、脱線、事実の捏造）だが、自然な言い回しのためにハイブリッドシステム内で使われている。DialoGPTスタイルのデコーダのみモデルには明示的なターン区切りとEOS処理が必要だが、FLAN-T5のtext2textパイプラインは教育例としてそのまま動作する。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### ステップ4: LLMエージェントループ

2026年の本番形態：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

3つの要素を名付けておく。ツールはLLMが呼び出せる関数。ループはLLMがツール呼び出しではなく最終回答を返すと終了する。ステップ予算は曖昧なタスクでの無限ループを防ぐ。

実際の本番環境では以下を追加する：検索優先グラウンディング（各LLM呼び出し前に関連文書を注入）、ガードレール（確認なしに破壊的アクションを拒否）、可観測性（全ステップのログ記録）、評価（エージェントの動作が仕様通りかを自動チェック）。

### ステップ5: ハイブリッドルーティング

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

パターン：破壊的なものは決定論的ルール、定型FAQには検索、それ以外はLLMエージェント。これが2026年のカスタマーサポートシステムで採用されているものだ。

## 使ってみる

2026年のスタック：

| ユースケース | アーキテクチャ |
|---------|---------------|
| 予約、決済、認証 | ルールベースのステートマシン＋スロット充填 |
| カスタマーサポートFAQ | キュレーションされた回答の検索 |
| オープンエンドなヘルプチャット | RAG＋ツール呼び出し付きLLMエージェント |
| 内部ツール/IDEアシスタント | ツール呼び出し付きLLMエージェント（検索、読込、書込） |
| コンパニオン/キャラクターチャットボット | ペルソナシステムプロンプト＋知識の検索付きチューニング済みLLM |

本番環境では必ずハイブリッドルーティングを使用すること。一つのアーキテクチャですべてのリクエストをうまく処理できるものはない。ルーティング層自体は一般的に小さなインテント分類器である。

## 今でも出荷されている失敗モード

- **自信を持った捏造。** LLMエージェントが実行していないアクションを完了したと主張する。緩和策：結果を検証し、ツール呼び出しをログに記録し、成功したツールの戻りがなければLLMが何かを実行したと主張させない。
- **プロンプトインジェクション。** ユーザーがシステムプロンプトを上書きするテキストを挿入する。OWASP Top 10 for LLM Applications 2025でLLM01としてランク付けされている。2つのフレーバーがある：直接インジェクション（チャットに貼り付け）と間接インジェクション（エージェントが読む文書、メール、ツール出力に隠れている）。

  攻撃成功率はシナリオによって異なる。一般的なツール使用やコーディングベンチマークでの測定された成功率はフロンティアモデルで約0.5〜8.5%に及ぶ。特定のハイリスクな設定（AIコーディングエージェントへの適応的攻撃、脆弱なオーケストレーション）では約84%に達している。本番CVEにはEchoLeak（CVE-2025-32711、CVSS 9.3）が含まれ、これは攻撃者が制御するメールによってトリガーされるMicrosoft 365 Copilotのゼロクリックデータ流出の欠陥である。

  緩和策：ループ全体でユーザー入力を信頼できないものとして扱う；ツール呼び出し前にサニタイズする；ツール出力をメインプロンプトから分離する；エージェントが最初に計画を立て、実行前に各アクションをその計画に対して検証するPlan-Verify-Execute（PVE）パターンを使用する（これによりツール結果が新しい未計画のアクションを注入するのを防ぐ）；破壊的アクションにはユーザー確認を要求する；ツールのスコープに最小権限を適用する。

  プロンプトエンジニアリングだけではこのリスクを完全に排除できない。外部ランタイム防御層（LLM Guard、許可リスト検証、意味的異常検知）が必要である。
- **スコープクリープ。** ツール呼び出しが間接的に関連する情報を返したためにエージェントがタスクから外れる。緩和策：ツールコントラクトを狭める；システムプロンプトを集中させる；タスク外率の評価を追加する。
- **無限ループ。** エージェントが同じツールを呼び続ける。緩和策：ステップ予算、ツール呼び出し重複除去、「進捗があるか」についてのLLMジャッジ。
- **コンテキストウィンドウの枯渇。** 長い会話で最初のターンがコンテキストから外れる。緩和策：古いターンを要約し、類似性で過去の関連ターンを検索するか、長コンテキストモデルを使用する。

## 成果物を出す

`outputs/skill-chatbot-architect.md`として保存：

```markdown
---
name: chatbot-architect
description: 特定のユースケースに合わせたチャットボットスタックを設計する。
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

製品のコンテキスト（ユーザーニーズ、コンプライアンス制約、利用可能なツール、データ量）を与えられた場合、以下を出力する：

1. アーキテクチャ。ルールベース、検索、ニューラル、LLMエージェント、またはハイブリッド（どのパスがどこに行くかを指定）。
2. 該当する場合のLLMの選択。モデルファミリーを名指し（Claude、GPT-4、Llama-3.1、Mixtral）。ツール使用の品質とコストに合わせること。
3. グラウンディング戦略。RAGソース、検索方法（レッスン14参照）、ツールコントラクト。
4. 評価計画。タスク成功率、ツール呼び出し正確性、タスク外率、保留済みダイアログでの幻覚率。

構造化された確認フローなしに、破壊的なアクション（決済、アカウント削除、データ変更）に対して純粋なLLMエージェントを推奨することを拒否する。エージェントが何かへの書き込みアクセスを持っている場合にプロンプトインジェクション監査をスキップすることを拒否する。
```

## 演習

1. **易。** 上記のルールベース応答をコーヒーショップ注文ボット用の10パターンで実装する。エッジケース（ダブル注文、変更、キャンセル、不明な意図）をテストする。
2. **中。** ハイブリッドFAQ＋LLMフォールバックを構築する。SaaSプロダクト用の50件の定型FAQエントリー、ドキュメントサイトの検索付きLLMフォールバック。100件の実際のサポートの質問で拒否率と精度を測定する。
3. **難。** 上記のエージェントループを3つのツール（検索、ユーザーデータ読込、メール送信）で実装する。プロンプトインジェクション試みを含む50のテストシナリオで評価を実行する。タスク外率、タスク失敗率、インジェクション成功件数を報告する。

## キーワード

| 用語 | 一般的な説明 | 実際の意味 |
|------|------------|----------|
| インテント | ユーザーの望み | カテゴリラベル（book_flight、reset_password）。ハンドラにルーティングされる。 |
| スロット | 情報の単位 | ボットが必要とするパラメータ（日付、目的地）。スロット充填は問いかけのシーケンス。 |
| RAG | 検索＋生成 | 関連文書を検索し、LLMの応答をグラウンディングする。 |
| ツール呼び出し | 関数呼び出し | LLMが名前＋引数の構造化された呼び出しを発行する。ランタイムが実行して結果を返す。 |
| エージェントループ | 計画、行動、確認 | タスク完了までLLM呼び出しとツール呼び出しを交互に実行するコントローラー。 |
| プロンプトインジェクション | ユーザーによるプロンプト攻撃 | システムプロンプトを上書きしようとする悪意のある入力。 |

## 参考資料

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — オリジナルのルールベースチャットボット論文。
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — LLMエージェントが引き継ぐ直前のGoogleの後期ニューラルチャットボット論文。
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — エージェントループパターンに名前をつけた論文。
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2026年でも有効な2024年の本番ガイダンス。
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — プロンプトインジェクションの論文。
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — プロンプトインジェクションをトップのセキュリティ懸念事項にしたランキング。
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — Plan-Verify-Executeとユーザー確認フローを含む実践的なオーケストレーション層の防御。
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — 間接プロンプトインジェクションによるゼロクリックデータ流出CVEの典型例。
