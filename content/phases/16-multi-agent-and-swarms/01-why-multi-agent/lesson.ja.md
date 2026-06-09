# なぜマルチエージェントなのか

> エージェントが1つでは壁に当たる。賢い選択は、より大きなエージェントではなく——より多くのエージェントだ。


## 学習目標

- シングルエージェントの限界（コンテキスト溢れ、専門性の混在、逐次処理のボトルネック）を特定し、複数エージェントへの分割が適切な判断となる場面を説明できる
- オーケストレーションパターン（パイプライン、並列ファンアウト、スーパーバイザー、階層型）を比較し、与えられたタスク構造に適したものを選択できる
- 明確な役割の境界、共有状態、通信コントラクトを持つマルチエージェントシステムを設計できる
- マルチエージェントの複雑さ（レイテンシ、コスト、デバッグの難しさ）とシングルエージェントのシンプルさのトレードオフを分析できる

## 問題

フェーズ14でシングルエージェントを構築した。それは動く。ファイルを読み、コマンドを実行し、APIを呼び出し、結果を推論できる。それを実際のコードベースに向けてみよう。ファイルが200個、3言語、インフラに依存するテスト、そして外部APIを調査してからコードを書くという要件がある。

エージェントは詰まる。LLMが愚かいわけではなく、タスクが1つのエージェントループで処理できる範囲を超えているからだ。コンテキストウィンドウがファイルの内容で埋まってしまう。40回のツール呼び出し前に読んだ内容を忘れてしまう。リサーチャー、コーダー、レビュアーを一度に担おうとして、3つすべてを中途半端にやってしまう。

これがシングルエージェントの限界だ。タスクが以下を要求するたびに直面する：

- **1つのウィンドウに収まらないコンテキスト** — 50ファイルを読むと20万トークンを超える
- **ステージごとに異なる専門性** — リサーチにはコード生成とは異なるプロンプティングが必要
- **並行して実行できる作業** — 3つのファイルを順番に読む必要があるか？同時に読めばよい

## 概念

### シングルエージェントの限界

シングルエージェントは、1つのループ、1つのコンテキストウィンドウ、1つのシステムプロンプトだ。イメージはこうだ：

```
┌─────────────────────────────────────────┐
│            シングルエージェント            │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         コンテキストウィンドウ      │  │
│  │                                   │  │
│  │  リサーチノート                     │  │
│  │  + コードファイル                   │  │
│  │  + テスト出力                       │  │
│  │  + レビューフィードバック             │  │
│  │  + APIドキュメント                   │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ 満杯 ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  1つのシステムプロンプトが               │
│  リサーチ＋コーディング＋レビュー＋テスト │
│  すべてをカバーしようとする               │
│                                         │
│  結果：すべてが中途半端                  │
└─────────────────────────────────────────┘
```

3つのことが壊れる：

1. **コンテキストの飽和** — ツールの結果が積み重なる。30ターン目になると、エージェントはファイルの内容、コマンド出力、事前推論で15万トークンを消費している。5ターン目の重要な詳細は失われている。

2. **役割の混乱** — 「あなたはリサーチャー、コーダー、レビュアー、テスターです」と言うシステムプロンプトは、半分リサーチして、半分コーディングして、レビューを終えることのないエージェントを生み出す。

3. **逐次処理のボトルネック** — エージェントはファイルAを読み、次にB、次にC。3つの逐次LLM呼び出し。3つの逐次ツール実行。並列性なし。

### マルチエージェントの解決策

作業を分割しよう。各エージェントに1つの仕事、1つのコンテキストウィンドウ、その仕事に合わせた1つのシステムプロンプトを与える：

```
┌──────────────────────────────────────────────────────────┐
│                    オーケストレーター                       │
│                                                          │
│  「ユーザー管理のためのREST APIを構築する」                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │リサーチャー│ │ コーダー  │ │レビュアー │ │テスター  │  │
│   │          │ │          │ │          │ │          │  │
│   │ ドキュメ  │ │ リサーチ  │ │ コード   │ │ テスト   │  │
│   │ ントを   │ │ ＋仕様に  │ │ 品質を   │ │ を実行   │  │
│   │ 読み、   │ │ 基づいて  │ │ チェック │ │ して結果 │  │
│   │ パターン │ │ コードを  │ │ しバグを │ │ を報告  │  │
│   │ を探す  │ │ 書く      │ │ 探す     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     結果をマージ                          │
└──────────────────────────────────────────────────────────┘
```

各エージェントには：
- 焦点を絞ったシステムプロンプト（「あなたはコードレビュアーです。バグを見つけることだけが仕事です。」）
- 独自のコンテキストウィンドウ（他のエージェントの作業で汚染されない）
- 明確な入出力コントラクト（リサーチノートを受け取り、コードを出力する）

### 実際にこれを行っているシステム

**Claude Codeのサブエージェント** — Claude CodeがTaskでサブエージェントを生成するとき、スコープを絞ったタスクを持つ子エージェントを作成する。親はコンテキストをきれいに保つ。子は焦点を絞った作業をして要約を返す。

**Devin** — プランナーエージェント、コーダーエージェント、ブラウザエージェントを実行する。プランナーは作業をステップに分解する。コーダーはコードを書く。ブラウザはドキュメントを調査する。それぞれが別のコンテキストを持つ。

**マルチエージェントコーディングチーム（SWE-bench）** — SWE-benchでトップパフォーマンスのシステムは、コードベースを読むリサーチャー、修正を設計するプランナー、実装するコーダーを使用している。シングルエージェントシステムはスコアが低い。

**ChatGPT Deep Research** — 複数の検索エージェントを並列で起動し、それぞれが異なる角度を探索し、結果を統合する。

### スペクトラム

マルチエージェントは二項対立ではない。スペクトラムだ：

```
シンプル ──────────────────────────────────────── 複雑

 シングル     サブ         パイプ      チーム      スワーム
 エージェント  エージェント  ライン

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │共有   │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ 状態  │
             └───┘          └───┘───┘  │  メッ  │    └───────┘
                                       │  セージ│
 1ループ     親＋            ステージ    │  バス  │    N個のピア
 1コンテキスト 子タスク      ごと        │       │    創発的挙動
                                       └───────┘
                                       明示的な役割
```

**シングルエージェント** — 1つのループ、1つのプロンプト。シンプルなタスクに適している。

**サブエージェント** — 親が焦点を絞ったサブタスクのために子を生成する。親はプランを維持する。子は報告する。Claude Codeが行っていることだ。

**パイプライン** — エージェントが順番に実行される。エージェントAの出力がエージェントBの入力になる。段階的なワークフロー（リサーチ→コード→レビュー→テスト）に適している。

**チーム** — エージェントが共有メッセージバスを使って並列に実行される。各エージェントに役割がある。オーケストレーターが調整する。異なるスキルが同時に必要な場合に適している。

**スワーム** — 多数の同一または近似のエージェントが共有状態を持つ。固定のオーケストレーターなし。エージェントはキューから作業を取り出す。高スループットの並列タスクに適している。

### 4つのマルチエージェントパターン

#### パターン1：パイプライン

```
入力 ──▶ エージェントA ──▶ エージェントB ──▶ エージェントC ──▶ 出力
          (リサーチ)       (コード)           (レビュー)
```

各エージェントがデータを変換して前に渡す。推論がシンプル。1つのステージの失敗が残りをブロックする。

#### パターン2：ファンアウト／ファンイン

```
                ┌──▶ エージェントA ──┐
                │                   │
入力 ──▶ 分割 ├──▶ エージェントB ──├──▶ マージ ──▶ 出力
                │                   │
                └──▶ エージェントC ──┘
```

並列エージェントに作業を分割し、結果をマージする。独立したサブタスクに分解できるタスクに適している。

#### パターン3：オーケストレーター・ワーカー

```
                    ┌──────────┐
                    │  オーケ  │
                    │ストレーター│
                    └──┬───┬───┘
                  タスク│   │タスク
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ ワーカーA │   │ ワーカーB │
           └──────────┘   └──────────┘
```

賢いオーケストレーターが何をするかを決定し、ワーカーに委任して結果を統合する。オーケストレーター自体がワーカーを生成するためのツールを持つエージェントだ。

#### パターン4：ピアスワーム

```
         ┌───┐ ◄──── メッセージ ────▶ ┌───┐
         │ A │                       │ B │
         └─┬─┘                       └─┬─┘
           │                           │
      メッセージ│    ┌───────────┐      │メッセージ
           └───▶│  共有状態  │◄────┘
                │  ／キュー  │
           ┌───▶│            │◄────┐
           │    └───────────┘     │
      メッセージ│                    │メッセージ
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── メッセージ ────▶ │ D │
         └───┘                  └───┘
```

中央のオーケストレーターなし。エージェントがピアツーピアで通信する。決定はインタラクションから創発する。デバッグが難しいが、多数のエージェントにスケールする。

### マルチエージェントを使わないべき場合

マルチエージェントは複雑さを増す。エージェント間のすべてのメッセージが潜在的な障害点になる。デバッグは「1つの会話を読む」から「5つのエージェントをまたいでメッセージをトレースする」になる。

**シングルエージェントに留まる場合：**
- タスクが1つのコンテキストウィンドウに収まる（作業データが約10万トークン未満）
- 異なるステージに異なるシステムプロンプトが不要
- 逐次実行で十分に速い
- タスクが十分シンプルで分割するオーバーヘッドが価値を上回る

**複雑さのコスト：**
- すべてのエージェント境界は損失のある圧縮ステップ：エージェントAの完全なコンテキストがエージェントBへのメッセージに要約される
- 調整ロジック（誰が何を、いつ、どの順番で行うか）それ自体がバグの源
- レイテンシが増加：Nエージェントは最低でもN回の逐次LLM呼び出しを意味し、双方向の通信が必要なら更に増える
- コストが倍増：各エージェントが独立してトークンを消費する

経験則：タスクが20回未満のツール呼び出しで10万トークンに収まるなら、シングルエージェントのままにしよう。

## 実装する

### ステップ1：過負荷のシングルエージェント

これはすべてをやろうとするシングルエージェントだ。1つの巨大なシステムプロンプトと、リサーチ、コード、レビューを保持する1つのコンテキストウィンドウを持つ：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

このアプローチの問題点：
- コンテキストウィンドウがステージごとに増えていく。レビューステップの時点では、リサーチノート、コード、事前推論が含まれている。
- システムプロンプトが汎用的。各ステージに合わせて調整できない。
- 並列実行がない。

### ステップ2：スペシャリストエージェント

分割しよう。各エージェントに1つの仕事を与える：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

各スペシャリストに焦点を絞ったプロンプトがある。各スペシャリストは必要な入力だけを持つきれいなコンテキストウィンドウを受け取る。

### ステップ3：メッセージパッシングで調整する

明示的なメッセージパッシングでスペシャリストを接続する：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

各エージェントは自分宛のメッセージだけを受け取る。コンテキストの汚染なし。リサーチャーが読んだ5万トークン分のドキュメントは、レビュアーのコンテキストに入らない。

### ステップ4：比較する

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

マルチエージェント版は合計トークン数が多い（3エージェント、3回の別々のLLM呼び出し）が、各エージェントのコンテキストはきれいに保たれる。システムプロンプトが専門化されているため、各ステージの品質が向上する。

## 使ってみる

このレッスンでは、マルチエージェントに移行するタイミングを決定するための再利用可能なプロンプトを作成する。`outputs/prompt-multi-agent-decision.md`を参照。

## 演習

1. 4番目のスペシャリスト「テスター」エージェントを追加する。コーダーからコードを受け取り、レビュアーからレビューフィードバックを受け取り、テストを書く
2. レビュアーがコーダーにフィードバックを送り返して修正ループを実現するようにパイプラインを変更する（最大2ラウンド）
3. 逐次パイプラインをファンアウトに変換する：リサーチャーと「要件アナライザー」エージェントを並列で実行し、コーダーに渡す前にその出力をマージする

## キーワード

| 用語 | よく言われること | 実際の意味 |
|------|----------------|----------------------|
| スワーム | 「AIエージェントのハイブマインド」 | 共有状態を持ち固定のリーダーのないピアエージェントのセット。動作はローカルインタラクションから創発する。 |
| オーケストレーター | 「ボスエージェント」 | 他のエージェントを生成・管理するツールを持つエージェント。計画と委任を行うが、実際の作業は行わない場合がある。 |
| コーディネーター | 「交通整理役」 | エージェント間でルールに基づいてメッセージをルーティングする非エージェントコンポーネント（多くの場合、LLMではなくコード）。 |
| コンセンサス | 「エージェントが合意する」 | 複数のエージェントが前進する前に合意に達しなければならないプロトコル。矛盾する出力を解決する必要がある場合に使用される。 |
| 創発的動作 | 「エージェントが自分で考え出した」 | エージェントインタラクションから生まれるシステムレベルのパターンだが、明示的にプログラムされていない。有益にも有害にもなり得る。 |
| ファンアウト／ファンイン | 「エージェントのマップリデュース」 | 並列エージェントにタスクを分割し（ファンアウト）、その結果を結合する（ファンイン）。 |
| メッセージパッシング | 「エージェント同士が話す」 | エージェント間の通信メカニズム：一方のエージェントからもう一方に送られる構造化データで、共有コンテキストウィンドウに代わるもの。 |

## 参考資料

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) — マルチエージェントパターンの調査
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) — Microsoftのマルチエージェント会話フレームワーク
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) — Claude CodeがTaskで委任する方法
- [CrewAI documentation](https://docs.crewai.com/) — 役割ベースのマルチエージェントフレームワーク
