# LangGraph — エージェントのためのステートマシン

> 手書きのReActループは `while True` だ。LangGraphで書かれたReActループはチェックポイント、割り込み、分岐、タイムトラベルができるグラフだ。エージェントは変わっていない。それを囲むハーネスが変わった。


## 問題

関数呼び出しエージェントを出荷する。3ターンうまく動作し、その後何かがおかしくなる: モデルが500を返すツールを試みる、ユーザーがタスクの途中で気が変わる、またはエージェントが人間のサインオフなしに注文の払い戻しを決める。`while True:` ループにはフックがない。一時停止できない、巻き戻せない、モデルが別のツールを選んでいたら「もしも」の分岐に入れない。デモを超えて出荷した瞬間、エージェントは動作したか否かのブラックボックスになる。

一度見れば次のステップは明らかだ。エージェントはすでにステートマシンだ——システムプロンプト + メッセージ履歴 + 保留中のツール呼び出し + 次のアクション。ステートマシンを明示的にする: 「モデルが思考する」「ツールが実行される」「人間が承認する」のノード、そしてそれらの間の条件付き遷移のエッジ。グラフが明示的になれば、ハーネスは4つのものを無料で得る: チェックポインティング（ステップ間で状態を保存）、割り込み（人間のために一時停止）、ストリーミング（トークンと中間イベントをストリーム）、タイムトラベル（以前の状態に巻き戻して別の分岐を試みる）。

LangGraphはこの抽象化を提供するライブラリだ。「AgentExecutorがある、頑張れ」というLangChain的なエージェントフレームワークではない。ファーストクラスの状態、ファーストクラスの永続性、ファーストクラスの割り込みを持つグラフランタイムだ。エージェントループは手書きするものではなく、描くものだ。

## 概念

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

`StateGraph` は3つのものを持つ。

1. **状態。** グラフを流れる型付きdict（TypedDictまたはPydanticモデル）。すべてのノードは完全な状態を受け取り、部分的な更新を返す。LangGraphはフィールドごとの*リデューサー*を使ってそれらをマージする——累積されるべきリストには `operator.add`、デフォルトでは上書き。
2. **ノード。** Pythonの関数 `state -> partial_state`。各ノードは独立したステップだ: 「モデルを呼び出す」「ツールを実行する」「要約する」。
3. **エッジ。** ノード間の遷移。静的エッジは1つの場所に行く。条件付きエッジはルーター関数 `state -> next_node_name` を取り、グラフがモデル出力に基づいて分岐できる。

グラフをコンパイルする。コンパイルはトポロジーをバインドし、チェックポインター（オプションだが本番に必須）をアタッチし、実行可能なものを返す。初期状態と `thread_id` で呼び出す。実行のすべてのステップは `(thread_id, checkpoint_id)` でキー付けされたチェックポイントを永続化する。

### 4つのスーパーパワー

**チェックポインティング。** すべてのノード遷移が新しい状態をストア（テスト用インメモリ、本番用Postgres/Redis/SQLite）に書き込む。同じ `thread_id` でグラフを再び呼び出して再開する。グラフは一時停止したところから再開する。

**割り込み。** ノードを `interrupt_before=["human_review"]` でマークすると、そのノードが実行される前に実行が停止する。状態が永続化される。APIは「承認待ち」でユーザーに応答する。同じ `thread_id` への後のリクエストで `Command(resume=...)` を使って実行を再開する。

**ストリーミング。** `graph.stream(state, mode="updates")` は状態デルタを発生するたびにyieldする。`mode="messages"` はモデルノード内のLLMトークンをストリームする。`mode="values"` は完全なスナップショットをyieldする。UIに何を表示するかを選択する。

**タイムトラベル。** `graph.get_state_history(thread_id)` は完全なチェックポイントログを返す。以前の `checkpoint_id` を `graph.invoke` に渡してその時点からフォークする。デバッグ（「モデルがツールBを選んでいたら？」）と本番トレースをリプレイする回帰テストに最適。

### リデューサーがポイントだ

すべての状態フィールドにリデューサーがある。ほとんどのデフォルトは問題ない——新しい値が古い値を上書きする。しかしメッセージリストは新しいメッセージが置き換えではなく追加されるよう `operator.add` が必要だ。並列エッジはリデューサーを通じて更新をマージする。2つのノードが両方 `messages` を更新し `Annotated[list, add_messages]` を忘れると、2番目が静かに勝ち半分のターンを失う。リデューサーはライブラリで唯一微妙なもの; 正しく理解すれば残りは合成できる。

### 4ノードでのReActグラフ

本番ReActエージェントは4ノードと2エッジだ:

1. `agent` — 現在のメッセージ履歴でLLMを呼び出す。アシスタントメッセージを返す（tool_callsを含む場合がある）。
2. `tools` — 最後のアシスタントメッセージのすべての tool_calls を実行し、ツール結果をツールメッセージとして追加する。
3. `agent` から条件付きエッジ——最後のメッセージに tool_calls がある場合は `tools` に、そうでなければ `END` にルーティングする。
4. `tools` から `agent` への静的エッジ。

それだけだ。チェックポインティング、割り込み、ストリーミングを持つ完全なReActループ（思考 → 行動 → 観察 → 思考 → …）が約40行のコードで実現できる。

### StateGraph vs Send（ファンアウト）

`Send(node_name, state)` はノードが並列サブグラフをディスパッチできるようにする。例: エージェントが3つのリトリーバーを同時にクエリすることを決める。各 `Send` がターゲットノードの並列実行を生成し、出力は状態リデューサーを通じてマージされる。これがLangGraphがスレッドプリミティブなしにオーケストレーター-ワーカーパターンを表現する方法だ。

### サブグラフ

コンパイルされたグラフは別のグラフのノードになれる。外側のグラフは単一のノードを見る; 内側のグラフは独自の状態と独自のチェックポイントを持つ。これがチームがスーパーバイザー-ワーカーエージェントを構築する方法だ: スーパーバイザーグラフがドメインごとのワーカーサブグラフにユーザーの意図をルーティングする。

## 実装する

### ステップ1: 状態とノード

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` はメッセージリストを上書きではなく累積させるリデューサーだ。これを忘れることが最も一般的なLangGraphのバグだ。

### ステップ2: スレッドで実行する

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

すべての更新は `{node_name: state_delta}` のdictだ。フロントエンドはこれらをUIにストリームし、ユーザーが「エージェントが考えています… search_webを呼び出しています… 結果を得ました… 回答しています。」と見えるようにできる。

### ステップ3: ヒューマンインザループ割り込みを追加する

ノードが実行される前に実行を一時停止するようマークする。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

状態、チェックポイント、スレッドはすべて割り込み間で永続化される。実行中以外はメモリには何もない。

### ステップ4: デバッグのためのタイムトラベル

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

入力として `None` を渡すと与えられたチェックポイントから再生する; 値を渡すとそのチェックポイントの状態への更新として追加してから再開する。これが会話全体を再実行せずに悪いエージェントの実行を再現する方法だ。

### ステップ5: 本番用にチェックポインターを交換する

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite、Redis、Postgresが搭載されている。`MemorySaver` はテスト用だ。再起動後も永続化が必要なものはすべて実際のストアが必要だ。

## スキル

> エージェントを `while True` ループではなくグラフとして構築する。

LangGraphに手を伸ばす前に、60秒の設計をする:

1. **ノードに名前を付ける。** すべての独立した決定または副作用のあるアクションがノードだ。「エージェントが考える」「ツールが実行される」「レビュアーが承認する」「レスポンスがストリームされる」。リストできないなら、タスクはまだエージェント形状でない。
2. **状態を宣言する。** すべてのリストフィールドにリデューサーを持つ最小限のTypedDict。すべてを `messages` に詰め込まない; タスク固有のフィールド（作業中の `plan`、`budget` カウンター、`retrieved_docs` リスト）をトップレベルに引き上げる。
3. **エッジを描く。** 次のステップがモデル出力に依存しない限り静的。すべての条件付きエッジには名前付き分岐を持つルーター関数が必要だ。
4. **チェックポインターを事前に選択する。** テストには `MemorySaver`、それ以外はすべてPostgres/Redis/SQLite。チェックポインターなしで出荷しない——チェックポインターがなければ再開なし、割り込みなし、タイムトラベルなし。
5. **ツールが実行される前に割り込みを決める、後ではなく。** 承認は副作用のあるノードへのエッジに置き、実行前にキャンセルできるようにする; バリデーションはモデルからのエッジに置き、悪い呼び出しを安価に拒否できるようにする。
6. **デフォルトでストリームする。** UIには `mode="updates"`、モデルノード内のトークンレベルストリーミングには `mode="messages"`、評価中のフルスナップショットには `mode="values"`。

チェックポインターのないLangGraphエージェントを出荷することを拒否する。副作用の*後*に割り込むものを出荷することを拒否する。リデューサーとして `add_messages` のない `messages` フィールドを出荷することを拒否する。

## 演習

1. **Easy（簡単）。** 上記の4ノードReActグラフを計算機ツールとWeb検索ツールで実装する。2ターンの会話について `list(app.get_state_history(config))` が少なくとも4つのチェックポイントを返すことを確認する。
2. **Medium（中級）。** `agent` の前に実行され、構造化された `plan: list[str]` を状態に書き込む `planner` ノードを追加する。`agent` がプランのステップを完了済みにマークする。チェックポイントの再開後に `plan` が失われた場合にテストが失敗することを確認する（間違ったリデューサー）。
3. **Hard（上級）。** `Send` を使って3つのサブグラフ（`researcher`、`writer`、`reviewer`）間でルーティングするスーパーバイザーグラフを構築する。各サブグラフは独自の状態とチェックポインターを持つ。`interrupt_before=["writer"]` を外側のグラフに追加し、人間が研究ブリーフを承認できるようにする。以前のチェックポイントからのタイムトラベルが分岐したブランチのみを再実行することを確認する。

## キーワード

| 用語 | よく言われる表現 | 実際の意味 |
|------|-----------------|-----------------------|
| StateGraph | 「LangGraphのグラフ」 | コンパイル前にノードとエッジを追加するビルダーオブジェクト |
| リデューサー | 「フィールドのマージ方法」 | ノードがそのフィールドの更新を返すときに適用される関数 `(old, new) -> merged`; デフォルトは上書き、`add_messages` は追加 |
| スレッド | 「会話ID」 | 1つのセッションのすべてのチェックポイントをスコープする `thread_id` 文字列 |
| チェックポイント | 「一時停止した状態」 | ノード遷移後の完全なグラフ状態の永続化されたスナップショット（`(thread_id, checkpoint_id)` でキー付け） |
| 割り込み | 「人間のための一時停止」 | `interrupt_before` / `interrupt_after` がノード境界で実行を停止; `Command(resume=...)` で再開 |
| タイムトラベル | 「以前のステップからフォーク」 | `graph.invoke(None, config_with_old_checkpoint_id)` がそのチェックポイントから前方に再生 |
| Send | 「並列サブグラフディスパッチ」 | ノードがN個の並列実行をターゲットノードに生成するために返せるコンストラクタ |
| サブグラフ | 「ノードとしてのコンパイル済みグラフ」 | 別のグラフのノードとして使用されるコンパイル済みStateGraph; 独自の状態スコープを保持 |

## 参考資料

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — StateGraph、リデューサー、チェックポインター、割り込みのカノニカルリファレンス
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) — このレッスンが使用するメンタルモデル
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) — Postgres/SQLite/Redisストア、チェックポイント名前空間、スレッドIDの詳細
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`、`interrupt_after`、`Command(resume=...)`、状態編集パターン
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — すべてのLangGraphエージェントが実装するパターン
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — どのグラフ形状（チェーン、ルーター、オーケストレーター-ワーカー、評価器-最適化器）をいつ好むか
- フェーズ11・09（関数呼び出し）— すべてのLangGraphエージェントノードが再利用するツール呼び出しプリミティブ
- フェーズ11・14（Model Context Protocol）— LangGraph `ToolNode` にMCPアダプター経由でプラグインする外部ツールディスカバリー
- フェーズ11・17（エージェントフレームワークのトレードオフ）— LangGraphとCrewAI、AutoGen、Agnoをいつ選ぶか
