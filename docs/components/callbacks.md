# CallbackManager

`Optimizer`は、最適化の進捗をイベントとして外部へ通知する仕組みを`CallbackManager`という差し替え可能なコンポーネントに委ねている。
ログ記録、収束履歴の蓄積、条件付きのコンポーネント差し替えなど、パイプライン本体を書き換えずに観察したり介入したりする際に使う。

## CallbackManagerの役割

`CallbackManager`は、イベント型ごとにハンドラのリストを保持する。

| メソッド | 内容 |
|---|---|
| `register(event_type, func)` | `event_type`のイベントが発火するたびに`func(event)`を呼ぶよう登録する |
| `dispatch(event)` | 登録済みの全ハンドラを登録順に呼ぶ |
| `unregister(event_type, func)` | 登録済みハンドラを削除する |
| `replace(event_type, old, new)` | 登録済みハンドラを別のハンドラに差し替える |

`Event`基底クラスは`ctx`（`OptimizationState`）だけを持つ。
慣例として`ctx`は読み取り専用として扱い、値の変更はコンポーネントのライフサイクルフック（`with_post`など）で行う。

## 利用可能なイベント一覧

| イベント | 発火タイミング | 主なフィールド |
|---|---|---|
| `RunStartEvent` | 実行開始時に1回 | — |
| `RunEndEvent` | 実行終了時に1回 | — |
| `GenerationStartEvent` | 各世代の開始時 | — |
| `GenerationEndEvent` | 各世代の終了時（状態をyieldする前） | — |
| `SurrogateStartEvent` / `SurrogateEndEvent` | サロゲートによるスコアリングの前後 | `offspring` |
| `PostCrossoverEvent` | 交叉と修復の後 | `candidates` |
| `PostMutationEvent` | 突然変異と修復の後 | `candidates` |
| `PostAskEvent` | `ask()`全体（交叉と突然変異）の後 | `candidates` |
| `PostSurrogateFitEvent` | サロゲートのフィット後 | `surrogate`, `train_x`, `train_f` |
| `PostEvaluationEvent` | 選ばれた候補の真の評価後 | `offspring` |
| `InitialEvaluationStartEvent` | 初期サンプリング後、初期評価の前 | `candidates_x` |
| `InitialEvaluationEndEvent` | 初期評価後、アーカイブのソート前 | `archive` |

```{note}
`PostSurrogateFitEvent`は型として定義され公開されているが、現行の組み込みパイプライン（`Stage`群）からは発火されない。
将来組み込まれる可能性のある拡張点として捉え、現時点では自分のパイプラインで明示的に発火させない限り観察できない。
```

## デフォルトのログ出力

`Optimizer`は構築時に`logging_generation`を`GenerationStartEvent`へ自動登録する。
標準ライブラリの`logging`モジュールを設定すれば、世代ごとの進捗（評価回数、最良目的値、または多目的の場合は第一フロントのサイズと範囲）がログに出力される。

## カスタムハンドラの登録と収束履歴の記録

`cbmanager.register(EventType, handler)`で任意のハンドラを登録できる。
収束履歴を記録したい場合は、クロージャで蓄積するリストを持つハンドラを登録する。

```python
from saealib import GenerationEndEvent

history = []


def record_best(event):
    f = event.ctx.archive.get_array("f")
    history.append(float(f.min()))


optimizer.cbmanager.register(GenerationEndEvent, record_best)
ctx = optimizer.run()
print(history)
```

## ハイパーボリューム追跡

`logging_generation_hv(reference_point)`は、指定した参照点に対する第一フロントのハイパーボリュームを世代ごとにログ出力するハンドラを返す。

```python
from saealib import GenerationStartEvent, logging_generation_hv

optimizer.cbmanager.register(
    GenerationStartEvent,
    logging_generation_hv(reference_point=np.array([1.1, 1.1])),
)
```

## デフォルトハンドラの差し替え

`unregister(event_type, func)`で自動登録された`logging_generation`を外したり、`replace(event_type, old, new)`で別のハンドラに差し替えたりできる。

```python
from saealib import GenerationStartEvent, logging_generation

optimizer.cbmanager.unregister(GenerationStartEvent, logging_generation)
```

## candidatesフィールドは観測目的

`PostCrossoverEvent`/`PostMutationEvent`/`PostAskEvent`の`candidates`フィールドは観測用であり、ハンドラ内で再代入（`event.candidates = new_array`）してもパイプラインの出力には反映されない。
`GA`はイベント発火後も自分が持つローカルな配列参照をそのまま使い続けるため、in-place変更（`event.candidates[:] = ...`）であれば`GA`には反映される。
一方`PSO`は、`Population.extend()`で候補群のコピーが完了した後にこのイベントを発火するため、in-place変更すら手遅れで一切反映されない。

候補配列そのものを差し替えたい場合は、`CallbackManager`ではなく[Crossover](crossover.md)/[Mutation](mutation.md)の`with_post(fn)`を使う。
`CallbackManager`は観測（ログ、記録、条件付きの分岐判断）のための仕組みであり、パイプラインのデータを書き換える手段ではない、という設計方針として理解する。

## コンポーネントの実行時差し替え

`Event`はハンドラに`ctx`だけを渡し、`Optimizer`インスタンス自体は渡さない。
実行時にコンポーネントを差し替えたい場合は、ハンドラのクロージャで`Optimizer`インスタンスを直接捕捉し、`optimizer.algorithm`/`strategy`/`surrogate_manager`/`termination`を再代入するか、既存コンポーネントのパラメータを直接変更する。

```python
from saealib import GenerationStartEvent


def widen_mutation_at_gen5(event):
    if event.ctx.gen == 5:
        optimizer.algorithm.mutation.prob = 1.0


optimizer.cbmanager.register(GenerationStartEvent, widen_mutation_at_gen5)
```

[strategies](strategies.md)で説明したとおり、各Strategyは`step()`のたびにパイプラインを再構築するため、この差し替えは次世代から確実に反映される。

## iterate()との使い分け

| 観点 | CallbackManager | `iterate()` |
|---|---|---|
| 呼び出しの粒度 | 特定イベントの発生時 | 世代ごと（ホスト側のforループ） |
| 主な用途 | ログ記録、観察、条件付きの副作用 | サロゲート精度に応じたコンポーネント切り替えなど、ループ構造そのものへの介入 |
| `run()`との関係 | `run()`/`iterate()`どちらでも動作する | `run()`の代わりにこちらを使う |

[サロゲート精度評価と動的切り替え](surrogate_switching.md)のSwitcher系は、`iterate()`ループの中で使うことを前提にしている。

## CheckpointCallback

`CheckpointCallback`は、あらかじめ用意されたCallbackの実例である。
一定世代ごとに[OptimizationState](optimization_state.md)の`save()`を呼び、npz/pickle形式でチェックポイントを保存する。
詳しい使い方は[チェックポイント](../tutorials/checkpoint.md)を参照する。
独自のCallbackを自作する際のお手本の1つとしても参照できる。

## 関連コンポーネント

- [拡張のガイドライン](extension_guidelines.md) — `with_post`系フックとの使い分け
- [Crossover](crossover.md) / [Mutation](mutation.md) — 候補配列を実際に差し替える手段
- [strategies](strategies.md) — 実行時のコンポーネント差し替えが反映されるタイミング
- [サロゲート精度評価と動的切り替え](surrogate_switching.md) — `iterate()`ループでの動的切り替え
- [チェックポイント](../tutorials/checkpoint.md) — `CheckpointCallback`の使い方

## 参照

- {py:class}`saealib.CallbackManager`
- {py:class}`saealib.RunStartEvent`
- {py:class}`saealib.RunEndEvent`
- {py:class}`saealib.GenerationStartEvent`
- {py:class}`saealib.GenerationEndEvent`
- {py:class}`saealib.PostEvaluationEvent`
- {py:class}`saealib.InitialEvaluationStartEvent`
- {py:class}`saealib.InitialEvaluationEndEvent`
- {py:func}`saealib.logging_generation`
- {py:func}`saealib.logging_generation_hv`
- {py:class}`saealib.CheckpointCallback`
