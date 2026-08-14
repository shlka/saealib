---
primary_layer: layer4
---

# CallbackManager

`Optimizer` は、実行経過を外部へ通知する処理を `CallbackManager` に委譲します。
CallbackManagerは、ログ記録、収束履歴の収集、条件付きのComponent差し替えに使えます。
Eventは実行境界の観測用であり、Eventのフィールドを書き換えてPipelineの入力を差し替える仕組みではありません。

## CallbackManager's role

`CallbackManager` keeps a list of handlers per event type.

| Method | Description |
|---|---|
| `register(event_type, func)` | Registers `func(event)` to be called every time an event of `event_type` fires |
| `dispatch(event)` | Calls every registered handler, in registration order |
| `unregister(event_type, func)` | Removes a registered handler |
| `replace(event_type, old, new)` | Replaces a registered handler with a different one |

`Event` の `ctx` は、`archive`、`comparator`、`n_obj`、`gen`、`fe` などを読み取るための `_EventContext` Protocolです。
通常のsequential実行では実体が `OptimizationState` ですが、Eventの公開契約はその読み取り能力に限定されます。
`ctx` は読み取り専用として扱い、値を変更するときは `with_post`、Component、Stageなどの正式な境界を使います。

## List of available events

| Event | Fired when | Main fields |
|---|---|---|
| `RunStartEvent` | Once, at the start of a run | — |
| `RunEndEvent` | Once, at the end of a run | — |
| `GenerationStartEvent` | At the start of each generation | — |
| `GenerationEndEvent` | At the end of each generation (before the state is yielded) | — |
| `SurrogateStartEvent` / `SurrogateEndEvent` | Before/after surrogate-based scoring | `offspring` |
| `AcquisitionStartEvent` / `AcquisitionEndEvent` | Before/after acquisition scoring | 開始時は `offspring`、終了時は `offspring` と `result` |
| `PostCrossoverEvent` | After crossover and repair | `candidates` |
| `PostMutationEvent` | After mutation and repair | `candidates` |
| `PostAskEvent` | After all of `ask()` (crossover and mutation) | `candidates` |
| `PostSurrogateFitEvent` | After the surrogate is fitted | `surrogate`。`train_x` と `train_f` は任意 |
| `PostEvaluationEvent` | After true evaluation of the chosen candidates | `offspring` |
| `InitialEvaluationStartEvent` | After initial sampling, before initial evaluation | `candidates_x` |
| `InitialEvaluationEndEvent` | After initial evaluation, before the archive is sorted | `archive` |

`PostSurrogateFitEvent` は、Surrogateのfit後に組み込みStageから発火します。
組み込みStageが発火するイベントでは、`train_x` と `train_f` が設定されない場合があります。
`PostEvaluationEvent` には、評価対象の `offspring` に加えて `request_id`、`candidate_ids`、`status` が含まれる場合があります。

## Default logging output

`Optimizer` automatically registers `logging_generation` on `GenerationStartEvent` at construction time.
Configuring the standard library's `logging` module makes per-generation progress (evaluation count, best objective value, or for multi-objective, the size and range of the first front) appear in the log.

## Registering custom handlers and recording convergence history

Register any handler with `cbmanager.register(EventType, handler)`.
To record convergence history, register a handler holding a list accumulated via closure.

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

## Tracking hypervolume

`logging_generation_hv(reference_point)` returns a handler that logs the first front's hypervolume relative to the specified reference point, every generation.

```python
from saealib import GenerationStartEvent, logging_generation_hv

optimizer.cbmanager.register(
    GenerationStartEvent,
    logging_generation_hv(reference_point=np.array([1.1, 1.1])),
)
```

## Replacing the default handler

You can remove the auto-registered `logging_generation` with `unregister(event_type, func)`, or replace it with a different handler via `replace(event_type, old, new)`.

```python
from saealib import GenerationStartEvent, logging_generation

optimizer.cbmanager.unregister(GenerationStartEvent, logging_generation)
```

## The candidates field is for observation only

The `candidates` field on `PostCrossoverEvent`/`PostMutationEvent`/`PostAskEvent` is for observation; reassigning it inside a handler (`event.candidates = new_array`) has no effect on the pipeline's output.
`GA` keeps using its own local array reference after the event fires, so an in-place change (`event.candidates[:] = ...`) does reach `GA`.
`PSO`, on the other hand, fires this event after `Population.extend()` has already finished copying the candidates, so even an in-place change is too late and has no effect at all.

If you want to actually swap out the candidate array, use `with_post(fn)` on [Crossover](../search_algorithms/crossover.md)/[Mutation](../search_algorithms/mutation.md) instead of `CallbackManager`.
Think of `CallbackManager` as a mechanism for observation (logging, recording, conditional branching decisions) by design, not a means of rewriting pipeline data.

## Swapping components at runtime

`Event` only passes `ctx` to handlers, not the `Optimizer` instance itself.
To swap a component at runtime, capture the `Optimizer` instance directly in the handler's closure, and either reassign `optimizer.algorithm`/`strategy`/`surrogate_manager`/`termination`, or change a parameter on an existing component directly.

```python
from saealib import GenerationStartEvent


def widen_mutation_at_gen5(event):
    if event.ctx.gen == 5:
        optimizer.algorithm.mutation.prob = 1.0


optimizer.cbmanager.register(GenerationStartEvent, widen_mutation_at_gen5)
```

As explained in [strategies](../execution_and_evaluation/strategies.md), each Strategy rebuilds the pipeline on every call to `step()`, so this kind of swap reliably takes effect from the next generation onward.

## When to use CallbackManager vs. iterate()

| Aspect | CallbackManager | `iterate()` |
|---|---|---|
| Granularity of invocation | When a specific event occurs | Per generation (a `for` loop on the host side) |
| Main use | Logging, observation, conditional side effects | Intervening in the loop structure itself, e.g. switching components based on surrogate accuracy |
| Relationship to `run()` | Works with either `run()` or `iterate()` | Used in place of `run()` |

The switcher classes in [Surrogate accuracy evaluation and dynamic switching](../surrogate_modeling/surrogate_switching.md) are meant to be used inside an `iterate()` loop.

## CheckpointCallback

`CheckpointCallback` はCallbackの実装例です。
npz形式では `OptimizationState.save()` を使い、pickle形式では `Optimizer.save_pickle()` を使います。
形式によって保存対象と再開条件が異なるため、詳細は [Checkpointing](../../tutorials/checkpoint.md) を参照してください。
独自Callbackを実装するときの登録処理も、このCallbackを参考にできます。

## Related components

- [Extension guidelines](../extension_guidelines.md): When to use `with_post`-style hooks instead
- [Crossover](../search_algorithms/crossover.md) / [Mutation](../search_algorithms/mutation.md): The actual means of swapping the candidate array
- [strategies](../execution_and_evaluation/strategies.md): When a runtime component swap takes effect
- [Surrogate accuracy evaluation and dynamic switching](../surrogate_modeling/surrogate_switching.md): Dynamic switching inside an `iterate()` loop
- [Checkpointing](../../tutorials/checkpoint.md): How to use `CheckpointCallback`

## References

- {py:class}`saealib.CallbackManager`
- {py:class}`saealib.RunStartEvent`
- {py:class}`saealib.RunEndEvent`
- {py:class}`saealib.GenerationStartEvent`
- {py:class}`saealib.GenerationEndEvent`
- {py:class}`saealib.AcquisitionStartEvent`
- {py:class}`saealib.AcquisitionEndEvent`
- {py:class}`saealib.PostSurrogateFitEvent`
- {py:class}`saealib.PostEvaluationEvent`
- {py:class}`saealib.InitialEvaluationStartEvent`
- {py:class}`saealib.InitialEvaluationEndEvent`
- {py:func}`saealib.logging_generation`
- {py:func}`saealib.logging_generation_hv`
- {py:class}`saealib.CheckpointCallback`
