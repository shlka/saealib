# Stage

The built-in `OptimizationStrategy`s (IB/GB/PS/Direct) split a single generation's processing into units called `Stage`, executed in order via `Pipeline`.
The "Pipeline/Stage" section of [Extension guidelines](extension_guidelines.md) covered how to rearrange stages via `pipeline.replace`/`find`.
This page covers the contract each `Stage` satisfies, the details of the 11 built-in ones, and how to implement a custom `Stage`.

## Stage's role

`Stage` requires only one method, `execute(state: OptimizationState) -> OptimizationState`, to be implemented.
It receives an `OptimizationState`, performs one well-defined operation, and returns the updated (or same) state.

There are three class attributes.

- **`name`**: A machine-readable identifier used for lookup via `pipeline["name"]`
- **`label`**: A human-readable description
- **`notation`**: The LaTeX notation used by `to_pseudocode()`

## Pipeline

`Pipeline(Stage)` is a composite class that runs a list of `Stage`s in sequence via `functools.reduce`.
Because `Pipeline` itself is a `Stage` subclass, it can be nested inside another `Pipeline`.

| Operation | Description |
|---|---|
| `pipeline["name"]` | Looks up a stage by `name` |
| `pipeline.replace(name, stage)` | Replaces the stage named `name` with a different `Stage` |
| `pipeline.find(name, *, recursive=False)` | With `recursive=True`, also searches inside nested stages like `SurrogateOnlyLoopStage` |

## Built-in Stages

The table shows how each Stage reads and writes `OptimizationState`'s standard fields: `offspring`/`scores`/`predictions`/`evaluated_offspring`.

| Class | Reads | Writes |
|---|---|---|
| `CountGenerationStage()` | — | `gen` |
| `AskStage(algorithm, n_offspring=None, cbmanager=None)` | — | `offspring` |
| `SurrogateScoreStage(surrogate_manager, cbmanager=None, *, refit=True)` | `offspring` | `scores`, `predictions` |
| `SurrogateFitStage(surrogate_manager)` | `archive` | — |
| `TopKSelectionStage(k)` | `offspring`, `scores` | `offspring` (only the top k) |
| `SortByScoreStage()` | `offspring`, `scores` | `offspring`, `scores` (all entries, reordered descending) |
| `TrueEvaluationStage(evaluator, cbmanager=None, n_eval=None)` | `offspring` | `evaluated_offspring`, `fe` |
| `ArchiveUpdateStage()` | `evaluated_offspring` | `archive`, `pareto_archive` |
| `TellStage(algorithm)` | `offspring` | `population` |
| `SurrogateOnlyLoopStage(algorithm, surrogate_manager, gen_ctrl, cbmanager=None)` | — | The entire inner loop |
| `InitializationStage(initializer, provider, problem)` | — | Rebuilds the entire state |

`AskStage` calls `algorithm.ask()`, writes to `state.offspring`, and fires PostCrossover/PostMutation/PostAskEvent via `cbmanager`.

`SurrogateScoreStage` scores via `surrogate_manager.score_candidates()`, writing to `state.scores`/`state.predictions` while also setting each candidate's `tell_f`.

`SurrogateFitStage` is used to pre-fit the surrogate once, ahead of an inner loop where the archive doesn't change.
It's used together with passing `refit=False` to the downstream `SurrogateScoreStage`.

`TopKSelectionStage` keeps only the top k entries of `state.offspring` by descending `state.scores`, discarding the rest.
Unlike `TopKSelectionStage`, `SortByScoreStage` keeps every candidate and just reorders them descending; it's used by IB-family strategies.

`TrueEvaluationStage` evaluates the first `n_eval` entries of `state.offspring` (all of them if `None`; you can specify an `int` or a `Callable[[OptimizationState], int]`) with the true objective function.

`SurrogateOnlyLoopStage` is a composite stage used by `GenerationBasedStrategy`.
It repeats an inner loop of `CountGeneration → Ask → SurrogateScore(refit=False) → Tell` `gen_ctrl` times.
It's a no-op when `gen_ctrl=0`.

```{note}
The `state` argument passed to `InitializationStage`'s `execute()` is ignored — it always builds a fresh state from initialization.
Used at the head of a user-defined pipeline when you want initialization itself to be treated as part of the pipeline.
```

See [strategies](strategies.md) and the pipeline diagram in [Components overview](index.md) for how the 4 built-in Strategies (IB/GB/PS/Direct) combine these Stages into a pipeline.
This page covers the contract of each Stage in isolation.

## Implementing a custom Stage

If you need a custom pipeline stage, subclass `Stage` and implement only `execute()`.
Follow `OptimizationState`'s update pattern, building a new state immutably via `state.replace(...)`.

```python
from saealib import Stage


class LogGenerationStage(Stage):
    """A custom stage that just prints the generation number to stdout."""

    name = "log_generation"
    label = "Log generation number"

    def execute(self, state):
        print(f"generation {state.gen}")
        return state
```

If you need to carry a custom field, use `OptimizationState`'s `data` dictionary, meant for extension.
Add a value via `state.replace(data={**state.data, "key": value})`.

## `to_pseudocode`

`to_pseudocode(expand=False, indent=0)` is a mechanism that outputs each Stage's `notation` as pseudocode for a paper (LaTeX algorithmic notation).
`AskStage`/`TellStage`/`SurrogateOnlyLoopStage` have custom implementations that expand `Algorithm.ask_notation`/`tell_notation` when `expand=True`.

## Related components

- [Extension guidelines](extension_guidelines.md): How to rearrange stages via `pipeline.replace`/`find`
- [strategies](strategies.md): How the built-in Strategies combine Stages
- [OptimizationState](optimization_state.md): The state object `execute()` reads and writes
- [Components overview](index.md): The diagram of the overall pipeline structure

## References

- {py:class}`saealib.Stage`
- {py:class}`saealib.Pipeline`
- {py:class}`saealib.CountGenerationStage`
- {py:class}`saealib.AskStage`
- {py:class}`saealib.SurrogateScoreStage`
- {py:class}`saealib.SurrogateFitStage`
- {py:class}`saealib.TopKSelectionStage`
- {py:class}`saealib.SortByScoreStage`
- {py:class}`saealib.TrueEvaluationStage`
- {py:class}`saealib.ArchiveUpdateStage`
- {py:class}`saealib.TellStage`
- {py:class}`saealib.SurrogateOnlyLoopStage`
- {py:class}`saealib.InitializationStage`
