# OptimizationState

The components making up the optimization pipeline don't hold running state directly — they communicate through a single value object called `OptimizationState`.
It appears in nearly every signature as `ctx` or `state`, e.g. `Algorithm.ask(ctx, ...)` or `Stage.execute(state)`.

## What OptimizationState represents

`OptimizationState` is designed as an immutable-style value object: instead of rewriting fields directly, an updated copy is made with `replace()` and passed along.
`Initializer` constructs the first state at the start of a run, and from then on, each `Stage` passes the copy it updated via `replace()` to the next `Stage`.

However, this immutability has two controlled exceptions.

**`archive`** is append-only, and in-place appending is permitted.
Copying it on every evaluation would cost proportional to the square of the evaluation count, so this part is deliberately made mutable.

**`rng`** has the side effect of advancing its internal state every time it's called.
This is a property of NumPy's `Generator` itself, not a design specific to `OptimizationState`.

Aside from these two, every other field follows the principle of immutable updates via `replace()`.

## Main fields

| Field | Content |
|---|---|
| `problem` | The [Problem](problem.md) being solved |
| `population` | The current generation's [Population](population.md) |
| `archive` | The [Archive](population.md) accumulating evaluated solutions |
| `pareto_archive` | The [ParetoArchive](population.md) maintaining the non-dominated solution set |
| `rng` | The random number generator |
| `fe` | The evaluation count |
| `gen` | The generation count |
| `data` | A free-form dictionary for user extension |

There are also typed fields that each pipeline [Stage](stage.md) reads and writes.

| Field | Written by | Read by |
|---|---|---|
| `offspring` | `AskStage` | Each subsequent stage |
| `evaluated_offspring` | `TrueEvaluationStage` | `ArchiveUpdateStage` |
| `scores` / `predictions` | `SurrogateScoreStage` | Each subsequent stage |

`data` is a dictionary for user extension, used as a place for a custom `Stage` or `Callback` to add arbitrary values.
Instead of a direct mutation like `state.data["key"] = value`, pass a newly built dictionary via `state.replace(data={**state.data, "key": value})`.

## Convenience properties

`dim`/`n_obj`/`lb`/`ub`/`direction`/`comparator` are all delegating properties to `state.problem.xxx`.
They're provided as shorthand so you can write `ctx.dim` instead of `ctx.problem.dim`.

## Updating via replace

`replace(**kwargs) -> OptimizationState` is a wrapper around `dataclasses.replace`, the most frequently used update method throughout the pipeline.
For example, `CountGenerationStage`, which advances the generation count, updates the field as `state.replace(gen=state.gen + 1)`.

`OptimizationState` also provides helper methods `count_fe(count=1)`/`count_generation()` that increment `fe`/`gen`, but these are one-off mutations that bypass `replace()`.
The only place in the built-in pipeline where these two are actually called is where [Initializer](initialization.md) adds the initial evaluation count to `fe`; the per-generation updates of `gen`/`fe` (`CountGenerationStage`/`TrueEvaluationStage`) are both unified on the path that uses `replace()`.
When writing a custom `Stage`, it's easier to preserve consistency by sticking to the `replace()` path as well.

## Checkpointing

`save(path)`/`load(path, problem)` (the latter a classmethod) save and restore `OptimizationState` in npz format.
Only the `archive`/`population`/`pareto_archive` arrays and `rng`'s complete bit-generator state are saved, and reproducibility is only guaranteed as far as reasonably possible (bit-exact resumption is intended within the same NumPy version and environment, but reproducibility across versions is not guaranteed).

Component-specific internal rngs, such as the one [NSGA3Comparator](comparators.md) holds (spawned from `state.rng`), are not included when saving.
On resume, such internal rngs are freshly re-spawned from `state.rng`.

See [Checkpointing](../tutorials/checkpoint.md) for how to use automatic checkpoint saving.
The `save`/`load` described here are the contract of `OptimizationState` itself, used internally by that feature.

## Related components

- [Population](population.md): The actual data behind the `population`/`archive`/`pareto_archive` fields
- [Initializer](initialization.md): Constructs the first `OptimizationState`
- [Stage](stage.md): Advances the pipeline while updating state via `state.replace(...)`
- [Checkpointing](../tutorials/checkpoint.md): How to use automatic saving via `CheckpointCallback`

## References

- {py:class}`saealib.OptimizationState`
