# SurrogateManager

Where [Surrogate](surrogate.md) handles only fit/predict, `SurrogateManager` coordinates the entire fit, predict, and scoring pipeline.
Because `score_candidates()` returns not just the scalar score but also the original predictions, the caller (such as `IndividualBasedStrategy`) can assign predicted objective values to offspring.

`Optimizer.set_surrogate_manager()` is a separate top-level swap point from `Optimizer.set_surrogate()` (a simplified version that wraps a [Surrogate](surrogate.md) in `LocalSurrogateManager`).

## SurrogateManager's role

`SurrogateManager`'s only abstract method is `score_candidates()`; everything else is a hook with a default implementation.

**`score_candidates(candidates_x, archive, ctx=None, *, refit=True) -> tuple[np.ndarray, list[SurrogatePrediction]]`** (abstract): Scores the candidates.
With `refit=True` (the default), the surrogate is retrained before scoring.

**`fit(archive, ctx=None) -> None`**: A no-op by default.
A pre-fit hook meant to be called once before a series of `score_candidates(..., refit=False)` calls, used in situations where the archive doesn't change (such as `GenerationBasedStrategy`'s surrogate-only inner loop).

**`last_accuracy: SurrogateAccuracy | None`** (class attribute): The accuracy metric computed by the most recent `fit`.
Covered in detail in [Surrogate accuracy evaluation and dynamic switching](surrogate_switching.md).

**`iter_acquisitions() -> Iterator[AcquisitionFunction]`**: An internal hook `Optimizer` uses to auto-inject `problem.direction` into each [AcquisitionFunction](acquisition_functions.md).
The default returns `self.acquisition` if it exists.
A manager with no acquisition function, like `PairwiseSurrogateManager`, returns nothing, and `CompositeSurrogateManager` delegates to its sub-managers.

**`post_score(scores, predictions, ctx=None)`** / **`with_post_score(fn)`**: A post-scoring lifecycle hook.
The same shape as [Surrogate](surrogate.md)'s `with_post_fit`; `with_post_score` returns a copy with the hook added, without modifying the original instance.

**`on_generation_end(gen, archive, ctx=None)`** / **`with_on_generation_end(fn)`**: An end-of-generation hook.
Also extensible via the same copy-and-chain approach.

## Built-in SurrogateManagers

| Class | Approach |
|---|---|
| `GlobalSurrogateManager` | Fits globally once over the entire archive, and predicts/scores every candidate at once |
| `LocalSurrogateManager` | Fits locally per candidate, using its k nearest neighbors |
| `CompositeSurrogateManager` | Combines scores from multiple managers |
| `PairwiseSurrogateManager` | Scoring via a pairwise-comparison surrogate |

`GlobalSurrogateManager(surrogate, acquisition, training_set=None, accuracy_evaluator=None)` uses `ArchiveObjectiveSet()` when `training_set` is omitted.

`LocalSurrogateManager(surrogate, acquisition, training_set=None, accuracy_evaluator=None)` uses `KNNObjectiveSet(n_neighbors=50)` when `training_set` is omitted.
`n_neighbors` isn't a constructor argument of `LocalSurrogateManager` itself — it's a parameter of the default `training_set`.
Because it reuses and refits the same `surrogate` instance across candidates, it isn't thread-safe.

`CompositeSurrogateManager(managers, combine_fn)` calls each manager in `managers`'s `score_candidates` independently, and combines the resulting score arrays via `combine_fn`.
`product_combine` (element-wise product, e.g. EI×PoF) and `rank_weighted_combine(weights=None)` (generates a function that returns a rank-normalized weighted average) are provided as functions to pass as `combine_fn`.

`PairwiseSurrogateManager(surrogate, training_set=None, n_ref=10)` uses `PairwiseComparisonSet()` when `training_set` is omitted.

See [TrainingSet](training_set.md) and [Surrogate accuracy evaluation and dynamic switching](surrogate_switching.md), respectively, for details on each manager's `training_set`/`accuracy_evaluator` arguments.

### ArchiveBasedManager: the family that doesn't train a surrogate

`ArchiveBasedManager` is an abstract subclass of `SurrogateManager` that trains no surrogate model at all, scoring candidates directly from the archive.
Its only abstract method is `compute_scores(candidates_x, archive, ctx=None) -> np.ndarray`; `score_candidates()` wraps `compute_scores()`'s result in a `SurrogatePrediction` with `tell_f=NaN`.
Since the score isn't an objective value, using it directly as `tell_f` would contaminate things like pbest — this design prevents that.

| Class | Parameters | Meaning of the score |
|---|---|---|
| `NoveltyManager` | `k=1` | Higher when the average distance to the k nearest archive points is larger |
| `DensityManager` | `eps=1.0` | The reciprocal of the ε-neighborhood density (prioritizes sparse regions) |
| `NichingManager` | None | The minimum distance among candidates + the minimum distance to the archive |

## Implementing a custom SurrogateManager

If you need a custom scoring scheme, subclass `SurrogateManager` and implement only `score_candidates()`.
For a pattern that scores directly from the archive without training a surrogate, it's lighter-weight to subclass `ArchiveBasedManager` and implement only `compute_scores()`.

```python
import numpy as np
from saealib import ArchiveBasedManager


class ConstantScoreManager(ArchiveBasedManager):
    """A minimal surrogate manager that always returns a constant score, never referencing the archive."""

    def compute_scores(self, candidates_x, archive, ctx=None):
        return np.ones(len(candidates_x))
```

## Related components

- [Surrogate](surrogate.md): The fit/predict implementation `SurrogateManager` coordinates
- [TrainingSet](training_set.md): Used by each `SurrogateManager` to extract training data
- [AcquisitionFunction](acquisition_functions.md): The `acquisition` argument of `GlobalSurrogateManager`/`LocalSurrogateManager`
- [Surrogate accuracy evaluation and dynamic switching](surrogate_switching.md): Details of `accuracy_evaluator`/`last_accuracy`
- [strategies](strategies.md): The caller of `score_candidates()`

## References

- {py:class}`saealib.SurrogateManager`
- {py:class}`saealib.GlobalSurrogateManager`
- {py:class}`saealib.LocalSurrogateManager`
- {py:class}`saealib.CompositeSurrogateManager`
- {py:class}`saealib.PairwiseSurrogateManager`
- {py:func}`saealib.product_combine`
- {py:func}`saealib.rank_weighted_combine`
- {py:class}`saealib.ArchiveBasedManager`
- {py:class}`saealib.NoveltyManager`
- {py:class}`saealib.DensityManager`
- {py:class}`saealib.NichingManager`
