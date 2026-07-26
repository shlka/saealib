# OptimizationStrategy

`saealib` delegates the decision of which candidate solutions receive an expensive true evaluation to `OptimizationStrategy`, a swappable component.
`step()` executes one generation's "generate, score, evaluate, update."

## OptimizationStrategy's role

`OptimizationStrategy` requires only one method, `step(ctx, provider) -> OptimizationState | None`, to be implemented.
Returning `None` is a convention for legacy-style implementations that update `ctx` in-place; all 4 built-in strategies return an updated `OptimizationState`.

The class attribute `requires_surrogate: bool` indicates whether this strategy needs a `SurrogateManager`.
`Optimizer.validate()` checks this attribute to confirm you aren't trying to use a strategy with `requires_surrogate=True` while `surrogate_manager` is unset.

## Built-in Strategies

| Class | Parameters | Approach |
|---|---|---|
| `DirectStrategy` | None | Uses no surrogate; truly evaluates every generated candidate |
| `IndividualBasedStrategy` | `evaluation_ratio: float = 0.1` | Scores every candidate with the surrogate, and truly evaluates only the top `evaluation_ratio` fraction |
| `PreSelectionStrategy` | `n_candidates: int, n_select: int` (both required) | Generates `n_candidates`, scores them, and truly evaluates only the top `n_select` |
| `GenerationBasedStrategy` | `gen_ctrl: int` (required) | Advances `gen_ctrl` generations using only the surrogate, then truly evaluates a single generation |

`IndividualBasedStrategy` selects by fraction of individuals, while `PreSelectionStrategy` selects by count of individuals.
`GenerationBasedStrategy` switches between surrogate and true evaluation by generation, not by individual.
`DirectStrategy` is the comparison baseline that uses no surrogate at all, with `requires_surrogate=False`.

### Each Strategy's pipeline structure

| Class | Pipeline |
|---|---|
| `DirectStrategy` | CountGeneration → Ask → TrueEvaluation → ArchiveUpdate → Tell |
| `IndividualBasedStrategy` | CountGeneration → Ask → SurrogateScore → SortByScore → TrueEvaluation (ratio-based) → ArchiveUpdate → Tell |
| `PreSelectionStrategy` | CountGeneration → Ask (n_candidates) → SurrogateScore → TopKSelection(k=n_select) → TrueEvaluation → ArchiveUpdate → Tell |
| `GenerationBasedStrategy` | SurrogateOnlyLoop (gen_ctrl times) → CountGeneration → Ask → TrueEvaluation → ArchiveUpdate → Tell |

See [Stage](stage.md) for the contract each individual stage satisfies.
See [Components overview](index.md) for the overall pipeline diagram.

## Which Strategy to choose

For problems where evaluation cost is extremely high, `IndividualBasedStrategy` or `PreSelectionStrategy`, which use the surrogate to filter out most candidates, are effective.
Early in the search, when the surrogate's reliability is still low, or when you don't want to pay the surrogate's own training cost frequently, `GenerationBasedStrategy`, which advances multiple generations at once using only the surrogate, is suited.
For problems where the surrogate's approximation error itself is unacceptable, or evaluation cost is sufficiently low, relying solely on true evaluation via `DirectStrategy` is reasonable.

## Behavior of runtime swapping

Every Strategy's `step()` unconditionally executes `self.pipeline = self._build_pipeline(provider)` before running the pipeline, every time it's called.
Because the pipeline isn't cached, swapping `provider.algorithm` or `provider.surrogate_manager` mid-run is reliably reflected from the next generation onward.

## Implementing a custom Strategy

If you need a custom candidate-selection scheme, there are two approaches.

**Subclass `OptimizationStrategy` directly**: Implement `step()` yourself.
This is also the form you'd use when building a new pipeline by combining [Pipeline/Stage](extension_guidelines.md).

```python
from saealib import OptimizationStrategy, Pipeline
from saealib.stages import (
    CountGenerationStage, AskStage, TrueEvaluationStage,
    ArchiveUpdateStage, TellStage,
)


class SimpleDirectStrategy(OptimizationStrategy):
    """An example reproducing the same content as DirectStrategy by assembling a Pipeline yourself."""

    requires_surrogate = False

    def step(self, ctx, provider):
        cbmanager = getattr(provider, "cbmanager", None)
        pipeline = Pipeline([
            CountGenerationStage(),
            AskStage(provider.algorithm, cbmanager=cbmanager),
            TrueEvaluationStage(provider.evaluator, cbmanager=cbmanager),
            ArchiveUpdateStage(),
            TellStage(provider.algorithm),
        ])
        return pipeline.execute(ctx)
```

If you just want to fine-tune an existing strategy's pipeline, it's lighter-weight to swap out only part of the built-in pipeline via [Pipeline.replace/find](extension_guidelines.md), rather than writing a new `OptimizationStrategy`.

## Related components

- [Stage](stage.md): The contract of each individual pipeline stage a Strategy combines
- [SurrogateManager](surrogate_manager.md): The scoring mechanism used by strategies with `requires_surrogate=True`
- [Extension guidelines](extension_guidelines.md): Rearranging stages via `Pipeline.replace`/`find`
- [Components overview](index.md): The diagram of the overall pipeline structure

## References

- {py:class}`saealib.OptimizationStrategy`
- {py:class}`saealib.IndividualBasedStrategy`
- {py:class}`saealib.GenerationBasedStrategy`
- {py:class}`saealib.PreSelectionStrategy`
- {py:class}`saealib.DirectStrategy`
