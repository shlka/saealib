---
primary_layer: layer3
related_layers: [layer2]
page_type: concept
---

# OptimizationStrategy

`saealib` delegates the selection of candidates sent for expensive true evaluation to `OptimizationStrategy`.
The Strategy describes candidate generation, prediction, evaluation planning, Feedback, and Population updates as one graph.

## OptimizationStrategy's role

The graph-native extension point is `build_graph(provider) -> ComponentGraph`.
Before execution, Optimizer passes this graph to the Compiler to create an `ExecutablePlan`.

`build_pipeline(provider) -> Pipeline` is the Stage compatibility path and an optional compatibility representation for Strategies that use the existing Pipeline DSL.
Even in this form, a Stage inserted into a structured Pipeline must be wrapped explicitly with `stage_component(stage)`.

`step(ctx, provider)` is the remaining boundary on the Stage compatibility path.
A new graph-native component reads a `StateView` and returns a `StatePatch` or `NodeResult`.

The class attribute `requires_surrogate: bool` indicates whether this strategy needs a `SurrogateManager`.
`Optimizer.validate()` checks this attribute to confirm you aren't trying to use a strategy with `requires_surrogate=True` while `surrogate_manager` is unset.

## Built-in Strategies

| Class | Parameters | Approach |
|---|---|---|
| `DirectStrategy` | None | Uses no surrogate; truly evaluates every generated candidate |
| `SteadyStateStrategy` | None | Generates one candidate per step and supports asynchronous refill |
| `IndividualBasedStrategy` | `evaluation_ratio: float = 0.1` | Scores every candidate with the surrogate, and truly evaluates only the top `evaluation_ratio` fraction |
| `PreSelectionStrategy` | `n_candidates: int, n_select: int` (both required) | Generates `n_candidates`, scores them, and truly evaluates only the top `n_select` |
| `GenerationBasedStrategy` | `gen_ctrl: int` (required) | Advances `gen_ctrl` generations using only the surrogate, then truly evaluates a single generation |

`IndividualBasedStrategy` selects by fraction of individuals, while `PreSelectionStrategy` selects by count of individuals.
`GenerationBasedStrategy` switches between surrogate and true evaluation by generation, not by individual.
`DirectStrategy` is the comparison baseline that uses no surrogate at all, with `requires_surrogate=False`.

For steady-state execution, combine `DirectStrategy` with an
`AsyncEvaluationScheduler`. The scheduler fills available worker slots, polls without
blocking, and commits each completed update in the lifecycle order.

### Each Strategy's pipeline structure

| Class | Pipeline |
|---|---|
| `DirectStrategy` | CountGeneration → Ask → TrueEvaluation → ArchiveUpdate → Tell |
| `IndividualBasedStrategy` | CountGeneration → Ask → SurrogatePredictStage → AcquisitionStage → SortByScore → TrueEvaluation (ratio-based) → ArchiveUpdate → Tell |
| `PreSelectionStrategy` | CountGeneration → Ask (n_candidates) → SurrogatePredictStage → AcquisitionStage → TopKSelection(k=n_select) → TrueEvaluation → ArchiveUpdate → Tell |
| `GenerationBasedStrategy` | SurrogateOnlyLoop (gen_ctrl times) → CountGeneration → Ask → TrueEvaluation → ArchiveUpdate → Tell |

Each Strategy's components include the `EvaluationPlan`, asynchronous evaluation submission and collection, Feedback, and Tell.
A Stage is a compatibility execution unit for part of that work, not the same execution contract as the entire structured Pipeline.
See [Stage](../observation_and_state/stage.md) and [Framework](../../framework/index.md) for details.

## Which Strategy to choose

For problems where evaluation cost is extremely high, `IndividualBasedStrategy` or `PreSelectionStrategy`, which use the surrogate to filter out most candidates, are effective.
Early in the search, when the surrogate's reliability is still low, or when you don't want to pay the surrogate's own training cost frequently, `GenerationBasedStrategy`, which advances multiple generations at once using only the surrogate, is suited.
For problems where the surrogate's approximation error itself is unacceptable, or evaluation cost is sufficiently low, relying solely on true evaluation via `DirectStrategy` is reasonable.

## Behavior of runtime swapping

When you change configuration such as `provider.algorithm` or `provider.surrogate_manager` at a step boundary in `iterate()` or `run()`, the execution environment detects the change, recompiles the plan, and applies it from the next generation.
This procedure works on both the Stage compatibility path and the graph-native path.
When a Component requests recompilation, the plan node returns `RuntimeCommand` or `RequestRecompile` through `NodeResult.commands`.
The request is accepted when the Runtime environment provides `recompile()`, and recompilation happens between steps. Rejection is also a normal result.

## Implementing a custom Strategy

When implementing a custom candidate-selection method, choose separately between the graph-native path and the Stage compatibility path.

**Graph-native path**: Implement `build_graph()` and return a `ComponentGraph`.
For a gradual migration using the Pipeline DSL, wrap each Stage with `stage_component()` before passing it to `lower_pipeline()`.

```python
from saealib import OptimizationStrategy, Pipeline
from saealib.core import lower_pipeline
from saealib.stages import stage_component


class CustomStrategy(OptimizationStrategy):
    """Skeleton for building a Strategy ComponentGraph from the Pipeline DSL."""

    requires_surrogate = False

    def build_graph(self, provider):
        pipeline = Pipeline(
            steps=[
                stage_component(first_stage),
                stage_component(second_stage),
            ]
        )
        return lower_pipeline(pipeline)
```

`first_stage` and `second_stage` stand for the Stage instances to combine.
This example shows only the boundary for building a ComponentGraph from a Pipeline; add evaluation-plan and Feedback contracts according to the Stage composition.

When adjusting only part of an existing Strategy, changing the compatibility Pipeline with [Pipeline.replace/find](../extension_guidelines.md) is preferable to creating a new Strategy.

## Related components

- [Stage](../observation_and_state/stage.md): The contract of each individual pipeline stage a Strategy combines
- [SurrogateManager](../surrogate_modeling/surrogate_manager.md): The prediction mechanism used by strategies with `requires_surrogate=True`
- [Extension guidelines](../extension_guidelines.md): Rearranging stages via `Pipeline.replace`/`find`
- [Components overview](../index.md): The diagram of the overall pipeline structure

## References

- {py:class}`saealib.OptimizationStrategy`
- {py:class}`saealib.IndividualBasedStrategy`
- {py:class}`saealib.GenerationBasedStrategy`
- {py:class}`saealib.PreSelectionStrategy`
- {py:class}`saealib.DirectStrategy`
- {py:class}`saealib.AsyncEvaluationScheduler`
