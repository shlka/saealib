---
primary_layer: layer4
---

# OptimizationStrategy

`saealib` は、高コストな真の評価へ送る候補の選択を `OptimizationStrategy` に委譲します。
Strategyは、候補生成、予測、評価計画、Feedback、Population更新を一つのグラフとして記述します。

## OptimizationStrategyの役割

現行の正規の拡張点は `build_graph(provider) -> ComponentGraph` です。
Optimizerはこのグラフを実行前にCompilerへ渡し、`ExecutablePlan`を作成します。

`build_pipeline(provider) -> Pipeline` は、既存のPipeline DSLを利用するStrategyが提供できる互換性用の記述形式です。
この形式を使う場合も、構造化Pipelineへ入れるStageは `stage_component(stage)` で明示的に包みます。

`step(ctx, provider)` はsequential compatibility経路に残る境界です。
新しいgraph-native componentは `StateView` を読み、`StatePatch` または `NodeResult` を返します。

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

各Strategyの構成要素には、EvaluationPlan、非同期評価のsubmitとcollect、Feedback、Tellが含まれます。
Stageはその一部を担う互換性用の実行単位であり、構造化Pipeline全体と同一の実行契約ではありません。
詳細は [Stage](../observation_and_state/stage.md) と [Framework](../../framework/index.md) を参照してください。

## Which Strategy to choose

For problems where evaluation cost is extremely high, `IndividualBasedStrategy` or `PreSelectionStrategy`, which use the surrogate to filter out most candidates, are effective.
Early in the search, when the surrogate's reliability is still low, or when you don't want to pay the surrogate's own training cost frequently, `GenerationBasedStrategy`, which advances multiple generations at once using only the surrogate, is suited.
For problems where the surrogate's approximation error itself is unacceptable, or evaluation cost is sufficiently low, relying solely on true evaluation via `DirectStrategy` is reasonable.

## Behavior of runtime swapping

Every Strategy's `step()` builds the current pipeline from `provider` before running it.
Because the pipeline isn't cached, swapping `provider.algorithm` or `provider.surrogate_manager` mid-run is reliably reflected from the next generation onward.

## Implementing a custom Strategy

独自の候補選択方式を実装するときは、graph-nativeの経路とStage互換の経路を分けて選びます。

**Graph-nativeの経路**：`build_graph()` を実装し、ComponentGraphを返します。
Pipeline DSLを使って段階的に移行する場合は、各Stageを `stage_component()` で包んでから `lower_pipeline()` に渡します。

```python
from saealib import OptimizationStrategy, Pipeline
from saealib.core import lower_pipeline
from saealib.stages import stage_component


class CustomStrategy(OptimizationStrategy):
    """Pipeline DSLからStrategyのComponentGraphを作る骨格。"""

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

`first_stage` と `second_stage` は、実際に組み合わせるStageインスタンスを表します。
この例はPipelineからComponentGraphを作る境界だけを示しており、評価計画やFeedbackの契約は各Stageの構成に応じて追加します。

既存Strategyの一部だけを調整する場合は、新しいStrategyを作るよりも [Pipeline.replace/find](../extension_guidelines.md) で互換性用Pipelineを変更する方が適しています。

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
