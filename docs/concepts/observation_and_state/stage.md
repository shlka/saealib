---
primary_layer: layer2
---

# Stage

組み込みStrategyは、世代処理を `Stage` という単位に分けて実行します。
Stageは既存AlgorithmとカスタムStageを支えるsequential compatibility surfaceです。

`Pipeline` は構造化DSLです。
graph-native component、入れ子のPipeline、`Repeat`、`Loop`、`Branch`を記述しますが、`OptimizationState` を直接実行しません。
Pipelineを実行可能な計画へ変換する処理は、Optimizerが内部でCompilerへ委譲します。
通常の利用者がCompilerを直接呼び出す必要はありません。

このページでは、Stageの互換性用契約、組み込みStage、カスタムStageの実装方法を説明します。
graph-native componentの実装方法は [Framework](../../framework/index.md) を参照してください。

## Stageの役割

`Stage` が実装する境界は `execute(state: OptimizationState) -> OptimizationState` です。
Stageは `OptimizationState` を受け取り、一つの処理を行って更新後のStateを返します。
この境界はgraph-native componentの `execute(StateView) -> StatePatch` とは異なります。

There are three class attributes.

- **`name`**: A machine-readable identifier used for lookup via `pipeline["name"]`
- **`label`**: A human-readable description
- **`notation`**: The LaTeX notation used by `to_pseudocode()`

## Pipeline

`Pipeline` is a structural sequence.  Its entries can be graph-native
components, nested `Pipeline` values, or structured control values.  Lowering
preserves nested names as graph regions and keeps repeat/loop progress
resumable at runtime.

| Operation | Description |
|---|---|
| `pipeline["name"]` | Looks up a structural entry by `name` |
| `pipeline.replace(name, entry)` | Replaces a top-level structural entry |
| `pipeline.find(name, *, recursive=False)` | With `recursive=True`, searches nested pipelines and control bodies |

## Built-in Stages

The following table is generated from each operational Stage's production
`contract()` (`reads`, `writes`, and `exports`). It is refreshed with:
`python scripts/generate_stage_docs.py`.

<!-- BEGIN GENERATED STAGE CONTRACTS: do not edit -->

| Class | Name | Label | Notation | Reads | Writes | Exports |
|---|---|---|---|---|---|---|
| CountGenerationStage | count_generation | Count generation | $gen \leftarrow gen + 1$ | `runtime.generation`, `evaluations.pending` | `runtime.generation` | — |
| AskStage | ask | Generate offspring | $\mathcal{Q} \leftarrow \text{ask}(P, n)$ | `evaluations.plan`, `evaluations.plan_state`, `runtime.candidate_id_allocator` | `proposals.offspring`, `proposals.current`, `runtime.candidate_id_allocator`, `evaluations.evaluated_offspring` | — |
| SurrogatePredictStage | surrogate_predict | Surrogate prediction | $\hat{y} \leftarrow \text{predict}(\mathcal{Q}, \mathcal{A})$ | `proposals.offspring`, `populations.main`, `archives.main` | `proposals.offspring`, `surrogates.predictions` | — |
| PendingEvaluationContextStage | pending_evaluation_context | Pending evaluation context | $C \leftarrow \text{pending}(C)$ | — | — | — |
| AcquisitionStage | acquisition | Acquisition scoring | $\mathbf{s} \leftarrow \text{acquire}(\mathcal{Q}, \hat{y}, \mathcal{A})$ | `proposals.offspring`, `surrogates.predictions`, `archives.main`, `runtime.generation`, `runtime.rng` | `evaluations.scores`, `evaluations.acquisition_result` | — |
| SurrogateFitStage | surrogate_fit | Fit surrogate | $\hat{f} \leftarrow \text{fit}(\mathcal{A})$ | `populations.main`, `archives.main` | — | — |
| TopKSelectionStage | top_k_selection | Top-k pre-selection | $\mathcal{Q} \leftarrow \text{top-}k(\mathcal{Q}, \mathbf{s})$ | `proposals.offspring`, `evaluations.scores` | `proposals.offspring` | — |
| SortByScoreStage | sort_by_score | Sort offspring by score | $\mathcal{Q} \leftarrow \text{sort\_desc}(\mathcal{Q},\,\mathbf{s})$ | `proposals.offspring`, `evaluations.scores` | `proposals.offspring`, `evaluations.scores` | — |
| EvaluationPlanStage | evaluation_plan | Plan evaluation | $R \leftarrow \text{plan}(Q)$ | `proposals.offspring`, `evaluations.pending`, `evaluations.request`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.handles`, `evaluations.owners`, `evaluations.acquisition_result`, `evaluations.scores`, `surrogates.predictions`, `runtime.request_id_allocator` | `evaluations.request`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.plan_updates`, `evaluations.pending`, `evaluations.updates`, `evaluations.update_new_ids`, `evaluations.new_ids`, `evaluations.handles`, `evaluations.owners`, `runtime.request_id_allocator` | — |
| AsyncEvaluationSubmitStage | async_evaluation_submit | Submit asynchronous evaluation |  | `proposals.offspring`, `evaluations.acquisition_result`, `evaluations.scores`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.pending`, `evaluations.handles`, `runtime.request_id_allocator` | `evaluations.request`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.plan_updates`, `evaluations.pending`, `runtime.request_id_allocator` | — |
| EvaluationSubmitStage | evaluation_submit | Submit evaluation | $H \leftarrow \text{submit}(R)$ | `evaluations.request`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.pending`, `evaluations.handles`, `evaluations.owners` | `evaluations.pending`, `evaluations.handles`, `evaluations.plan_state` | — |
| EvaluationCollectStage | evaluation_collect | Collect evaluation | $U \leftarrow \text{collect}(H)$ | `evaluations.plan`, `evaluations.plan_state`, `evaluations.pending`, `evaluations.handles`, `evaluations.plan_updates`, `evaluations.request` | `evaluations.updates`, `evaluations.request`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.plan_updates`, `evaluations.pending`, `evaluations.update_new_ids`, `evaluations.new_ids` | — |
| EvaluationApplyStage | evaluation_apply | Apply evaluation | $Q \leftarrow \text{apply}(U)$ | `proposals.offspring`, `evaluations.request`, `evaluations.updates`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.plan_updates`, `evaluations.pending` | `proposals.offspring`, `evaluations.evaluated_offspring`, `evaluations.new_ids`, `evaluations.update_new_ids`, `evaluations.pending` | — |
| EvaluationAcknowledgeStage | evaluation_acknowledge | Acknowledge evaluation | $H \leftarrow \text{ack}(U)$ | `proposals.offspring`, `evaluations.request`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.pending`, `evaluations.handles`, `evaluations.updates`, `evaluations.update_new_ids`, `evaluations.plan_updates`, `evaluations.count` | `evaluations.plan`, `evaluations.plan_state`, `evaluations.plan_updates`, `evaluations.pending`, `evaluations.handles`, `evaluations.count` | — |
| TrueEvaluationStage | true_evaluation | True objective evaluation | $\mathcal{Q}_{eval} \leftarrow \text{eval}(\mathcal{Q})$ | `proposals.offspring` | `proposals.offspring`, `evaluations.evaluated_offspring`, `evaluations.count` | — |
| ArchiveUpdateStage | archive_update | Archive update | $\mathcal{A} \leftarrow \mathcal{A} \cup \mathcal{Q}_{eval}$ | `evaluations.evaluated_offspring`, `archives.main`, `archives.pareto`, `evaluations.plan`, `evaluations.plan_state` | `archives.main`, `archives.pareto`, `evaluations.evaluated_offspring` | — |
| FeedbackStage | feedback | Apply feedback | $\mathcal{Q} \leftarrow \mathrm{feedback}(\mathcal{Q})$ | `proposals.offspring`, `evaluations.evaluated_offspring`, `evaluations.new_ids`, `surrogates.predictions`, `evaluations.plan`, `evaluations.plan_state` | `proposals.offspring`, `feedback.result` | — |
| TellStage | tell | Update population | $P \leftarrow \text{tell}(P, \mathcal{Q})$ | `proposals.offspring`, `proposals.current`, `feedback.result`, `evaluations.evaluated_offspring`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.updates` | — | — |
| SurrogateOnlyLoopStage | surrogate_only_loop | Surrogate-only generations | $\text{for}\;i=1\dots gen\_ctrl$: $P \leftarrow \mathrm{tell}(P,\,\mathrm{acquire}(\mathrm{predict}(\mathrm{ask}(P))))$ | `archives.main` | — | — |
| InitializationStage | initialization | Initialize population | $\mathcal{A}_0,\,P_0 \leftarrow \mathrm{init}(n_{\mathrm{init}})$ | — | — | — |
<!-- END GENERATED STAGE CONTRACTS -->

`TellStage` declares `writes=()` because its state contract describes the
Stage boundary. When a legacy algorithm is used, the legacy adapter can still
mutate the algorithm's population during execution; that runtime behavior is
not added to the Stage contract.

The contract table includes all operational stages, including stages used by
asynchronous evaluation and nested surrogate-only loops.

`AskStage` calls `algorithm.ask()`, writes to `state.offspring`, and fires PostCrossover/PostMutation/PostAskEvent via `cbmanager`.

`SurrogatePredictStage` obtains prediction data; `AcquisitionStage` computes scores and preserves the complete `AcquisitionResult` for planning.

`SurrogateFitStage` is used to pre-fit the surrogate once, ahead of an inner loop where the archive doesn't change.
It is used before a surrogate prediction stage when an inner loop can reuse a fit.

`SurrogateOnlyLoopStage` remains available to stage-based compatibility
strategies.  `GenerationBasedStrategy` represents the same control flow with
the structured `Repeat` region instead.

```{note}
The `state` argument passed to `InitializationStage`'s `execute()` is ignored — it always builds a fresh state from initialization.
Used at the head of a user-defined pipeline when you want initialization itself to be treated as part of the pipeline.
```

See [strategies](../execution_and_evaluation/strategies.md) and the pipeline diagram in [Components overview](../index.md) for how the built-in strategies combine these stages and structured regions.
This page covers the contract of each Stage in isolation.

## Implementing a custom Stage

sequential compatibility経路にカスタムStageを追加するときは、`Stage` を継承して `execute()` を実装します。
`state.replace(...)` で新しいStateを返し、Stageから直接Compiler用のStatePatchを操作しないようにします。

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

互換性用Stageで追加の値を渡す場合は、`OptimizationState.data` に保存します。
値を追加するときは、`state.replace(data={**state.data, "key": value})` のように新しい辞書を渡します。
graph-native componentで同じことを行う場合は、宣言した `StateKey` と `StatePatch` を使います。

## `to_pseudocode`

`to_pseudocode(expand=False, indent=0)` is a mechanism that outputs each Stage's `notation` as pseudocode for a paper (LaTeX algorithmic notation).
`AskStage`/`TellStage`/`SurrogateOnlyLoopStage` have custom implementations that expand `Algorithm.ask_notation`/`tell_notation` when `expand=True`.

## Related components

- [Extension guidelines](../extension_guidelines.md): How to rearrange stages via `pipeline.replace`/`find`
- [strategies](../execution_and_evaluation/strategies.md): How the built-in Strategies combine Stages
- [OptimizationState](optimization_state.md): The state object `execute()` reads and writes
- [Components overview](../index.md): The diagram of the overall pipeline structure

## References

- {py:class}`saealib.Stage`
- {py:class}`saealib.Pipeline`
- {py:class}`saealib.CountGenerationStage`
- {py:class}`saealib.AskStage`
- {py:class}`saealib.SurrogatePredictStage`
- {py:class}`saealib.SurrogateFitStage`
- {py:class}`saealib.ArchiveUpdateStage`
- {py:class}`saealib.TellStage`
- {py:class}`saealib.SurrogateOnlyLoopStage`
- {py:class}`saealib.InitializationStage`
