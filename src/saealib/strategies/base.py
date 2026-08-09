"""Base class for optimization strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

from saealib.context import OptimizationState
from saealib.core.compiler.graph import ComponentGraph
from saealib.core.contracts import ComponentContract, PortContract, StateContract
from saealib.core.graph_builder import build_component_graph
from saealib.core.state import PENDING_EVALUATIONS
from saealib.optimizer import ComponentProvider
from saealib.pipeline import Pipeline, Stage
from saealib.policies.evaluation import EvaluateAll, EvaluationPlanner
from saealib.policies.feedback import FeedbackBuilder, MixedFeedback
from saealib.stages import (
    AsyncEvaluationSubmitStage,
    EvaluationPlanStage,
    PendingEvaluationContextStage,
    RuntimeNoOpStage,
    RuntimeStage,
)


class _SyncGraphProvider:
    """Forward a provider while disabling its async scheduler for graph shape."""

    def __init__(self, provider: ComponentProvider) -> None:
        self._provider = provider

    @property
    def async_evaluation_scheduler(self) -> None:
        return None

    def __getattr__(self, name: str) -> object:
        return getattr(self._provider, name)


def _graph_stages(pipeline: Pipeline) -> tuple[Stage, ...]:
    """Remove typed pending-context stages from one graph-only pipeline."""
    return tuple(
        stage
        for stage in pipeline.stages
        if not isinstance(stage, PendingEvaluationContextStage)
    )


def build_runtime_neutral_graph(
    strategy: OptimizationStrategy, provider: ComponentProvider
) -> ComponentGraph:
    """Build a canonical graph while retaining runtime-specific execution."""
    scheduler = getattr(provider, "async_evaluation_scheduler", None)
    sync_provider = _SyncGraphProvider(provider) if scheduler is not None else provider
    sync_stages = _graph_stages(
        strategy.build_pipeline(cast(ComponentProvider, sync_provider))
    )
    if scheduler is None:
        return build_component_graph(Pipeline(list(sync_stages)))

    async_stages = _graph_stages(strategy.build_pipeline(provider))
    sync_tail = next(
        index
        for index, stage in enumerate(sync_stages)
        if isinstance(stage, EvaluationPlanStage)
    )
    async_tail = next(
        index
        for index, stage in enumerate(async_stages)
        if isinstance(stage, AsyncEvaluationSubmitStage)
    )
    if sync_tail != async_tail:
        raise TypeError("sync and async strategy graph prefixes must have equal shape")

    bridged: list[Stage] = []
    for index, sync_stage in enumerate(sync_stages):
        if index < sync_tail:
            async_stage = async_stages[index]
        elif index == sync_tail:
            async_stage = async_stages[async_tail]
        else:
            async_stage = RuntimeNoOpStage()
        bridged.append(RuntimeStage(sync_stage, async_stage, async_mode=True))
    return build_component_graph(Pipeline(bridged))


class OptimizationStrategy(ABC):
    """Base class for optimization strategies.

    Strategies compose their generation logic from a pipeline built for each
    step.  ``build_pipeline`` is the public extension contract, so provider
    replacements and strategy parameter changes are observed on the next
    execution.
    """

    # Optimizer.validate() checks this to ensure surrogate_manager is configured.
    requires_surrogate: bool = False

    evaluation_planner: EvaluationPlanner = EvaluateAll()

    feedback_builder: FeedbackBuilder = MixedFeedback()

    def contract(self) -> ComponentContract:
        """Return the strategy contract."""
        return ComponentContract(
            ports={"strategy": PortContract()},
            state=StateContract(
                reads=(PENDING_EVALUATIONS,),
            ),
        )

    @abstractmethod
    def build_pipeline(self, provider: ComponentProvider):
        """Build the next pipeline from the current component provider."""
        ...

    def step(
        self, ctx: OptimizationState, provider: ComponentProvider
    ) -> OptimizationState:
        """
        Perform one generation step: generate, score, evaluate, and update.

        Parameters
        ----------
        ctx : OptimizationState
            Current optimization context.
        provider : ComponentProvider
            Component provider.

        Returns
        -------
        OptimizationState
            Updated state returned by the pipeline.
        """
        return self.build_pipeline(provider).execute(ctx)
