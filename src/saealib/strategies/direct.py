"""Direct strategy: plain EA without surrogate evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from saealib.core.compiler.graph import ComponentGraph
from saealib.pipeline import Pipeline
from saealib.policies.evaluation import EvaluateAll
from saealib.policies.feedback import FeedbackBuilder, TrueOnlyFeedback
from saealib.registry import register
from saealib.stages import (
    ArchiveUpdateStage,
    AskStage,
    AsyncEvaluationSubmitStage,
    CountGenerationStage,
    EvaluationAcknowledgeStage,
    EvaluationApplyStage,
    EvaluationCollectStage,
    EvaluationPlanStage,
    EvaluationSubmitStage,
    FeedbackStage,
    PendingEvaluationContextStage,
    TellStage,
)
from saealib.strategies.base import (
    OptimizationStrategy,
    build_runtime_neutral_graph,
)

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.optimizer import ComponentProvider


class DirectStrategy(OptimizationStrategy):
    """Plain EA strategy without surrogate scoring.

    Every candidate produced by :meth:`~saealib.algorithms.Algorithm.ask` is
    evaluated with the true objective function.  No surrogate manager is
    required.
    """

    requires_surrogate: bool = False

    evaluation_planner = EvaluateAll()
    feedback_builder = TrueOnlyFeedback()

    def __init__(self, n_offspring: int | None = None) -> None:
        if n_offspring is not None and n_offspring < 1:
            raise ValueError("n_offspring must be positive")
        self.n_offspring = n_offspring
        self.pipeline: Pipeline | None = None

    def build_pipeline(self, provider: ComponentProvider) -> Pipeline:
        """Build the current direct evaluation pipeline."""
        cbmanager = getattr(provider, "cbmanager", None)
        scheduler = getattr(provider, "async_evaluation_scheduler", None)
        evaluation_planner = (
            getattr(provider, "evaluation_planner", None) or self.evaluation_planner
        )
        feedback_builder = cast(
            FeedbackBuilder | None, getattr(provider, "feedback_builder", None)
        )
        if feedback_builder is None:
            feedback_builder = self.feedback_builder
        if scheduler is not None:
            evaluation_tail = [
                AsyncEvaluationSubmitStage(
                    scheduler,
                    evaluation_planner,
                    feedback_builder,
                    provider.algorithm,
                    cbmanager,
                )
            ]
        else:
            evaluation_tail = [
                EvaluationPlanStage(evaluation_planner),
                EvaluationSubmitStage(provider.evaluator),
                EvaluationCollectStage(provider.evaluator),
                EvaluationApplyStage(),
                ArchiveUpdateStage(),
                FeedbackStage(feedback_builder),
                TellStage(provider.algorithm),
                EvaluationAcknowledgeStage(provider.evaluator, cbmanager),
            ]
        return Pipeline(
            [
                CountGenerationStage(),
                *(
                    [PendingEvaluationContextStage(scheduler)]
                    if scheduler is not None
                    else []
                ),
                AskStage(
                    provider.algorithm,
                    n_offspring=self.n_offspring,
                    cbmanager=cbmanager,
                ),
                *evaluation_tail,
            ]
        )

    def build_graph(self, provider: ComponentProvider) -> ComponentGraph:
        """Build the graph view of the current pipeline configuration."""
        return build_runtime_neutral_graph(self, provider)

    def step(
        self, ctx: OptimizationState, provider: ComponentProvider
    ) -> OptimizationState:
        """Evaluate all offspring with the true objective function.

        Rebuilds the pipeline each call so component/parameter changes take
        effect immediately.

        Parameters
        ----------
        ctx : OptimizationState
            Current optimization context.
        provider : ComponentProvider
            Component provider.
        """
        scheduler = getattr(provider, "async_evaluation_scheduler", None)
        if scheduler is not None and ctx.pending_evaluations:
            ctx = scheduler.poll(ctx, wait=False)
            if not ctx.pending_evaluations:
                return ctx
            if len(ctx.pending_evaluations) >= scheduler.max_pending:
                return ctx
        self.pipeline = self.build_pipeline(provider)
        return self.pipeline.execute(ctx)


@register()
class SteadyStateStrategy(DirectStrategy):
    """Evaluate one candidate per step and refill asynchronous worker slots."""

    def __init__(self) -> None:
        super().__init__(n_offspring=1)

    def build_graph(self, provider: ComponentProvider) -> ComponentGraph:
        """Build the graph view of the steady-state pipeline."""
        return build_runtime_neutral_graph(self, provider)
