"""
Generation-based strategy.

Each call to ``step`` runs ``gen_ctrl`` surrogate-only generations followed
by one generation of true objective evaluations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from saealib.core.compiler.graph import ComponentGraph
from saealib.pipeline import Pipeline
from saealib.policies.feedback import (
    FeedbackBuilder,
    MixedFeedback,
    PredictedFeedback,
    TrueOnlyFeedback,
)
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
    SurrogateOnlyLoopStage,
    TellStage,
)
from saealib.strategies.base import (
    OptimizationStrategy,
    build_runtime_neutral_graph,
)

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.optimizer import ComponentProvider

_MISSING = object()


@register()
class GenerationBasedStrategy(OptimizationStrategy):
    """Generation-based strategy.

    Parameters
    ----------
    gen_ctrl : int
        Number of surrogate-only generations executed inside each :meth:`step`
        call before one generation of true objective evaluation.
    """

    requires_surrogate: bool = True
    feedback_builder = MixedFeedback()
    true_feedback_builder = TrueOnlyFeedback()

    def __init__(self, gen_ctrl: int) -> None:
        self.gen_ctrl = gen_ctrl
        self.pipeline: Pipeline | None = None

    def build_pipeline(self, provider: ComponentProvider) -> Pipeline:
        """Build the current generation-based pipeline."""
        cbmanager = getattr(provider, "cbmanager", None)
        scheduler = getattr(provider, "async_evaluation_scheduler", None)
        evaluation_planner = (
            getattr(provider, "evaluation_planner", None) or self.evaluation_planner
        )
        builder_explicit = getattr(provider, "feedback_builder_explicit", None)
        configured_builder = getattr(provider, "feedback_builder", None)
        feedback_builder: FeedbackBuilder | None = None
        if builder_explicit:
            feedback_builder = cast(FeedbackBuilder | None, configured_builder)
        if feedback_builder is None:
            feedback_builder = self.true_feedback_builder
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
                SurrogateOnlyLoopStage(
                    provider.algorithm,
                    provider.surrogate_manager,
                    self.gen_ctrl,
                    cbmanager,
                    acquisition=provider.acquisition,
                    feedback_builder=PredictedFeedback(),
                ),
                CountGenerationStage(),
                *(
                    [PendingEvaluationContextStage(scheduler)]
                    if scheduler is not None
                    else []
                ),
                AskStage(provider.algorithm, cbmanager=cbmanager),
                *evaluation_tail,
            ]
        )

    def build_graph(self, provider: ComponentProvider) -> ComponentGraph:
        """Build the graph view of the current generation pipeline."""
        return build_runtime_neutral_graph(self, provider)

    def step(
        self, ctx: OptimizationState, provider: ComponentProvider
    ) -> OptimizationState:
        """Run ``gen_ctrl`` surrogate-only generations, then one true-evaluation step.

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
