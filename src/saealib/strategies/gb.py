"""
Generation-based strategy.

Each call to ``step`` runs ``gen_ctrl`` surrogate-only generations followed
by one generation of true objective evaluations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from saealib.pipeline import Pipeline
from saealib.policies.feedback import MixedFeedback, PredictedFeedback, TrueOnlyFeedback
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
from saealib.strategies.base import OptimizationStrategy

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.optimizer import ComponentProvider


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
    feedback_policy = MixedFeedback()
    true_feedback_policy = TrueOnlyFeedback()

    def __init__(self, gen_ctrl: int) -> None:
        self.gen_ctrl = gen_ctrl
        self.pipeline: Pipeline | None = None

    def _build_pipeline(self, provider: ComponentProvider) -> Pipeline:
        cbmanager = getattr(provider, "cbmanager", None)
        scheduler = getattr(provider, "async_scheduler", None)
        evaluation_policy = (
            getattr(provider, "evaluation_policy", None) or self.evaluation_policy
        )
        feedback_policy = (
            getattr(provider, "feedback_policy", None)
            if getattr(provider, "feedback_policy_explicit", False)
            else None
        ) or self.true_feedback_policy
        if scheduler is not None:
            evaluation_tail = [
                AsyncEvaluationSubmitStage(
                    scheduler,
                    evaluation_policy,
                    feedback_policy,
                    provider.algorithm,
                    cbmanager,
                )
            ]
        else:
            evaluation_tail = [
                EvaluationPlanStage(evaluation_policy),
                EvaluationSubmitStage(provider.evaluator),
                EvaluationCollectStage(provider.evaluator),
                EvaluationApplyStage(),
                ArchiveUpdateStage(),
                FeedbackStage(feedback_policy),
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
                    feedback_policy=PredictedFeedback(),
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
        scheduler = getattr(provider, "async_scheduler", None)
        if scheduler is not None and ctx.pending_evaluations:
            ctx = scheduler.poll(ctx, wait=False)
            if not ctx.pending_evaluations:
                return ctx
            if len(ctx.pending_evaluations) >= scheduler.max_pending:
                return ctx
        self.pipeline = self._build_pipeline(provider)
        return self.pipeline.execute(ctx)
