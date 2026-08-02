"""Direct strategy: plain EA without surrogate evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from saealib.pipeline import Pipeline
from saealib.policies.evaluation import EvaluateAll
from saealib.policies.feedback import TrueOnlyFeedback
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
from saealib.strategies.base import OptimizationStrategy

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

    evaluation_policy = EvaluateAll()
    feedback_policy = TrueOnlyFeedback()

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None

    def _build_pipeline(self, provider: ComponentProvider) -> Pipeline:
        cbmanager = getattr(provider, "cbmanager", None)
        scheduler = getattr(provider, "async_scheduler", None)
        evaluation_policy = (
            getattr(provider, "evaluation_policy", None) or self.evaluation_policy
        )
        feedback_policy = (
            getattr(provider, "feedback_policy", None) or self.feedback_policy
        )
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
        scheduler = getattr(provider, "async_scheduler", None)
        if scheduler is not None and ctx.pending_evaluations:
            ctx = scheduler.poll(ctx, wait=False)
            if not ctx.pending_evaluations:
                return ctx
            if len(ctx.pending_evaluations) >= scheduler.max_pending:
                return ctx
        self.pipeline = self._build_pipeline(provider)
        return self.pipeline.execute(ctx)
