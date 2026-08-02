"""Pre-selection strategy.

Generates a large pool of candidates, screens them with the surrogate,
and selects the top-k for true objective evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from saealib.pipeline import Pipeline
from saealib.policies.evaluation import TopKEvaluation
from saealib.policies.feedback import FeedbackBuilder, TrueOnlyFeedback
from saealib.registry import register
from saealib.stages import (
    AcquisitionStage,
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
    SurrogatePredictStage,
    TellStage,
)
from saealib.strategies.base import OptimizationStrategy

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.optimizer import ComponentProvider

_MISSING = object()


@register()
class PreSelectionStrategy(OptimizationStrategy):
    """Pre-selection strategy.

    Generates ``n_candidates`` offspring, scores them with the surrogate,
    and true-evaluates only the top ``n_select`` candidates.

    Parameters
    ----------
    n_candidates : int
        Number of candidates to generate and score with the surrogate.
    n_select : int
        Number of top-scoring candidates to evaluate with the true
        objective function.
    """

    requires_surrogate: bool = True

    def __init__(self, n_candidates: int, n_select: int):
        """
        Initialize PreSelectionStrategy.

        Parameters
        ----------
        n_candidates : int
            Number of candidates to generate and score with the surrogate.
        n_select : int
            Number of top-scoring candidates to evaluate with the true
            objective function.
        """
        self.n_candidates = n_candidates
        self.n_select = n_select
        self.pipeline: Pipeline | None = None

    @property
    def evaluation_planner(self):
        """Return the current top-k planner."""
        return TopKEvaluation(
            min(self.n_select, self.n_candidates), sanitize_nonfinite=True
        )

    feedback_builder = TrueOnlyFeedback()

    def build_pipeline(self, provider: ComponentProvider) -> Pipeline:
        """Build the current pre-selection pipeline."""
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
                    n_offspring=self.n_candidates,
                    cbmanager=cbmanager,
                ),
                SurrogatePredictStage(provider.surrogate_manager, cbmanager=cbmanager),
                *(
                    [PendingEvaluationContextStage(scheduler)]
                    if scheduler is not None
                    else []
                ),
                AcquisitionStage(
                    provider.acquisition,
                    cbmanager=cbmanager,
                ),
                *evaluation_tail,
            ]
        )

    def step(
        self, ctx: OptimizationState, provider: ComponentProvider
    ) -> OptimizationState:
        """
        Generate a large candidate pool, screen with surrogate, true-evaluate top-k.

        Rebuilds the pipeline each call so component/parameter changes take
        effect immediately.

        Parameters
        ----------
        ctx : OptimizationState
            Current optimization context.
        provider : ComponentProvider
            Component provider.

        Returns
        -------
        OptimizationState
            Updated state after one generation step.
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
