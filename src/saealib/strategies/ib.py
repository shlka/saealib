"""
Individual-based / Pre-selection / Local-modeling.

For each offspring, the surrogate manager scores candidates using a local
surrogate model. The top-evaluation_ratio fraction are selected for true evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from saealib.pipeline import Pipeline
from saealib.policies.evaluation import RatioEvaluation
from saealib.policies.feedback import FeedbackBuilder, MixedFeedback
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
class IndividualBasedStrategy(OptimizationStrategy):
    """Individual-based strategy."""

    requires_surrogate: bool = True

    def __init__(self, evaluation_ratio: float = 0.1):
        """
        Initialize IndividualBasedStrategy.

        Parameters
        ----------
        evaluation_ratio : float
            Ratio of offspring selected for true objective evaluation.
            The top ``evaluation_ratio`` fraction of offspring are evaluated
            with the real objective function; the rest are discarded after
            surrogate scoring.
            The number of neighbors for local surrogate fitting is now
            configured on the SurrogateManager (e.g. LocalSurrogateManager).
        """
        self.evaluation_ratio = evaluation_ratio
        self.pipeline: Pipeline | None = None

    @property
    def evaluation_planner(self):
        """Return the current ratio planner."""
        return RatioEvaluation(self.evaluation_ratio, sanitize_nonfinite=True)

    feedback_builder = MixedFeedback()

    def build_pipeline(self, provider: ComponentProvider) -> Pipeline:
        """Build the current individual-based pipeline."""
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
                AskStage(provider.algorithm, cbmanager=cbmanager),
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
        Score all offspring with the surrogate, then true-evaluate the top fraction.

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
