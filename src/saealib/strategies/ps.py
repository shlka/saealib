"""Pre-selection strategy.

Generates a large pool of candidates, screens them with the surrogate,
and selects the top-k for true objective evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from saealib.core.compiler.graph import ComponentGraph
from saealib.core.graph_builder import (
    NodeAdapterSpec,
    StageContractNodeAdapter,
    build_decomposed_component_graph_from_specs,
)
from saealib.exceptions import ValidationError
from saealib.pipeline import Pipeline
from saealib.policies.evaluation import TopKEvaluation
from saealib.policies.feedback import FeedbackBuilder, TrueOnlyFeedback
from saealib.registry import register
from saealib.stages import (
    AcquisitionStage,
    ArchiveUpdateStage,
    AskStage,
    CountGenerationStage,
    EvaluationAcknowledgeStage,
    EvaluationApplyStage,
    EvaluationCollectStage,
    EvaluationPlanStage,
    EvaluationSubmitStage,
    FeedbackStage,
    SurrogatePredictStage,
    TellStage,
)
from saealib.strategies.base import (
    OptimizationStrategy,
)

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
        if not isinstance(n_select, int) or isinstance(n_select, bool) or n_select < 1:
            raise ValidationError("n_select must be a positive integer")
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

    def build_graph(self, provider: ComponentProvider) -> ComponentGraph:
        """Build the canonical pre-selection strategy graph."""
        cbmanager = getattr(provider, "cbmanager", None)
        evaluation_planner = (
            getattr(provider, "evaluation_planner", None) or self.evaluation_planner
        )
        feedback_builder = cast(
            FeedbackBuilder | None, getattr(provider, "feedback_builder", None)
        )
        if feedback_builder is None:
            feedback_builder = self.feedback_builder
        evaluation_tail = [
            EvaluationPlanStage(
                evaluation_planner,
                cors_runtime_warning=getattr(provider, "_cors_runtime_warning", None),
            ),
            EvaluationSubmitStage(provider.evaluator),
            EvaluationCollectStage(provider.evaluator),
            EvaluationApplyStage(),
            ArchiveUpdateStage(),
            FeedbackStage(feedback_builder),
            TellStage(provider.algorithm),
            EvaluationAcknowledgeStage(provider.evaluator, cbmanager),
        ]
        stages = (
            CountGenerationStage(),
            AskStage(
                provider.algorithm,
                n_offspring=self.n_candidates,
                cbmanager=cbmanager,
            ),
            SurrogatePredictStage(provider.surrogate_manager, cbmanager=cbmanager),
            AcquisitionStage(
                provider.acquisition,
                cbmanager=cbmanager,
            ),
            *evaluation_tail,
        )
        specs = tuple(
            NodeAdapterSpec(
                component_id=stage.name,
                adapter=StageContractNodeAdapter(stage, node_path=stage.name),
            )
            for stage in stages
        )
        return build_decomposed_component_graph_from_specs(specs)

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
        return super().step(ctx, provider)
