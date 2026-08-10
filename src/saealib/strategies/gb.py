"""
Generation-based strategy.

Each call to ``step`` runs ``gen_ctrl`` surrogate-only generations followed
by one generation of true objective evaluations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from saealib.core.compiler.graph import ComponentGraph
from saealib.core.compiler.lowerer import lower_pipeline
from saealib.core.contracts.observation import SURROGATE
from saealib.pipeline import Pipeline, Repeat
from saealib.policies.feedback import (
    FeedbackBuilder,
    MixedFeedback,
    PredictedFeedback,
    TrueOnlyFeedback,
)
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
    SurrogateFitStage,
    SurrogatePredictStage,
    TellStage,
    stage_component,
)
from saealib.strategies.base import (
    OptimizationStrategy,
)

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
    feedback_builder = MixedFeedback()
    true_feedback_builder = TrueOnlyFeedback()

    def __init__(self, gen_ctrl: int) -> None:
        self.gen_ctrl = gen_ctrl
        self.pipeline: Pipeline | None = None

    def _feedback_builder(self, provider: ComponentProvider) -> FeedbackBuilder:
        builder_explicit = getattr(provider, "feedback_builder_explicit", None)
        configured_builder = getattr(provider, "feedback_builder", None)
        feedback_builder: FeedbackBuilder | None = None
        if builder_explicit:
            feedback_builder = cast(FeedbackBuilder | None, configured_builder)
        if feedback_builder is None:
            feedback_builder = self.true_feedback_builder
        return feedback_builder

    def build_pipeline(self, provider: ComponentProvider) -> Pipeline:
        """Describe the strategy as a structured Pipeline DSL."""
        cbmanager = getattr(provider, "cbmanager", None)
        evaluation_planner = (
            getattr(provider, "evaluation_planner", None) or self.evaluation_planner
        )
        feedback_builder = self._feedback_builder(provider)

        surrogate_generation = Pipeline(
            name="surrogate_generation",
            steps=[
                stage_component(CountGenerationStage()),
                stage_component(AskStage(provider.algorithm, cbmanager=cbmanager)),
                stage_component(
                    SurrogatePredictStage(
                        provider.surrogate_manager,
                        cbmanager=cbmanager,
                        refit=False,
                    )
                ),
                stage_component(AcquisitionStage(provider.acquisition, cbmanager)),
                stage_component(FeedbackStage(PredictedFeedback())),
                stage_component(TellStage(provider.algorithm, channel=SURROGATE)),
            ],
        )
        true_generation = Pipeline(
            name="true_generation",
            steps=[
                stage_component(CountGenerationStage()),
                stage_component(AskStage(provider.algorithm, cbmanager=cbmanager)),
                stage_component(EvaluationPlanStage(evaluation_planner)),
                stage_component(EvaluationSubmitStage(provider.evaluator)),
                stage_component(EvaluationCollectStage(provider.evaluator)),
                stage_component(EvaluationApplyStage()),
                stage_component(ArchiveUpdateStage()),
                stage_component(FeedbackStage(feedback_builder)),
                stage_component(TellStage(provider.algorithm)),
                stage_component(
                    EvaluationAcknowledgeStage(provider.evaluator, cbmanager)
                ),
            ],
        )
        return Pipeline(
            name="generation_based",
            steps=[
                stage_component(
                    SurrogateFitStage(
                        provider.surrogate_manager,
                        cbmanager=cbmanager,
                    )
                ),
                Repeat(
                    surrogate_generation,
                    count=self.gen_ctrl,
                    name="surrogate_generations",
                ),
                true_generation,
            ],
        )

    def build_graph(self, provider: ComponentProvider) -> ComponentGraph:
        """Lower the structured generation-based Pipeline to its graph."""
        return lower_pipeline(self.build_pipeline(provider))

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
        return super().step(ctx, provider)
