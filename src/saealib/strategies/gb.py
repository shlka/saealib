"""
Generation-based strategy.

Each call to ``step`` runs ``gen_ctrl`` surrogate-only generations followed
by one generation of true objective evaluations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from saealib.core.compiler.graph import ComponentGraph
from saealib.core.graph_builder import (
    NodeAdapterSpec,
    StageContractNodeAdapter,
    build_decomposed_component_graph_from_specs,
)
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
    CountGenerationStage,
    EvaluationAcknowledgeStage,
    EvaluationApplyStage,
    EvaluationCollectStage,
    EvaluationPlanStage,
    EvaluationSubmitStage,
    FeedbackStage,
    SurrogateOnlyLoopStage,
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

    def build_graph(self, provider: ComponentProvider) -> ComponentGraph:
        """Build the canonical generation-based strategy graph."""
        cbmanager = getattr(provider, "cbmanager", None)
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
        stages = (
            SurrogateOnlyLoopStage(
                provider.algorithm,
                provider.surrogate_manager,
                self.gen_ctrl,
                cbmanager,
                acquisition=provider.acquisition,
                feedback_builder=PredictedFeedback(),
            ),
            CountGenerationStage(),
            AskStage(provider.algorithm, cbmanager=cbmanager),
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
