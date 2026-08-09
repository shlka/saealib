"""Direct strategy: plain EA without surrogate evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from saealib.core.compiler.graph import ComponentGraph
from saealib.pipeline import Pipeline, Stage
from saealib.policies.evaluation import EvaluateAll
from saealib.policies.feedback import FeedbackBuilder, TrueOnlyFeedback
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
    TellStage,
)
from saealib.strategies.base import (
    OptimizationStrategy,
    build_pipeline_from_graph,
    build_runtime_neutral_graph,
)

if TYPE_CHECKING:
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

    def _build_stages(self, provider: ComponentProvider) -> tuple[Stage, ...]:
        """Build the canonical direct Stage sequence."""
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
            EvaluationPlanStage(evaluation_planner),
            EvaluationSubmitStage(provider.evaluator),
            EvaluationCollectStage(provider.evaluator),
            EvaluationApplyStage(),
            ArchiveUpdateStage(),
            FeedbackStage(feedback_builder),
            TellStage(provider.algorithm),
            EvaluationAcknowledgeStage(provider.evaluator, cbmanager),
        ]
        return (
            CountGenerationStage(),
            AskStage(
                provider.algorithm,
                n_offspring=self.n_offspring,
                cbmanager=cbmanager,
            ),
            *evaluation_tail,
        )

    def build_pipeline(self, provider: ComponentProvider) -> Pipeline:
        """Return the legacy pipeline facade for the canonical graph."""
        return build_pipeline_from_graph(self.build_graph(provider))

    def build_graph(self, provider: ComponentProvider) -> ComponentGraph:
        """Build the canonical direct strategy graph."""
        return build_runtime_neutral_graph(self, provider)


@register()
class SteadyStateStrategy(DirectStrategy):
    """Evaluate one candidate per step and refill asynchronous worker slots."""

    def __init__(self) -> None:
        super().__init__(n_offspring=1)

    def build_graph(self, provider: ComponentProvider) -> ComponentGraph:
        """Build the graph view of the steady-state pipeline."""
        return build_runtime_neutral_graph(self, provider)
