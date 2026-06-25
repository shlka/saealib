"""
Individual-based / Pre-selection / Local-modeling.

For each offspring, the surrogate manager scores candidates using a local
surrogate model. The top-evaluation_ratio fraction are selected for true evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from saealib.pipeline import Pipeline
from saealib.stages import (
    ArchiveUpdateStage,
    AskStage,
    CountGenerationStage,
    SortByScoreStage,
    SurrogateScoreStage,
    TellStage,
    TrueEvaluationStage,
)
from saealib.strategies.base import OptimizationStrategy

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.optimizer import ComponentProvider


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

    def step(
        self, ctx: OptimizationState, provider: ComponentProvider
    ) -> OptimizationState:
        """
        Score all offspring with the surrogate, then true-evaluate the top fraction.

        Parameters
        ----------
        ctx : OptimizationState
            Current optimization context.
        provider : ComponentProvider
            Component provider.
        """
        cbmanager = getattr(provider, "cbmanager", None)
        ratio = self.evaluation_ratio
        pipeline = Pipeline(
            [
                CountGenerationStage(),
                AskStage(provider.algorithm, cbmanager=cbmanager),
                SurrogateScoreStage(provider.surrogate_manager, cbmanager=cbmanager),
                SortByScoreStage(),
                TrueEvaluationStage(
                    provider.evaluator,
                    cbmanager=cbmanager,
                    n_eval=lambda s: max(1, int(ratio * len(s.offspring))),
                ),
                ArchiveUpdateStage(),
                TellStage(provider.algorithm),
            ]
        )
        return pipeline.execute(ctx)
