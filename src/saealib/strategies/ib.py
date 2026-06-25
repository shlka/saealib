"""
Individual-based / Pre-selection / Local-modeling.

For each offspring, the surrogate manager scores candidates using a local
surrogate model. The top-evaluation_ratio fraction are selected for true evaluation.
"""

from __future__ import annotations

import functools
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


def _n_eval_by_ratio(state: OptimizationState, *, ratio: float) -> int:
    assert state.offspring is not None
    return max(1, int(ratio * len(state.offspring)))


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

    def _build_pipeline(self, provider: ComponentProvider) -> Pipeline:
        cbmanager = getattr(provider, "cbmanager", None)
        return Pipeline(
            [
                CountGenerationStage(),
                AskStage(provider.algorithm, cbmanager=cbmanager),
                SurrogateScoreStage(provider.surrogate_manager, cbmanager=cbmanager),
                SortByScoreStage(),
                TrueEvaluationStage(
                    provider.evaluator,
                    cbmanager=cbmanager,
                    n_eval=functools.partial(
                        _n_eval_by_ratio, ratio=self.evaluation_ratio
                    ),
                ),
                ArchiveUpdateStage(),
                TellStage(provider.algorithm),
            ]
        )

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
        if self.pipeline is None:
            self.pipeline = self._build_pipeline(provider)
        return self.pipeline.execute(ctx)
