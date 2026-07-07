"""Pre-selection strategy.

Generates a large pool of candidates, screens them with the surrogate,
and selects the top-k for true objective evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from saealib.pipeline import Pipeline
from saealib.registry import register
from saealib.stages import (
    ArchiveUpdateStage,
    AskStage,
    CountGenerationStage,
    SurrogateScoreStage,
    TellStage,
    TopKSelectionStage,
    TrueEvaluationStage,
)
from saealib.strategies.base import OptimizationStrategy

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.optimizer import ComponentProvider


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

    def _build_pipeline(self, provider: ComponentProvider) -> Pipeline:
        cbmanager = getattr(provider, "cbmanager", None)
        return Pipeline(
            [
                CountGenerationStage(),
                AskStage(
                    provider.algorithm,
                    n_offspring=self.n_candidates,
                    cbmanager=cbmanager,
                ),
                SurrogateScoreStage(provider.surrogate_manager, cbmanager=cbmanager),
                TopKSelectionStage(k=self.n_select),
                TrueEvaluationStage(provider.evaluator, cbmanager=cbmanager),
                ArchiveUpdateStage(),
                TellStage(provider.algorithm),
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
        self.pipeline = self._build_pipeline(provider)
        return self.pipeline.execute(ctx)
