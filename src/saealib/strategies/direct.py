"""Direct strategy: plain EA without surrogate evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from saealib.pipeline import Pipeline
from saealib.stages import (
    ArchiveUpdateStage,
    AskStage,
    CountGenerationStage,
    TellStage,
    TrueEvaluationStage,
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

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None

    def _build_pipeline(self, provider: ComponentProvider) -> Pipeline:
        cbmanager = getattr(provider, "cbmanager", None)
        return Pipeline(
            [
                CountGenerationStage(),
                AskStage(provider.algorithm, cbmanager=cbmanager),
                TrueEvaluationStage(provider.evaluator, cbmanager=cbmanager),
                ArchiveUpdateStage(),
                TellStage(provider.algorithm),
            ]
        )

    def step(
        self, ctx: OptimizationState, provider: ComponentProvider
    ) -> OptimizationState:
        """Evaluate all offspring with the true objective function.

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
