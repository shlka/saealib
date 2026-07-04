"""
Generation-based strategy.

Each call to ``step`` runs ``gen_ctrl`` surrogate-only generations followed
by one generation of true objective evaluations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from saealib.pipeline import Pipeline
from saealib.registry import register
from saealib.stages import (
    ArchiveUpdateStage,
    AskStage,
    CountGenerationStage,
    SurrogateOnlyLoopStage,
    TellStage,
    TrueEvaluationStage,
)
from saealib.strategies.base import OptimizationStrategy

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

    def __init__(self, gen_ctrl: int) -> None:
        self.gen_ctrl = gen_ctrl
        self.pipeline: Pipeline | None = None

    def _build_pipeline(self, provider: ComponentProvider) -> Pipeline:
        cbmanager = getattr(provider, "cbmanager", None)
        return Pipeline(
            [
                SurrogateOnlyLoopStage(
                    provider.algorithm,
                    provider.surrogate_manager,
                    self.gen_ctrl,
                    cbmanager,
                ),
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
        """Run ``gen_ctrl`` surrogate-only generations, then one true-evaluation step.

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
