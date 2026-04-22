"""
Pre-selection strategy.

Generates a large pool of candidates, screens them with the surrogate,
and selects the top-k for true objective evaluation.
"""

import numpy as np

from saealib.callback import PostEvaluationEvent, SurrogateEndEvent, SurrogateStartEvent
from saealib.context import OptimizationContext
from saealib.optimizer import ComponentProvider
from saealib.strategies.base import OptimizationStrategy


class PreSelectionStrategy(OptimizationStrategy):
    """Pre-selection strategy."""

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

    def step(self, ctx: OptimizationContext, provider: ComponentProvider) -> None:
        """
        Perform one iteration of optimization processing.

        Parameters
        ----------
        ctx : OptimizationContext
            A dataclass object that holds internal information about the Optimizer.
        provider : ComponentProvider
            Objects of the class in which the component is exposed (ex. Optimizer).
        """
        ctx.count_generation()

        candidates = provider.algorithm.ask(
            ctx, provider, n_offspring=self.n_candidates
        )

        provider.dispatch(
            SurrogateStartEvent(ctx=ctx, provider=provider, offspring=candidates)
        )

        scores, predictions = provider.surrogate_manager.score_candidates(
            candidates.x, ctx.archive, provider, ctx
        )
        for i, pred in enumerate(predictions):
            candidates[i].f = pred.mean[0]

        provider.dispatch(
            SurrogateEndEvent(ctx=ctx, provider=provider, offspring=candidates)
        )

        idx = np.argsort(-scores)
        candidates = candidates.extract(idx)
        n_eval = min(self.n_select, len(candidates))

        for i in range(n_eval):
            candidates[i].f = ctx.problem.evaluate(candidates[i].x)
            g, cv = ctx.problem.evaluate_constraints(candidates[i].x)
            candidates[i].cv = cv
            candidates[i].g = g
            ctx.archive.add(
                {"x": candidates[i].x, "f": candidates[i].f, "g": g, "cv": cv}
            )
        ctx.count_fe(n_eval)

        evaluated = candidates.extract(list(range(n_eval)))
        provider.dispatch(
            PostEvaluationEvent(ctx=ctx, provider=provider, offspring=evaluated)
        )

        provider.algorithm.tell(ctx, provider, candidates)
