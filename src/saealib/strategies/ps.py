"""
Pre-selection strategy.

Generates a large pool of candidates, screens them with the surrogate,
and selects the top-k for true objective evaluation.
"""

import numpy as np

from saealib.callback import PostEvaluationEvent, SurrogateEndEvent, SurrogateStartEvent
from saealib.context import OptimizationContext
from saealib.optimizer import ComponentProvider
from saealib.strategies.base import OptimizationStrategy, assign_tell_f


class PreSelectionStrategy(OptimizationStrategy):
    """Pre-selection strategy."""

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

    def step(self, ctx: OptimizationContext, provider: ComponentProvider) -> None:
        """
        Generate a large candidate pool, screen with surrogate, true-evaluate top-k.

        Parameters
        ----------
        ctx : OptimizationContext
            Current optimization context.
        provider : ComponentProvider
            Component provider.
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
            assign_tell_f(candidates[i], pred, ctx)

        provider.dispatch(
            SurrogateEndEvent(ctx=ctx, provider=provider, offspring=candidates)
        )

        idx = np.argsort(-scores)
        candidates = candidates.extract(idx)
        n_eval = min(self.n_select, len(candidates))

        result = provider.evaluator.evaluate_batch(candidates.x[:n_eval], ctx.problem)
        for i in range(n_eval):
            f_i, g_i, cv_i = result.f[i], result.g[i], float(result.cv[i])
            candidates[i].f = f_i
            candidates[i].g = g_i
            candidates[i].cv = cv_i
            ctx.archive.add({"x": candidates[i].x, "f": f_i, "g": g_i, "cv": cv_i})
        ctx.count_fe(n_eval)

        evaluated = candidates.extract(list(range(n_eval)))
        provider.dispatch(
            PostEvaluationEvent(ctx=ctx, provider=provider, offspring=evaluated)
        )

        provider.algorithm.tell(ctx, provider, candidates)
