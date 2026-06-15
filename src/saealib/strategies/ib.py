"""
Individual-based / Pre-selection / Local-modeling.

For each offspring, the surrogate manager scores candidates using a local
surrogate model. The top-evaluation_ratio fraction are selected for true evaluation.
"""

import numpy as np

from saealib.callback import PostEvaluationEvent, SurrogateEndEvent, SurrogateStartEvent
from saealib.context import OptimizationContext
from saealib.optimizer import ComponentProvider
from saealib.strategies.base import OptimizationStrategy, assign_tell_f


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

    def step(self, ctx: OptimizationContext, provider: ComponentProvider) -> None:
        """
        Score all offspring with the surrogate, then true-evaluate the top fraction.

        Parameters
        ----------
        ctx : OptimizationContext
            Current optimization context.
        provider : ComponentProvider
            Component provider.
        """
        ctx.count_generation()

        # 1. get candidate solutions as "offspring" (algorithm.ask)
        offspring = provider.algorithm.ask(ctx, provider)
        n_offspring = len(offspring)
        n_eval = max(1, int(self.evaluation_ratio * n_offspring))

        provider.dispatch(
            SurrogateStartEvent(ctx=ctx, offspring=offspring)
        )

        # 2. score candidates via surrogate manager
        scores, predictions = provider.surrogate_manager.score_candidates(
            offspring.x, ctx.archive, ctx
        )
        for i, pred in enumerate(predictions):
            assign_tell_f(offspring[i], pred, ctx)

        provider.dispatch(
            SurrogateEndEvent(ctx=ctx, offspring=offspring)
        )

        # 3. top-k selection and true evaluation
        idx = np.argsort(-scores)
        offspring = offspring.extract(idx)

        result = provider.evaluator.evaluate_batch(offspring.x[:n_eval], ctx.problem)
        for i in range(n_eval):
            f_i, g_i, cv_i = result.f[i], result.g[i], float(result.cv[i])
            offspring[i].f = f_i
            offspring[i].g = g_i
            offspring[i].cv = cv_i
            ctx.archive.add({"x": offspring[i].x, "f": f_i, "g": g_i, "cv": cv_i})
            ctx.pareto_archive.add(
                {"x": offspring[i].x, "f": f_i, "g": g_i, "cv": cv_i}
            )
        ctx.count_fe(n_eval)

        evaluated = offspring.extract(list(range(n_eval)))
        provider.dispatch(
            PostEvaluationEvent(ctx=ctx, offspring=evaluated)
        )

        # 4. update algorithm with offspring (algorithm.tell)
        provider.algorithm.tell(ctx, provider, offspring)
