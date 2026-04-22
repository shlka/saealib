"""
Individual-based / Pre-selection / Local-modeling.

For each offspring, the surrogate manager scores candidates using a local
surrogate model. The top-evaluation_ratio fraction are selected for true evaluation.
"""

import numpy as np

from saealib.callback import PostEvaluationEvent, SurrogateEndEvent, SurrogateStartEvent
from saealib.context import OptimizationContext
from saealib.optimizer import ComponentProvider
from saealib.strategies.base import OptimizationStrategy


class IndividualBasedStrategy(OptimizationStrategy):
    """Individual-based strategy."""

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
        Perform one iteration of optimization processing.

        Parameters
        ----------
        ctx : OptimizationContext
            A dataclass object that holds internal information about the Optimizer.
        provider : ComponentProvider
            Objects of the class in which the component is exposed (ex. Optimizer).
        """
        ctx.count_generation()

        # 1. get candidate solutions as "offspring" (algorithm.ask)
        offspring = provider.algorithm.ask(ctx, provider)
        n_offspring = len(offspring)
        n_eval = max(1, int(self.evaluation_ratio * n_offspring))

        provider.dispatch(
            SurrogateStartEvent(ctx=ctx, provider=provider, offspring=offspring)
        )

        # 2. score candidates via surrogate manager
        scores, predictions = provider.surrogate_manager.score_candidates(
            offspring.x, ctx.archive, provider, ctx
        )
        for i, pred in enumerate(predictions):
            offspring[i].f = pred.mean[0]  # assign predicted objectives (n_obj,)

        provider.dispatch(
            SurrogateEndEvent(ctx=ctx, provider=provider, offspring=offspring)
        )

        # 3. top-k selection and true evaluation
        idx = np.argsort(-scores)
        offspring = offspring.extract(idx)

        for i in range(n_eval):
            offspring[i].f = ctx.problem.evaluate(offspring[i].x)
            g, cv = ctx.problem.evaluate_constraints(offspring[i].x)
            offspring[i].cv = cv
            offspring[i].g = g
            ctx.archive.add(
                {"x": offspring[i].x, "f": offspring[i].f, "g": g, "cv": cv}
            )
        ctx.count_fe(n_eval)

        evaluated = offspring.extract(list(range(n_eval)))
        provider.dispatch(
            PostEvaluationEvent(ctx=ctx, provider=provider, offspring=evaluated)
        )

        # 4. update algorithm with offspring (algorithm.tell)
        provider.algorithm.tell(ctx, provider, offspring)
