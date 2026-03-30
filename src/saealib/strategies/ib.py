"""
Individual-based / Pre-selection / Local-modeling.

For each offspring, the surrogate manager scores candidates using a local
surrogate model. The top-rsm fraction are selected for true evaluation.
"""

import numpy as np

from saealib.callback import CallbackEvent
from saealib.context import OptimizationContext
from saealib.optimizer import ComponentProvider
from saealib.strategies.base import OptimizationStrategy


class IndividualBasedStrategy(OptimizationStrategy):
    """Individual-based strategy."""

    def __init__(self, rsm: float = 0.1):
        """
        Initialize IndividualBasedStrategy.

        Parameters
        ----------
        rsm : float
            Ratio of surrogate model usage. The top ``rsm`` fraction of
            offspring are selected for true objective evaluation.
            The number of neighbors for local surrogate fitting is now
            configured on the SurrogateManager (e.g. LocalSurrogateManager).
        """
        self.rsm = rsm

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
        n_eval = max(1, int(self.rsm * n_offspring))

        provider.dispatch(CallbackEvent.SURROGATE_START, ctx=ctx)

        # 2. score candidates via surrogate manager
        reference = ctx.archive.f.min(axis=0)  # (n_obj,) component-wise best
        scores, predictions = provider.surrogate_manager.score_candidates(
            offspring.x, ctx.archive, reference
        )
        for i, pred in enumerate(predictions):
            offspring[i].f = pred.mean[0]  # assign predicted objectives (n_obj,)

        provider.dispatch(CallbackEvent.SURROGATE_END, ctx=ctx)

        # 3. top-k selection and true evaluation
        idx = np.argsort(-scores)
        offspring = offspring.extract(idx)

        for i in range(n_eval):
            offspring[i].f = ctx.problem.evaluate(offspring[i].x)
            ctx.archive.add(x=offspring[i].x, f=offspring[i].f)
        ctx.count_fe(n_eval)

        # 4. update algorithm with offspring (algorithm.tell)
        provider.algorithm.tell(ctx, provider, offspring)
