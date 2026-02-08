"""
Individual-based / Pre-selection / Local-modeling.

For each offspring, construct a surrogate model to obtain a predicted value.
Select the top models based on the RSM ratio for actual evaluation.
The surrogate model is a local approximation of the objective function.

"""

from saealib.callback import CallbackEvent
from saealib.context import OptimizationContext
from saealib.optimizer import ComponentProvider
from saealib.strategies.base import OptimizationStrategy


class IndividualBasedStrategy(OptimizationStrategy):
    """Individual-based strategy."""

    def __init__(self, n_train: int = 50, rsm: float = 0.1):
        """Initialize IndividualBasedStrategy class."""
        self.n_train = n_train
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

        # 2. predict obj value using surrogate (surrogate.fit, surrogate.predict)
        for i in range(n_offspring):
            train_idx, _ = ctx.archive.get_knn(offspring[i].x, self.n_train)
            train_x = ctx.archive.x[train_idx]
            train_f = ctx.archive.f[train_idx]
            provider.surrogate.fit(train_x, train_f)
            offspring[i].f = provider.surrogate.predict(offspring[i].x)

        provider.dispatch(CallbackEvent.SURROGATE_END, ctx=ctx)

        # 3. top-k selection and true evaluation
        idx = ctx.comparator.sort(offspring.f, offspring.cv)
        offspring = offspring.extract(idx)

        for i in range(n_eval):
            offspring[i].f = ctx.problem.evaluate(offspring[i].x)
            ctx.archive.add(x=offspring[i].x, f=offspring[i].f)
        ctx.count_fe(n_eval)

        # 4. update algorithm with offspring (algorithm.tell)
        provider.algorithm.tell(ctx, provider, offspring)
