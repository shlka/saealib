"""
Generation-based strategy.

Each call to ``step`` runs ``gen_ctrl`` surrogate-only generations followed
by one generation of true objective evaluations.
"""

from saealib.callback import PostEvaluationEvent, SurrogateEndEvent, SurrogateStartEvent
from saealib.context import OptimizationContext
from saealib.optimizer import ComponentProvider
from saealib.strategies.base import OptimizationStrategy


class GenerationBasedStrategy(OptimizationStrategy):
    """Generation-based strategy."""

    def __init__(self, gen_ctrl: int):
        """
        Initialize GenerationBasedStrategy.

        Parameters
        ----------
        gen_ctrl : int
            Number of surrogate-only generations executed inside each ``step``
            call before one generation of true objective evaluation.
        """
        self.gen_ctrl = gen_ctrl

    def step(self, ctx: OptimizationContext, provider: ComponentProvider) -> None:
        """
        Perform one iteration of optimization processing.

        Runs ``gen_ctrl`` surrogate generations internally, then evaluates
        all offspring with the true objective function.

        Parameters
        ----------
        ctx : OptimizationContext
            A dataclass object that holds internal information about the Optimizer.
        provider : ComponentProvider
            Objects of the class in which the component is exposed (ex. Optimizer).
        """
        # --- surrogate inner loop ---
        for _ in range(self.gen_ctrl):
            ctx.count_generation()
            offspring = provider.algorithm.ask(ctx, provider)

            provider.dispatch(
                SurrogateStartEvent(ctx=ctx, provider=provider, offspring=offspring)
            )

            _, predictions = provider.surrogate_manager.score_candidates(
                offspring.x, ctx.archive, provider, ctx
            )
            for i, pred in enumerate(predictions):
                offspring[i].f = pred.mean[0]

            provider.dispatch(
                SurrogateEndEvent(ctx=ctx, provider=provider, offspring=offspring)
            )

            provider.algorithm.tell(ctx, provider, offspring)

        # --- real evaluation generation ---
        ctx.count_generation()
        offspring = provider.algorithm.ask(ctx, provider)
        n_offspring = len(offspring)

        for i in range(n_offspring):
            offspring[i].f = ctx.problem.evaluate(offspring[i].x)
            g, cv = ctx.problem.evaluate_constraints(offspring[i].x)
            offspring[i].cv = cv
            offspring[i].g = g
            ctx.archive.add(
                {"x": offspring[i].x, "f": offspring[i].f, "g": g, "cv": cv}
            )
        ctx.count_fe(n_offspring)

        provider.dispatch(
            PostEvaluationEvent(ctx=ctx, provider=provider, offspring=offspring)
        )

        provider.algorithm.tell(ctx, provider, offspring)
