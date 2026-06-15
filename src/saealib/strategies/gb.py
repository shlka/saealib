"""
Generation-based strategy.

Each call to ``step`` runs ``gen_ctrl`` surrogate-only generations followed
by one generation of true objective evaluations.
"""

from saealib.callback import PostEvaluationEvent, SurrogateEndEvent, SurrogateStartEvent
from saealib.context import OptimizationContext
from saealib.optimizer import ComponentProvider
from saealib.strategies.base import OptimizationStrategy, assign_tell_f


class GenerationBasedStrategy(OptimizationStrategy):
    """Generation-based strategy."""

    requires_surrogate: bool = True

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
        Run ``gen_ctrl`` surrogate-only generations, then one true-evaluation step.

        Parameters
        ----------
        ctx : OptimizationContext
            Current optimization context.
        provider : ComponentProvider
            Component provider.
        """
        # --- surrogate inner loop ---
        for _ in range(self.gen_ctrl):
            ctx.count_generation()
            offspring = provider.algorithm.ask(ctx, provider)

            provider.dispatch(
                SurrogateStartEvent(ctx=ctx, offspring=offspring)
            )

            _, predictions = provider.surrogate_manager.score_candidates(
                offspring.x, ctx.archive, ctx
            )
            for i, pred in enumerate(predictions):
                assign_tell_f(offspring[i], pred, ctx)

            provider.dispatch(
                SurrogateEndEvent(ctx=ctx, offspring=offspring)
            )

            provider.algorithm.tell(ctx, provider, offspring)

        # --- real evaluation generation ---
        ctx.count_generation()
        offspring = provider.algorithm.ask(ctx, provider)
        n_offspring = len(offspring)

        result = provider.evaluator.evaluate_batch(offspring.x, ctx.problem)
        for i in range(n_offspring):
            f_i, g_i, cv_i = result.f[i], result.g[i], float(result.cv[i])
            offspring[i].f = f_i
            offspring[i].g = g_i
            offspring[i].cv = cv_i
            ctx.archive.add({"x": offspring[i].x, "f": f_i, "g": g_i, "cv": cv_i})
            ctx.pareto_archive.add(
                {"x": offspring[i].x, "f": f_i, "g": g_i, "cv": cv_i}
            )
        ctx.count_fe(n_offspring)

        provider.dispatch(
            PostEvaluationEvent(ctx=ctx, offspring=offspring)
        )

        provider.algorithm.tell(ctx, provider, offspring)
