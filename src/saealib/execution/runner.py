"""Runner: drives the optimization loop and yields context each generation."""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

from saealib.callback import (
    GenerationEndEvent,
    GenerationStartEvent,
    RunEndEvent,
    RunStartEvent,
)
from saealib.context import OptimizationState

if TYPE_CHECKING:
    from saealib.optimizer import Optimizer


class Runner:
    """
    Run the optimization process as Generator.

    Attributes
    ----------
    optimizer : Optimizer
        The optimizer instance.
    """

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def run(self) -> OptimizationState:
        """Run to completion and return the final context."""
        for ctx in self.iterate():
            pass
        return ctx

    def run_from(self, ctx: OptimizationState) -> OptimizationState:
        """Resume from an existing context and run to completion."""
        for ctx in self.iterate_from(ctx):
            pass
        return ctx

    def iterate(self) -> Generator[OptimizationState, None, None]:
        """
        Iterate the optimization loop, yielding the context after each generation.

        Returns
        -------
        Generator[OptimizationState, None, None]
        """
        opt = self.optimizer
        assert opt.initializer is not None
        ctx = opt.initializer.initialize(opt, opt.problem)
        yield from self.iterate_from(ctx)

    def iterate_from(
        self, ctx: OptimizationState
    ) -> Generator[OptimizationState, None, None]:
        """
        Resume the optimization loop from an existing context.

        Skips initialization; useful after loading a checkpoint.

        Parameters
        ----------
        ctx : OptimizationState
            Previously saved (or freshly constructed) context to resume from.

        Returns
        -------
        Generator[OptimizationState, None, None]
        """
        opt = self.optimizer
        ctx.comparator.eps_cv = ctx.problem.handler.feasibility_threshold

        opt.dispatch(RunStartEvent(ctx=ctx))
        yield ctx

        while not opt.termination.is_terminated(ctx):
            opt.dispatch(GenerationStartEvent(ctx=ctx))
            result = opt.strategy.step(ctx, opt)
            if result is not None:
                ctx = result
            handler = ctx.problem.handler
            handler.on_generation_end(ctx.gen, ctx.population)
            ctx.comparator.eps_cv = handler.feasibility_threshold
            sm = getattr(opt, "surrogate_manager", None)
            if sm is not None:
                sm.on_generation_end(ctx.gen, ctx.archive, ctx)
            opt.dispatch(GenerationEndEvent(ctx=ctx))
            yield ctx

        opt.dispatch(RunEndEvent(ctx=ctx))
