"""
Runner module.

Run the optimization process as Generator.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

from saealib.callback import CallbackEvent
from saealib.context import OptimizationContext

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

    def run(self) -> OptimizationContext:
        """
        Run the optimization process.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer instance.

        Returns
        -------
        OptimizationContext
            The optimization context.
        """
        for ctx in self.iterate():
            pass
        return ctx

    def iterate(self) -> Generator[OptimizationContext, None, None]:
        """
        Iterate the optimization process.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer instance.

        Returns
        -------
        Generator[OptimizationContext, None, None]
            The optimization context.
        """
        ctx = self.optimizer.initializer.initialize(
            self.optimizer,
            self.optimizer.problem,
        )
        self.optimizer.dispatch(CallbackEvent.RUN_START, ctx=ctx)
        yield ctx

        while not self.optimizer.termination.is_terminated(fe=ctx.fe):
            self.optimizer.dispatch(CallbackEvent.GENERATION_START, ctx=ctx)
            self.optimizer.strategy.step(ctx, self.optimizer)
            yield ctx
            self.optimizer.dispatch(CallbackEvent.GENERATION_END, ctx=ctx)

        self.optimizer.dispatch(CallbackEvent.RUN_END, ctx=ctx)
