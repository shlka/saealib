"""
Runner module.

Run the optimization process as Generator.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

from saealib.callback import (
    GenerationEndEvent,
    GenerationStartEvent,
    RunEndEvent,
    RunStartEvent,
)
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
        opt = self.optimizer
        ctx = opt.initializer.initialize(opt, opt.problem)

        opt.dispatch(RunStartEvent(ctx=ctx, provider=opt))
        yield ctx

        while not opt.termination.is_terminated(ctx):
            opt.dispatch(GenerationStartEvent(ctx=ctx, provider=opt))
            opt.strategy.step(ctx, opt)
            opt.dispatch(GenerationEndEvent(ctx=ctx, provider=opt))
            yield ctx

        opt.dispatch(RunEndEvent(ctx=ctx, provider=opt))
