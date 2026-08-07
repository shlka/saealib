"""Runner: drives the optimization loop and yields context each generation."""

from __future__ import annotations

import time
from collections.abc import Generator
from typing import TYPE_CHECKING

from saealib.callback import (
    GenerationEndEvent,
    GenerationStartEvent,
    RunEndEvent,
    RunStartEvent,
)
from saealib.comparators import NSGA3Comparator
from saealib.context import OptimizationState
from saealib.exceptions import EvaluationFatalError

if TYPE_CHECKING:
    from saealib.execution.scheduler import AsyncEvaluationScheduler
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

    def _sync_eps_cv(self, ctx: OptimizationState) -> None:
        """Sync ``eps_cv`` on ``comparator`` and ``pareto_archive`` from the handler."""
        threshold = ctx.problem.handler.feasibility_threshold
        ctx.comparator.eps_cv = threshold
        ctx.pareto_archive.eps_cv = threshold

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
        self._sync_eps_cv(ctx)
        if isinstance(ctx.comparator, NSGA3Comparator) and ctx.comparator._rng is None:
            ctx.comparator.rng = ctx.rng.spawn(1)[0]

        opt.dispatch(RunStartEvent(ctx=ctx))
        yield ctx

        generation_open = bool(ctx.pending_evaluations)
        while True:
            scheduler = getattr(opt, "async_evaluation_scheduler", None)
            if ctx.async_fatal:
                raise EvaluationFatalError(
                    str(ctx.async_fatal.get("reason", "async fatal")), ctx
                )
            if ctx.pending_evaluations:
                (
                    ctx,
                    generation_open,
                    generation_finished,
                    continue_loop,
                ) = self._process_pending(ctx, generation_open, scheduler)
                if generation_finished:
                    yield ctx
                if continue_loop:
                    continue
            if opt.termination.is_terminated(ctx):
                break
            opt.dispatch(GenerationStartEvent(ctx=ctx))
            generation_open = True
            result = opt.strategy.step(ctx, opt)
            if result is not None:
                ctx = result
            if not ctx.pending_evaluations:
                self._finish_generation(ctx)
                generation_open = False
                yield ctx

        opt.dispatch(RunEndEvent(ctx=ctx))

    def _process_pending(
        self,
        ctx: OptimizationState,
        generation_open: bool,
        scheduler: AsyncEvaluationScheduler | None,
    ) -> tuple[OptimizationState, bool, bool, bool]:
        opt = self.optimizer
        if scheduler is None:
            result = opt.strategy.step(ctx, opt)
            if result is not None:
                ctx = result
            if ctx.pending_evaluations:
                return ctx, generation_open, False, True
        else:
            if set(ctx.pending_evaluations) != set(ctx.evaluation_handles):
                ctx = scheduler.reattach(ctx)
            before = ctx
            ctx = scheduler.poll(ctx, wait=False)
            if ctx.pending_evaluations:
                if opt.termination.is_terminated(ctx):
                    if ctx is before:
                        time.sleep(0.001)
                    return ctx, generation_open, False, True
                result = opt.strategy.step(ctx, opt)
                if result is not None:
                    ctx = result
                if ctx.pending_evaluations:
                    if ctx is before:
                        time.sleep(0.001)
                    return ctx, generation_open, False, True

        if generation_open:
            self._finish_generation(ctx)
            return ctx, False, True, scheduler is None
        return ctx, generation_open, False, scheduler is None

    def _finish_generation(self, ctx: OptimizationState) -> None:
        handler = ctx.problem.handler
        handler.on_generation_end(ctx.gen, ctx.population)
        self._sync_eps_cv(ctx)
        sm = getattr(self.optimizer, "surrogate_manager", None)
        if sm is not None:
            sm.on_generation_end(ctx.gen, ctx.archive, ctx)
        self.optimizer.dispatch(GenerationEndEvent(ctx=ctx))
