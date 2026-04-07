"""
Callback module.

This module contains the implementation of callback events and manager.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from saealib.problem import non_dominated_sort
from saealib.utils.indicators import hypervolume

if TYPE_CHECKING:
    from saealib.context import OptimizationContext
    from saealib.optimizer import ComponentProvider
    from saealib.population import Population

logger = logging.getLogger(__name__)


@dataclass
class CallbackArgs:
    """Base argument object passed to every callback handler."""

    ctx: OptimizationContext
    provider: ComponentProvider | None = None


@dataclass
class SurrogateArgs(CallbackArgs):
    """Arguments for surrogate-related callback events."""

    offspring: Population | None = None


@dataclass
class PostAskArgs(CallbackArgs):
    """Arguments for post-ask callback events (crossover/mutation/ask)."""

    candidates: np.ndarray | None = None


class CallbackEvent(Enum):
    """
    Enum class for callback events.

    Attributes
    ----------
    RUN_START
        Triggered when the optimization run starts.
    RUN_END
        Triggered when the optimization run ends.
    GENERATION_START
        Triggered when a new generation starts.
    GENERATION_END
        Triggered when a generation ends.
    SURROGATE_START
        Triggered when surrogate model training starts.
    SURROGATE_END
        Triggered when surrogate model training ends.
    POST_CROSSOVER
        Triggered after crossover operation.
    POST_MUTATION
        Triggered after mutation operation.
    POST_SURROGATE_FIT
        Triggered after surrogate model fitting.
    """

    # Optimizer.run events
    RUN_START = auto()
    RUN_END = auto()
    GENERATION_START = auto()
    GENERATION_END = auto()
    SURROGATE_START = auto()
    SURROGATE_END = auto()
    # Algorithm.ask events
    POST_CROSSOVER = auto()
    POST_MUTATION = auto()
    POST_ASK = auto()
    # ModelManager.run events (commented out for future use)
    POST_SURROGATE_FIT = auto()
    # POST_SURROGATE_PREDICT = auto()


class CallbackManager:
    """
    Manages callback events and their handlers.

    Attributes
    ----------
    handlers : defaultdict[CallbackEvent, list[callable]]
        Dictionary mapping events to list of callback functions.
    """

    def __init__(self) -> None:
        """Initialize CallbackManager."""
        self.handlers = defaultdict(list)

    def register(self, event: CallbackEvent, func: callable) -> None:
        """
        Register a callback function for a event.

        Parameters
        ----------
        event : CallbackEvent
            The event to register the callback for.
        func : callable
            The callback function to register.

        Returns
        -------
        None
        """
        self.handlers[event].append(func)

    def dispatch(self, event: CallbackEvent, args: CallbackArgs) -> None:
        """
        Dispatch a callback event.

        Parameters
        ----------
        event : CallbackEvent
            The event to dispatch.
        args : CallbackArgs
            Argument object passed to each handler. Handlers must not return
            a value; the args object may be read but should not be mutated.

        Returns
        -------
        None
        """
        for handler in self.handlers[event]:
            handler(args)

    def unregister(self, event: CallbackEvent, func: callable) -> None:
        """
        Unregister a callback function from an event.

        Parameters
        ----------
        event : CallbackEvent
            The event to unregister the callback from.
        func : callable
            The callback function to remove. Raises ValueError if not found.

        Returns
        -------
        None
        """
        self.handlers[event].remove(func)

    def replace(self, event: CallbackEvent, old: callable, new: callable) -> None:
        """
        Replace a registered callback with another.

        Parameters
        ----------
        event : CallbackEvent
            The event whose handler list to modify.
        old : callable
            The handler to replace. Raises ValueError if not found.
        new : callable
            The replacement handler.

        Returns
        -------
        None
        """
        idx = self.handlers[event].index(old)
        self.handlers[event][idx] = new


def logging_generation(args: CallbackArgs) -> None:
    """
    Log generation start event.

    For single-objective problems, logs the best objective value determined
    by the comparator. For multi-objective problems, logs the size of the
    first Pareto front and the per-objective value ranges of that front.

    Parameters
    ----------
    args : CallbackArgs
        Callback argument object. Must contain a valid ``ctx``.

    Returns
    -------
    None
    """
    ctx: OptimizationContext = args.ctx

    if ctx.n_obj == 1:
        cmp = ctx.comparator
        sorted_idxs = cmp.sort_population(ctx.archive)
        best_idx = sorted_idxs[0]
        best_f = ctx.archive.get("f")[best_idx]
        logger.info(f"Generation {ctx.gen} started. fe: {ctx.fe}. Best f: {best_f}")
    else:
        f = ctx.archive.get("f")
        _, fronts = non_dominated_sort(f)
        front1_idxs = fronts[0] if fronts else []
        front1_size = len(front1_idxs)
        if front1_size > 0:
            f_front1 = f[front1_idxs]
            f_min = np.min(f_front1, axis=0)
            f_max = np.max(f_front1, axis=0)
            ranges_str = ", ".join(
                f"f[{i}]=[{f_min[i]:.4g}, {f_max[i]:.4g}]" for i in range(ctx.n_obj)
            )
        else:
            ranges_str = "n/a"
        logger.info(
            f"Generation {ctx.gen} started. fe: {ctx.fe}. "
            f"Front1 size: {front1_size}. {ranges_str}"
        )


def logging_generation_hv(reference_point: np.ndarray):
    """
    Return a callback that logs the hypervolume per generation.

    Computes the hypervolume of the first Pareto front in the archive with
    respect to the given reference point (minimization convention).

    Parameters
    ----------
    reference_point : np.ndarray
        Reference (nadir) point, shape (n_obj,). Each component should be
        strictly greater than the best achievable value per objective.

    Returns
    -------
    callable
        A callback function compatible with CallbackManager.register.

    Examples
    --------
    >>> optimizer.cbmanager.register(
    ...     CallbackEvent.GENERATION_START,
    ...     logging_generation_hv(np.array([1.1, 1.1]))
    ... )
    """
    ref = np.asarray(reference_point, dtype=float)

    def _callback(args: CallbackArgs) -> None:
        ctx: OptimizationContext = args.ctx
        f = ctx.archive.get("f")
        _, fronts = non_dominated_sort(f)
        if not fronts or not fronts[0]:
            return
        f_front1 = f[fronts[0]]
        hv = hypervolume(f_front1, ref)
        logger.info(f"Generation {ctx.gen}. fe: {ctx.fe}. HV: {hv:.6g}")

    return _callback
