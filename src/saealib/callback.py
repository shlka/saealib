"""
Callback module.

This module contains event classes and the callback manager for
the optimization lifecycle.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, TypeVar

import numpy as np

from saealib.problem import non_dominated_sort
from saealib.utils.indicators import hypervolume

if TYPE_CHECKING:
    from saealib.context import OptimizationContext
    from saealib.optimizer import ComponentProvider
    from saealib.population import Population

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event hierarchy
# ---------------------------------------------------------------------------


@dataclass
class Event:
    """
    Base class for all callback events.

    Attributes
    ----------
    ctx : OptimizationContext
        The current optimization context.
    provider : ComponentProvider
        The component provider (e.g. Optimizer) that fired this event.
    """

    ctx: OptimizationContext
    provider: ComponentProvider


# --- Optimizer.run events ---


@dataclass
class RunStartEvent(Event):
    """Fired once when the optimization run starts."""


@dataclass
class RunEndEvent(Event):
    """Fired once when the optimization run ends."""


@dataclass
class GenerationStartEvent(Event):
    """Fired at the beginning of each generation."""


@dataclass
class GenerationEndEvent(Event):
    """Fired at the end of each generation, before yielding the context."""


# --- Surrogate events ---


@dataclass
class SurrogateStartEvent(Event):
    """Fired before surrogate-based candidate scoring."""

    offspring: Population | None = None


@dataclass
class SurrogateEndEvent(Event):
    """Fired after surrogate-based candidate scoring."""

    offspring: Population | None = None


# --- Algorithm.ask events ---


@dataclass
class PostCrossoverEvent(Event):
    """
    Fired after crossover and repair.

    Handlers may replace ``candidates`` with a modified array.
    """

    candidates: np.ndarray | None = None


@dataclass
class PostMutationEvent(Event):
    """
    Fired after mutation and repair.

    Handlers may replace ``candidates`` with a modified array.
    """

    candidates: np.ndarray | None = None


@dataclass
class PostAskEvent(Event):
    """
    Fired after the full ask step (post crossover and mutation).

    Handlers may replace ``candidates`` with a modified array.
    """

    candidates: np.ndarray | None = None


# --- Model events ---


@dataclass
class PostSurrogateFitEvent(Event):
    """Fired after the surrogate model is fitted."""


# ---------------------------------------------------------------------------
# CallbackManager
# ---------------------------------------------------------------------------

E = TypeVar("E", bound=Event)


class CallbackManager:
    """
    Manages event handlers.

    Handlers are registered per concrete event type and called in
    registration order when an event of that type is dispatched.

    Attributes
    ----------
    handlers : defaultdict[type[Event], list[Callable]]
        Mapping from event class to list of registered handler functions.
    """

    def __init__(self) -> None:
        """Initialize CallbackManager."""
        self.handlers: defaultdict[type[Event], list] = defaultdict(list)

    def register(self, event_type: type[E], func: Callable[[E], None]) -> None:
        """
        Register a handler for an event type.

        Parameters
        ----------
        event_type : type[E]
            The concrete event class to listen for.
        func : Callable[[E], None]
            Handler function. Receives the event object and returns nothing.

        Returns
        -------
        None
        """
        self.handlers[event_type].append(func)

    def dispatch(self, event: Event) -> None:
        """
        Invoke all handlers registered for the type of *event*.

        Parameters
        ----------
        event : Event
            The event object to dispatch. Handlers receive this object
            directly and may modify its mutable fields.

        Returns
        -------
        None
        """
        for handler in self.handlers[type(event)]:
            handler(event)

    def unregister(self, event_type: type[E], func: Callable[[E], None]) -> None:
        """
        Remove a previously registered handler.

        Parameters
        ----------
        event_type : type[E]
            The event class the handler was registered for.
        func : Callable[[E], None]
            The handler to remove. Raises ``ValueError`` if not found.

        Returns
        -------
        None
        """
        self.handlers[event_type].remove(func)

    def replace(
        self,
        event_type: type[E],
        old: Callable[[E], None],
        new: Callable[[E], None],
    ) -> None:
        """
        Replace a registered handler with another.

        Parameters
        ----------
        event_type : type[E]
            The event class whose handler list to modify.
        old : Callable[[E], None]
            The handler to replace. Raises ``ValueError`` if not found.
        new : Callable[[E], None]
            The replacement handler.

        Returns
        -------
        None
        """
        idx = self.handlers[event_type].index(old)
        self.handlers[event_type][idx] = new


# ---------------------------------------------------------------------------
# Built-in handlers
# ---------------------------------------------------------------------------


def logging_generation(event: GenerationStartEvent) -> None:
    """
    Log the state at the start of each generation.

    For single-objective problems, logs the best objective value determined
    by the comparator. For multi-objective problems, logs the size of the
    first Pareto front and the per-objective value ranges of that front.

    Parameters
    ----------
    event : GenerationStartEvent
        The generation-start event.

    Returns
    -------
    None
    """
    ctx: OptimizationContext = event.ctx

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
    Return a handler that logs the hypervolume per generation.

    Computes the hypervolume of the first Pareto front in the archive with
    respect to the given reference point (minimization convention).

    Parameters
    ----------
    reference_point : np.ndarray
        Reference (nadir) point, shape (n_obj,). Each component should be
        strictly greater than the best achievable value per objective.

    Returns
    -------
    Callable[[GenerationStartEvent], None]
        A handler compatible with ``CallbackManager.register``.

    Examples
    --------
    >>> optimizer.cbmanager.register(
    ...     GenerationStartEvent,
    ...     logging_generation_hv(np.array([1.1, 1.1]))
    ... )
    """
    ref = np.asarray(reference_point, dtype=float)

    def _callback(event: GenerationStartEvent) -> None:
        ctx: OptimizationContext = event.ctx
        f = ctx.archive.get("f")
        _, fronts = non_dominated_sort(f)
        if not fronts or not fronts[0]:
            return
        f_front1 = f[fronts[0]]
        hv = hypervolume(f_front1, ref)
        logger.info(f"Generation {ctx.gen}. fe: {ctx.fe}. HV: {hv:.6g}")

    return _callback
