"""
Callback module.

This module contains the implementation of callback events and manager.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from saealib.problem import non_dominated_sort
from saealib.utils.indicators import hypervolume

if TYPE_CHECKING:
    from saealib.optimizer import OptimizationContext
    # from saealib.optimizer import ComponentProvider

logger = logging.getLogger(__name__)


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

    def dispatch(self, event: CallbackEvent, data: any, **kwargs) -> any:
        """
        Dispatch a callback event.

        Parameters
        ----------
        event : CallbackEvent
            The event to dispatch.
        data : any
            The data to pass to the callback functions.
            Can be modified by each callback.

        **kwargs : any
            Additional keyword arguments to pass to the callback functions.
            Read-only.

        Returns
        -------
        any
            The result of the last callback function.
        """
        cur_data = data
        for handler in self.handlers[event]:
            cur_data = handler(data=cur_data, **kwargs)
        return cur_data


def logging_generation(data, **kwargs):
    """
    Log generation start event.

    For single-objective problems, logs the best objective value determined
    by the comparator. For multi-objective problems, logs the size of the
    first Pareto front and the per-objective value ranges of that front.

    Parameters
    ----------
    data : any
        The data passed to the callback. Returned unchanged.
    **kwargs : any
        Additional keyword arguments. Should contain 'ctx'.

    Returns
    -------
    any
        data, unchanged.
    """
    ctx: OptimizationContext = kwargs.get("ctx")

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

    return data

