"""
Built-in callback handlers for common logging use cases.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from saealib.callback.events import GenerationStartEvent
from saealib.comparators import non_dominated_sort
from saealib.utils.indicators import hypervolume

if TYPE_CHECKING:
    from saealib.context import OptimizationContext

logger = logging.getLogger(__name__)


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
