"""Shared helpers for :mod:`saealib.viz` plot functions.

This module imports neither matplotlib nor any heavy dependency at module
level. Matplotlib is only touched inside :func:`_resolve_axes`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from saealib.exceptions import ValidationError

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from saealib.api import Result

from saealib.viz._matplotlib import _require_matplotlib


def _resolve_axes(ax: Axes | None = None, **kwargs: object) -> tuple[Figure, Axes]:
    """Resolve a figure/axes pair for a plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Target axes. When ``None`` a new ``Figure`` and ``Axes`` are created.
    **kwargs
        Forwarded to ``Figure.add_subplot`` when new axes are created (e.g.
        ``projection="3d"``). When ``ax`` is supplied, a requested projection
        is checked against ``ax.name`` and other arguments are ignored.

    Returns
    -------
    tuple[Figure, Axes]
        The root figure owning ``ax`` and the axes to draw on.
    """
    if ax is None:
        _require_matplotlib()
        from matplotlib.figure import Figure

        fig = Figure()
        ax = fig.add_subplot(111, **kwargs)
        return fig, ax

    required_projection = kwargs.get("projection")
    if required_projection is not None and ax.name != required_projection:
        raise ValidationError(
            f'This plot requires an Axes with projection="{required_projection}". '
            "Pass an Axes created with "
            f'fig.add_subplot(111, projection="{required_projection}") or equivalent.'
        )

    from matplotlib.figure import SubFigure

    fig = ax.figure
    while isinstance(fig, SubFigure):
        fig = fig.figure
    return fig, ax


def _direction(result: Result, n_obj: int | None = None) -> np.ndarray:
    """Return the objective direction as a one-dimensional array."""
    try:
        direction = np.asarray(result.ctx.problem.direction, dtype=float).reshape(-1)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValidationError("A valid objective direction is required.") from exc
    if direction.size == 0 or (n_obj is not None and direction.size != n_obj):
        raise ValidationError("Objective direction does not match the objective data.")
    return direction


def _minimize_sign(direction: float | np.ndarray | None) -> float | np.ndarray:
    """Return the sign converting objectives into minimize space.

    Mirrors :func:`saealib.acquisition.base.direction_to_minimize_sign` without
    reaching into the (unexported) acquisition internals.

    Parameters
    ----------
    direction : float, np.ndarray, or None
        Per-objective optimization direction (``+1`` = maximize, ``-1`` =
        minimize). ``None`` means the quantity is already in minimize space.

    Returns
    -------
    float or np.ndarray
        ``-direction`` when *direction* is given, else the scalar ``1.0``.

    Notes
    -----
    Do not multiply standard deviations or uncertainties by this sign; it
    applies only to the mean objective value.
    """
    return -direction if direction is not None else 1.0
