"""Plots for optimization history series."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from saealib.result import Result
from saealib.viz._common import _resolve_axes
from saealib.viz._matplotlib import _require_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from saealib.context import OptimizationState


def plot_history(
    source: Result | OptimizationState,
    value: str | Callable[[Mapping[str, Any]], float],
    *,
    x: str = "fe",
    channel: str | None = None,
    ax: Axes | None = None,
    **value_kwargs: Any,
) -> Figure:
    """Plot a scalar series from an optimization history."""
    _require_matplotlib()
    result = source if isinstance(source, Result) else Result.from_state(source)
    series = result.history_series(value, x=x, channel=channel, **value_kwargs)
    fig, axes = _resolve_axes(ax)
    axes.plot(series.x, series.y)
    axes.set_xlabel(series.x_name)
    axes.set_ylabel(series.y_name)
    return fig
