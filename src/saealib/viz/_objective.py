"""Objective-space plots for optimization results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from saealib.exceptions import ValidationError
from saealib.viz._common import _direction, _minimize_sign, _resolve_axes
from saealib.viz._history import _front_history, _history_column
from saealib.viz._matplotlib import _require_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from saealib.api import Result


def _objective_matrix(result: Result, function: str) -> np.ndarray:
    try:
        values = np.array(result.f, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"{function} requires a two-dimensional multi-objective result.f. "
            "Use plot_convergence for a single objective."
        ) from exc
    if values.ndim == 1:
        raise ValidationError(
            f"{function} requires a multi-objective result.f. "
            "Use plot_convergence for a single objective."
        )
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValidationError(
            f"{function} requires at least two objectives. "
            "Use plot_convergence for a single objective."
        )
    return values


def _select_objectives(
    n_obj: int,
    objectives: Sequence[int] | None,
    function: str,
) -> tuple[int, ...]:
    if objectives is None:
        if n_obj == 2 or n_obj == 3:
            return tuple(range(n_obj))
        raise ValidationError(
            f"{function} requires objectives=(i, j) or objectives=(i, j, k) "
            "when there are four or more objectives."
        )
    try:
        selected = tuple(objectives)
    except TypeError as exc:
        raise ValidationError(
            f"{function} objectives must contain two or three indices."
        ) from exc
    if len(selected) not in (2, 3):
        raise ValidationError(
            f"{function} objectives must contain two or three indices."
        )
    for index in selected:
        if (
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            or index < 0
            or index >= n_obj
        ):
            raise ValidationError(
                f"{function} objectives contain an index outside 0..{n_obj - 1}."
            )
    return tuple(int(index) for index in selected)


def _normalize_objectives(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.empty(values.shape, dtype=float)
    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    span = maximum - minimum
    normalized = np.full(values.shape, 0.5, dtype=float)
    np.divide(
        values - minimum,
        span,
        out=normalized,
        where=span != 0,
    )
    return normalized


def plot_pareto(
    result: Result,
    *,
    objectives: Sequence[int] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot the result's Pareto objective values.

    Two selected objectives are shown as a two-dimensional scatter plot and
    three selected objectives as a three-dimensional scatter plot. Duplicate
    archive points are retained.

    Parameters
    ----------
    result : saealib.api.Result
        Optimization result whose ``f`` values are plotted.
    objectives : sequence of int or None, optional
        Objective indices to plot. For four or more objectives this must have
        length two or three.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    _require_matplotlib()
    values = _objective_matrix(result, "plot_pareto")
    selected = _select_objectives(values.shape[1], objectives, "plot_pareto")
    if len(selected) == 2:
        fig, axes = _resolve_axes(ax)
        axes.scatter(values[:, selected[0]], values[:, selected[1]])
    else:
        fig, axes = _resolve_axes(ax, projection="3d")
        axes.scatter(
            values[:, selected[0]],
            values[:, selected[1]],
            values[:, selected[2]],
        )
    for axis, index in zip("xyz", selected):
        getattr(axes, f"set_{axis}label")(f"f{index}")
    return fig


def plot_pareto_evolution(
    result: Result,
    *,
    objectives: Sequence[int] | None = None,
    cmap: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot recorded Pareto fronts colored by generation.

    Empty recorded fronts are skipped, while all points in non-empty fronts,
    including duplicates, are retained.

    Parameters
    ----------
    result : saealib.api.Result
        Optimization result with ``front`` history.
    objectives : sequence of int or None, optional
        Objective indices to plot. For four or more objectives this must have
        length two or three.
    cmap : str or None, optional
        Matplotlib colormap used for generation values.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot and generation colorbar.
    """
    matplotlib = _require_matplotlib()
    values = _objective_matrix(result, "plot_pareto_evolution")
    selected = _select_objectives(values.shape[1], objectives, "plot_pareto_evolution")
    _, columns, blocks = _front_history(result, "plot_pareto_evolution")
    generations = _history_column(
        columns, "gen", len(blocks), "plot_pareto_evolution"
    ).astype(float, copy=False)
    fronts: list[np.ndarray] = []
    front_generations: list[float] = []
    for front, generation in zip(blocks, generations):
        try:
            front_array = np.array(front, dtype=float, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "plot_pareto_evolution requires two-dimensional objective fronts."
            ) from exc
        if front_array.ndim != 2 or front_array.shape[1] != values.shape[1]:
            raise ValidationError(
                "plot_pareto_evolution front dimensions do not match result.f."
            )
        if len(front_array):
            fronts.append(front_array)
            front_generations.append(float(generation))

    if len(selected) == 2:
        fig, axes = _resolve_axes(ax)
    else:
        fig, axes = _resolve_axes(ax, projection="3d")

    if len(generations):
        vmin = float(np.min(generations))
        vmax = float(np.max(generations))
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
    else:
        vmin, vmax = 0.0, 1.0
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array(generations)
    for front, generation in zip(fronts, front_generations):
        color = scalar_mappable.to_rgba(generation)
        if len(selected) == 2:
            axes.scatter(front[:, selected[0]], front[:, selected[1]], color=color)
        else:
            axes.scatter(
                front[:, selected[0]],
                front[:, selected[1]],
                front[:, selected[2]],
                color=color,
            )
    colorbar = fig.colorbar(scalar_mappable, ax=axes)
    colorbar.set_label("Generation")
    for axis, index in zip("xyz", selected):
        getattr(axes, f"set_{axis}label")(f"f{index}")
    return fig


def plot_pcp(result: Result, *, ax: Axes | None = None) -> Figure:
    """Plot objective values as normalized parallel coordinates.

    Each objective column is min--max normalized independently. A constant
    objective column is represented by ``0.5`` for every solution.

    Parameters
    ----------
    result : saealib.api.Result
        Optimization result whose ``f`` values are plotted.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    _require_matplotlib()
    values = _objective_matrix(result, "plot_pcp")
    normalized = _normalize_objectives(values)
    fig, axes = _resolve_axes(ax)
    positions = np.arange(values.shape[1])
    for row in normalized:
        axes.plot(positions, row)
    axes.set_xticks(positions)
    axes.set_xticklabels([f"f{index}" for index in positions])
    axes.set_ylim(0.0, 1.0)
    return fig


def plot_radar(result: Result, *, ax: Axes | None = None) -> Figure:
    """Plot normalized objective values as closed radar polygons.

    Each objective column is min--max normalized independently. A constant
    objective column is represented by ``0.5`` for every solution.

    Parameters
    ----------
    result : saealib.api.Result
        Optimization result whose ``f`` values are plotted.
    ax : matplotlib.axes.Axes or None, optional
        Polar axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the radar plot.
    """
    _require_matplotlib()
    values = _objective_matrix(result, "plot_radar")
    if values.shape[1] < 3:
        raise ValidationError(
            "plot_radar requires at least three objectives. Use plot_pareto for "
            "two objectives or plot_pcp for parallel coordinates."
        )
    normalized = _normalize_objectives(values)
    fig, axes = _resolve_axes(ax, projection="polar")
    angles = np.linspace(0.0, 2.0 * np.pi, values.shape[1], endpoint=False)
    closed_angles = np.concatenate((angles, angles[:1]))
    for row in normalized:
        axes.plot(closed_angles, np.concatenate((row, row[:1])))
    axes.set_xticks(angles)
    axes.set_xticklabels([f"f{index}" for index in range(values.shape[1])])
    axes.set_ylim(0.0, 1.0)
    return fig


def plot_objective_heatmap(
    result: Result,
    *,
    cmap: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot the normalized objective matrix as a heatmap.

    Rows are sorted by the ascending value of objective zero in minimization
    space, after applying the problem's objective direction. Constant columns
    are represented by ``0.5``.

    Parameters
    ----------
    result : saealib.api.Result
        Optimization result whose ``f`` values are plotted.
    cmap : str or None, optional
        Matplotlib colormap used for normalized objective values.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the heatmap and colorbar.
    """
    _require_matplotlib()
    values = _objective_matrix(result, "plot_objective_heatmap")
    direction = _direction(result, values.shape[1])
    normalized = _normalize_objectives(values)
    sign = np.asarray(_minimize_sign(direction), dtype=float)
    order = np.argsort((values * sign)[:, 0], kind="stable")
    sorted_values = normalized[order]
    fig, axes = _resolve_axes(ax)
    if cmap is None:
        image = axes.imshow(sorted_values, aspect="auto")
    else:
        image = axes.imshow(sorted_values, aspect="auto", cmap=cmap)
    colorbar = fig.colorbar(image, ax=axes)
    colorbar.set_label("Normalized objective value")
    axes.set_xticks(np.arange(values.shape[1]))
    axes.set_xticklabels([str(index) for index in range(values.shape[1])])
    axes.set_xlabel("Objective index")
    axes.set_ylabel("Solution index")
    return fig
