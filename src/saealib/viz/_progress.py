"""Progress plots for optimization results."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from saealib.exceptions import ValidationError
from saealib.utils import gd, gd_plus, hypervolume, igd, igd_plus, spacing, spread
from saealib.viz._common import _direction, _minimize_sign, _resolve_axes
from saealib.viz._history import (
    _front_history,
    _history_column,
    _require_channel,
)
from saealib.viz._matplotlib import _require_matplotlib
from saealib.viz._trials import _aggregate_convergence, _prepare_convergence

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from saealib.api import Result


def _front_arrays(
    fronts: tuple[np.ndarray, ...], n_obj: int, function: str
) -> tuple[np.ndarray, ...]:
    result: list[np.ndarray] = []
    for front in fronts:
        try:
            values = np.array(front, dtype=float, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"{function} requires two-dimensional objective front blocks."
            ) from exc
        if values.ndim != 2 or values.shape[1] != n_obj:
            raise ValidationError(
                f"{function} requires front blocks with {n_obj} objectives."
            )
        result.append(values)
    return tuple(result)


def plot_convergence(
    result: Result | Sequence[Result],
    *,
    groups: Sequence[Hashable] | None = None,
    labels: str | Mapping[Hashable, str] | None = None,
    fe_range: str = "common",
    ax: Axes | None = None,
) -> Figure:
    """Plot convergence for one result or aggregate several results.

    A single result is plotted using the objective value recorded for each
    generation's Pareto archive. Multiple results are converted to per-run
    best-so-far step functions and summarized with a median line and an
    interquartile band on a common function-evaluation axis.

    Parameters
    ----------
    result : saealib.api.Result or sequence of saealib.api.Result
        Optimization result or results with ``summary`` history.
    groups : sequence of hashable or None, optional
        Group key for each result. Results sharing a key are aggregated into
        separate median and interquartile curves.
    labels : str, mapping, or None, optional
        Display label for an ungrouped curve, or a mapping from group keys to
        display labels.
    fe_range : {"common", "full"}, optional
        Function-evaluation range used for multi-result aggregation. ``"common"``
        ends at the shortest run; ``"full"`` extends to the longest run and
        holds shorter runs at their final values.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.

    Notes
    -----
    ``summary.fe`` is recorded at generation boundaries. Synchronous and
    asynchronous runtimes advance it at different lifecycle points, so
    aggregating runs across execution modes can introduce a bounded offset on
    the function-evaluation axis. The ``evaluation`` channel can provide a
    different source when that distinction is required, but it is not used by
    this function.

    ``f_min_0`` and ``f_max_0`` describe the best value in the Pareto archive
    at each generation, not a best-so-far value. For multi-result aggregation,
    each run is therefore converted to best-so-far before the statistics are
    computed. With ``fe_range="full"``, a run that finishes earlier contributes
    its final value over the remaining range. This frozen tail is a limitation
    of the full-range view, not an observation after the run ended.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    _require_matplotlib()
    group_values, series, direction = _prepare_convergence(
        result, groups, labels, fe_range
    )

    fig, axes = _resolve_axes(ax)
    if len(series) == 1:
        fe, values = series[0]
        line_label: str | None = None
        if groups is None:
            if isinstance(labels, str):
                line_label = labels
        elif isinstance(labels, Mapping):
            line_label = labels[group_values[0]]
        line_kwargs = {"label": line_label} if line_label is not None else {}
        axes.plot(fe, values, **line_kwargs)
        if line_label is not None:
            axes.legend()
    else:
        grid, aggregates = _aggregate_convergence(
            series, group_values, direction, fe_range
        )
        for group, median, q1, q3 in aggregates:
            line_label = None
            if groups is None:
                if isinstance(labels, str):
                    line_label = labels
            elif isinstance(labels, Mapping):
                line_label = labels[group]
            line_kwargs = {"label": line_label} if line_label is not None else {}
            axes.plot(grid, median, **line_kwargs)
            axes.fill_between(grid, q1, q3, alpha=0.25)
        if labels is not None:
            axes.legend()
    axes.set_xlabel("Function evaluations")
    axes.set_ylabel("Objective value")
    return fig


def plot_hypervolume(
    result: Result, reference_point: np.ndarray, *, ax: Axes | None = None
) -> Figure:
    """Plot hypervolume over function evaluations.

    ``reference_point`` is supplied in the raw objective space. Both each
    recorded front and the reference point are converted to minimization space
    before the minimization-convention hypervolume is computed.

    Parameters
    ----------
    result : saealib.api.Result
        Optimization result with ``front`` history.
    reference_point : array-like
        Reference point in the raw objective space.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    _require_matplotlib()
    _, columns, blocks = _front_history(result, "plot_hypervolume")
    direction = _direction(result)
    n_obj = direction.size
    try:
        reference = np.asarray(reference_point, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "reference_point must have one value per objective."
        ) from exc
    if reference.shape != (n_obj,):
        raise ValidationError("reference_point must have one value per objective.")
    fronts = _front_arrays(blocks, n_obj, "plot_hypervolume")
    fe = _history_column(columns, "fe", len(fronts), "plot_hypervolume")
    sign = np.asarray(_minimize_sign(direction), dtype=float)
    reference_min = reference * sign
    values = np.array(
        [
            np.nan if len(front) == 0 else hypervolume(front * sign, reference_min)
            for front in fronts
        ],
        dtype=float,
    )

    fig, axes = _resolve_axes(ax)
    axes.plot(fe, values)
    axes.set_xlabel("Function evaluations")
    axes.set_ylabel("Hypervolume")
    return fig


def plot_indicator(
    result: Result,
    indicator: str,
    reference_front: np.ndarray | None = None,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plot a quality indicator over function evaluations.

    Fronts and ``reference_front`` are accepted in the raw objective space and
    are both converted to minimization space before the indicator is called.

    Parameters
    ----------
    result : saealib.api.Result
        Optimization result with ``front`` history.
    indicator : str
        One of ``"gd"``, ``"gd_plus"``, ``"igd"``, ``"igd_plus"``,
        ``"spacing"``, or ``"spread"``.
    reference_front : array-like or None, optional
        Reference front in raw objective space. Required except for
        ``"spacing"``.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    _require_matplotlib()
    valid_indicators = ("gd", "gd_plus", "igd", "igd_plus", "spacing", "spread")
    if not isinstance(indicator, str) or indicator not in valid_indicators:
        valid = ", ".join(valid_indicators)
        raise ValidationError(
            f"Unknown indicator {indicator!r}. Choose one of: {valid}."
        )

    _, columns, blocks = _front_history(result, "plot_indicator")
    direction = _direction(result)
    n_obj = direction.size
    reference_min: np.ndarray | None = None
    if indicator != "spacing":
        if reference_front is None:
            raise ValidationError(
                f'plot_indicator indicator "{indicator}" requires reference_front.'
            )
        try:
            reference = np.asarray(reference_front, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f'plot_indicator indicator "{indicator}" requires a reference '
                "front with one column per objective."
            ) from exc
        if reference.ndim == 1 and reference.size == n_obj:
            reference = reference.reshape(1, n_obj)
        if reference.ndim != 2 or reference.shape[1] != n_obj or len(reference) == 0:
            raise ValidationError(
                f'plot_indicator indicator "{indicator}" requires a non-empty '
                "reference front with one column per objective."
            )
        sign = np.asarray(_minimize_sign(direction), dtype=float)
        reference_min = reference * sign

    fronts = _front_arrays(blocks, n_obj, "plot_indicator")
    fe = _history_column(columns, "fe", len(fronts), "plot_indicator")
    sign = np.asarray(_minimize_sign(direction), dtype=float)
    values: list[float] = []
    for front in fronts:
        if len(front) == 0:
            values.append(np.nan)
        elif indicator == "spacing":
            values.append(float(spacing(front * sign)))
        else:
            assert reference_min is not None
            if indicator == "gd":
                values.append(float(gd(front * sign, reference_min)))
            elif indicator == "gd_plus":
                values.append(float(gd_plus(front * sign, reference_min)))
            elif indicator == "igd":
                values.append(float(igd(front * sign, reference_min)))
            elif indicator == "igd_plus":
                values.append(float(igd_plus(front * sign, reference_min)))
            else:
                values.append(float(spread(front * sign, reference_min)))

    fig, axes = _resolve_axes(ax)
    axes.plot(fe, np.asarray(values, dtype=float))
    axes.set_xlabel("Function evaluations")
    axes.set_ylabel(indicator)
    return fig


def plot_constraint_violation(result: Result, *, ax: Axes | None = None) -> Figure:
    """Plot population constraint violation and feasibility by generation.

    ``min_cv`` and ``feasible_ratio`` are calculated from the population at
    each generation, not from the best constraint violation seen so far. Their
    values are therefore not guaranteed to be monotonic. Empty populations are
    recorded as ``NaN``.

    Parameters
    ----------
    result : saealib.api.Result
        Optimization result with ``summary`` history.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot and its secondary axis.
    """
    _require_matplotlib()
    summary = _require_channel(result, "summary", "plot_constraint_violation")
    count = len(summary.get("fe", ()))
    fe = _history_column(summary, "fe", count, "plot_constraint_violation")
    min_cv = _history_column(summary, "min_cv", count, "plot_constraint_violation")
    feasible_ratio = _history_column(
        summary, "feasible_ratio", count, "plot_constraint_violation"
    )

    fig, axes = _resolve_axes(ax)
    secondary = axes.twinx()
    left_line = axes.plot(fe, min_cv, label="min_cv")[0]
    right_line = secondary.plot(fe, feasible_ratio, label="feasible_ratio")[0]
    axes.set_xlabel("Function evaluations")
    axes.set_ylabel("Minimum constraint violation")
    secondary.set_ylabel("Feasible ratio")
    axes.legend(handles=[left_line, right_line])
    return fig


def plot_running_metric(
    result: Result,
    *,
    window: int | None = None,
    significance: float = 0.005,
    ax: Axes | None = None,
) -> Figure:
    """Plot the Blank--Deb (2020) running performance metric.

    The metric uses the same definition as pymoo's running metric: every
    recorded front is normalized by the final front's ideal and nadir, and
    ``delta_f`` is the IGD to that final normalized front. Significant ideal or
    nadir movement is marked with a vertical line and marker.

    Parameters
    ----------
    result : saealib.api.Result
        Optimization result with ``front`` history.
    window : int or None, optional
        Number of recorded fronts retained from the end before empty fronts
        are excluded.
    significance : float, optional
        Threshold for significant ideal or nadir movement.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the running metric.
    """
    _require_matplotlib()
    if window is not None and (
        isinstance(window, (bool, np.bool_))
        or not isinstance(window, (int, np.integer))
        or window <= 0
    ):
        raise ValidationError("window must be a positive integer or None.")

    _, columns, blocks = _front_history(result, "plot_running_metric")
    direction = _direction(result)
    n_obj = direction.size
    fronts = _front_arrays(blocks, n_obj, "plot_running_metric")
    generations = _history_column(
        columns, "gen", len(fronts), "plot_running_metric"
    ).astype(float, copy=False)
    records = list(zip(fronts, generations))
    if window is not None:
        records = records[-int(window) :]
    non_empty = [(front, generation) for front, generation in records if len(front)]

    fig, axes = _resolve_axes(ax)
    if non_empty:
        sign = np.asarray(_minimize_sign(direction), dtype=float)
        selected_fronts = tuple(front * sign for front, _ in non_empty)
        selected_generations = np.asarray(
            [generation for _, generation in non_empty], dtype=float
        )
        current = selected_fronts[-1]
        ideal = current.min(axis=0)
        nadir = current.max(axis=0)
        norm = nadir - ideal
        norm = np.where(ideal == nadir, 1.0, norm)
        current_normalized = (current - ideal) / norm
        normalized = [(front - ideal) / norm for front in selected_fronts]
        delta_f = np.asarray(
            [igd(front, current_normalized) for front in normalized], dtype=float
        )
        ideals = np.asarray([front.min(axis=0) for front in selected_fronts])
        nadirs = np.asarray([front.max(axis=0) for front in selected_fronts])
        ideal_changes = np.max(np.abs(np.diff(ideals, axis=0)) / norm, axis=1)
        nadir_changes = np.max(np.abs(np.diff(nadirs, axis=0)) / norm, axis=1)
        delta_ideal = np.concatenate((ideal_changes, np.array([0.0])))
        delta_nadir = np.concatenate((nadir_changes, np.array([0.0])))
    else:
        selected_generations = np.empty(0, dtype=float)
        delta_f = np.empty(0, dtype=float)
        delta_ideal = np.empty(0, dtype=float)
        delta_nadir = np.empty(0, dtype=float)

    axes.plot(selected_generations, delta_f)
    significant = np.maximum(delta_ideal, delta_nadir) > significance
    for generation, metric, is_significant in zip(
        selected_generations, delta_f, significant
    ):
        if is_significant:
            axes.vlines(generation, 0.0, metric)
            axes.plot([generation], [metric], marker="o")
    axes.set_yscale("symlog")
    axes.set_xlabel("Generation")
    axes.set_ylabel("$\\Delta f$")
    return fig
