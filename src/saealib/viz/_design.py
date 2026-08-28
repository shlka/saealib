"""Design-space plots for optimization results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from saealib.exceptions import ValidationError
from saealib.space import BoundsService, DenseNumericView
from saealib.viz._common import _resolve_axes
from saealib.viz._history import _history_column, _require_block, _require_history
from saealib.viz._matplotlib import _require_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from saealib.api import Result


_DENSE_ERROR = "The search space does not provide a dense numeric view."


class _ServiceRegistry(Protocol):
    def get(self, name: str) -> object | None: ...


def _services(
    result: Result, function: str
) -> tuple[DenseNumericView, _ServiceRegistry]:
    try:
        services = result.ctx.problem.space.services
        dense = services.get("DenseNumericView")
    except AttributeError as exc:
        raise ValidationError(f"{function}: {_DENSE_ERROR}") from exc
    if dense is None:
        raise ValidationError(f"{function}: {_DENSE_ERROR}")
    return cast(DenseNumericView, dense), cast(_ServiceRegistry, services)


def _archive_values(
    result: Result, function: str
) -> tuple[np.ndarray, np.ndarray, _ServiceRegistry]:
    dense, services = _services(result, function)
    archive = result.ctx.archive
    if len(archive) == 0:
        raise ValidationError(f"{function} requires a non-empty archive.")
    try:
        values = np.asarray(dense.get_view(archive.genomes), dtype=float)
        objectives = np.asarray(archive.get_array("f"), dtype=float)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValidationError(f"{function} requires numeric archive data.") from exc
    if values.ndim != 2 or objectives.ndim != 2 or len(values) != len(objectives):
        raise ValidationError(f"{function} requires a two-dimensional archive.")
    return values, objectives, services


def _variables(
    dimension: int,
    variables: Sequence[int] | None,
    function: str,
    *,
    pair: bool = False,
) -> tuple[int, ...]:
    if variables is None:
        if pair:
            if dimension == 2:
                return (0, 1)
            raise ValidationError(
                f"{function} requires variables=(i, j) when dimension is "
                "greater than two."
            )
        return tuple(range(dimension))
    try:
        selected = tuple(variables)
    except TypeError as exc:
        raise ValidationError(
            f"{function} variables must contain valid indices."
        ) from exc
    if (pair and len(selected) != 2) or (not pair and not selected):
        raise ValidationError(f"{function} variables have an invalid length.")
    if any(
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        or index < 0
        or index >= dimension
        for index in selected
    ):
        raise ValidationError(
            f"{function} variables contain an index outside 0..{dimension - 1}."
        )
    return tuple(int(index) for index in selected)


def _bounds(services: _ServiceRegistry, function: str) -> tuple[np.ndarray, np.ndarray]:
    bounds_service = services.get("BoundsService")
    if bounds_service is None:
        raise ValidationError(f"{function} requires the BoundsService service.")
    try:
        lower, upper = cast(BoundsService, bounds_service).bounds
        lower_array = np.asarray(lower, dtype=float).reshape(-1)
        upper_array = np.asarray(upper, dtype=float).reshape(-1)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValidationError(f"{function} requires valid bounds.") from exc
    if lower_array.shape != upper_array.shape:
        raise ValidationError(f"{function} requires matching lower and upper bounds.")
    return lower_array, upper_array


def plot_archive(
    result: Result,
    *,
    variables: Sequence[int] | None = None,
    objective: int | None = None,
    cmap: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot two design variables from the archive, colored by an objective.

    Parameters
    ----------
    result : saealib.api.Result
        Optimization result whose archive is plotted.
    variables : sequence of int or None, optional
        Two design-variable indices. For a two-dimensional space, ``None``
        selects both variables.
    objective : int or None, optional
        Objective index used for point colors. Required for multi-objective
        archives and defaults to zero for single-objective archives.
    cmap : str or None, optional
        Matplotlib colormap passed to the scatter plot.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the scatter plot and colorbar.
    """
    _require_matplotlib()
    values, objectives, _ = _archive_values(result, "plot_archive")
    selected = _variables(values.shape[1], variables, "plot_archive", pair=True)
    n_obj = objectives.shape[1]
    if objective is None:
        if n_obj > 1:
            raise ValidationError(
                "plot_archive requires objective for multi-objective archives."
            )
        objective = 0
    if (
        isinstance(objective, (bool, np.bool_))
        or not isinstance(objective, (int, np.integer))
        or not 0 <= objective < n_obj
    ):
        raise ValidationError(
            f"plot_archive objective must be an index in 0..{n_obj - 1}."
        )
    fig, axes = _resolve_axes(ax)
    scatter = axes.scatter(
        values[:, selected[0]],
        values[:, selected[1]],
        c=objectives[:, int(objective)],
        cmap=cmap,
    )
    colorbar = fig.colorbar(scatter, ax=axes)
    colorbar.set_label(f"f{int(objective)}")
    axes.set_xlabel(f"x{selected[0]}")
    axes.set_ylabel(f"x{selected[1]}")
    return fig


def plot_design_pcp(
    result: Result, *, variables: Sequence[int] | None = None, ax: Axes | None = None
) -> Figure:
    """Plot archive design variables as bounds-normalized parallel coordinates.

    Bounds normalization uses the search-space bounds rather than observed
    archive minima and maxima, so axes remain comparable across runs. Fixed
    variables are represented by ``0.5``.

    Parameters
    ----------
    result : saealib.api.Result
        Optimization result whose archive is plotted.
    variables : sequence of int or None, optional
        Variables to display, in plotting order. ``None`` displays all.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the parallel-coordinate plot.
    """
    _require_matplotlib()
    values, _, services = _archive_values(result, "plot_design_pcp")
    selected = _variables(values.shape[1], variables, "plot_design_pcp")
    lower, upper = _bounds(services, "plot_design_pcp")
    if len(lower) != values.shape[1]:
        raise ValidationError("plot_design_pcp bounds do not match design dimensions.")
    span = upper - lower
    normalized = np.full(values[:, selected].shape, 0.5)
    np.divide(
        values[:, selected] - lower[list(selected)],
        span[list(selected)],
        out=normalized,
        where=span[list(selected)] != 0,
    )
    fig, axes = _resolve_axes(ax)
    for row in normalized:
        axes.plot(np.arange(len(selected)), row)
    axes.set_xticks(np.arange(len(selected)))
    axes.set_xticklabels([f"x{index}" for index in selected])
    axes.set_ylim(0.0, 1.0)
    return fig


def _selected_generations(
    generations: np.ndarray, requested: Sequence[int] | None, function: str
) -> np.ndarray:
    if requested is not None:
        try:
            selected = np.asarray(tuple(requested), dtype=int)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"{function} generations must be integer generation numbers."
            ) from exc
        if selected.ndim != 1 or any(
            int(generation) not in generations for generation in selected
        ):
            raise ValidationError(
                f"{function} generations include an unrecorded generation."
            )
        return selected
    if len(generations) <= 20:
        return generations.astype(int, copy=True)
    return generations[np.linspace(0, len(generations) - 1, 20, dtype=int)].astype(int)


def plot_variable_distribution(
    result: Result,
    *,
    variable: int | None = None,
    generations: Sequence[int] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot generation-wise population distributions for one design variable.

    Parameters
    ----------
    result : saealib.api.Result
        Optimization result with population history.
    variable : int or None, optional
        Design-variable index. It may be omitted only for one-dimensional data.
    generations : sequence of int or None, optional
        Recorded generations to display. By default, at most 20 evenly spaced
        recorded generations are selected for readable figures.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the box plots.
    """
    _require_matplotlib()
    dense, _ = _services(result, "plot_variable_distribution")
    del dense
    history = _require_history(result, "plot_variable_distribution")
    blocks = _require_block(history, "population", "x", "plot_variable_distribution")
    columns = history.channel("population")
    recorded = _history_column(
        columns, "gen", len(blocks), "plot_variable_distribution"
    )
    dimension = next(
        (block.shape[1] for block in blocks if block.ndim == 2 and block.shape[1]), 0
    )
    if dimension == 0:
        raise ValidationError(
            "plot_variable_distribution requires population design values."
        )
    if variable is None:
        if dimension != 1:
            raise ValidationError(
                "plot_variable_distribution requires variable for dimension "
                "two or greater."
            )
        variable = 0
    if (
        isinstance(variable, (bool, np.bool_))
        or not isinstance(variable, (int, np.integer))
        or not 0 <= variable < dimension
    ):
        raise ValidationError(
            f"plot_variable_distribution variable must be an index in "
            f"0..{dimension - 1}."
        )
    selected = _selected_generations(
        recorded, generations, "plot_variable_distribution"
    )
    by_generation = {int(gen): block for gen, block in zip(recorded, blocks)}
    data = [
        np.asarray(by_generation[int(gen)])[:, int(variable)]
        for gen in selected
        if len(by_generation[int(gen)])
    ]
    labels = [int(gen) for gen in selected if len(by_generation[int(gen)])]
    if not data:
        raise ValidationError(
            "plot_variable_distribution has no non-empty generations to plot."
        )
    fig, axes = _resolve_axes(ax)
    axes.boxplot(data, tick_labels=[str(label) for label in labels])
    axes.set_xlabel("Generation")
    axes.set_ylabel(f"x{int(variable)}")
    return fig


def plot_diversity(result: Result, *, ax: Axes | None = None) -> Figure:
    """Plot bounds-normalized mean pairwise population diversity.

    Diversity is normalized with search-space bounds, not each generation's
    observed range, so the time series retains a stable scale.

    Parameters
    ----------
    result : saealib.api.Result
        Optimization result with population history.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing diversity against function evaluations.
    """
    _require_matplotlib()
    _, services = _services(result, "plot_diversity")
    lower, upper = _bounds(services, "plot_diversity")
    history = _require_history(result, "plot_diversity")
    blocks = _require_block(history, "population", "x", "plot_diversity")
    columns = history.channel("population")
    evaluations = _history_column(columns, "fe", len(blocks), "plot_diversity")
    active = upper != lower
    if not np.any(active):
        raise ValidationError(
            "plot_diversity requires at least one non-fixed variable."
        )
    if len(lower) == 0 or any(
        block.ndim != 2 or block.shape[1] != len(lower) for block in blocks
    ):
        raise ValidationError(
            "plot_diversity population dimensions do not match bounds."
        )
    values = []
    scale = np.sqrt(np.count_nonzero(active))
    for block in blocks:
        if len(block) < 2:
            values.append(np.nan)
            continue
        normalized = (np.asarray(block, dtype=float)[:, active] - lower[active]) / (
            upper[active] - lower[active]
        )
        distances = np.linalg.norm(
            normalized[:, None, :] - normalized[None, :, :], axis=2
        )
        values.append(float(np.mean(distances[np.triu_indices(len(block), 1)]) / scale))
    fig, axes = _resolve_axes(ax)
    axes.plot(evaluations, values)
    axes.set_xlabel("Function evaluations")
    axes.set_ylabel("Normalized population diversity")
    return fig
