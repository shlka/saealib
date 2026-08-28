"""Plots for decision-space structures and island migration."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from saealib.exceptions import ValidationError
from saealib.viz._common import _resolve_axes
from saealib.viz._history import _history_column, _require_channel
from saealib.viz._matplotlib import _require_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from saealib.context import OptimizationState
    from saealib.islands import IslandModel


def _objective_indices(n_obj: int, objectives: Sequence[int] | None) -> tuple[int, ...]:
    if objectives is None:
        if n_obj in (2, 3):
            return tuple(range(n_obj))
        raise ValidationError(
            "plot_weight_vectors requires objectives=(i, j) or "
            "objectives=(i, j, k) for four or more objectives."
        )
    try:
        selected = tuple(objectives)
    except TypeError as exc:
        raise ValidationError(
            "plot_weight_vectors objectives must contain two or three indices."
        ) from exc
    if len(selected) not in (2, 3):
        raise ValidationError(
            "plot_weight_vectors objectives must contain two or three indices."
        )
    if any(
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        or index < 0
        or index >= n_obj
        for index in selected
    ):
        raise ValidationError(
            f"plot_weight_vectors objectives contain an index outside 0..{n_obj - 1}."
        )
    return tuple(int(index) for index in selected)


def plot_weight_vectors(
    vectors: np.ndarray,
    *,
    objectives: Sequence[int] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot weight vectors in a selected objective projection.

    Parameters
    ----------
    vectors : numpy.ndarray
        Weight vectors with shape ``(n_vectors, n_objectives)``. A one-
        dimensional array is treated as one vector.
    objectives : sequence of int or None, optional
        Two or three objective indices. For two or three objectives, all
        objectives are selected by default. Four or more objectives require
        an explicit selection.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. Three selected objectives require a three-dimensional
        axes.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the scatter plot.
    """
    _require_matplotlib()
    if not isinstance(vectors, np.ndarray):
        raise ValidationError("plot_weight_vectors requires a numpy.ndarray.")
    try:
        values = np.asarray(vectors, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError("plot_weight_vectors requires numeric vectors.") from exc
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValidationError(
            "plot_weight_vectors requires a non-empty two-dimensional array "
            "with at least two objectives."
        )
    selected = _objective_indices(values.shape[1], objectives)
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
        getattr(axes, f"set_{axis}label")(f"w{index}")
    return fig


def _root_figure(axes: Sequence[Axes]) -> Figure:
    try:
        if any(
            getattr(axis, "name", None) is None or getattr(axis, "figure", None) is None
            for axis in axes
        ):
            raise ValidationError("axes must contain two matplotlib Axes.")
        first_figure, _ = _resolve_axes(axes[0])
        for axis in axes[1:]:
            figure, _ = _resolve_axes(axis)
            if figure is not first_figure:
                raise ValidationError("axes must belong to the same Figure.")
    except ValidationError:
        raise
    except (AttributeError, TypeError) as exc:
        raise ValidationError("axes must contain two matplotlib Axes.") from exc
    return first_figure


def _state_convergence(
    state: OptimizationState, island: int
) -> tuple[np.ndarray, np.ndarray]:
    function = "plot_island_migration"
    try:
        problem = state.problem
        n_obj = int(problem.n_obj)
        direction = np.asarray(problem.direction, dtype=float).reshape(-1)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValidationError(
            f"{function} requires valid problem metadata for island {island}."
        ) from exc
    if n_obj != 1 or direction.size != 1:
        raise ValidationError(
            f"{function} requires single-objective islands. Use "
            "plot_hypervolume or plot_indicator for multi-objective islands."
        )
    if not np.isfinite(direction[0]) or abs(direction[0]) != 1:
        raise ValidationError(f"{function} requires ±1 objective directions.")
    summary = _require_channel(state, "summary", function)
    try:
        raw_fe = np.asarray(summary["fe"])
    except KeyError as exc:
        raise ValidationError(
            f'{function} requires the "fe" column in the summary channel.'
        ) from exc
    if raw_fe.ndim != 1 or len(raw_fe) == 0:
        raise ValidationError(f"{function} requires a non-empty valid summary.")
    value_name = "f_min_0" if direction[0] < 0 else "f_max_0"
    fe = _history_column(summary, "fe", len(raw_fe), function)
    values = _history_column(summary, value_name, len(raw_fe), function)
    try:
        fe = np.asarray(fe, dtype=float)
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{function} requires numeric summary columns.") from exc
    finite = np.isfinite(fe) & np.isfinite(values)
    if not np.any(finite):
        raise ValidationError(f"{function} requires a non-empty finite summary.")
    order = np.argsort(fe[finite], kind="stable")
    return fe[finite][order], values[finite][order]


def plot_island_migration(
    model: IslandModel,
    states: Sequence[OptimizationState],
    *,
    axes: Sequence[Axes] | None = None,
) -> Figure:
    """Plot island migration frequencies and island convergence.

    The migration panel shows only edges represented by actual migration
    events, not the topology configured on the model. An edge that never
    fired is therefore omitted. Unlike other plotting functions, this
    function takes ``axes=`` because two panels are essential.

    Parameters
    ----------
    model : saealib.islands.IslandModel
        Island model providing optimizers and recorded migration events.
    states : sequence of saealib.context.OptimizationState
        Final state for each island.
    axes : sequence of matplotlib.axes.Axes or None, optional
        Exactly two axes for the migration and convergence panels. Unlike
        other plotting functions, two panels are essential for this plot.

    Returns
    -------
    matplotlib.figure.Figure
        Root figure containing the two panels.
    """
    _require_matplotlib()
    try:
        n_islands = len(model.optimizers)
        state_values = tuple(states)
    except (AttributeError, TypeError) as exc:
        raise ValidationError(
            "plot_island_migration requires an island model and states."
        ) from exc
    if len(state_values) != n_islands:
        raise ValidationError("states must contain one state per optimizer.")
    if axes is None:
        from matplotlib.figure import Figure

        figure = Figure()
        panel_axes = (figure.add_subplot(121), figure.add_subplot(122))
    else:
        try:
            if len(axes) != 2:
                raise ValidationError("axes must contain exactly two Axes.")
            panel_axes = tuple(axes)
            figure = _root_figure(panel_axes)
        except TypeError as exc:
            raise ValidationError("axes must contain exactly two Axes.") from exc

    migration_axes, convergence_axes = panel_axes
    angles = np.linspace(0.0, 2.0 * np.pi, n_islands, endpoint=False)
    node_x = np.cos(angles)
    node_y = np.sin(angles)
    migration_axes.scatter(node_x, node_y)
    for index, (x, y) in enumerate(zip(node_x, node_y)):
        migration_axes.annotate(
            f"{index}", (x, y), xytext=(4, 4), textcoords="offset points"
        )

    try:
        events = tuple(model.migration_events)
    except (AttributeError, TypeError) as exc:
        raise ValidationError(
            "model.migration_events must be a sequence of edges."
        ) from exc
    counts: Counter[tuple[int, int]] = Counter()
    for event in events:
        try:
            source, target = event
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "migration events must be (source, target) pairs."
            ) from exc
        if any(
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            or index < 0
            or index >= n_islands
            for index in (source, target)
        ):
            raise ValidationError("migration event index is outside the island range.")
        counts[(int(source), int(target))] += 1
    for (source, target), count in counts.items():
        migration_axes.annotate(
            "",
            xy=(node_x[target], node_y[target]),
            xytext=(node_x[source], node_y[source]),
            arrowprops={"arrowstyle": "->", "lw": 1.0 + count, "alpha": 0.75},
        )
        midpoint = (
            (node_x[source] + node_x[target]) / 2,
            (node_y[source] + node_y[target]) / 2,
        )
        migration_axes.annotate(str(count), midpoint, ha="center", va="center")
    migration_axes.set_title("Migration")
    migration_axes.set_aspect("equal")
    migration_axes.set_axis_off()

    for index, state in enumerate(state_values):
        fe, values = _state_convergence(state, index)
        convergence_axes.plot(fe, values, label=f"island {index}")
    convergence_axes.set_xlabel("Function evaluations")
    convergence_axes.set_ylabel("Objective value")
    convergence_axes.set_title("Convergence")
    if n_islands:
        convergence_axes.legend()
    return figure
