"""Surrogate diagnostics and prescreening plots."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from saealib.exceptions import ValidationError
from saealib.population import DenseVectorBatch
from saealib.space import BoundsService, DenseNumericView
from saealib.space.space import encode_features
from saealib.surrogate import Surrogate
from saealib.viz._common import _resolve_axes
from saealib.viz._history import (
    _history_column,
    _require_block,
    _require_channel,
    _require_history,
)
from saealib.viz._matplotlib import _require_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from saealib.acquisition import AcquisitionFunction
    from saealib.api import Result


_DENSE_ERROR = "The search space does not provide a dense numeric view."


def _landscape_grid(
    result: Result,
    function: str,
    variables: tuple[int, int] | None,
    fixed: Mapping[int, float] | None,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Validate landscape inputs and return design and encoded grid arrays."""
    try:
        services = result.ctx.problem.space.services
        dense = services.get("DenseNumericView")
        bounds_service = services.get("BoundsService")
    except AttributeError as exc:
        raise ValidationError(f"{function}: {_DENSE_ERROR}") from exc
    if dense is None:
        raise ValidationError(f"{function}: {_DENSE_ERROR}")
    if bounds_service is None:
        raise ValidationError(f"{function} requires the BoundsService service.")
    try:
        lower, upper = cast(BoundsService, bounds_service).bounds
        lower = np.asarray(lower, dtype=float).reshape(-1)
        upper = np.asarray(upper, dtype=float).reshape(-1)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValidationError(f"{function} requires valid bounds.") from exc
    if (
        lower.shape != upper.shape
        or lower.size < 2
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(upper < lower)
    ):
        raise ValidationError(f"{function} requires at least two matching bounds.")
    dimension = lower.size
    if (
        not isinstance(resolution, (int, np.integer))
        or isinstance(resolution, (bool, np.bool_))
        or resolution < 2
    ):
        raise ValidationError("resolution must be an integer of at least 2.")
    if variables is None:
        if dimension != 2:
            raise ValidationError(
                f"{function} requires variables=(i, j) when dimension is greater "
                "than two."
            )
        selected = (0, 1)
    else:
        try:
            selected_raw = tuple(variables)
        except TypeError as exc:
            raise ValidationError(
                f"{function} variables must contain two indices."
            ) from exc
        if len(selected_raw) != 2 or any(
            isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
            for i in selected_raw
        ):
            raise ValidationError(f"{function} variables must contain two indices.")
        selected = (int(selected_raw[0]), int(selected_raw[1]))
        if selected[0] == selected[1] or any(i < 0 or i >= dimension for i in selected):
            raise ValidationError(f"{function} variables contain invalid indices.")
    if fixed is None:
        fixed_items = ()
    elif isinstance(fixed, Mapping):
        fixed_items = fixed.items()
    else:
        raise ValidationError(f"{function} fixed must be a mapping of index to value.")
    fixed_values: dict[int, float] = {}
    for raw_index, value in fixed_items:
        index = raw_index
        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index, (int, np.integer)
        ):
            raise ValidationError(f"{function} fixed contains an invalid index.")
        index = int(index)
        if index < 0 or index >= dimension:
            raise ValidationError(
                f"{function} fixed contains an index outside the bounds."
            )
        if index in selected:
            raise ValidationError(f"{function} fixed overlaps variables.")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{function} fixed values must be numeric.") from exc
        if not np.isfinite(numeric) or numeric < lower[index] or numeric > upper[index]:
            raise ValidationError(f"{function} fixed value is outside its bounds.")
        fixed_values[index] = numeric
    missing = set(range(dimension)) - set(selected) - set(fixed_values)
    if missing:
        raise ValidationError(
            f"{function} requires fixed values for dimensions {sorted(missing)}."
        )
    axis_x = np.linspace(lower[selected[0]], upper[selected[0]], int(resolution))
    axis_y = np.linspace(lower[selected[1]], upper[selected[1]], int(resolution))
    grid = np.empty((int(resolution) ** 2, dimension), dtype=float)
    grid[:] = 0.0
    for index, value in fixed_values.items():
        grid[:, index] = value
    xx, yy = np.meshgrid(axis_x, axis_y)
    grid[:, selected[0]] = xx.ravel()
    grid[:, selected[1]] = yy.ravel()
    batch = DenseVectorBatch(grid)
    try:
        features = np.asarray(
            encode_features(result.ctx.problem.space, batch), dtype=np.float64
        )
    except ValidationError:
        raise
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(
            f"{function} could not encode the landscape grid."
        ) from exc
    if features.ndim != 2 or features.shape[0] != len(grid):
        raise ValidationError(
            f"{function} requires a two-dimensional encoded grid with one row "
            "per design point."
        )
    return grid, features, selected


def _landscape_objective(result: Result, objective: int | None, function: str) -> int:
    try:
        n_obj_value = result.ctx.problem.n_obj
    except AttributeError as exc:
        raise ValidationError(f"{function} requires objective metadata.") from exc
    if (
        isinstance(n_obj_value, (bool, np.bool_))
        or not isinstance(n_obj_value, (int, np.integer))
        or n_obj_value < 1
    ):
        raise ValidationError(f"{function} requires a positive objective count.")
    n_obj = int(n_obj_value)
    if n_obj > 1 and objective is None:
        raise ValidationError(
            f"{function} requires objective for multi-objective data."
        )
    index = 0 if objective is None else objective
    if (
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        or index < 0
        or index >= n_obj
    ):
        raise ValidationError(
            f"{function} objective is outside the available objective range."
        )
    return int(index)


def _landscape_plot(
    result: Result,
    surrogate: Surrogate,
    *,
    variables: tuple[int, int] | None,
    fixed: Mapping[int, float] | None,
    resolution: int,
    cmap: str | None,
    ax: Axes | None,
    objective: int | None,
    uncertainty: bool = False,
) -> Figure:
    _require_matplotlib()
    if not isinstance(surrogate, Surrogate):
        raise ValidationError("surrogate must be a fitted Surrogate, not a manager.")
    grid, features, selected = _landscape_grid(
        result,
        "plot_surrogate_uncertainty" if uncertainty else "plot_surrogate",
        variables,
        fixed,
        resolution,
    )
    index = _landscape_objective(
        result,
        objective,
        "plot_surrogate_uncertainty" if uncertainty else "plot_surrogate",
    )
    prediction = surrogate.predict(features)
    values = prediction.std if uncertainty else prediction.value
    if uncertainty and values is None:
        raise ValidationError(
            "This surrogate does not provide uncertainty. Use a surrogate with "
            "Surrogate.provides_uncertainty=True, such as SklearnGPRSurrogate."
        )
    assert values is not None
    values = np.asarray(values)
    expected = int(resolution) ** 2
    if values.ndim != 2 or values.shape[0] != expected or values.shape[1] <= index:
        raise ValidationError(
            "surrogate prediction shape does not match the landscape grid."
        )
    fig, ax = _resolve_axes(ax)
    image = ax.imshow(
        values[:, index].reshape((int(resolution), int(resolution))),
        origin="lower",
        extent=(
            grid[:, selected[0]].min(),
            grid[:, selected[0]].max(),
            grid[:, selected[1]].min(),
            grid[:, selected[1]].max(),
        ),
        cmap=cmap,
        aspect="auto",
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(
        "Prediction standard deviation" if uncertainty else "Predicted objective"
    )
    if not uncertainty:
        dense = cast(
            DenseNumericView, result.ctx.problem.space.services.get("DenseNumericView")
        )
        try:
            points = np.asarray(dense.get_view(result.ctx.archive.genomes), dtype=float)
        except ValidationError:
            raise
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(
                "plot_surrogate requires numeric archive design data."
            ) from exc
        if points.ndim != 2 or points.shape[1] != grid.shape[1]:
            raise ValidationError("archive design data is not dense numeric data.")
        ax.scatter(points[:, selected[0]], points[:, selected[1]])
    ax.set_xlabel(f"x{selected[0]}")
    ax.set_ylabel(f"x{selected[1]}")
    return fig


def _accuracy_rows(
    result: Result, function: str
) -> tuple[Mapping[str, np.ndarray], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    history = _require_history(result, function)
    columns = _require_channel(result, "surrogate_accuracy", function)
    predicted = _require_block(history, "surrogate_accuracy", "predicted", function)
    true = _require_block(history, "surrogate_accuracy", "true", function)
    if len(predicted) != len(true):
        raise ValidationError(
            f"{function} requires matching predicted and true history blocks."
        )
    return columns, predicted, true


def _objective_index(
    blocks: tuple[np.ndarray, ...], objective: int | None, function: str
) -> int:
    widths = [block.shape[1] for block in blocks if block.ndim == 2]
    n_obj = max(widths, default=0)
    if n_obj == 0:
        raise ValidationError(f"{function} has no objective values to plot.")
    if n_obj > 1 and objective is None:
        raise ValidationError(
            f"{function} requires objective for multi-objective data."
        )
    index = 0 if objective is None else objective
    if (
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        or index < 0
        or index >= n_obj
    ):
        raise ValidationError(
            f"objective={objective!r} is outside the available objective range "
            f"0..{n_obj - 1}."
        )
    return int(index)


def _accuracy_data_for_objective(
    columns: Mapping[str, np.ndarray],
    predicted_blocks: tuple[np.ndarray, ...],
    true_blocks: tuple[np.ndarray, ...],
    index: int,
    function: str,
    *,
    with_metadata: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, float]]]:
    count = len(predicted_blocks)
    size = _history_column(columns, "size", count, function)
    values_pred: list[np.ndarray] = []
    values_true: list[np.ndarray] = []
    rows: list[int] = []
    for row, (predicted, true) in enumerate(zip(predicted_blocks, true_blocks)):
        if int(size[row]) == 0:
            continue
        if predicted.ndim != 2 or true.ndim != 2 or predicted.shape != true.shape:
            raise ValidationError(
                f"{function} requires matching two-dimensional predicted and "
                "true blocks."
            )
        if predicted.shape[1] <= index:
            raise ValidationError(
                f"objective={index} is outside the available objective range."
            )
        predicted_values = np.asarray(predicted[:, index], dtype=float)
        true_values = np.asarray(true[:, index], dtype=float)
        finite = np.isfinite(predicted_values) & np.isfinite(true_values)
        values_pred.append(predicted_values[finite])
        values_true.append(true_values[finite])
        rows.extend([row] * int(np.count_nonzero(finite)))
    if not values_pred or sum(map(len, values_pred)) == 0:
        raise ValidationError(
            f"{function} found no valid predicted/true pairs; run an "
            "optimization using a surrogate."
        )
    metadata: list[tuple[int, int, float]] = []
    if with_metadata:
        gen = _history_column(columns, "gen", count, function)
        fe_after = _history_column(columns, "fe_after", count, function)
        metadata = [(int(gen[row]), row, float(fe_after[row])) for row in rows]
    return np.concatenate(values_pred), np.concatenate(values_true), metadata


def _prepare_accuracy(result: Result, objective: int | None, function: str):
    columns, predicted_blocks, true_blocks = _accuracy_rows(result, function)
    index = _objective_index(predicted_blocks, objective, function)
    return _accuracy_data_for_objective(
        columns, predicted_blocks, true_blocks, index, function
    )


def plot_surrogate_accuracy(
    result: Result, *, objective: int | None = None, ax: Axes | None = None
) -> Figure:
    """Plot pooled surrogate predictions against observed objective values.

    All valid prediction/observation pairs are pooled before calculating the
    metrics. Rows with ``size=0`` are skipped, and pairs containing a non-finite
    predicted or true value are excluded.

    Parameters
    ----------
    result : saealib.api.Result
        Optimization result with ``surrogate_accuracy`` history.
    objective : int or None, optional
        Objective column to plot. Defaults to zero for single-objective data;
        required for multi-objective data.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the prediction scatter, identity line, and metrics.

    Notes
    -----
    RMSE and R² are calculated from the pooled pairs rather than per-history
    row metrics. When the true values have zero total sum of squares,
    ``SS_tot == 0`` and R² is reported as NaN.
    """
    _require_matplotlib()
    predicted, true, _ = _prepare_accuracy(result, objective, "plot_surrogate_accuracy")
    fig, ax = _resolve_axes(ax)
    ax.scatter(true, predicted)
    low = float(np.nanmin(true))
    high = float(np.nanmax(true))
    ax.plot([low, high], [low, high], linestyle="--")
    residual = predicted - true
    rmse = float(np.sqrt(np.mean(residual**2)))
    centered = true - np.mean(true)
    total = float(np.sum(centered**2))
    r2 = float("nan") if total == 0 else 1.0 - float(np.sum(residual**2)) / total
    r2_text = "NaN (constant true values)" if np.isnan(r2) else f"{r2:.6g}"
    ax.text(
        0.05,
        0.95,
        f"RMSE = {rmse:.6g}\nR² = {r2_text}",
        transform=ax.transAxes,
        va="top",
    )
    ax.set_xlabel("True objective value")
    ax.set_ylabel("Predicted objective value")
    return fig


def plot_surrogate_error_history(
    result: Result,
    *,
    objective: int | None = None,
    window: int | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot surrogate prediction error over function evaluations.

    Error is defined as RMSE. R² is not shown because a history row commonly
    contains only one prediction pair, for which R² is not defined. With no
    window, one RMSE is plotted for each generation group. With a window, the
    RMSE uses the most recent number of valid pairs in row order.

    Parameters
    ----------
    result : saealib.api.Result
        Optimization result with ``surrogate_accuracy`` history.
    objective : int or None, optional
        Objective column to plot. Defaults to zero for single-objective data;
        required for multi-objective data.
    window : int or None, optional
        Positive number of valid pairs included in each moving RMSE. ``None``
        groups valid pairs by generation instead.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the RMSE history.

    Notes
    -----
    Rows with ``size=0`` and pairs with a non-finite predicted or true value
    are excluded before grouping or applying the moving window.
    """
    _require_matplotlib()
    if window is not None and (
        not isinstance(window, (int, np.integer))
        or isinstance(window, bool)
        or window <= 0
    ):
        raise ValidationError("window must be a positive integer.")
    columns, predicted_blocks, true_blocks = _accuracy_rows(
        result, "plot_surrogate_error_history"
    )
    index = _objective_index(
        predicted_blocks, objective, "plot_surrogate_error_history"
    )
    predicted, true, metadata = _accuracy_data_for_objective(
        columns,
        predicted_blocks,
        true_blocks,
        index,
        "plot_surrogate_error_history",
        with_metadata=True,
    )
    errors = (predicted - true) ** 2
    if window is None:
        groups: dict[int, list[int]] = {}
        for i, (gen, _, _) in enumerate(metadata):
            groups.setdefault(gen, []).append(i)
        xs = [max(metadata[i][2] for i in indexes) for indexes in groups.values()]
        ys = [float(np.sqrt(np.mean(errors[indexes]))) for indexes in groups.values()]
    else:
        xs = []
        ys = []
        row_values = np.asarray([meta[1] for meta in metadata], dtype=int)
        boundaries = np.flatnonzero(
            np.concatenate(
                (np.array([True]), row_values[1:] != row_values[:-1], np.array([True]))
            )
        )
        prefix = np.concatenate((np.array([0.0]), np.cumsum(errors, dtype=float)))
        for start, end in pairwise(boundaries):
            start_index = max(0, int(end) - int(window))
            xs.append(max(metadata[i][2] for i in range(int(start), int(end))))
            total = prefix[int(end)] - prefix[start_index]
            ys.append(float(np.sqrt(total / (int(end) - start_index))))
    fig, ax = _resolve_axes(ax)
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Surrogate RMSE")
    return fig


def _prescreen_blocks(result: Result, function: str):
    history = _require_history(result, function)
    columns = _require_channel(result, "decision_candidates", function)
    count = len(columns.get("size", ()))
    selected = _require_block(history, "decision_candidates", "selected", function)
    scores = _require_block(
        history, "decision_candidates", "acquisition_scores", function
    )
    return history, columns, selected, scores, count


def plot_prescreening(
    result: Result,
    *,
    decision: int = -1,
    variables: tuple[int, int] | None = None,
    cmap: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot selected and rejected candidates for one prescreening decision.

    When the history contains a ``candidates`` block, candidates are drawn in
    a two-dimensional design-space projection. Without that block, the plot
    falls back to acquisition-score rank on the x-axis. Prediction standard
    deviation is not used.

    Parameters
    ----------
    result : saealib.api.Result
        Optimization result with ``decision_candidates`` history.
    decision : int, optional
        Recorded decision index. Negative indices select from the end.
    variables : tuple of int or None, optional
        Two design-variable indices for the dense candidate projection. When
        omitted, a two-dimensional design space uses ``(0, 1)``.
    cmap : str or None, optional
        Colormap for acquisition-score coloring. When ``None``, Matplotlib's
        default colormap is used without passing a colormap override.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the selected and rejected candidate points.

    Notes
    -----
    The presence of a ``candidates`` block selects the design-space path. If
    it is absent, acquisition-score descending rank is used as the x-axis.
    A ``prediction_std`` history column is intentionally not required.
    """
    _require_matplotlib()
    history, columns, selected_blocks, score_blocks, count = _prescreen_blocks(
        result, "plot_prescreening"
    )
    if (
        not isinstance(decision, (int, np.integer))
        or isinstance(decision, bool)
        or decision < -count
        or decision >= count
    ):
        raise ValidationError(
            f"decision index {decision!r} is outside {count} recorded decisions."
        )
    row = int(decision) % count
    size = int(_history_column(columns, "size", count, "plot_prescreening")[row])
    if size == 0:
        raise ValidationError(f"decision {row} has size=0 and cannot be plotted.")
    selected = np.asarray(selected_blocks[row]).reshape(-1).astype(bool)
    scores = np.asarray(score_blocks[row], dtype=float).reshape(-1)
    if len(selected) != size or len(scores) != size:
        raise ValidationError(
            "plot_prescreening requires selected and acquisition_scores blocks "
            "matching size."
        )
    try:
        candidates = history.blocks("decision_candidates", "candidates")[row]
    except ValidationError:
        candidates = None
    fig, ax = _resolve_axes(ax)
    finite = np.isfinite(scores)
    if candidates is not None:
        if candidates.ndim != 2 or len(candidates) != size:
            raise ValidationError(
                "plot_prescreening requires a candidates block matching size."
            )
        dim = candidates.shape[1]
        if variables is None:
            if dim != 2:
                raise ValidationError(
                    "variables must be specified when candidates have 3 or "
                    "more dimensions."
                )
            variables = (0, 1)
        if len(variables) != 2 or any(
            isinstance(v, (bool, np.bool_))
            or not isinstance(v, (int, np.integer))
            or v < 0
            or v >= dim
            for v in variables
        ):
            raise ValidationError(
                f"variables must contain two indices in the range 0..{dim - 1}."
            )
        colors = scores if np.any(finite) else None
        if colors is None:
            rejected_kwargs: dict[str, Any] = {}
            selected_kwargs: dict[str, Any] = {}
        else:
            value_min = float(np.nanmin(colors))
            value_max = float(np.nanmax(colors))
            rejected_kwargs = {
                "c": colors[~selected],
                "vmin": value_min,
                "vmax": value_max,
            }
            selected_kwargs = {
                "c": colors[selected],
                "vmin": value_min,
                "vmax": value_max,
            }
            if cmap is not None:
                rejected_kwargs["cmap"] = cmap
                selected_kwargs["cmap"] = cmap
        plot_ax = cast(Any, ax)
        plot_ax.scatter(
            candidates[~selected, variables[0]],
            candidates[~selected, variables[1]],
            marker="x",
            **rejected_kwargs,
        )
        collection = plot_ax.scatter(
            candidates[selected, variables[0]],
            candidates[selected, variables[1]],
            marker="o",
            **selected_kwargs,
        )
        if colors is not None:
            fig.colorbar(collection, ax=ax, label="Acquisition score")
        ax.set_xlabel(f"x{variables[0]}")
        ax.set_ylabel(f"x{variables[1]}")
        return fig
    if not np.any(finite):
        raise ValidationError(
            "plot_prescreening cannot plot opaque candidates: acquisition "
            "scores and candidates are unavailable."
        )
    order = np.argsort(np.where(finite, -scores, np.inf), kind="stable")
    ranks = np.empty(size, dtype=float)
    ranks[order] = np.arange(1, size + 1)
    ax.scatter(ranks[~selected], scores[~selected], marker="x")
    ax.scatter(ranks[selected], scores[selected], marker="o")
    ax.set_xlabel("Candidate rank by acquisition score")
    ax.set_ylabel("Acquisition score")
    return fig


def plot_surrogate(
    result: Result,
    surrogate: Surrogate,
    *,
    variables: tuple[int, int] | None = None,
    fixed: Mapping[int, float] | None = None,
    resolution: int = 50,
    objective: int | None = None,
    cmap: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot the mean landscape of a fitted surrogate.

    This is a visualization of the snapshot of the fitted surrogate passed to
    this function. It is not the global landscape representing the
    candidate-wise policy of a ``LocalSurrogateManager``. ``manager.surrogate``
    is an attribute of the concrete global/local manager classes, not part of
    the ``SurrogateManager`` ABC contract. For a local manager, the surrogate
    immediately after a run may be fitted around only the last evaluated
    candidate. To see a global landscape, refit a surrogate on the complete
    archive and pass that fitted surrogate.

    Parameters
    ----------
    result : saealib.api.Result
        Optimization result providing the search space and archive.
    surrogate : saealib.surrogate.Surrogate
        Fitted surrogate snapshot to evaluate.
    variables : tuple of int or None, optional
        Two axes to plot. ``None`` selects ``(0, 1)`` only in two dimensions.
    fixed : mapping of int to float or None, optional
        Values for every non-plotted dimension.
    resolution : int, optional
        Number of grid points per axis.
    objective : int or None, optional
        Objective to plot; required for multi-objective problems.
    cmap : str or None, optional
        Matplotlib colormap.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the mean image, colorbar, and archive points.
    """
    return _landscape_plot(
        result,
        surrogate,
        variables=variables,
        fixed=fixed,
        resolution=resolution,
        cmap=cmap,
        ax=ax,
        objective=objective,
    )


def plot_surrogate_uncertainty(
    result: Result,
    surrogate: Surrogate,
    *,
    variables: tuple[int, int] | None = None,
    fixed: Mapping[int, float] | None = None,
    resolution: int = 50,
    objective: int | None = None,
    cmap: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot the predictive standard-deviation landscape.

    This is a visualization of the snapshot of the fitted surrogate passed to
    this function, not the global candidate-wise policy of a
    ``LocalSurrogateManager``. ``manager.surrogate`` is an attribute of the
    concrete manager classes and is not an ABC contract of
    ``SurrogateManager``. For a local manager, the surrogate immediately after
    a run may be fitted around only the last evaluated candidate. For a global
    landscape, refit on the complete archive and pass that surrogate.

    Parameters
    ----------
    result : saealib.api.Result
        Optimization result providing the search space.
    surrogate : saealib.surrogate.Surrogate
        Fitted surrogate snapshot providing uncertainty.
    variables : tuple of int or None, optional
        Two axes to plot; ``None`` is valid only for two dimensions.
    fixed : mapping of int to float or None, optional
        Values for every non-plotted dimension.
    resolution : int, optional
        Number of grid points per axis.
    objective : int or None, optional
        Objective to plot; required for multi-objective problems.
    cmap : str or None, optional
        Matplotlib colormap.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the uncertainty image and colorbar.
    """
    return _landscape_plot(
        result,
        surrogate,
        variables=variables,
        fixed=fixed,
        resolution=resolution,
        cmap=cmap,
        ax=ax,
        objective=objective,
        uncertainty=True,
    )


def plot_acquisition(
    result: Result,
    surrogate: Surrogate,
    acquisition: AcquisitionFunction,
    *,
    variables: tuple[int, int] | None = None,
    fixed: Mapping[int, float] | None = None,
    resolution: int = 50,
    rng: np.random.Generator | int | None = None,
    cmap: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot acquisition scores over a two-dimensional design landscape.

    This is a visualization of the snapshot of the fitted surrogate passed to
    this function, not the global candidate-wise policy of a
    ``LocalSurrogateManager``. ``manager.surrogate`` belongs to concrete
    manager classes and is not part of the ``SurrogateManager`` ABC contract.
    For a local manager, the surrogate immediately after a run may be fitted
    around only the last evaluated candidate. To obtain a global landscape,
    refit on the complete archive and pass that surrogate.

    Parameters
    ----------
    result : saealib.api.Result
        Optimization result providing the search space, archive, and context.
    surrogate : saealib.surrogate.Surrogate
        Fitted surrogate snapshot.
    acquisition : saealib.acquisition.AcquisitionFunction
        Acquisition function used to score the encoded grid.
    variables : tuple of int or None, optional
        Two axes to plot; ``None`` is valid only for two dimensions.
    fixed : mapping of int to float or None, optional
        Values for every non-plotted dimension.
    resolution : int, optional
        Number of grid points per axis.
    rng : numpy.random.Generator, int, or None, optional
        Detached random generator, seed, or a request for a fresh generator.
    cmap : str or None, optional
        Matplotlib colormap.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the acquisition-score image and colorbar.
    """
    _require_matplotlib()
    if not isinstance(surrogate, Surrogate):
        raise ValidationError("surrogate must be a fitted Surrogate, not a manager.")
    grid, features, selected = _landscape_grid(
        result, "plot_acquisition", variables, fixed, resolution
    )
    prediction = surrogate.predict(features)
    if rng is None:
        plot_rng = np.random.default_rng()
    elif isinstance(rng, np.random.Generator):
        plot_rng = rng
    elif isinstance(rng, (int, np.integer)) and not isinstance(rng, (bool, np.bool_)):
        plot_rng = np.random.default_rng(int(rng))
    else:
        raise ValidationError("rng must be a numpy Generator, integer seed, or None.")
    plot_ctx = result.ctx.replace(rng=plot_rng)
    outcome = acquisition.evaluate(features, prediction, result.ctx.archive, plot_ctx)
    scores = None if outcome is None else outcome.scores
    expected = int(resolution) ** 2
    if scores is None or np.asarray(scores).shape != (expected,):
        raise ValidationError("acquisition scores must have one value per grid point.")
    fig, ax = _resolve_axes(ax)
    image = ax.imshow(
        np.asarray(scores).reshape((int(resolution), int(resolution))),
        origin="lower",
        extent=(
            grid[:, selected[0]].min(),
            grid[:, selected[0]].max(),
            grid[:, selected[1]].min(),
            grid[:, selected[1]].max(),
        ),
        cmap=cmap,
        aspect="auto",
    )
    fig.colorbar(image, ax=ax, label="Acquisition score")
    ax.set_xlabel(f"x{selected[0]}")
    ax.set_ylabel(f"x{selected[1]}")
    return fig
