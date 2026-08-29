"""Unified plots for optimization results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from saealib.exceptions import ValidationError
from saealib.result import Result
from saealib.space import DenseNumericView
from saealib.viz._common import _resolve_axes
from saealib.viz._matplotlib import _require_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from saealib.context import OptimizationState


class _ServiceRegistry(Protocol):
    def get(self, name: str) -> object | None: ...


def _select_dimensions(
    dimension: int, dimensions: Sequence[int] | None
) -> tuple[int, ...]:
    if dimensions is None:
        return tuple(range(dimension))
    try:
        selected = tuple(dimensions)
    except TypeError as exc:
        raise ValidationError("dimensions must contain valid indices.") from exc
    if not selected:
        raise ValidationError("dimensions must not be empty.")
    if any(
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        or index < 0
        or index >= dimension
        for index in selected
    ):
        raise ValidationError(
            f"dimensions contain an index outside 0..{dimension - 1}."
        )
    return tuple(int(index) for index in selected)


def _matrix(
    values: object, function: str, *, reshape_vector: bool = False
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"{function} requires numeric two-dimensional data."
        ) from exc
    if reshape_vector and array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValidationError(f"{function} requires two-dimensional data.")
    if len(array) == 0:
        raise ValidationError(f"{function} requires a non-empty result set.")
    return array


def _services(result: Result, function: str) -> DenseNumericView:
    try:
        services = result.ctx.problem.space.services
        dense = services.get("DenseNumericView")
    except AttributeError as exc:
        raise ValidationError(
            f"{function}: The search space does not provide a dense numeric view."
        ) from exc
    if dense is None:
        raise ValidationError(
            f"{function}: The search space does not provide a dense numeric view."
        )
    return cast(DenseNumericView, dense)


def _values(result: Result, space: str, source: str) -> np.ndarray:
    function = "plot_result"
    if space == "objective" and source == "result":
        return _matrix(result.f, function, reshape_vector=True)
    if space == "decision" and source == "result":
        return _matrix(result.x, function, reshape_vector=True)

    collection = result.archive if source == "archive" else result.population
    if len(collection) == 0:
        raise ValidationError("plot_result requires a non-empty result set.")
    if space == "objective":
        return _matrix(collection.get_array("f"), function)

    dense = _services(result, function)
    try:
        return _matrix(dense.get_view(collection.genomes), function)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValidationError("plot_result requires numeric decision data.") from exc


def _normalize(values: np.ndarray) -> np.ndarray:
    minimum = values.min(axis=0)
    span = values.max(axis=0) - minimum
    normalized = np.full(values.shape, 0.5, dtype=float)
    np.divide(values - minimum, span, out=normalized, where=span != 0)
    return normalized


def plot_result(
    result: Result | OptimizationState,
    *,
    space: str = "objective",
    source: str = "result",
    kind: str = "auto",
    dimensions: Sequence[int] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot objective or decision values from an optimization result."""
    _require_matplotlib()
    if not isinstance(space, str) or space not in {"objective", "decision"}:
        raise ValidationError("space must be 'objective' or 'decision'.")
    if not isinstance(source, str) or source not in {
        "result",
        "archive",
        "population",
    }:
        raise ValidationError("source must be 'result', 'archive', or 'population'.")
    if not isinstance(kind, str) or kind not in {"auto", "scatter", "parallel"}:
        raise ValidationError("kind must be 'auto', 'scatter', or 'parallel'.")

    normalized_result = (
        result if isinstance(result, Result) else Result.from_state(result)
    )
    values = _values(normalized_result, space, source)
    selected = _select_dimensions(values.shape[1], dimensions)
    values = values[:, selected]
    n_dimensions = values.shape[1]
    if n_dimensions < 2:
        raise ValidationError("plot_result requires at least two dimensions.")
    if kind == "auto":
        if n_dimensions in (2, 3):
            kind = "scatter"
        elif n_dimensions >= 4:
            kind = "parallel"
    if kind == "scatter" and n_dimensions not in (2, 3):
        raise ValidationError("scatter requires exactly two or three dimensions.")
    prefix = "f" if space == "objective" else "x"
    labels = [f"{prefix}{index}" for index in selected]
    if kind == "scatter":
        if n_dimensions == 2:
            fig, axes = _resolve_axes(ax)
            axes.scatter(values[:, 0], values[:, 1])
        else:
            fig, axes = _resolve_axes(ax, projection="3d")
            axes.scatter(values[:, 0], values[:, 1], values[:, 2])
        for axis, label in zip("xyz", labels):
            getattr(axes, f"set_{axis}label")(label)
        return fig

    fig, axes = _resolve_axes(ax)
    for row in _normalize(values):
        axes.plot(np.arange(n_dimensions), row)
    axes.set_xticks(np.arange(n_dimensions))
    axes.set_xticklabels(labels)
    axes.set_ylim(0.0, 1.0)
    return fig
