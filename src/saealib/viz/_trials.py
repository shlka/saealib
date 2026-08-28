"""Helpers for aggregating convergence histories."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, cast

import numpy as np

from saealib.exceptions import ValidationError
from saealib.viz._common import _direction
from saealib.viz._history import _history_column, _require_channel

if TYPE_CHECKING:
    from saealib.api import Result


def _normalize_results(result: Result | Sequence[Result]) -> tuple[Result, ...]:
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        results = tuple(cast("Sequence[Result]", result))
    else:
        results = (cast("Result", result),)
    if not results:
        raise ValidationError("plot_convergence requires at least one result.")
    return results


def _validate_fe_range(fe_range: str) -> None:
    if not isinstance(fe_range, str) or fe_range not in ("common", "full"):
        raise ValidationError('fe_range must be either "common" or "full".')


def _validate_groups_and_labels(
    count: int,
    groups: Sequence[Hashable] | None,
    labels: str | Mapping[Hashable, str] | None,
) -> tuple[Hashable | None, ...]:
    if groups is None:
        if isinstance(labels, Mapping):
            raise ValidationError(
                "labels must be a string when groups is not specified."
            )
        if labels is not None and not isinstance(labels, str):
            raise ValidationError("labels must be a string or a mapping.")
        return (None,) * count

    try:
        group_values = tuple(groups)
    except TypeError as exc:
        raise ValidationError("groups must be a sequence of group keys.") from exc
    if len(group_values) != count:
        raise ValidationError("groups must have one key per result.")
    try:
        for key in group_values:
            hash(key)
    except TypeError as exc:
        raise ValidationError("groups must contain hashable keys.") from exc

    if labels is not None and not isinstance(labels, Mapping):
        raise ValidationError("labels must be a mapping when groups is specified.")
    if isinstance(labels, Mapping):
        expected = set(group_values)
        provided = set(labels)
        if expected != provided:
            raise ValidationError("labels keys must match the groups exactly.")
        if any(not isinstance(value, str) for value in labels.values()):
            raise ValidationError("labels mapping values must be strings.")
    return group_values


def _summary_series(result: Result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    function = "plot_convergence"
    try:
        problem = result.ctx.problem
        n_obj = problem.n_obj
    except AttributeError as exc:
        raise ValidationError("plot_convergence requires Result objects.") from exc
    if not isinstance(n_obj, (int, np.integer)) or n_obj < 1:
        raise ValidationError("plot_convergence requires a valid objective count.")
    if n_obj > 1:
        raise ValidationError(
            "plot_convergence requires a single objective. Use "
            "plot_hypervolume or plot_indicator for multi-objective results."
        )

    direction = _direction(result, int(n_obj))
    if not np.all(np.isfinite(direction)) or not np.all(np.abs(direction) == 1):
        raise ValidationError("plot_convergence requires ±1 objective directions.")

    summary = _require_channel(result, "summary", function)
    try:
        raw_fe = np.asarray(summary["fe"])
    except KeyError as exc:
        raise ValidationError(
            'plot_convergence requires the "fe" column in the summary channel.'
        ) from exc
    if raw_fe.ndim != 1:
        raise ValidationError(
            'plot_convergence requires a one-dimensional "fe" history column.'
        )
    count = len(raw_fe)
    if count == 0:
        raise ValidationError("plot_convergence cannot use a zero-row summary.")

    value_name = "f_min_0" if direction[0] < 0 else "f_max_0"
    fe_values = _history_column(summary, "fe", count, function)
    objective_values = _history_column(summary, value_name, count, function)
    try:
        fe = np.asarray(fe_values, dtype=float)
        values = np.asarray(objective_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "plot_convergence requires numeric summary fe and objective columns."
        ) from exc
    finite = np.isfinite(fe) & np.isfinite(values)
    if not np.any(finite):
        raise ValidationError("plot_convergence requires a non-empty finite summary.")
    fe = fe[finite]
    values = values[finite]
    order = np.argsort(fe, kind="stable")
    return direction.copy(), fe[order], values[order]


def _prepare_convergence(
    result: Result | Sequence[Result],
    groups: Sequence[Hashable] | None,
    labels: str | Mapping[Hashable, str] | None,
    fe_range: str,
) -> tuple[
    tuple[Hashable | None, ...],
    tuple[tuple[np.ndarray, np.ndarray], ...],
    np.ndarray,
]:
    _validate_fe_range(fe_range)
    results = _normalize_results(result)
    group_values = _validate_groups_and_labels(len(results), groups, labels)
    series: list[tuple[np.ndarray, np.ndarray]] = []
    reference_direction: np.ndarray | None = None
    for run in results:
        direction, fe, values = _summary_series(run)
        if reference_direction is None:
            reference_direction = direction
        elif not np.array_equal(direction, reference_direction):
            raise ValidationError(
                "plot_convergence requires all results to have the same "
                "objective direction."
            )
        series.append((fe, values))
    assert reference_direction is not None
    return group_values, tuple(series), reference_direction


def _aggregate_convergence(
    series: tuple[tuple[np.ndarray, np.ndarray], ...],
    groups: tuple[Hashable | None, ...],
    direction: np.ndarray,
    fe_range: str,
) -> tuple[
    np.ndarray,
    tuple[
        tuple[Hashable | None, np.ndarray, np.ndarray, np.ndarray],
        ...,
    ],
]:
    _validate_fe_range(fe_range)
    starts = np.asarray([fe[0] for fe, _ in series], dtype=float)
    ends = np.asarray([fe[-1] for fe, _ in series], dtype=float)
    left = float(np.max(starts))
    right = float(np.min(ends) if fe_range == "common" else np.max(ends))
    if fe_range == "common" and left > right:
        raise ValidationError(
            'The results have no common FE range; use fe_range="full".'
        )

    grid_parts = [fe[(fe >= left) & (fe <= right)] for fe, _ in series]
    grid = np.unique(np.concatenate(grid_parts))
    if len(grid) == 0:
        raise ValidationError("plot_convergence produced an empty FE grid.")

    expanded: list[np.ndarray] = []
    for fe, values in series:
        best = (
            np.minimum.accumulate(values)
            if direction[0] < 0
            else np.maximum.accumulate(values)
        )
        indices = np.searchsorted(fe, grid, side="right") - 1
        if np.any(indices < 0):
            raise ValidationError(
                "plot_convergence cannot extrapolate before a result's first FE."
            )
        expanded.append(best[indices])
    expanded_array = np.asarray(expanded, dtype=float)

    unique_groups: list[Hashable | None] = []
    for group in groups:
        if group not in unique_groups:
            unique_groups.append(group)

    aggregates: list[tuple[Hashable | None, np.ndarray, np.ndarray, np.ndarray]] = []
    for group in unique_groups:
        indices = [index for index, value in enumerate(groups) if value == group]
        values = expanded_array[indices]
        q1, median, q3 = np.percentile(values, [25.0, 50.0, 75.0], axis=0)
        aggregates.append((group, median, q1, q3))
    return grid, tuple(aggregates)
