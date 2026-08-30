"""Built-in values for :meth:`Result.history_series`.

A front-based value is computed in minimize-space: the recorded front and the
caller's reference are multiplied by the problem's minimize sign before the
indicator runs, and neither is normalized.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from saealib.acquisition.base import direction_to_minimize_sign
from saealib.exceptions import ValidationError
from saealib.utils import gd, gd_plus, hypervolume, igd, igd_plus, spacing, spread


@dataclass(frozen=True)
class _ValueSpec:
    channel: str
    column: str | None = None
    reference: str | None = None
    compute: Callable[..., float] | None = None


def _indicator(name: str) -> Callable[..., float]:
    indicators = {
        "gd": gd,
        "gd_plus": gd_plus,
        "igd": igd,
        "igd_plus": igd_plus,
        "spread": spread,
    }

    def compute(front: np.ndarray, *, reference_front: np.ndarray) -> float:
        return float(indicators[name](front, reference_front))

    return compute


def _hypervolume(
    front: np.ndarray, *, reference_point: np.ndarray, sign: np.ndarray
) -> float:
    if len(front) == 0:
        return np.nan
    return float(hypervolume(front * sign, reference_point * sign))


def _spacing(front: np.ndarray) -> float:
    return float(spacing(front))


def _mean_normalized_pairwise_distance(
    block: np.ndarray, *, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Return mean pairwise distance in bounds-normalized design space.

    The distance is divided by the unit-hypercube diagonal to yield values in [0, 1].
    """
    if len(block) < 2:
        return np.nan
    active = upper != lower
    normalized = (block[:, active] - lower[active]) / (upper[active] - lower[active])
    distances = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=2)
    return float(
        np.mean(distances[np.triu_indices(len(block), 1)])
        / np.sqrt(np.count_nonzero(active))
    )


_HISTORY_VALUES: dict[str, _ValueSpec] = {
    "best": _ValueSpec("summary", "best_f"),
    "min_cv": _ValueSpec("summary", "min_cv"),
    "feasible_ratio": _ValueSpec("summary", "feasible_ratio"),
    "front_size": _ValueSpec("summary", "front_size"),
    "hypervolume": _ValueSpec(
        "front", reference="reference_point", compute=_hypervolume
    ),
    "gd": _ValueSpec("front", reference="reference_front", compute=_indicator("gd")),
    "gd_plus": _ValueSpec(
        "front", reference="reference_front", compute=_indicator("gd_plus")
    ),
    "igd": _ValueSpec("front", reference="reference_front", compute=_indicator("igd")),
    "igd_plus": _ValueSpec(
        "front", reference="reference_front", compute=_indicator("igd_plus")
    ),
    "spread": _ValueSpec(
        "front", reference="reference_front", compute=_indicator("spread")
    ),
    "spacing": _ValueSpec("front", compute=_spacing),
    "mean_normalized_pairwise_distance": _ValueSpec(
        "population", column="x", compute=_mean_normalized_pairwise_distance
    ),
}


def _reference_front(value: Any, n_obj: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"{name} requires a non-empty reference front with one column per "
            "objective."
        ) from exc
    if array.ndim == 1 and array.size == n_obj:
        array = array.reshape(1, n_obj)
    if array.ndim != 2 or array.shape[1] != n_obj or len(array) == 0:
        raise ValidationError(
            f"{name} requires a non-empty reference front with one column per "
            "objective."
        )
    return array


def _reference_point(value: Any, n_obj: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "reference_point must have one value per objective."
        ) from exc
    if array.ndim != 1 or array.size != n_obj:
        raise ValidationError("reference_point must have one value per objective.")
    return array


def _service_bounds(result: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        services = result.problem.space.services
        dense_service = services.get("DenseNumericView")
        bounds_service = services.get("BoundsService")
    except AttributeError as exc:
        raise ValidationError(
            "mean_normalized_pairwise_distance requires DenseNumericView and "
            "BoundsService services."
        ) from exc
    if dense_service is None:
        raise ValidationError(
            "mean_normalized_pairwise_distance requires the DenseNumericView service."
        )
    if bounds_service is None:
        raise ValidationError(
            "mean_normalized_pairwise_distance requires the BoundsService service."
        )
    try:
        lower, upper = bounds_service.bounds
        lower = np.asarray(lower, dtype=float).reshape(-1)
        upper = np.asarray(upper, dtype=float).reshape(-1)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValidationError(
            "mean_normalized_pairwise_distance requires valid bounds."
        ) from exc
    if lower.shape != upper.shape or not np.any(upper != lower):
        raise ValidationError(
            "mean_normalized_pairwise_distance requires at least one non-fixed "
            "variable."
        )
    return lower, upper


def _builtin_values(
    result: Any, value: str, spec: _ValueSpec, kwargs: Mapping[str, Any]
) -> list[float]:
    history = result.history
    assert history is not None
    if spec.channel == "summary":
        if value == "best" and result.problem.n_obj > 1:
            raise ValidationError(
                '"best" is available only for single-objective results; use '
                '"hypervolume" or an indicator for multi-objective results.'
            )
        column = history.get("summary", spec.column or value)
        return [float(item) for item in column]

    if value == "mean_normalized_pairwise_distance":
        lower, upper = _service_bounds(result)
        blocks = history.get("population", "x")
        values = []
        for block in blocks:
            array = np.asarray(block, dtype=float)
            if array.ndim != 2 or array.shape[1] != len(lower):
                raise ValidationError(
                    "mean_normalized_pairwise_distance population dimensions do "
                    "not match bounds."
                )
            values.append(
                _mean_normalized_pairwise_distance(array, lower=lower, upper=upper)
            )
        return values

    blocks = history.get(spec.channel, spec.column or "f")
    sign = np.asarray(direction_to_minimize_sign(result.problem.direction), dtype=float)
    compute = spec.compute
    if compute is None:
        raise RuntimeError(f"History value {value!r} has no computation.")
    n_obj = result.problem.n_obj
    front_values = []
    for block in blocks:
        array = np.asarray(block, dtype=float)
        if array.ndim != 2 or array.shape[1] != n_obj:
            raise ValidationError(
                f"{value} requires front blocks with {n_obj} objectives."
            )
        front_values.append(array)
    if spec.reference == "reference_point":
        reference = _reference_point(kwargs.get("reference_point"), n_obj)
        return [
            compute(front, reference_point=reference, sign=sign)
            for front in front_values
        ]
    if spec.reference == "reference_front":
        if "reference_front" not in kwargs:
            raise ValidationError(f"{value} requires reference_front.")
        reference = _reference_front(kwargs["reference_front"], n_obj, value) * sign
        return [
            compute(front * sign, reference_front=reference) for front in front_values
        ]
    return [compute(front * sign) for front in front_values]
