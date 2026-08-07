from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from saealib.core.contracts import (
    MANY,
    OPTIONAL,
    ComponentContract,
    DataSpec,
    PortContract,
    PortDirection,
    PortSpec,
    ServiceRequirement,
)
from saealib.exceptions import ValidationError
from saealib.execution.evaluator import EvaluationResult
from saealib.registry import register

if TYPE_CHECKING:
    from saealib.population import Population


@dataclass(frozen=True)
class FeedbackResult:
    """Validated values supplied to an algorithm tell operation."""

    candidate_ids: np.ndarray
    f: np.ndarray
    g: np.ndarray | None
    cv: np.ndarray | None
    evaluated_mask: np.ndarray
    source: np.ndarray
    artifacts: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize owned arrays."""
        ids = _array(self.candidate_ids, np.int64, "candidate_ids")
        f = _array(self.f, np.float64, "f")
        mask = _array(self.evaluated_mask, bool, "evaluated_mask")
        source = _array(self.source, np.uint8, "source")
        if ids.ndim != 1 or f.ndim != 2 or len(ids) != len(f):
            raise ValidationError("feedback candidate_ids and f are misaligned")
        if mask.shape != ids.shape or source.shape != ids.shape:
            raise ValidationError("feedback masks must align with candidate_ids")
        if len(np.unique(ids)) != len(ids):
            raise ValidationError("feedback candidate_ids must be unique")
        if self.g is not None:
            g = _array(self.g, np.float64, "g")
            if g.ndim != 2 or g.shape[0] != len(ids):
                raise ValidationError("feedback g has an invalid shape")
            object.__setattr__(self, "g", _readonly(g))
        if self.cv is not None:
            cv = _array(self.cv, np.float64, "cv")
            if cv.shape != (len(ids),):
                raise ValidationError("feedback cv has an invalid shape")
            object.__setattr__(self, "cv", _readonly(cv))
        for name, arr in (
            ("candidate_ids", ids),
            ("f", f),
            ("evaluated_mask", mask),
            ("source", source),
        ):
            object.__setattr__(self, name, _readonly(arr))
        artifacts = {}
        for name, value in self.artifacts.items():
            arr = np.array(value, copy=True)
            if arr.dtype == object or arr.ndim == 0 or arr.shape[0] != len(ids):
                raise ValidationError(
                    "feedback artifacts must have the candidate row count"
                )
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)
            artifacts[name] = _readonly(arr)
        object.__setattr__(self, "artifacts", artifacts)


def _readonly(arr: np.ndarray) -> np.ndarray:
    arr.flags.writeable = False
    return arr


def _array(value, dtype, name: str) -> np.ndarray:
    arr = np.asarray(value)
    expected = np.dtype(dtype)
    if arr.dtype == object or arr.dtype != expected:
        raise ValidationError(f"feedback {name} must have dtype {expected}")
    return np.array(arr, dtype=expected, order="C", copy=True)


def _ids(candidates: Population) -> np.ndarray:
    if "id" not in candidates.schema:
        return np.arange(len(candidates), dtype=np.int64)
    return np.array(candidates.get_array("id"), dtype=np.int64, copy=True)


def _true_rows(
    candidates: Population, evaluation: EvaluationResult | None
) -> tuple[np.ndarray, np.ndarray]:
    ids = _ids(candidates)
    if evaluation is None or evaluation.candidate_ids is None:
        return ids, np.zeros(len(ids), dtype=bool)
    lookup = {int(value): i for i, value in enumerate(evaluation.candidate_ids)}
    mask = np.array([int(value) in lookup for value in ids], dtype=bool)
    return ids, mask


def _empty(n_obj: int) -> FeedbackResult:
    return FeedbackResult(
        np.empty(0, dtype=np.int64),
        np.empty((0, n_obj), dtype=np.float64),
        None,
        None,
        np.empty(0, dtype=bool),
        np.empty(0, dtype=np.uint8),
    )


class FeedbackBuilder(ABC):
    """Build algorithm feedback from true and predicted values."""

    def contract(self) -> ComponentContract:
        """Return the feedback-builder family contract."""
        return ComponentContract(
            ports={
                "feedback_builder": PortContract(
                    inputs=(
                        PortSpec(
                            name="candidates",
                            direction=PortDirection.INPUT,
                            data=DataSpec(kind="Population"),
                            cardinality=MANY,
                        ),
                        PortSpec(
                            name="prediction",
                            direction=PortDirection.INPUT,
                            data=DataSpec(kind="SurrogatePrediction"),
                            cardinality=OPTIONAL,
                            optional=True,
                        ),
                        PortSpec(
                            name="evaluation",
                            direction=PortDirection.INPUT,
                            data=DataSpec(kind="ObservationBatch"),
                            cardinality=OPTIONAL,
                            optional=True,
                        ),
                    ),
                    outputs=(
                        PortSpec(
                            name="feedback",
                            direction=PortDirection.OUTPUT,
                            data=DataSpec(kind="FeedbackBatch"),
                            cardinality=MANY,
                        ),
                    ),
                ),
            }
        )

    @abstractmethod
    def build(self, candidates, prediction, evaluation, evaluated_indices, ctx):
        """Return feedback aligned to candidate IDs."""


@register()
class TrueOnlyFeedback(FeedbackBuilder):
    """Return completed true objective rows."""

    def __init__(self) -> None:
        pass

    def build(self, candidates, prediction, evaluation, evaluated_indices, ctx):
        """Build true-only feedback."""
        if evaluation is None or evaluation.candidate_ids is None:
            return _empty(ctx.n_obj)
        ids = _ids(candidates)
        evaluated = set(map(int, evaluation.candidate_ids))
        rows = np.array([int(value) in evaluated for value in ids], dtype=bool)
        selected = ids[rows]
        positions = [
            int(np.flatnonzero(evaluation.candidate_ids == value)[0])
            for value in selected
        ]
        return _result_from_evaluation(
            evaluation, positions, np.ones(len(positions), dtype=bool), 0
        )


@register()
class PredictedFeedback(FeedbackBuilder):
    """Return the objective prediction channel."""

    def __init__(self) -> None:
        pass

    def build(self, candidates, prediction, evaluation, evaluated_indices, ctx):
        """Build predicted feedback."""
        if prediction is None or "objective" not in prediction.channels:
            raise ValidationError("PredictedFeedback requires an objective channel")
        ids = _ids(candidates)
        n = len(ids)
        values = prediction.value
        if values.shape[0] != n:
            raise ValidationError("prediction rows do not match candidates")
        mask = np.ones(n, dtype=bool)
        return FeedbackResult(ids, values, None, None, mask, np.ones(n, dtype=np.uint8))


@register()
class MixedFeedback(FeedbackBuilder):
    """Prefer true rows and fill the remainder from objective predictions."""

    def __init__(self) -> None:
        pass

    def build(self, candidates, prediction, evaluation, evaluated_indices, ctx):
        """Build mixed feedback."""
        ids = _ids(candidates)
        n = len(ids)
        predicted = None if prediction is None else prediction.channels.get("objective")
        if predicted is None:
            n_obj = evaluation.f.shape[1] if evaluation is not None else ctx.n_obj
        else:
            n_obj = predicted.value.shape[1]
        f = np.empty((n, n_obj), dtype=np.float64)
        source = np.ones(n, dtype=np.uint8)
        f.fill(np.nan)
        if predicted is not None:
            if predicted.value.shape != f.shape:
                raise ValidationError(
                    "prediction objective shape does not match candidates"
                )
            f[:] = predicted.value
        evaluated_mask = np.zeros(n, dtype=bool)
        if evaluation is not None and evaluation.candidate_ids is not None:
            lookup = {int(value): i for i, value in enumerate(evaluation.candidate_ids)}
            for row, candidate_id in enumerate(ids):
                if int(candidate_id) in lookup:
                    f[row] = evaluation.f[lookup[int(candidate_id)]]
                    evaluated_mask[row] = True
                    source[row] = 0
        return FeedbackResult(ids, f, None, None, evaluated_mask, source)


@register()
class NoFeedback(FeedbackBuilder):
    """Return an empty feedback batch."""

    def __init__(self) -> None:
        pass

    def build(self, candidates, prediction, evaluation, evaluated_indices, ctx):
        """Build empty feedback."""
        return _empty(ctx.n_obj)


@register()
class ComparatorWorstFallback(FeedbackBuilder):
    """Fill missing rows with the comparator's worst population objective."""

    def __init__(self, inner: FeedbackBuilder | None = None) -> None:
        self.inner = inner or MixedFeedback()

    def contract(self) -> ComponentContract:
        """Return the feedback contract requiring population comparison."""
        family = super().contract()
        builder = family.ports["feedback_builder"]
        candidates = replace(
            builder.inputs[0],
            required_services=(ServiceRequirement(name="ComparisonService"),),
        )
        return replace(
            family,
            ports={
                **family.ports,
                "feedback_builder": replace(
                    builder, inputs=(candidates, *builder.inputs[1:])
                ),
            },
        )

    def build(self, candidates, prediction, evaluation, evaluated_indices, ctx):
        """Build feedback and replace missing rows."""
        result = self.inner.build(
            candidates, prediction, evaluation, evaluated_indices, ctx
        )
        if len(result.candidate_ids) == 0:
            return result
        missing = np.flatnonzero(
            (result.source != 0) & np.any(~np.isfinite(result.f), axis=1)
        )
        if len(missing) == 0:
            return result
        order = ctx.problem.comparator.sort_population(ctx.population)
        fallback = np.array(
            ctx.population.get_array("f")[order[-1]], dtype=np.float64, copy=True
        )
        f = np.array(result.f, copy=True)
        f[missing] = fallback
        source = np.array(result.source, copy=True)
        source[missing] = 2
        return FeedbackResult(
            result.candidate_ids,
            f,
            result.g,
            result.cv,
            result.evaluated_mask,
            source,
            result.artifacts,
        )


def _result_from_evaluation(evaluation, positions, mask, source):
    g = None if evaluation.g is None else evaluation.g[positions]
    cv = None if evaluation.cv is None else evaluation.cv[positions]
    return FeedbackResult(
        evaluation.candidate_ids[positions],
        evaluation.f[positions],
        g,
        cv,
        np.array(mask, dtype=bool, copy=True),
        np.full(len(positions), source, dtype=np.uint8),
    )
