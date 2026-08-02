from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from saealib.acquisition.base import AcquisitionResult
from saealib.exceptions import ValidationError
from saealib.execution.evaluator import EvaluationRequest
from saealib.registry import register

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.population import Population


def _scores(
    acquisition: AcquisitionResult | None, n: int, *, sanitize_nonfinite: bool = False
) -> np.ndarray:
    if acquisition is None or acquisition.scores is None:
        raise ValidationError("an acquisition score array is required")
    scores = np.asarray(acquisition.scores)
    if scores.shape != (n,) or scores.dtype != np.float64:
        raise ValidationError("acquisition scores must be float64 with shape (n,)")
    if not np.all(np.isfinite(scores)) and sanitize_nonfinite:
        scores = np.array(scores, copy=True)
        scores[~np.isfinite(scores)] = np.finfo(np.float64).min
    return scores


def select_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Return stable descending top-k indices."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or scores.dtype != np.float64:
        raise ValidationError("scores must be a float64 vector")
    if not np.all(np.isfinite(scores)):
        raise ValidationError("acquisition scores must be finite")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise ValidationError("k must be an integer")
    if k < 0 or k > len(scores):
        raise ValidationError("k must be within the score vector")
    return np.array(np.argsort(-scores, kind="stable")[:k], dtype=np.intp, copy=True)


def select_ratio(scores: np.ndarray, ratio: float) -> np.ndarray:
    """Return stable descending indices for the ratio-selected prefix."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or scores.dtype != np.float64:
        raise ValidationError("scores must be a float64 vector")
    if not np.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
        raise ValidationError("ratio must be finite and in [0, 1]")
    n = max(1, int(ratio * len(scores))) if len(scores) else 0
    return select_top_k(scores, n)


class EvaluationPolicy(ABC):
    """Construct an evaluation request from a candidate batch."""

    @abstractmethod
    def plan(
        self,
        candidates: Population,
        acquisition: AcquisitionResult | None,
        ctx: OptimizationState,
    ) -> EvaluationRequest:
        """Plan one request."""
        ...

    @staticmethod
    def _request(
        candidates: Population, indices: np.ndarray, ctx: OptimizationState
    ) -> EvaluationRequest:
        if "id" not in candidates.schema:
            ids = np.arange(len(candidates), dtype=np.int64)
        else:
            current = np.asarray(candidates.get_array("id"), dtype=np.int64)
            missing = np.flatnonzero(current < 0)
            if len(missing):
                ids = ctx.candidate_id_allocator.allocate(len(missing))
                candidates._assign_ids(missing, ids)
        if "id" in candidates.schema:
            ids = np.array(
                candidates.get_array("id")[indices], dtype=np.int64, copy=True
            )
        else:
            ids = np.array(indices, dtype=np.int64, copy=True)
        if len(ids) != len(np.unique(ids)) or np.any(ids < 0):
            raise ValidationError("candidate IDs must be unique and assigned")
        request_id = ctx.request_id_allocator.allocate(1)[0]
        x = np.array(candidates.x[indices], dtype=np.float64, order="C", copy=True)
        x.flags.writeable = False
        return EvaluationRequest(
            np.int64(request_id), ids, x, metadata={"row_indices": indices.tolist()}
        )


@register()
class EvaluateAll(EvaluationPolicy):
    """Evaluate every candidate."""

    def plan(self, candidates, acquisition, ctx):
        """Build a request containing every candidate."""
        return self._request(candidates, np.arange(len(candidates), dtype=np.intp), ctx)


@register()
class TopKEvaluation(EvaluationPolicy):
    """Evaluate the stable top-k acquisition scores."""

    def __init__(self, k: int, sanitize_nonfinite: bool = False) -> None:
        self.k = k
        self.sanitize_nonfinite = sanitize_nonfinite

    def plan(self, candidates, acquisition, ctx):
        """Build a request for the selected prefix."""
        indices = select_top_k(
            _scores(
                acquisition,
                len(candidates),
                sanitize_nonfinite=self.sanitize_nonfinite,
            ),
            self.k,
        )
        return self._request(candidates, indices, ctx)


@register()
class RatioEvaluation(EvaluationPolicy):
    """Evaluate the stable prefix selected by a ratio."""

    def __init__(self, ratio: float, sanitize_nonfinite: bool = False) -> None:
        self.ratio = ratio
        self.sanitize_nonfinite = sanitize_nonfinite

    def plan(self, candidates, acquisition, ctx):
        """Build a request for the ratio-selected prefix."""
        scores = _scores(
            acquisition,
            len(candidates),
            sanitize_nonfinite=self.sanitize_nonfinite,
        )
        indices = select_ratio(scores, self.ratio)
        return self._request(candidates, indices, ctx)
