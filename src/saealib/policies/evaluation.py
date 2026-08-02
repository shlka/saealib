from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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


@register()
class RepeatedEvaluation(EvaluationPolicy):
    """Request candidates with an explicit replicate number."""

    def __init__(self, replicates: int = 2):
        if (
            not isinstance(replicates, int)
            or isinstance(replicates, bool)
            or replicates < 1
        ):
            raise ValidationError("replicates must be a positive integer")
        self.replicates = replicates

    def plan(self, candidates, acquisition, ctx):
        """Build one request and attach its replicate number."""
        request = EvaluateAll().plan(candidates, acquisition, ctx)
        metadata = dict(request.metadata)
        metadata["replicate"] = int(metadata.get("replicate", 0)) % self.replicates
        return EvaluationRequest(
            request.request_id,
            request.candidate_ids,
            request.x,
            request.outputs,
            metadata,
        )

    def plan_replicates(self, candidates, ctx):
        """Build one request per replicate for the same candidate IDs."""
        first = EvaluateAll().plan(candidates, None, ctx)
        requests = []
        for replicate in range(self.replicates):
            request_id = first.request_id
            if replicate:
                request_id = np.int64(ctx.request_id_allocator.allocate(1)[0])
            requests.append(
                EvaluationRequest(
                    request_id,
                    first.candidate_ids,
                    first.x,
                    first.outputs,
                    {**first.metadata, "replicate": replicate},
                )
            )
        return tuple(requests)


@dataclass(frozen=True)
class ReplicateSummary:
    """Aggregated observations for one candidate batch."""

    candidate_ids: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray

    def __post_init__(self):
        """Own arrays and validate aggregate shapes."""
        ids = np.array(self.candidate_ids, dtype=np.int64, copy=True)
        mean = np.array(self.mean, dtype=np.float64, copy=True)
        std = np.array(self.std, dtype=np.float64, copy=True)
        count = np.array(self.count, dtype=np.int64, copy=True)
        if mean.ndim != 2 or std.shape != mean.shape or count.shape != (len(ids),):
            raise ValidationError("replicate summary shapes are inconsistent")
        object.__setattr__(self, "candidate_ids", ids)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "count", count)


def aggregate_replicates(candidate_ids, observations) -> ReplicateSummary:
    """Aggregate repeated objective observations by stable candidate ID."""
    ids = np.asarray(candidate_ids, dtype=np.int64)
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != len(ids):
        raise ValidationError(
            "observations must have shape (replicate, candidate, objective)"
        )
    return ReplicateSummary(
        ids,
        values.mean(axis=0),
        values.std(axis=0),
        np.full(len(ids), values.shape[0], dtype=np.int64),
    )


@register()
class FidelityEvaluation(EvaluationPolicy):
    """Attach an explicit fidelity level to an evaluation request."""

    def __init__(self, fidelity: int = 0):
        if not isinstance(fidelity, int) or isinstance(fidelity, bool) or fidelity < 0:
            raise ValidationError("fidelity must be a non-negative integer")
        self.fidelity = fidelity

    def plan(self, candidates, acquisition, ctx):
        """Build one request and attach its fidelity level."""
        request = EvaluateAll().plan(candidates, acquisition, ctx)
        metadata = dict(request.metadata)
        metadata["fidelity"] = self.fidelity
        return EvaluationRequest(
            request.request_id,
            request.candidate_ids,
            request.x,
            request.outputs,
            metadata,
        )


@register()
class FidelityPromotion(FidelityEvaluation):
    """Represent an explicit promotion from one fidelity level to another."""

    def __init__(self, fidelity: int = 0, next_fidelity: int | None = None):
        super().__init__(fidelity)
        if next_fidelity is not None and (
            not isinstance(next_fidelity, int)
            or isinstance(next_fidelity, bool)
            or next_fidelity <= fidelity
        ):
            raise ValidationError("next_fidelity must exceed fidelity")
        self.next_fidelity = next_fidelity

    def promote(self, request, ctx):
        """Create a new request for the next fidelity level."""
        if self.next_fidelity is None:
            raise ValidationError("next_fidelity is not configured")
        request_id = np.int64(ctx.request_id_allocator.allocate(1)[0])
        return EvaluationRequest(
            request_id,
            request.candidate_ids,
            request.x,
            request.outputs,
            {**request.metadata, "fidelity": self.next_fidelity},
        )
