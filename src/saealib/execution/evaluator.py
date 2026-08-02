"""
Evaluator abstraction.

An ``Evaluator`` is the single entry point through which Strategies and
Initializers turn a batch of design vectors into objective values, raw
constraint values, and aggregate constraint violations. Centralizing
evaluation here enables pluggable execution backends (serial, parallel, ...)
without touching the pipeline code.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np

from saealib.exceptions import EvaluationProtocolError, ValidationError

if TYPE_CHECKING:
    from saealib.problem import Problem


class EvaluationStatus(Enum):
    """State of a submitted evaluation."""

    PENDING = auto()
    RUNNING = auto()
    PARTIAL = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


def _owned_array(value: Any, *, dtype: np.dtype, ndim: int, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype != dtype or arr.ndim != ndim:
        raise ValidationError(f"{name} must have dtype {dtype} and ndim {ndim}")
    result = np.array(arr, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


@dataclass
class EvaluationResult:
    """Validated numeric result for a batch of candidates."""

    f: np.ndarray
    g: np.ndarray
    cv: np.ndarray
    candidate_ids: np.ndarray | None = None
    cost: np.ndarray | None = None
    noise: np.ndarray | None = None
    outputs: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and own all numeric channels."""
        self.f = _owned_array(self.f, dtype=np.dtype(np.float64), ndim=2, name="f")
        self.g = _owned_array(self.g, dtype=np.dtype(np.float64), ndim=2, name="g")
        self.cv = _owned_array(self.cv, dtype=np.dtype(np.float64), ndim=1, name="cv")
        n = len(self.f)
        if len(self.g) != n or len(self.cv) != n:
            raise ValidationError("EvaluationResult channel lengths must match")
        if self.candidate_ids is not None:
            self.candidate_ids = _owned_array(
                self.candidate_ids,
                dtype=np.dtype(np.int64),
                ndim=1,
                name="candidate_ids",
            )
            if len(self.candidate_ids) != n:
                raise ValidationError("candidate_ids length must match result rows")
            if len(np.unique(self.candidate_ids)) != n:
                raise ValidationError("candidate_ids must be unique")
        if self.cost is not None:
            self.cost = _owned_array(
                self.cost, dtype=np.dtype(np.float64), ndim=1, name="cost"
            )
            if len(self.cost) != n:
                raise ValidationError("cost length must match result rows")
        if self.noise is not None:
            self.noise = _owned_array(
                self.noise, dtype=np.dtype(np.float64), ndim=2, name="noise"
            )
            if self.noise.shape[0] != n or self.noise.shape[1] != self.f.shape[1]:
                raise ValidationError("noise must have shape (n, n_obj)")
        normalized: dict[str, np.ndarray] = {}
        for name, value in self.outputs.items():
            arr = np.asarray(value)
            if arr.dtype != np.float64 or arr.ndim == 0 or arr.shape[0] != n:
                raise ValidationError(
                    f"outputs[{name!r}] must be float64 with leading size {n}"
                )
            arr = np.array(arr, dtype=np.float64, order="C", copy=True)
            arr.flags.writeable = False
            normalized[name] = arr
        pickle.dumps(normalized)
        self.outputs = normalized


@dataclass(frozen=True)
class EvaluationErrorInfo:
    """Serializable error information for a failed evaluation."""

    error_type: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate error details are serializable."""
        pickle.dumps(dict(self.details))


@dataclass(frozen=True)
class EvaluationRequest:
    """Owned input snapshot for one evaluation request."""

    request_id: np.int64
    candidate_ids: np.ndarray
    x: np.ndarray
    outputs: tuple[str, ...] = ("f", "g", "cv")
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and own request arrays."""
        if np.asarray(self.request_id).shape != ():
            raise ValidationError("request_id must be scalar")
        object.__setattr__(self, "request_id", np.int64(self.request_id))
        ids = _owned_array(
            self.candidate_ids, dtype=np.dtype(np.int64), ndim=1, name="candidate_ids"
        )
        x = _owned_array(self.x, dtype=np.dtype(np.float64), ndim=2, name="x")
        if len(ids) != len(x) or len(ids) != len(np.unique(ids)):
            raise ValidationError("request candidate_ids must be unique and match x")
        object.__setattr__(self, "candidate_ids", ids)
        object.__setattr__(self, "x", x)
        metadata = dict(self.metadata)
        pickle.dumps(metadata)
        object.__setattr__(self, "metadata", metadata)


@dataclass
class EvaluationHandle:
    """Runtime reference to submitted evaluation work."""

    request_id: np.int64
    status: EvaluationStatus
    backend_token: Any = None
    _sync_result: EvaluationResult | None = None
    _sync_error: EvaluationErrorInfo | None = None
    _delivered_sequence: int = -1
    _acknowledged_sequence: int = -1


@dataclass(frozen=True)
class EvaluationUpdate:
    """One delivered evaluation update."""

    request_id: np.int64
    status: EvaluationStatus
    candidate_ids: np.ndarray
    result: EvaluationResult | None = None
    error: EvaluationErrorInfo | None = None
    sequence: int = 0

    def __post_init__(self) -> None:
        """Validate update row identity."""
        if self.sequence < 0:
            raise EvaluationProtocolError("update sequence must be non-negative")
        ids = _owned_array(
            self.candidate_ids, dtype=np.dtype(np.int64), ndim=1, name="candidate_ids"
        )
        if len(ids) != len(np.unique(ids)):
            raise EvaluationProtocolError("update candidate_ids must be unique")
        object.__setattr__(self, "candidate_ids", ids)
        if self.result is not None and self.result.candidate_ids is None:
            raise EvaluationProtocolError("lifecycle result requires candidate_ids")


@dataclass(frozen=True)
class PendingEvaluation:
    """Serializable record for submitted work."""

    request: EvaluationRequest
    status: EvaluationStatus
    applied_candidate_ids: np.ndarray
    last_delivered_sequence: int = -1
    last_acknowledged_sequence: int = -1

    def __post_init__(self) -> None:
        """Validate and own applied candidate IDs."""
        ids = _owned_array(
            self.applied_candidate_ids,
            dtype=np.dtype(np.int64),
            ndim=1,
            name="applied_candidate_ids",
        )
        object.__setattr__(self, "applied_candidate_ids", ids)


class Evaluator(ABC):
    """Base class for batch evaluators."""

    @abstractmethod
    def evaluate_batch(self, x: np.ndarray, problem: Problem) -> EvaluationResult:
        """
        Evaluate a batch of design vectors.

        Parameters
        ----------
        x : np.ndarray
            Design vectors to evaluate. shape = (n, dim)
        problem : Problem
            The optimization problem providing the objective and constraints.

        Returns
        -------
        EvaluationResult
            Batched objective values, raw constraint values, and violations.
        """
        ...

    def submit(self, request: EvaluationRequest, problem: Problem) -> EvaluationHandle:
        """Submit a request through the synchronous adapter."""
        try:
            result = self.evaluate_batch(request.x, problem)
        except Exception as exc:
            error = EvaluationErrorInfo(type(exc).__name__, str(exc))
            return EvaluationHandle(
                request.request_id, EvaluationStatus.FAILED, _sync_error=error
            )
        if result.candidate_ids is None:
            result.candidate_ids = request.candidate_ids
            result.__post_init__()
        elif not np.array_equal(result.candidate_ids, request.candidate_ids):
            raise EvaluationProtocolError("result candidate_ids do not match request")
        return EvaluationHandle(
            request.request_id, EvaluationStatus.COMPLETED, _sync_result=result
        )

    @classmethod
    def has_partial_lifecycle_override(cls) -> bool:
        """Return whether lifecycle adapter methods are only partly overridden."""
        methods = ("submit", "collect", "acknowledge")
        overridden = [
            getattr(cls, name) is not getattr(Evaluator, name) for name in methods
        ]
        return any(overridden) and not all(overridden)

    def collect(
        self, handle: EvaluationHandle, *, wait: bool = True
    ) -> list[EvaluationUpdate]:
        """Return unacknowledged updates for a handle."""
        if handle._sync_result is None and handle._sync_error is None:
            return []
        if handle._acknowledged_sequence >= 0:
            return []
        ids = (
            handle._sync_result.candidate_ids
            if handle._sync_result is not None
            and handle._sync_result.candidate_ids is not None
            else np.empty(0, dtype=np.int64)
        )
        update = EvaluationUpdate(
            request_id=handle.request_id,
            status=handle.status,
            candidate_ids=ids,
            result=handle._sync_result,
            error=handle._sync_error,
            sequence=0,
        )
        handle._delivered_sequence = 0
        return [update]

    def acknowledge(self, handle: EvaluationHandle, sequence: int) -> None:
        """Acknowledge one contiguous update sequence."""
        if (
            sequence != handle._acknowledged_sequence + 1
            or sequence > handle._delivered_sequence
        ):
            raise EvaluationProtocolError("evaluation sequences must be contiguous")
        handle._acknowledged_sequence = sequence


class SerialEvaluator(Evaluator):
    """Default evaluator: evaluates each candidate sequentially."""

    def evaluate_batch(self, x: np.ndarray, problem: Problem) -> EvaluationResult:
        """
        Evaluate ``x``, preferring ``problem.evaluate_batch`` when available.

        First tries ``problem.evaluate_batch(x)``: if the ``Problem`` overrides
        it to return raw ``(f_batch, g_batch)`` for the whole batch in one
        call, this skips ``problem.evaluate``/``problem.evaluate_constraints``
        entirely and only runs the (cheap) per-row constraint-handler calls
        (``handler.compute_cv`` / ``handler.augment_objective``) needed to turn
        the raw batch into final ``f`` / ``cv`` values. Otherwise it falls back
        to evaluating each row one at a time via
        :meth:`~saealib.problem.Problem.evaluate` /
        :meth:`~saealib.problem.Problem.evaluate_constraints`, reproducing the
        per-candidate evaluation loops that previously lived in each Strategy
        and Initializer.

        Parameters
        ----------
        x : np.ndarray
            Design vectors to evaluate. shape = (n, dim)
        problem : Problem
            The optimization problem providing the objective and constraints.

        Returns
        -------
        EvaluationResult
            Batched objective values, raw constraint values, and violations.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n = len(x)
        n_constraints = problem.n_constraints

        f = np.empty((n, problem.n_obj), dtype=float)
        g = np.empty((n, n_constraints), dtype=float)
        cv = np.zeros(n, dtype=float)

        raw = problem.evaluate_batch(x)
        if raw is not None:
            f_raw, g_raw = raw
            for i in range(n):
                cv[i] = float(
                    problem.handler.compute_cv(problem.constraints, x[i], g_raw[i])
                )
                f[i] = problem.handler.augment_objective(
                    f_raw[i], problem.constraints, x[i], g_raw[i]
                )
                g[i] = g_raw[i]
            return EvaluationResult(f=f, g=g, cv=cv)

        for i, xi in enumerate(x):
            g_i, cv_i = problem.evaluate_constraints(xi)
            f[i] = problem.evaluate(xi, g_i)
            g[i] = g_i
            cv[i] = cv_i

        return EvaluationResult(f=f, g=g, cv=cv)


class JoblibEvaluator(Evaluator):
    """
    Parallel evaluator backed by `joblib <https://joblib.readthedocs.io>`_.

    Candidates in each batch are evaluated in parallel using joblib's
    ``Parallel`` / ``delayed`` interface.  The default backend ``"loky"``
    uses cloudpickle, so problem functions defined as lambdas or closures
    are handled without extra serialisation work.

    Switching to Dask or Ray is a single-parameter change::

        JoblibEvaluator(n_jobs=-1, backend="dask")   # Dask cluster
        JoblibEvaluator(n_jobs=-1, backend="ray")    # Ray cluster

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers.  ``-1`` uses all available CPU cores
        (joblib convention).  ``1`` disables parallelism (equivalent to
        :class:`SerialEvaluator` but with joblib overhead).
    backend : str
        joblib backend name.  ``"loky"`` (default) launches fresh worker
        processes with cloudpickle serialisation.  Other built-in options:
        ``"threading"``, ``"multiprocessing"``.  Third-party backends
        ``"dask"`` and ``"ray"`` require the corresponding packages and a
        running cluster.
    **joblib_kwargs
        Additional keyword arguments forwarded verbatim to
        ``joblib.Parallel``.  Useful for ``verbose``, ``prefer``,
        ``require``, ``timeout``, etc.

    Raises
    ------
    ImportError
        If joblib is not installed.

    Notes
    -----
    **Island-model nested parallelism**: when multiple islands each own a
    ``JoblibEvaluator``, CPU cores may be over-subscribed.  Set
    ``n_jobs=1`` per island evaluator and let the island-level parallelism
    control concurrency, or use ``joblib.parallel_backend`` as a context
    manager to limit inner workers.

    Async evaluation is out of scope for this class.  Asynchronous
    candidate dispatch would require changes to the ``Algorithm.ask`` /
    ``Algorithm.tell`` interface and is tracked separately.

    This evaluator does not use ``Problem.evaluate_batch`` even when a
    ``Problem`` overrides it; combining per-row parallel dispatch (this
    class) with a single vectorized batch call is tracked separately.
    """

    def __init__(
        self,
        n_jobs: int = -1,
        backend: str = "loky",
        **joblib_kwargs: object,
    ) -> None:
        try:
            import joblib as _joblib  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "JoblibEvaluator requires joblib. "
                "Install it with: pip install saealib[parallel]"
            ) from exc
        self._n_jobs = n_jobs
        self._backend = backend
        self._joblib_kwargs = joblib_kwargs

    @property
    def n_jobs(self) -> int:
        """Number of parallel workers (joblib convention; ``-1`` = all cores)."""
        return self._n_jobs

    @property
    def backend(self) -> str:
        """Joblib backend name."""
        return self._backend

    def evaluate_batch(self, x: np.ndarray, problem: Problem) -> EvaluationResult:
        """
        Evaluate candidates in parallel using joblib.

        Parameters
        ----------
        x : np.ndarray
            Design vectors to evaluate.  shape = (n, dim)
        problem : Problem
            The optimization problem providing the objective and constraints.

        Returns
        -------
        EvaluationResult
            Batched objective values, raw constraint values, and violations.
        """
        from joblib import Parallel, delayed

        x = np.atleast_2d(np.asarray(x, dtype=float))
        n = len(x)

        def _eval_one(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
            g_i, cv_i = problem.evaluate_constraints(xi)
            f_i = problem.evaluate(xi, g_i)
            return f_i, g_i, cv_i

        results = Parallel(
            n_jobs=self._n_jobs,
            backend=self._backend,
            **self._joblib_kwargs,
        )(delayed(_eval_one)(x[i]) for i in range(n))

        f = np.array([r[0] for r in results], dtype=float).reshape(n, problem.n_obj)
        g = np.array([r[1] for r in results], dtype=float).reshape(
            n, problem.n_constraints
        )
        cv = np.array([r[2] for r in results], dtype=float)

        return EvaluationResult(f=f, g=g, cv=cv)
