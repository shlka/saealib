"""
Evaluator abstraction.

An ``Evaluator`` is the single entry point through which Strategies and
Initializers turn a batch of design vectors into objective values, raw
constraint values, and aggregate constraint violations. Centralizing
evaluation here enables pluggable execution backends (serial, parallel, ...)
without touching the pipeline code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from saealib.problem import Problem


@dataclass
class EvaluationResult:
    """
    Batched result of evaluating a set of design vectors.

    Attributes
    ----------
    f : np.ndarray
        Objective values. shape = (n, n_obj)
    g : np.ndarray
        Raw constraint values. shape = (n, n_constraints).
        Has shape (n, 0) when the problem defines no constraints.
    cv : np.ndarray
        Aggregate constraint violation per candidate. shape = (n, )
        All zeros when the problem defines no constraints.
    """

    f: np.ndarray
    g: np.ndarray
    cv: np.ndarray


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


class SerialEvaluator(Evaluator):
    """Default evaluator: evaluates each candidate sequentially."""

    def evaluate_batch(self, x: np.ndarray, problem: Problem) -> EvaluationResult:
        """
        Evaluate each row of ``x`` one at a time.

        Produces results equivalent to the per-candidate evaluation loops that
        previously lived in each Strategy and Initializer.

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
