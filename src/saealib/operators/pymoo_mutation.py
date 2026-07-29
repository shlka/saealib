"""Adapter exposing a constructed pymoo mutation operator as saealib's Mutation."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from saealib.operators.mutation import Mutation


class _PymooMutationLike(Protocol):
    """Structural interface of a ``pymoo.core.mutation.Mutation`` instance."""

    def _do(
        self,
        problem: object,
        x: np.ndarray,
        *args: object,
        random_state: object = None,
        **kwargs: object,
    ) -> np.ndarray:
        """Apply the mutation to a batch of individuals, (n, dim) -> (n, dim)."""
        ...


class PymooMutation(Mutation):
    """
    Adapter wrapping a pymoo mutation operator as a saealib ``Mutation``.

    Lets researchers who already have a pymoo ``Mutation`` (e.g. ``PM()``)
    reuse it unchanged inside saealib's ``GA``.

    Parameters
    ----------
    operator : pymoo.core.mutation.Mutation
        An already-constructed pymoo mutation operator instance.
    prob : float, optional
        Individual-level mutation probability. Unlike ``PymooCrossover``,
        this class applies the gate itself inside ``mutate_batch()``.
        Defaults to 1.0.

    Notes
    -----
    ``prob_var`` (pymoo's per-variable mutation probability) is deliberately
    **not** mirrored from the wrapped operator: pymoo stores it as a
    ``pymoo.core.variable.Real`` or ``None``, and the wrapped operator's
    ``_do()`` resolves its own per-variable rate internally regardless of
    what saealib's ``prob_var`` attribute says. ``self.prob_var`` stays
    ``None`` here so that any saealib code reading it (e.g. mixed-variable
    routing in ``GA``) falls back to its own default rather than seeing a
    foreign, potentially non-``float`` value.

    ``mutate_batch()`` calls the wrapped operator's ``_do()`` at most once,
    on the gated subset, and forwards ``rng`` via pymoo's ``random_state``
    parameter for reproducibility. The inherited ``mutate()`` derives a
    single-row call from that batch implementation. A minimal cached pymoo
    ``Problem`` shim is synthesized the same way.
    """

    def __init__(self, operator: _PymooMutationLike, *, prob: float = 1.0) -> None:
        super().__init__()
        self.operator = operator
        self.prob = prob
        self.prob_var = None
        self._problem: object | None = None
        self._problem_key: tuple[int, bytes, bytes] | None = None

    def _pymoo_problem(self, dim: int, mutate_range: tuple) -> object:
        """Return a cached pymoo ``Problem`` shim, rebuilt when dim/bounds change."""
        from pymoo.core.problem import Problem as PymooProblem

        lb = np.asarray(mutate_range[0], dtype=float)
        ub = np.asarray(mutate_range[1], dtype=float)
        key = (dim, lb.tobytes(), ub.tobytes())

        if key != self._problem_key:
            self._problem = PymooProblem(n_var=dim, n_obj=1, xl=lb, xu=ub)
            self._problem_key = key
        return self._problem

    def mutate_batch(
        self,
        candidates_batch: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute mutation on a batch of candidates via a single pymoo call.

        Unlike ``Crossover.crossover_batch``, this method owns its own
        ``prob`` gate: it draws one gate value per row (``rng.random(n) <
        self.prob``, mirroring ``mutate()``'s per-call ``rng.random() >=
        self.prob`` check) and only mutates the gated rows; ungated rows are
        returned unchanged. The wrapped operator's ``_do()`` is called at
        most once, on only the gated subset. If no row is gated
        (``gate.sum() == 0``), ``_do()`` is not called at all, avoiding a
        zero-length-batch call into the wrapped pymoo operator.

        Parameters
        ----------
        candidates_batch : np.ndarray
            Batch of candidate individuals. shape = (n, dim)
        mutate_range : tuple
            Tuple of (lower_bound, upper_bound) for mutation.
        rng : np.random.Generator, optional
            Forwarded to the wrapped operator as ``random_state``.

        Returns
        -------
        np.ndarray
            Mutated individuals. shape = (n, dim)

        Notes
        -----
        For more than one gated row, a loop of separate single-row
        ``mutate_batch`` calls (equivalently, separate :meth:`mutate` calls)
        does not reproduce the same per-row results as one batched call with
        the same seeded ``rng``. Pymoo operators such as PM draw each random
        phase across the whole gated batch, whereas single-row calls
        interleave phases per row.
        """
        candidates_batch = np.asarray(candidates_batch, dtype=float)
        n = candidates_batch.shape[0]
        gate = rng.random(n) < self.prob
        result = candidates_batch.copy()
        if not np.any(gate):
            return result
        problem = self._pymoo_problem(candidates_batch.shape[-1], mutate_range)
        mutated = self.operator._do(problem, candidates_batch[gate], random_state=rng)
        result[gate] = np.asarray(mutated, dtype=float)
        return result
