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
        this class applies the gate itself (saealib's ``GA`` calls
        ``mutate()`` unconditionally; each concrete ``Mutation`` owns its
        own probability check). Defaults to 1.0.

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

    Like :class:`PymooCrossover`, this calls the wrapped operator's ``_do()``
    once per individual (saealib's per-individual calling convention), pays
    the corresponding per-call overhead, and forwards ``rng`` via pymoo's
    ``random_state`` parameter for reproducibility. A minimal cached pymoo
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

    def mutate(
        self,
        p: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute mutation via the wrapped pymoo operator.

        Parameters
        ----------
        p : np.ndarray
            Parent individual. shape = (dim,)
        mutate_range : tuple
            Tuple of (lower_bound, upper_bound) for mutation.
        rng : np.random.Generator, optional
            Forwarded to the wrapped operator as ``random_state``.

        Returns
        -------
        np.ndarray
            Mutated individual. shape = (dim,)
        """
        if rng.random() >= self.prob:
            return p.copy()
        problem = self._pymoo_problem(p.shape[0], mutate_range)
        x_batch = np.asarray(p, dtype=float)[np.newaxis, :]
        xp_batch = self.operator._do(problem, x_batch, random_state=rng)
        return np.asarray(xp_batch, dtype=float)[0]
