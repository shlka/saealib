"""Adapter exposing a constructed pymoo crossover operator as saealib's Crossover."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from saealib.operators.crossover import Crossover


class _PymooVariableLike(Protocol):
    """Structural interface of a ``pymoo.core.variable.Variable`` (e.g. ``Real``)."""

    # pymoo's own stub types Variable.value as `object` (Variable.__init__'s
    # `value: Optional[object]` parameter), even though it always holds a
    # float at runtime for Real. Any is the honest type here, not a
    # loosened-for-convenience shortcut.
    value: Any


class _PymooCrossoverLike(Protocol):
    """Structural interface of a ``pymoo.core.crossover.Crossover`` instance."""

    n_parents: int
    n_offsprings: int

    @property
    def prob(self) -> _PymooVariableLike:
        """Individual-level crossover probability, wrapped as a pymoo Variable."""
        ...

    def _do(
        self,
        problem: object,
        x: np.ndarray,
        *args: object,
        random_state: object = None,
        **kwargs: object,
    ) -> np.ndarray:
        """Apply crossover. (n_parents, n_matings, dim) -> (n_offsprings, ..., dim)."""
        ...


class PymooCrossover(Crossover):
    """
    Adapter wrapping a pymoo crossover operator as a saealib ``Crossover``.

    Lets researchers who already have a pymoo ``Crossover`` (e.g. ``SBX()``,
    ``PCX()``) reuse it unchanged inside saealib's ``GA``.

    Parameters
    ----------
    operator : pymoo.core.crossover.Crossover
        An already-constructed pymoo crossover operator instance.
    prob : float, optional
        Individual-level crossover probability, gated by saealib's ``GA``
        before ``crossover()`` is even called. Defaults to the wrapped
        operator's own ``prob.value`` (pymoo wraps it as a
        ``pymoo.core.variable.Real``).
    n_parents : int, optional
        Defaults to ``operator.n_parents``.
    n_children : int, optional
        Defaults to ``operator.n_offsprings``.

    Notes
    -----
    ``crossover_batch()`` calls the wrapped operator's ``_do()`` (not
    ``.do()``, which would apply pymoo's own ``prob`` gate a second time on
    top of saealib's). The inherited ``crossover()`` derives a single-parent-
    group call from that batch implementation.

    A minimal internal ``pymoo.core.problem.Problem`` is synthesized to satisfy
    operators (e.g. SBX) that read ``problem.xl``/``problem.xu``/``problem.n_var``;
    it is cached and rebuilt only when ``dim`` or ``bounds`` changes. Most pymoo
    crossovers unconditionally require finite ``xl``/``xu`` and raise their own
    (pymoo-internal) error when ``bounds=None`` is passed through; saealib's own
    ``GA`` always supplies bounds, so this only matters for direct calls.

    ``rng`` is forwarded to the wrapped operator via pymoo's own ``random_state``
    parameter, so results are reproducible under saealib's seeding.
    """

    def __init__(
        self,
        operator: _PymooCrossoverLike,
        *,
        prob: float | None = None,
        n_parents: int | None = None,
        n_children: int | None = None,
    ) -> None:
        super().__init__()
        self.operator = operator
        self.n_parents = int(n_parents if n_parents is not None else operator.n_parents)
        self.n_children = int(
            n_children if n_children is not None else operator.n_offsprings
        )
        self.prob = float(prob if prob is not None else operator.prob.value)
        self._problem: object | None = None
        self._problem_key: tuple[int, bytes, bytes] | None = None

    def _pymoo_problem(
        self, dim: int, bounds: tuple[np.ndarray, np.ndarray] | None
    ) -> object:
        """Return a cached pymoo ``Problem`` shim, rebuilt when dim/bounds change."""
        from pymoo.core.problem import Problem as PymooProblem

        if bounds is not None:
            lb = np.asarray(bounds[0], dtype=float)
            ub = np.asarray(bounds[1], dtype=float)
            key = (dim, lb.tobytes(), ub.tobytes())
        else:
            lb = ub = None
            key = (dim, b"", b"")

        if key != self._problem_key:
            self._problem = PymooProblem(n_var=dim, n_obj=1, xl=lb, xu=ub)
            self._problem_key = key
        return self._problem

    def crossover_batch(
        self,
        parents_batch: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute crossover on a batch of parent pairs via a single pymoo call.

        This is the adapter's primary crossover implementation and calls the
        wrapped operator's ``_do()`` exactly once for the whole batch.

        Parameters
        ----------
        parents_batch : np.ndarray
            Batch of parent groups. shape = (n_pair, n_parents, dim)
        bounds : tuple of (np.ndarray, np.ndarray) or None
            Lower and upper bounds for each variable.
        rng : np.random.Generator, optional
            Forwarded to the wrapped operator as ``random_state``.

        Returns
        -------
        np.ndarray
            Offspring individuals. shape = (n_pair, n_children, dim)

        Notes
        -----
        For ``n_pair > 1``, a loop of separate single-pair
        ``crossover_batch`` calls (equivalently, separate :meth:`crossover`
        calls) does not reproduce the same per-row results as one batched
        call with the same seeded ``rng``. Operators such as pymoo's SBX draw
        each random phase across the whole batch, whereas single-pair calls
        interleave phases per row.
        """
        problem = self._pymoo_problem(parents_batch.shape[-1], bounds)
        # saealib's (n_pair, n_parents, dim) vs. pymoo's own
        # (n_parents, n_matings, dim) convention: swap the first two axes
        # (a view, no copy) rather than reshaping.
        x_pymoo_layout = np.swapaxes(np.asarray(parents_batch, dtype=float), 0, 1)
        q_pymoo_layout = self.operator._do(problem, x_pymoo_layout, random_state=rng)
        return np.asarray(np.swapaxes(q_pymoo_layout, 0, 1), dtype=float)
