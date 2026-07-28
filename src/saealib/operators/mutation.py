"""Mutation operators for evolutionary algorithms."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from saealib.registry import register

if TYPE_CHECKING:
    from saealib.context import OptimizationState


class Mutation(ABC):
    """Base class for mutation operators.

    Attributes
    ----------
    prob : float
        Individual-level mutation probability. When ``rng.random() >= prob``,
        the individual is returned unchanged.
    prob_var : float or None
        Per-variable mutation probability. ``None`` means the effective value
        is resolved at call time as ``min(0.5, 1 / dim)``.
    """

    prob: float = 1.0
    prob_var: float | None = None

    @abstractmethod
    def mutate(
        self,
        p: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute mutation.

        Parameters
        ----------
        p : np.ndarray
            Parent individual. shape = (dim,)
        mutate_range : tuple
            Tuple of (lower_bound, upper_bound) for mutation.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Mutated individual. shape = (dim,)
        """
        pass

    def mutate_batch(
        self,
        candidates_batch: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray | None:
        """
        Execute mutation on a batch of candidates at once.

        Default implementation returns ``None``, meaning this operator does
        not support batched mutation; the caller should fall back to
        invoking :meth:`mutate` once per candidate.

        Parameters
        ----------
        candidates_batch : np.ndarray
            Batch of candidate individuals. shape = (n, dim)
        mutate_range : tuple
            Tuple of (lower_bound, upper_bound) for mutation.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray or None
            Mutated individuals. shape = (n, dim), or ``None`` if batched
            mutation is unsupported for this particular call/shape.

        Notes
        -----
        Unlike ``Crossover.crossover_batch``, ``prob`` gating is NOT the
        caller's responsibility here — it must be applied by the overriding
        implementation itself, per row, mirroring how every concrete
        :meth:`mutate` self-gates today (``if rng.random() >= self.prob:
        return p.copy()``). Overriding implementations must draw one gate
        value per row and leave ungated rows unchanged, rather than expecting
        the caller to pre-filter ``candidates_batch``. This is a deliberate
        asymmetry with ``Crossover.crossover_batch``.
        """
        return None

    def post_mutation(
        self,
        offspring: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator,
        ctx: OptimizationState | None = None,
    ) -> np.ndarray:
        """Post-mutation lifecycle hook; override to inject custom processing.

        Parameters
        ----------
        offspring : np.ndarray
            Individual after mutation. shape = (dim,)
        mutate_range : tuple
            Tuple of (lower_bound, upper_bound) used for mutation.
        rng : np.random.Generator
            Random number generator.
        ctx : OptimizationState or None, optional
            Current optimization context.

        Returns
        -------
        np.ndarray
            Processed individual. shape = (dim,)
        """
        return offspring

    def with_post(
        self,
        fn: Callable[
            [np.ndarray, tuple, np.random.Generator, OptimizationState | None],
            np.ndarray,
        ],
    ) -> Mutation:
        """Return a copy of this operator with ``fn`` appended to the hook.

        Parameters
        ----------
        fn : callable
            ``fn(offspring, mutate_range, rng, ctx) -> np.ndarray``

        Returns
        -------
        Mutation
            Shallow copy with the hook registered.
        """
        new = copy.copy(self)
        prev = self.post_mutation
        new.post_mutation = lambda offspring, mutate_range, rng, ctx=None: fn(  # type: ignore  # lambda hook; slot type stricter than inferred lambda signature
            prev(offspring, mutate_range, rng, ctx), mutate_range, rng, ctx
        )
        return new


@register()
class MutationUniform(Mutation):
    """
    Uniform mutation operator.

    Attributes
    ----------
    prob : float
        Individual-level mutation probability.
    prob_var : float or None
        Per-variable mutation probability. Defaults to ``min(0.5, 1/dim)``
        when ``None``.
    """

    def __init__(self, prob: float = 1.0, *, prob_var: float | None = None):
        """
        Initialize uniform mutation operator.

        Parameters
        ----------
        prob : float, optional
            Individual-level mutation probability, by default 1.0.
        prob_var : float or None, optional
            Per-variable mutation probability. ``None`` uses ``min(0.5, 1/dim)``.
        """
        super().__init__()
        self.prob = prob
        self.prob_var = prob_var

    def mutate(
        self,
        p: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute uniform mutation.

        Parameters
        ----------
        p : np.ndarray
            Parent individual. shape = (dim,)
        mutate_range : tuple
            Tuple of (lower_bound, upper_bound) for mutation.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Mutated individual.
        """
        if rng.random() >= self.prob:
            return p.copy()
        dim = len(p)
        p_var = self.prob_var if self.prob_var is not None else min(0.5, 1.0 / dim)
        c = p.copy()
        lb, ub = mutate_range
        for i in range(dim):
            if rng.random() < p_var:
                c[i] = rng.uniform(lb[i], ub[i])
        return c

    def mutate_batch(
        self,
        candidates_batch: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray | None:
        """
        Execute uniform mutation on a batch of candidates at once.

        Applies the same two-level gating as :meth:`mutate`, but vectorized:
        an individual-level ``prob`` gate (one Bernoulli draw per row,
        ``rng.random(n) < self.prob``, mirroring ``mutate()``'s
        ``rng.random() >= self.prob`` early return) selects which rows are
        touched at all; a per-dimension ``prob_var`` gate is then drawn only
        for the selected rows, and only dimensions passing that gate are
        replaced with a value drawn ``Uniform(lb[i], ub[i])``. Rows that
        fail the individual-level gate are returned byte-identical to the
        input; dimensions failing the per-dimension gate on a selected row
        keep their original value exactly (``np.where`` copies rather than
        recomputes them). If no row passes the individual-level gate
        (``gate.sum() == 0``), no further random draws happen at all and the
        input is returned unchanged, mirroring
        ``PymooMutation.mutate_batch``'s empty-gate skip.

        Parameters
        ----------
        candidates_batch : np.ndarray
            Batch of candidate individuals. shape = (n, dim)
        mutate_range : tuple
            Tuple of (lower_bound, upper_bound) for mutation.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Mutated individuals. shape = (n, dim)

        Notes
        -----
        Unlike every ``Crossover.crossover_batch`` override (which draws a
        fixed, shape-determined number of random values per row regardless
        of outcome), :meth:`mutate` draws one gate value per dimension and,
        only when that gate passes, a second value for the replacement --
        an interleaved, data-dependent draw count (``dim + k`` draws, where
        ``k`` is however many dimensions happen to pass the gate). A
        vectorized implementation cannot reproduce that sequence: it must
        draw a full ``(dim,)`` (or, batched, ``(k, dim)``) gate array in one
        call and a full replacement-value array in another, since NumPy
        vectorized calls cannot conditionally skip drawing per element. That
        is a different total draw count and a different interleaving than
        ``mutate()``'s, for any ``dim > 1`` and generally even for
        ``dim == 1``. Consequently, this method's output is **not**
        bit-identical to calling :meth:`mutate` once per candidate in a
        loop with the same seeded ``rng``, for any batch size -- not even
        ``n == 1`` -- unlike the ``n_pair == 1`` exact-match guarantee that
        holds for every ``Crossover.crossover_batch`` override. No test
        should assert such equivalence; only the statistical/distributional
        semantics (independent per-dimension replacement at rate
        ``prob_var``, exact pass-through of ungated rows/dimensions) are
        guaranteed to match.

        The output is always ``float64`` (via the internal ``dtype=float``
        cast), regardless of the input array's dtype -- consistent with
        ``PymooMutation.mutate_batch``, but unlike :meth:`mutate`, which
        preserves whatever dtype ``p`` already has.
        """
        candidates_batch = np.asarray(candidates_batch, dtype=float)
        n, dim = candidates_batch.shape
        p_var = self.prob_var if self.prob_var is not None else min(0.5, 1.0 / dim)
        lb, ub = mutate_range
        gate = rng.random(n) < self.prob
        result = candidates_batch.copy()
        if not np.any(gate):
            return result
        sub = candidates_batch[gate]
        k = sub.shape[0]
        var_gate = rng.random((k, dim)) < p_var
        replacement = rng.uniform(lb, ub, size=(k, dim))
        mutated = np.where(var_gate, replacement, sub)
        result[gate] = mutated
        return result


class MutationPolynomial(Mutation):
    """
    Polynomial mutation operator.

    Attributes
    ----------
    prob : float
        Individual-level mutation probability.
    eta : float
        Distribution index. Larger values produce smaller perturbations.
    prob_var : float or None
        Per-variable mutation probability. Defaults to ``min(0.5, 1/dim)``
        when ``None``.

    Notes
    -----
    Originates from Deb & Goyal (1996); the primary paper has not been
    obtained (it is credited here by name only). The asymmetric
    delta1/delta2 perturbation formula implemented in :meth:`mutate` has
    been verified against Deb's own official NSGA-II reference
    implementation (``nsga2-gnuplot-v1.1.6/mutation.c``, function
    ``real_mutate_ind``).
    """

    def __init__(self, prob: float = 1.0, *, eta: float, prob_var: float | None = None):
        """
        Initialize polynomial mutation operator.

        Parameters
        ----------
        prob : float, optional
            Individual-level mutation probability, by default 1.0.
        eta : float
            Distribution index.
        prob_var : float or None, optional
            Per-variable mutation probability. ``None`` uses ``min(0.5, 1/dim)``.
        """
        super().__init__()
        self.prob = prob
        self.eta = eta
        self.prob_var = prob_var

    def mutate(
        self,
        p: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute polynomial mutation.

        Parameters
        ----------
        p : np.ndarray
            Parent individual. shape = (dim,)
        mutate_range : tuple
            Tuple of (lower_bound, upper_bound) for mutation.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Mutated individual.
        """
        if rng.random() >= self.prob:
            return p.copy()
        dim = len(p)
        p_var = self.prob_var if self.prob_var is not None else min(0.5, 1.0 / dim)
        c = p.copy()
        lb, ub = mutate_range
        for i in range(dim):
            if rng.random() < p_var:
                delta1 = (c[i] - lb[i]) / (ub[i] - lb[i])
                delta2 = (ub[i] - c[i]) / (ub[i] - lb[i])
                u = rng.random()
                if u <= 0.5:
                    delta_q = (
                        2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta1) ** (self.eta + 1)
                    ) ** (1.0 / (self.eta + 1)) - 1.0
                else:
                    delta_q = 1.0 - (
                        2.0 * (1.0 - u)
                        + 2.0 * (u - 0.5) * (1.0 - delta2) ** (self.eta + 1)
                    ) ** (1.0 / (self.eta + 1))
                c[i] = np.clip(c[i] + delta_q * (ub[i] - lb[i]), lb[i], ub[i])
        return c

    def mutate_batch(
        self,
        candidates_batch: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray | None:
        """
        Execute polynomial mutation on a batch of candidates at once.

        Applies the same two-level gating as :meth:`mutate`, but vectorized:
        an individual-level ``prob`` gate (one Bernoulli draw per row,
        ``rng.random(n) < self.prob``, mirroring ``mutate()``'s
        ``rng.random() >= self.prob`` early return) selects which rows are
        touched at all; a per-dimension ``prob_var`` gate is then drawn only
        for the selected rows, and only dimensions passing that gate are
        replaced. Unlike :meth:`~MutationUniform.mutate_batch`, the
        replacement value is not a plain uniform draw: it is the asymmetric
        ``delta1``/``delta2`` polynomial perturbation formula (Deb & Goyal,
        1996), which itself branches on a second per-dimension draw ``u``
        (``u <= 0.5`` vs. ``u > 0.5``). That inner branch is vectorized by
        computing both branches (``delta_q_lo``, ``delta_q_hi``) over the
        full ``(k, dim)`` array unconditionally and selecting between them
        with ``np.where(u <= 0.5, delta_q_lo, delta_q_hi)`` -- the same
        "compute both sides, select after" pattern already used by
        :meth:`CrossoverSBX.crossover_batch`'s ``_beta_q`` helper. Rows that
        fail the individual-level gate are returned byte-identical to the
        input; dimensions failing the per-dimension gate on a selected row
        keep their original value exactly (``np.where`` copies rather than
        recomputes them). If no row passes the individual-level gate
        (``gate.sum() == 0``), no further random draws happen at all and the
        input is returned unchanged, mirroring
        ``MutationUniform.mutate_batch``'s empty-gate skip.

        Parameters
        ----------
        candidates_batch : np.ndarray
            Batch of candidate individuals. shape = (n, dim)
        mutate_range : tuple
            Tuple of (lower_bound, upper_bound) for mutation.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Mutated individuals. shape = (n, dim)

        Notes
        -----
        Unlike every ``Crossover.crossover_batch`` override (which draws a
        fixed, shape-determined number of random values per row regardless
        of outcome), :meth:`mutate` draws one gate value per dimension and,
        only when that gate passes, a second value ``u`` used inside the
        polynomial perturbation formula -- an interleaved, data-dependent
        draw count (``dim + k`` draws, where ``k`` is however many
        dimensions happen to pass the gate). A vectorized implementation
        cannot reproduce that sequence: it must draw a full ``(k, dim)``
        gate array and a full ``(k, dim)`` ``u`` array in separate calls,
        since NumPy vectorized calls cannot conditionally skip drawing per
        element. That is a different total draw count and a different
        interleaving than ``mutate()``'s, for any ``dim > 1`` and generally
        even for ``dim == 1``. Consequently, this method's output is
        **not** bit-identical to calling :meth:`mutate` once per candidate
        in a loop with the same seeded ``rng``, for any batch size -- not
        even ``n == 1``. No test should assert such equivalence; only the
        statistical/distributional semantics (independent per-dimension
        mutation at rate ``prob_var`` using the correct polynomial formula,
        exact pass-through of ungated rows/dimensions) are guaranteed to
        match.

        The output is always ``float64`` (via the internal ``dtype=float``
        cast), regardless of the input array's dtype -- consistent with
        :meth:`MutationUniform.mutate_batch`, but unlike :meth:`mutate`,
        which preserves whatever dtype ``p`` already has.
        """
        candidates_batch = np.asarray(candidates_batch, dtype=float)
        n, dim = candidates_batch.shape
        p_var = self.prob_var if self.prob_var is not None else min(0.5, 1.0 / dim)
        lb, ub = mutate_range
        gate = rng.random(n) < self.prob
        result = candidates_batch.copy()
        if not np.any(gate):
            return result
        sub = candidates_batch[gate]
        k = sub.shape[0]
        var_gate = rng.random((k, dim)) < p_var
        delta1 = (sub - lb) / (ub - lb)
        delta2 = (ub - sub) / (ub - lb)
        u = rng.random((k, dim))
        delta_q_lo = (2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta1) ** (self.eta + 1)) ** (
            1.0 / (self.eta + 1)
        ) - 1.0
        delta_q_hi = 1.0 - (
            2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - delta2) ** (self.eta + 1)
        ) ** (1.0 / (self.eta + 1))
        delta_q = np.where(u <= 0.5, delta_q_lo, delta_q_hi)
        mutated = np.clip(sub + delta_q * (ub - lb), lb, ub)
        result_sub = np.where(var_gate, mutated, sub)
        result[gate] = result_sub
        return result


class MutationGaussian(Mutation):
    """
    Gaussian mutation operator.

    Attributes
    ----------
    prob : float
        Individual-level mutation probability.
    sigma : float
        Standard deviation of the Gaussian perturbation.
    prob_var : float or None
        Per-variable mutation probability. Defaults to ``min(0.5, 1/dim)``
        when ``None``.

    Notes
    -----
    Originates from Rechenberg (1973); the original (German-language)
    monograph has not been obtained and is credited here by name only.
    Section 2 of the survey below summarizes Rechenberg's (1+1)-ES with
    Gaussian mutation and its mutation-strength (standard deviation)
    parameterization, consistent with this operator.

    References
    ----------
    :cite:`beyer2002essurvey`: Beyer, H.-G., & Schwefel, H.-P. (2002).
    Evolution strategies -- A comprehensive introduction. *Natural
    Computing*, 1, 3-52.
    """

    def __init__(
        self, prob: float = 1.0, *, sigma: float, prob_var: float | None = None
    ):
        """
        Initialize Gaussian mutation operator.

        Parameters
        ----------
        prob : float, optional
            Individual-level mutation probability, by default 1.0.
        sigma : float
            Standard deviation of the Gaussian perturbation.
        prob_var : float or None, optional
            Per-variable mutation probability. ``None`` uses ``min(0.5, 1/dim)``.
        """
        super().__init__()
        self.prob = prob
        self.sigma = sigma
        self.prob_var = prob_var

    def mutate(
        self,
        p: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute Gaussian mutation.

        Parameters
        ----------
        p : np.ndarray
            Parent individual. shape = (dim,)
        mutate_range : tuple
            Tuple of (lower_bound, upper_bound) for mutation.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Mutated individual.
        """
        if rng.random() >= self.prob:
            return p.copy()
        dim = len(p)
        p_var = self.prob_var if self.prob_var is not None else min(0.5, 1.0 / dim)
        c = p.copy()
        for i in range(dim):
            if rng.random() < p_var:
                c[i] = c[i] + rng.normal(0.0, self.sigma)
        return c

    def mutate_batch(
        self,
        candidates_batch: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray | None:
        """
        Execute Gaussian mutation on a batch of candidates at once.

        Applies the same two-level gating as :meth:`mutate`, but vectorized:
        an individual-level ``prob`` gate (one Bernoulli draw per row,
        ``rng.random(n) < self.prob``, mirroring ``mutate()``'s
        ``rng.random() >= self.prob`` early return) selects which rows are
        touched at all; a per-dimension ``prob_var`` gate is then drawn only
        for the selected rows, and only dimensions passing that gate are
        replaced by adding ``Normal(0.0, self.sigma)`` noise. Unlike
        :meth:`~MutationUniform.mutate_batch` and
        :meth:`~MutationPolynomial.mutate_batch`, the replacement value has
        no branching formula -- it is simply ``sub + noise`` -- so this is
        the simplest of the three ``mutate_batch`` overrides. Rows that fail
        the individual-level gate are returned byte-identical to the input;
        dimensions failing the per-dimension gate on a selected row keep
        their original value exactly (``np.where`` copies rather than
        recomputes them). If no row passes the individual-level gate
        (``gate.sum() == 0``), no further random draws happen at all and the
        input is returned unchanged, mirroring
        ``MutationUniform.mutate_batch``'s empty-gate skip.

        Parameters
        ----------
        candidates_batch : np.ndarray
            Batch of candidate individuals. shape = (n, dim)
        mutate_range : tuple
            Tuple of (lower_bound, upper_bound). Accepted for interface
            uniformity with the ``Mutation`` ABC and the other
            ``mutate_batch`` overrides, but **unused** -- exactly like
            :meth:`mutate`, this operator adds unbounded Gaussian noise with
            no clipping to ``[lb, ub]``. Do not add a ``np.clip`` here; that
            would be a silent behavior change relative to :meth:`mutate`,
            not a faithful vectorization of it.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Mutated individuals. shape = (n, dim)

        Notes
        -----
        Unlike every ``Crossover.crossover_batch`` override (which draws a
        fixed, shape-determined number of random values per row regardless
        of outcome), :meth:`mutate` draws one gate value per dimension and,
        only when that gate passes, a second value for the noise -- an
        interleaved, data-dependent draw count (``dim + k`` draws, where
        ``k`` is however many dimensions happen to pass the gate). A
        vectorized implementation cannot reproduce that sequence: it must
        draw a full ``(k, dim)`` gate array and a full ``(k, dim)`` noise
        array in separate calls, since NumPy vectorized calls cannot
        conditionally skip drawing per element. That is a different total
        draw count and a different interleaving than ``mutate()``'s, for any
        ``dim > 1`` and generally even for ``dim == 1``. Consequently, this
        method's output is **not** bit-identical to calling :meth:`mutate`
        once per candidate in a loop with the same seeded ``rng``, for any
        batch size -- not even ``n == 1``. No test should assert such
        equivalence; only the statistical/distributional semantics
        (independent per-dimension noise addition at rate ``prob_var``,
        exact pass-through of ungated rows/dimensions) are guaranteed to
        match.

        The output is always ``float64`` (via the internal ``dtype=float``
        cast), regardless of the input array's dtype -- consistent with the
        other two ``mutate_batch`` overrides, but unlike :meth:`mutate`,
        which preserves whatever dtype ``p`` already has.
        """
        candidates_batch = np.asarray(candidates_batch, dtype=float)
        n, dim = candidates_batch.shape
        p_var = self.prob_var if self.prob_var is not None else min(0.5, 1.0 / dim)
        gate = rng.random(n) < self.prob
        result = candidates_batch.copy()
        if not np.any(gate):
            return result
        sub = candidates_batch[gate]
        k = sub.shape[0]
        var_gate = rng.random((k, dim)) < p_var
        noise = rng.normal(0.0, self.sigma, size=(k, dim))
        mutated = sub + noise
        result_sub = np.where(var_gate, mutated, sub)
        result[gate] = result_sub
        return result


class _MutationDiscreteUniform(Mutation):
    """Shared implementation for discrete uniform mutation.

    Replaces each dimension's value with a uniform random integer draw
    from ``[lb[i], ub[i]]`` (both inclusive).
    """

    def __init__(self, prob: float = 1.0, *, prob_var: float | None = None):
        super().__init__()
        self.prob = prob
        self.prob_var = prob_var

    def mutate(
        self,
        p: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute discrete uniform mutation.

        Parameters
        ----------
        p : np.ndarray
            Parent individual. shape = (dim,)
        mutate_range : tuple
            Tuple of (lower_bound, upper_bound) arrays.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Mutated individual.
        """
        if rng.random() >= self.prob:
            return p.copy()
        dim = len(p)
        p_var = self.prob_var if self.prob_var is not None else min(0.5, 1.0 / dim)
        c = p.copy()
        lb, ub = mutate_range
        for i in range(dim):
            if rng.random() < p_var:
                c[i] = float(rng.integers(int(lb[i]), int(ub[i]) + 1))
        return c

    def mutate_batch(
        self,
        candidates_batch: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray | None:
        """
        Execute discrete uniform mutation on a batch of candidates at once.

        Applies the same two-level gating as :meth:`mutate`, but vectorized:
        an individual-level ``prob`` gate (one Bernoulli draw per row,
        ``rng.random(n) < self.prob``, mirroring ``mutate()``'s
        ``rng.random() >= self.prob`` early return) selects which rows are
        touched at all; a per-dimension ``prob_var`` gate is then drawn only
        for the selected rows, and only dimensions passing that gate are
        replaced with a value drawn as a uniform random integer from
        ``[lb[i], ub[i]]`` (both inclusive) via
        ``rng.integers(lb, ub + 1, size=(k, dim))`` -- ``rng.integers``'s
        ``high`` argument is exclusive by default, so ``ub + 1`` reproduces
        the scalar :meth:`mutate`'s inclusive upper bound (``int(ub[i]) +
        1``). The integer draw is cast to ``float`` via ``.astype(float)``,
        consistent with :meth:`mutate`'s own ``float(rng.integers(...))``
        cast -- the whole codebase represents integer/categorical dimensions
        as floats internally. Rows that fail the individual-level gate are
        returned byte-identical to the input; dimensions failing the
        per-dimension gate on a selected row keep their original value
        exactly (``np.where`` copies rather than recomputes them). If no row
        passes the individual-level gate (``gate.sum() == 0``), no further
        random draws happen at all and the input is returned unchanged,
        mirroring ``MutationUniform.mutate_batch``'s empty-gate skip.

        Parameters
        ----------
        candidates_batch : np.ndarray
            Batch of candidate individuals. shape = (n, dim)
        mutate_range : tuple
            Tuple of (lower_bound, upper_bound) arrays.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Mutated individuals. shape = (n, dim)

        Notes
        -----
        Unlike every ``Crossover.crossover_batch`` override (which draws a
        fixed, shape-determined number of random values per row regardless
        of outcome), :meth:`mutate` draws one gate value per dimension and,
        only when that gate passes, a second value for the integer
        replacement -- an interleaved, data-dependent draw count (``dim +
        k`` draws, where ``k`` is however many dimensions happen to pass the
        gate). A vectorized implementation cannot reproduce that sequence:
        it must draw a full ``(k, dim)`` gate array and a full ``(k, dim)``
        replacement array in separate calls, since NumPy vectorized calls
        cannot conditionally skip drawing per element. That is a different
        total draw count and a different interleaving than ``mutate()``'s,
        for any ``dim > 1`` and generally even for ``dim == 1``.
        Consequently, this method's output is **not** bit-identical to
        calling :meth:`mutate` once per candidate in a loop with the same
        seeded ``rng``, for any batch size -- not even ``n == 1``. No test
        should assert such equivalence; only the statistical/distributional
        semantics (independent per-dimension replacement at rate
        ``prob_var``, exact pass-through of ungated rows/dimensions) are
        guaranteed to match.

        The output is always ``float64`` (via the internal ``dtype=float``
        cast), regardless of the input array's dtype -- consistent with the
        other ``mutate_batch`` overrides, but unlike :meth:`mutate`, which
        preserves whatever dtype ``p`` already has.
        """
        candidates_batch = np.asarray(candidates_batch, dtype=float)
        n, dim = candidates_batch.shape
        p_var = self.prob_var if self.prob_var is not None else min(0.5, 1.0 / dim)
        lb, ub = mutate_range
        gate = rng.random(n) < self.prob
        result = candidates_batch.copy()
        if not np.any(gate):
            return result
        sub = candidates_batch[gate]
        k = sub.shape[0]
        var_gate = rng.random((k, dim)) < p_var
        lb_int = np.asarray(lb, dtype=int)
        ub_int = np.asarray(ub, dtype=int)
        replacement = rng.integers(lb_int, ub_int + 1, size=(k, dim)).astype(float)
        mutated = np.where(var_gate, replacement, sub)
        result[gate] = mutated
        return result


class MutationIntegerUniform(_MutationDiscreteUniform):
    """
    Uniform integer mutation.

    Replaces each dimension's value with a uniform random integer draw
    from ``[lb[i], ub[i]]`` (both inclusive).

    Attributes
    ----------
    prob : float
        Individual-level mutation probability.
    prob_var : float or None
        Per-variable mutation probability. Defaults to ``min(0.5, 1/dim)``
        when ``None``.
    """

    def __init__(self, prob: float = 1.0, *, prob_var: float | None = None):
        """
        Initialize integer uniform mutation operator.

        Parameters
        ----------
        prob : float, optional
            Individual-level mutation probability, by default 1.0.
        prob_var : float or None, optional
            Per-variable mutation probability. ``None`` uses ``min(0.5, 1/dim)``.
        """
        super().__init__(prob, prob_var=prob_var)


class MutationCategorical(_MutationDiscreteUniform):
    """
    Uniform categorical mutation.

    Replaces each dimension's category index with a uniform random draw
    from ``{0, 1, ..., n_categories - 1}``.  The valid range is inferred
    from ``mutate_range``, where ``ub[i] == n_categories - 1``.

    Attributes
    ----------
    prob : float
        Individual-level mutation probability.
    prob_var : float or None
        Per-variable mutation probability. Defaults to ``min(0.5, 1/dim)``
        when ``None``.
    """

    def __init__(self, prob: float = 1.0, *, prob_var: float | None = None):
        """
        Initialize categorical mutation operator.

        Parameters
        ----------
        prob : float, optional
            Individual-level mutation probability, by default 1.0.
        prob_var : float or None, optional
            Per-variable mutation probability. ``None`` uses ``min(0.5, 1/dim)``.
        """
        super().__init__(prob, prob_var=prob_var)
