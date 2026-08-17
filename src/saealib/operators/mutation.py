"""Mutation operators for evolutionary algorithms."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import Self

from saealib.core.contracts import (
    MANY,
    ComponentContract,
    DataSpec,
    Fixed,
    PortContract,
    PortDirection,
    PortSpec,
    ServiceRequirement,
    Var,
)
from saealib.registry import register

if TYPE_CHECKING:
    from saealib.core.state import ExecutionContext


class Mutation(ABC):
    """Base class for mutation operators.

    Attributes
    ----------
    prob : float
        Individual-level mutation probability. Rows whose gate draw is
        greater than or equal to ``prob`` are returned unchanged.
    prob_var : float or None
        Per-variable mutation probability. ``None`` means the effective value
        is resolved at call time as ``min(0.5, 1 / dim)``.
    """

    prob: float = 1.0
    prob_var: float | None = None

    def contract(self) -> ComponentContract:
        """Return the mutation contract; mutate_range is required for every mutation."""
        representation = Var(name="R")
        candidate_count = Var(name="N")
        data = DataSpec(
            kind="GenomeBatch",
            bindings={
                "representation": representation,
                "candidate_count": candidate_count,
            },
        )
        bounds = (ServiceRequirement(name="BoundsService"),)
        return ComponentContract(
            ports={
                "mutation": PortContract(
                    inputs=(
                        PortSpec(
                            name="candidates",
                            direction=PortDirection.INPUT,
                            data=data,
                            cardinality=MANY,
                            required_services=bounds,
                        ),
                    ),
                    outputs=(
                        PortSpec(
                            name="mutants",
                            direction=PortDirection.OUTPUT,
                            data=data,
                            cardinality=MANY,
                        ),
                    ),
                )
            }
        )

    def mutate(
        self,
        p: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute mutation for one candidate.

        This method is derived from :meth:`mutate_batch` by passing a
        single-row batch.

        Notes
        -----
        Overriding only this method (not :meth:`mutate_batch`) has no effect
        under GA's default ``variation_execution="batch"``: the batch path
        calls :meth:`mutate_batch` directly and never calls ``mutate()``.
        Such an override only takes effect under
        ``variation_execution="sequential"``. To customize behavior under
        both modes, override :meth:`mutate_batch`.

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
            Mutated individual. shape = (dim,). The result is always
            ``float64`` because :meth:`mutate_batch` casts its input
            internally; the result does not preserve the input's dtype.
        """
        return self.mutate_batch(p[np.newaxis, :], mutate_range, rng)[0]

    @abstractmethod
    def mutate_batch(
        self,
        candidates_batch: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute mutation on a batch of candidates at once.

        This is the required primitive that every concrete mutation class
        must implement directly. Each implementation self-gates ``prob``
        internally; see Notes for why this differs from
        ``Crossover.crossover_batch``.

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
        Unlike ``Crossover.crossover_batch``, ``prob`` gating is NOT the
        caller's responsibility here — it must be applied by the overriding
        implementation itself, per row. Implementations must draw one gate
        value per row and leave ungated rows unchanged, rather than expecting
        the caller to pre-filter ``candidates_batch``. This is a deliberate
        asymmetry with ``Crossover.crossover_batch``.
        """
        pass

    def post_mutation(
        self,
        offspring: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator,
        ctx: ExecutionContext | None = None,
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
        ctx : ExecutionContext or None, optional
            Current optimization context.

        Returns
        -------
        np.ndarray
            Processed individual. shape = (dim,)
        """
        return offspring

    def post_mutation_batch(
        self,
        offspring_batch: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator,
        ctx: ExecutionContext | None = None,
    ) -> np.ndarray:
        """Run the post-mutation lifecycle hook for a batch.

        The default implementation calls :meth:`post_mutation` once per
        individual, in order. Override this method to provide genuinely
        vectorized post-processing.

        Parameters
        ----------
        offspring_batch : np.ndarray
            Offspring produced by mutation. shape = (n, dim)
        mutate_range : tuple
            Tuple of (lower_bound, upper_bound) used for mutation.
        rng : np.random.Generator
            Random number generator.
        ctx : ExecutionContext or None, optional
            Current optimization context.

        Returns
        -------
        np.ndarray
            Processed offspring. shape = (n, dim)

        Notes
        -----
        :meth:`with_post` reassigns only the instance's
        :meth:`post_mutation` hook. A subclass override of this method is
        responsible for composing that hook itself. If the override does not
        call ``self.post_mutation``, a hook installed with :meth:`with_post`
        will not run in GA batch mode. It still runs in GA sequential mode,
        which calls :meth:`post_mutation` directly.
        """
        result = np.empty_like(offspring_batch)
        for i in range(len(offspring_batch)):
            result[i] = self.post_mutation(offspring_batch[i], mutate_range, rng, ctx)
        return result

    def with_post(
        self,
        fn: Callable[
            [np.ndarray, tuple, np.random.Generator, ExecutionContext | None],
            np.ndarray,
        ],
    ) -> Self:
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

    def contract(self) -> ComponentContract:
        """Return the real-valued uniform mutation contract."""
        base = super().contract()
        role = base.ports["mutation"]
        input_data = role.inputs[0].data
        output_data = role.outputs[0].data
        input_bindings = dict(input_data.bindings)
        output_bindings = dict(output_data.bindings)
        input_bindings["representation"] = Fixed(value="real")
        output_bindings["representation"] = Fixed(value="real")
        return replace(
            base,
            ports={
                "mutation": replace(
                    role,
                    inputs=(
                        replace(
                            role.inputs[0],
                            data=replace(input_data, bindings=input_bindings),
                        ),
                    ),
                    outputs=(
                        replace(
                            role.outputs[0],
                            data=replace(output_data, bindings=output_bindings),
                        ),
                    ),
                )
            },
        )

    def mutate_batch(
        self,
        candidates_batch: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute uniform mutation on a batch of candidates at once.

        This is the primary uniform mutation implementation. An
        individual-level ``prob`` gate (one Bernoulli draw per row via
        ``rng.random(n) < self.prob``) selects which rows are touched at all;
        a per-dimension ``prob_var`` gate is then drawn only for the selected
        rows, and only dimensions passing that gate are
        replaced with a value drawn ``Uniform(lb[i], ub[i])``. The
        replacement is drawn only for the positions ``var_gate`` actually
        selects (via boolean fancy-indexing into the broadcast ``lb``/``ub``
        arrays, then ``Generator.uniform``'s array-valued ``low``/``high``
        form with no explicit ``size``), rather than for every ``(row, dim)``
        position unconditionally -- this avoids drawing (and validating the
        range of) a replacement for a dimension that ``var_gate`` will
        discard anyway, which matters when that dimension is unbounded (see
        Notes). Rows that fail the individual-level gate are returned
        byte-identical to the input; dimensions failing the per-dimension
        gate on a selected row keep their original value exactly. If no row
        passes the individual-level gate (``gate.sum() == 0``), no further
        random draws happen at all and the input is returned unchanged,
        mirroring ``PymooMutation.mutate_batch``'s empty-gate skip.

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
        For multiple rows, a loop of separate single-row ``mutate_batch``
        calls (equivalently, separate :meth:`mutate` calls) does not
        reproduce the same per-row results as one batched call with the same
        seeded ``rng``. Separate calls interleave each row's individual
        gate, full per-dimension gate array, and replacement draws before
        moving to the next row. The number of replacements is
        data-dependent (``dim + m`` post-individual-gate draws, where ``m``
        dimensions pass the variable gate). One batched call instead draws
        all individual gates, then a full ``(k, dim)`` variable-gate array
        for the ``k`` selected rows, followed by one replacement draw sized
        to all selected positions. NumPy vectorized calls cannot
        conditionally skip drawing per element within a single call, so the
        draw count and interleaving generally diverge. Only the statistical
        and distributional semantics are guaranteed to match.

        The output is always ``float64`` (via the internal ``dtype=float``
        cast), regardless of the input array's dtype.
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
        mutated = sub.copy()
        if np.any(var_gate):
            # Draw a replacement only for the positions var_gate actually
            # selects, rather than for every (row, dim) position
            # unconditionally: rng.uniform validates the range of every
            # position it draws for, even ones that would be discarded by
            # np.where afterwards, so an unfiltered draw raises OverflowError
            # as soon as any unbounded (lb=-inf/ub=inf) dimension is present
            # anywhere in the batch -- regardless of whether that dimension
            # is ever actually gated in.
            lb_b = np.broadcast_to(lb, (k, dim))
            ub_b = np.broadcast_to(ub, (k, dim))
            mutated[var_gate] = rng.uniform(lb_b[var_gate], ub_b[var_gate])
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
    delta1/delta2 perturbation formula implemented in
    :meth:`mutate_batch` has been verified against Deb's own official
    NSGA-II reference implementation
    (``nsga2-gnuplot-v1.1.6/mutation.c``, function ``real_mutate_ind``).
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

    def contract(self) -> ComponentContract:
        """Return the real-valued polynomial mutation contract."""
        base = super().contract()
        role = base.ports["mutation"]
        input_data = role.inputs[0].data
        output_data = role.outputs[0].data
        input_bindings = dict(input_data.bindings)
        output_bindings = dict(output_data.bindings)
        input_bindings["representation"] = Fixed(value="real")
        output_bindings["representation"] = Fixed(value="real")
        return replace(
            base,
            ports={
                "mutation": replace(
                    role,
                    inputs=(
                        replace(
                            role.inputs[0],
                            data=replace(input_data, bindings=input_bindings),
                        ),
                    ),
                    outputs=(
                        replace(
                            role.outputs[0],
                            data=replace(output_data, bindings=output_bindings),
                        ),
                    ),
                )
            },
        )

    def mutate_batch(
        self,
        candidates_batch: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute polynomial mutation on a batch of candidates at once.

        This is the primary polynomial mutation implementation. An
        individual-level ``prob`` gate (one Bernoulli draw per row via
        ``rng.random(n) < self.prob``) selects which rows are touched at all;
        a per-dimension ``prob_var`` gate is then drawn only for the selected
        rows, and only dimensions passing that gate are
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
        For multiple rows, a loop of separate single-row ``mutate_batch``
        calls (equivalently, separate :meth:`mutate` calls) does not
        reproduce the same per-row results as one batched call with the same
        seeded ``rng``. Separate calls interleave each row's individual
        gate, full per-dimension gate array, and full ``u`` array before
        moving to the next row. One batched call instead draws all
        individual gates, then the variable-gate arrays for all gated rows,
        then the ``u`` arrays for all of them. These phases therefore have a
        different draw order whenever more than one row is involved. Only
        the statistical and distributional semantics are guaranteed to
        match.

        The output is always ``float64`` (via the internal ``dtype=float``
        cast), regardless of the input array's dtype.
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
    The survey below summarizes Rechenberg's (1+1)-ES with
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

    def contract(self) -> ComponentContract:
        """Return the real-valued Gaussian mutation contract."""
        base = super().contract()
        role = base.ports["mutation"]
        input_data = role.inputs[0].data
        output_data = role.outputs[0].data
        input_bindings = dict(input_data.bindings)
        output_bindings = dict(output_data.bindings)
        input_bindings["representation"] = Fixed(value="real")
        output_bindings["representation"] = Fixed(value="real")
        return replace(
            base,
            ports={
                "mutation": replace(
                    role,
                    inputs=(
                        replace(
                            role.inputs[0],
                            data=replace(input_data, bindings=input_bindings),
                            required_services=(),
                        ),
                    ),
                    outputs=(
                        replace(
                            role.outputs[0],
                            data=replace(output_data, bindings=output_bindings),
                        ),
                    ),
                )
            },
        )

    def mutate_batch(
        self,
        candidates_batch: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute Gaussian mutation on a batch of candidates at once.

        This is the primary Gaussian mutation implementation. An
        individual-level ``prob`` gate (one Bernoulli draw per row via
        ``rng.random(n) < self.prob``) selects which rows are touched at all;
        a per-dimension ``prob_var`` gate is then drawn only for the selected
        rows, and only dimensions passing that gate are
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
            ``mutate_batch`` overrides, but **unused**. This operator adds
            unbounded Gaussian noise with no clipping to ``[lb, ub]``.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Mutated individuals. shape = (n, dim)

        Notes
        -----
        For multiple rows, a loop of separate single-row ``mutate_batch``
        calls (equivalently, separate :meth:`mutate` calls) does not
        reproduce the same per-row results as one batched call with the same
        seeded ``rng``. Separate calls interleave each row's individual
        gate, full per-dimension gate array, and full noise array before
        moving to the next row. One batched call instead draws all
        individual gates, then the variable-gate arrays for all gated rows,
        then the noise arrays for all of them. These phases therefore have a
        different draw order whenever more than one row is involved. Only
        the statistical and distributional semantics are guaranteed to
        match.

        The output is always ``float64`` (via the internal ``dtype=float``
        cast), regardless of the input array's dtype.
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

    def mutate_batch(
        self,
        candidates_batch: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute discrete uniform mutation on a batch of candidates at once.

        This is the primary discrete uniform mutation implementation. An
        individual-level ``prob`` gate (one Bernoulli draw per row via
        ``rng.random(n) < self.prob``) selects which rows are touched at all;
        a per-dimension ``prob_var`` gate is then drawn only for the selected
        rows, and only dimensions passing that gate are
        replaced with a value drawn as a uniform random integer from
        ``[lb[i], ub[i]]`` (both inclusive) via
        ``rng.integers(lb, ub + 1, size=(k, dim))`` -- ``rng.integers``'s
        ``high`` argument is exclusive by default, so ``ub + 1`` makes the
        upper bound inclusive. The integer draw is cast to ``float`` via
        ``.astype(float)`` because the whole codebase represents integer and
        categorical dimensions as floats internally. Rows that fail the
        individual-level gate are returned byte-identical to the input;
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
            Tuple of (lower_bound, upper_bound) arrays.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Mutated individuals. shape = (n, dim)

        Notes
        -----
        For multiple rows, a loop of separate single-row ``mutate_batch``
        calls (equivalently, separate :meth:`mutate` calls) does not
        reproduce the same per-row results as one batched call with the same
        seeded ``rng``. Separate calls interleave each row's individual
        gate, full per-dimension gate array, and full replacement array
        before moving to the next row. One batched call instead draws all
        individual gates, then the variable-gate arrays for all gated rows,
        then the replacement arrays for all of them. These phases therefore
        have a different draw order whenever more than one row is involved.
        Only the statistical and distributional semantics are guaranteed to
        match.

        The output is always ``float64`` (via the internal ``dtype=float``
        cast), regardless of the input array's dtype.
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

    def contract(self) -> ComponentContract:
        """Return the integer mutation contract."""
        base = super().contract()
        role = base.ports["mutation"]
        input_data = role.inputs[0].data
        output_data = role.outputs[0].data
        input_bindings = dict(input_data.bindings)
        output_bindings = dict(output_data.bindings)
        input_bindings["representation"] = Fixed(value="integer")
        output_bindings["representation"] = Fixed(value="integer")
        return replace(
            base,
            ports={
                "mutation": replace(
                    role,
                    inputs=(
                        replace(
                            role.inputs[0],
                            data=replace(input_data, bindings=input_bindings),
                        ),
                    ),
                    outputs=(
                        replace(
                            role.outputs[0],
                            data=replace(output_data, bindings=output_bindings),
                        ),
                    ),
                )
            },
        )


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

    def contract(self) -> ComponentContract:
        """Return the categorical mutation contract."""
        base = super().contract()
        role = base.ports["mutation"]
        input_data = role.inputs[0].data
        output_data = role.outputs[0].data
        input_bindings = dict(input_data.bindings)
        output_bindings = dict(output_data.bindings)
        input_bindings["representation"] = Fixed(value="categorical")
        output_bindings["representation"] = Fixed(value="categorical")
        return replace(
            base,
            ports={
                "mutation": replace(
                    role,
                    inputs=(
                        replace(
                            role.inputs[0],
                            data=replace(input_data, bindings=input_bindings),
                        ),
                    ),
                    outputs=(
                        replace(
                            role.outputs[0],
                            data=replace(output_data, bindings=output_bindings),
                        ),
                    ),
                )
            },
        )
