"""Mutation operators for evolutionary algorithms."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

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
                delta = min(c[i] - lb[i], ub[i] - c[i]) / (ub[i] - lb[i])
                u = rng.random()
                if u <= 0.5:
                    delta_q = (
                        2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta) ** (self.eta + 1)
                    ) ** (1.0 / (self.eta + 1)) - 1.0
                else:
                    delta_q = 1.0 - (
                        2.0 * (1.0 - u)
                        + 2.0 * (u - 0.5) * (1.0 - delta) ** (self.eta + 1)
                    ) ** (1.0 / (self.eta + 1))
                c[i] = np.clip(c[i] + delta_q * (ub[i] - lb[i]), lb[i], ub[i])
        return c


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
