"""Mutation operators for evolutionary algorithms."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from saealib.context import OptimizationContext


class Mutation(ABC):
    """Base class for mutation operators."""

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
        ctx: OptimizationContext | None = None,
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
        ctx : OptimizationContext or None, optional
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
            [np.ndarray, tuple, np.random.Generator, OptimizationContext | None],
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
        new.post_mutation = lambda offspring, mutate_range, rng, ctx=None: fn(
            prev(offspring, mutate_range, rng, ctx), mutate_range, rng, ctx
        )
        return new


class MutationUniform(Mutation):
    """
    Uniform mutation operator.

    Attributes
    ----------
    mutation_rate : float
        The probability of mutating each dimension.
    """

    def __init__(self, mutation_rate: float):
        """
        Initialize uniform mutation operator.

        Parameters
        ----------
        mutation_rate : float
            The probability of mutating each dimension.
        """
        super().__init__()
        self.mutation_rate = mutation_rate

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
        dim = len(p)
        c = p.copy()
        lb, ub = mutate_range
        for i in range(dim):
            if rng.random() < self.mutation_rate:
                c[i] = rng.uniform(lb[i], ub[i])
        return c


class MutationPolynomial(Mutation):
    """
    Polynomial mutation operator.

    Attributes
    ----------
    mutation_rate : float
        The probability of mutating each dimension.
    eta : float
        Distribution index. Larger values produce smaller perturbations.
    """

    def __init__(self, mutation_rate: float, eta: float):
        """
        Initialize polynomial mutation operator.

        Parameters
        ----------
        mutation_rate : float
            The probability of mutating each dimension.
        eta : float
            Distribution index.
        """
        super().__init__()
        self.mutation_rate = mutation_rate
        self.eta = eta

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
        c = p.copy()
        lb, ub = mutate_range
        for i in range(len(p)):
            if rng.random() < self.mutation_rate:
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
    mutation_rate : float
        The probability of mutating each dimension.
    sigma : float
        Standard deviation of the Gaussian perturbation.
    """

    def __init__(self, mutation_rate: float, sigma: float):
        """
        Initialize Gaussian mutation operator.

        Parameters
        ----------
        mutation_rate : float
            The probability of mutating each dimension.
        sigma : float
            Standard deviation of the Gaussian perturbation.
        """
        super().__init__()
        self.mutation_rate = mutation_rate
        self.sigma = sigma

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
        c = p.copy()
        for i in range(len(p)):
            if rng.random() < self.mutation_rate:
                c[i] = c[i] + rng.normal(0.0, self.sigma)
        return c


class _MutationDiscreteUniform(Mutation):
    """Shared implementation for discrete uniform mutation.

    Replaces each dimension's value with a uniform random integer draw
    from ``[lb[i], ub[i]]`` (both inclusive).
    """

    def __init__(self, mutation_rate: float):
        super().__init__()
        self.mutation_rate = mutation_rate

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
        c = p.copy()
        lb, ub = mutate_range
        for i in range(len(p)):
            if rng.random() < self.mutation_rate:
                c[i] = float(rng.integers(int(lb[i]), int(ub[i]) + 1))
        return c


class MutationIntegerUniform(_MutationDiscreteUniform):
    """
    Uniform integer mutation.

    Replaces each dimension's value with a uniform random integer draw
    from ``[lb[i], ub[i]]`` (both inclusive).

    Attributes
    ----------
    mutation_rate : float
        The probability of mutating each dimension.
    """

    def __init__(self, mutation_rate: float):
        """
        Initialize integer uniform mutation operator.

        Parameters
        ----------
        mutation_rate : float
            The probability of mutating each dimension.
        """
        super().__init__(mutation_rate)


class MutationCategorical(_MutationDiscreteUniform):
    """
    Uniform categorical mutation.

    Replaces each dimension's category index with a uniform random draw
    from ``{0, 1, ..., n_categories - 1}``.  The valid range is inferred
    from ``mutate_range``, where ``ub[i] == n_categories - 1``.

    Attributes
    ----------
    mutation_rate : float
        The probability of mutating each dimension.
    """

    def __init__(self, mutation_rate: float):
        """
        Initialize categorical mutation operator.

        Parameters
        ----------
        mutation_rate : float
            The probability of mutating each dimension.
        """
        super().__init__(mutation_rate)
