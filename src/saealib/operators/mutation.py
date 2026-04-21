"""
Mutation operators module.

This module defines mutation operators for evolutionary algorithms.
"""

from abc import ABC, abstractmethod

import numpy as np


class Mutation(ABC):
    """Base class for mutation operators."""

    @abstractmethod
    def mutate(
        self, p: np.ndarray, mutate_range: tuple, rng=np.random.default_rng()
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

    # TODO: mutate_range should be handled outside (__init__ method?)
    def mutate(
        self, p: np.ndarray, mutate_range: tuple, rng=np.random.default_rng()
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
        self, p: np.ndarray, mutate_range: tuple, rng=np.random.default_rng()
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
                    delta_q = (2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta) ** (self.eta + 1)) ** (
                        1.0 / (self.eta + 1)
                    ) - 1.0
                else:
                    delta_q = 1.0 - (
                        2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - delta) ** (self.eta + 1)
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
        self, p: np.ndarray, mutate_range: tuple, rng=np.random.default_rng()
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
