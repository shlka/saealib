"""
Mutation operators module.

This module defines mutation operators for evolutionary algorithms.
"""

from abc import ABC, abstractmethod

import numpy as np


class Mutation(ABC):
    """
    Base class for mutation operators.
    """

    @abstractmethod
    def mutate(
        self, p: np.ndarray, mutate_range: tuple, rng=np.random.default_rng()
    ) -> np.ndarray:
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
