"""
Crossover operators module.

This module defines crossover operators for evolutionary algorithms.
"""

from abc import ABC, abstractmethod

import numpy as np


class Crossover(ABC):
    """Base class for crossover operators."""

    @abstractmethod
    def crossover(self, parent: np.ndarray, rng=np.random.default_rng()) -> np.ndarray:
        """
        Execute crossover.

        Parameters
        ----------
        parent : np.ndarray
            Parent individuals. shape = (n_parents, dim)
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Offspring individuals. shape = (n_offspring, dim)
        """
        pass


class CrossoverBLXAlpha(Crossover):
    """
    BLX-alpha crossover operator.

    Attributes
    ----------
    crossover_rate : float
        Crossover rate.
    alpha : float
        Alpha parameter for BLX-alpha crossover. Controls how far outside
        the parents' range offspring can be generated.
    """

    def __init__(self, crossover_rate: float, alpha: float):
        """
        Initialize BLX-alpha crossover operator.

        Parameters
        ----------
        crossover_rate : float
            Crossover rate.
        alpha : float
            Alpha parameter for BLX-alpha crossover.
        """
        super().__init__()
        self.crossover_rate = crossover_rate
        self.alpha = alpha

    def crossover(self, p: np.ndarray, rng=np.random.default_rng()) -> np.ndarray:
        """
        Execute BLX-alpha crossover.

        Parameters
        ----------
        p : np.ndarray
            Parent individuals. shape = (2, dim)
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Offspring individuals. shape = (2, dim)
        """
        p1 = p[0]
        p2 = p[1]
        dim = len(p1)
        blend = rng.uniform(-self.alpha, 1 + self.alpha, size=dim)
        c1 = blend * p1 + (1 - blend) * p2
        c2 = (1 - blend) * p1 + blend * p2
        return np.array([c1, c2])

        return np.array([c1, c2])
