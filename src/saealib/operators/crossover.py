"""
Crossover operators module.

This module defines crossover operators for evolutionary algorithms.
"""

from abc import ABC, abstractmethod

import numpy as np


class Crossover(ABC):
    """Base class for crossover operators.

    Attributes
    ----------
    n_parents : int
        Number of parents required per crossover call. Default is 2.
        Subclasses may override this class attribute if they require a
        different number of parents.
    """

    n_parents: int = 2

    @abstractmethod
    def crossover(
        self, parent: np.ndarray, rng: np.random.Generator = np.random.default_rng()
    ) -> np.ndarray:
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

    def crossover(
        self, p: np.ndarray, rng: np.random.Generator = np.random.default_rng()
    ) -> np.ndarray:
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


class CrossoverSBX(Crossover):
    """
    Simulated Binary Crossover (SBX) operator.

    Attributes
    ----------
    crossover_rate : float
        Crossover rate.
    eta : float
        Distribution index. Larger values produce offspring closer to parents.
    """

    def __init__(self, crossover_rate: float, eta: float):
        """
        Initialize SBX crossover operator.

        Parameters
        ----------
        crossover_rate : float
            Crossover rate.
        eta : float
            Distribution index.
        """
        super().__init__()
        self.crossover_rate = crossover_rate
        self.eta = eta

    def crossover(
        self, p: np.ndarray, rng: np.random.Generator = np.random.default_rng()
    ) -> np.ndarray:
        """
        Execute SBX crossover.

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
        u = rng.uniform(0.0, 1.0, size=dim)
        beta_q = np.where(
            u <= 0.5,
            (2.0 * u) ** (1.0 / (self.eta + 1)),
            (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta + 1)),
        )
        mid = 0.5 * (p1 + p2)
        half_diff = 0.5 * beta_q * (p2 - p1)
        c1 = mid - half_diff
        c2 = mid + half_diff
        return np.array([c1, c2])


class CrossoverUniform(Crossover):
    """
    Uniform crossover operator.

    Each dimension is independently swapped between parents with probability
    ``swap_rate``.

    Attributes
    ----------
    crossover_rate : float
        Crossover rate.
    swap_rate : float
        Per-dimension swap probability. Default is 0.5.
    """

    def __init__(self, crossover_rate: float, swap_rate: float = 0.5):
        """
        Initialize uniform crossover operator.

        Parameters
        ----------
        crossover_rate : float
            Crossover rate.
        swap_rate : float, optional
            Per-dimension swap probability, by default 0.5.
        """
        super().__init__()
        self.crossover_rate = crossover_rate
        self.swap_rate = swap_rate

    def crossover(
        self, p: np.ndarray, rng: np.random.Generator = np.random.default_rng()
    ) -> np.ndarray:
        """
        Execute uniform crossover.

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
        mask = rng.random(len(p1)) < self.swap_rate
        c1 = np.where(mask, p2, p1)
        c2 = np.where(mask, p1, p2)
        return np.array([c1, c2])


class CrossoverOnePoint(Crossover):
    """
    One-point crossover operator.

    Attributes
    ----------
    crossover_rate : float
        Crossover rate.
    """

    def __init__(self, crossover_rate: float):
        """
        Initialize one-point crossover operator.

        Parameters
        ----------
        crossover_rate : float
            Crossover rate.
        """
        super().__init__()
        self.crossover_rate = crossover_rate

    def crossover(
        self, p: np.ndarray, rng: np.random.Generator = np.random.default_rng()
    ) -> np.ndarray:
        """
        Execute one-point crossover.

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
        point = rng.integers(1, dim)
        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])
        return np.array([c1, c2])


class CrossoverTwoPoint(Crossover):
    """
    Two-point crossover operator.

    Attributes
    ----------
    crossover_rate : float
        Crossover rate.
    """

    def __init__(self, crossover_rate: float):
        """
        Initialize two-point crossover operator.

        Parameters
        ----------
        crossover_rate : float
            Crossover rate.
        """
        super().__init__()
        self.crossover_rate = crossover_rate

    def crossover(
        self, p: np.ndarray, rng: np.random.Generator = np.random.default_rng()
    ) -> np.ndarray:
        """
        Execute two-point crossover.

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
        pts = np.sort(rng.choice(dim + 1, size=2, replace=False))
        pt1, pt2 = pts[0], pts[1]
        c1 = np.concatenate([p1[:pt1], p2[pt1:pt2], p1[pt2:]])
        c2 = np.concatenate([p2[:pt1], p1[pt1:pt2], p2[pt2:]])
        return np.array([c1, c2])
