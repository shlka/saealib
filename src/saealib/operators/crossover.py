"""Crossover operators for evolutionary algorithms."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from saealib.context import OptimizationContext


class Crossover(ABC):
    """Base class for crossover operators.

    Attributes
    ----------
    n_parents : int
        Number of parents required per crossover call. Default is 2.
        Subclasses may override this class attribute if they require a
        different number of parents.
    n_children : int
        Number of offspring produced per crossover call. Default is 2.
        Subclasses may override this class attribute if they produce a
        different number of offspring.
    """

    n_parents: int = 2
    n_children: int = 2
    crossover_rate: float = 1.0

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

    def post_crossover(
        self,
        offspring: np.ndarray,
        parents: np.ndarray,
        rng: np.random.Generator,
        ctx: OptimizationContext | None = None,
    ) -> np.ndarray:
        """Post-crossover lifecycle hook; override to inject custom processing.

        Parameters
        ----------
        offspring : np.ndarray
            Offspring produced by crossover. shape = (n_children, dim)
        parents : np.ndarray
            Parent individuals. shape = (n_parents, dim)
        rng : np.random.Generator
            Random number generator.
        ctx : OptimizationContext or None, optional
            Current optimization context.

        Returns
        -------
        np.ndarray
            Processed offspring. shape = (n_children, dim)
        """
        return offspring

    def with_post(
        self,
        fn: Callable[
            [np.ndarray, np.ndarray, np.random.Generator, OptimizationContext | None],
            np.ndarray,
        ],
    ) -> Crossover:
        """Return a copy of this operator with ``fn`` appended to the hook.

        Parameters
        ----------
        fn : callable
            ``fn(offspring, parents, rng, ctx) -> np.ndarray``

        Returns
        -------
        Crossover
            Shallow copy with the hook registered.
        """
        new = copy.copy(self)
        prev = self.post_crossover
        new.post_crossover = lambda offspring, parents, rng, ctx=None: fn(  # type: ignore
            prev(offspring, parents, rng, ctx), parents, rng, ctx
        )
        return new


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
        self, parent: np.ndarray, rng: np.random.Generator = np.random.default_rng()
    ) -> np.ndarray:
        """
        Execute BLX-alpha crossover.

        Parameters
        ----------
        parent : np.ndarray
            Parent individuals. shape = (2, dim)
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Offspring individuals. shape = (2, dim)
        """
        p1 = parent[0]
        p2 = parent[1]
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
        self, parent: np.ndarray, rng: np.random.Generator = np.random.default_rng()
    ) -> np.ndarray:
        """
        Execute SBX crossover.

        Parameters
        ----------
        parent : np.ndarray
            Parent individuals. shape = (2, dim)
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Offspring individuals. shape = (2, dim)
        """
        p1 = parent[0]
        p2 = parent[1]
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
        self, parent: np.ndarray, rng: np.random.Generator = np.random.default_rng()
    ) -> np.ndarray:
        """
        Execute uniform crossover.

        Parameters
        ----------
        parent : np.ndarray
            Parent individuals. shape = (2, dim)
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Offspring individuals. shape = (2, dim)
        """
        p1 = parent[0]
        p2 = parent[1]
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
        self, parent: np.ndarray, rng: np.random.Generator = np.random.default_rng()
    ) -> np.ndarray:
        """
        Execute one-point crossover.

        Parameters
        ----------
        parent : np.ndarray
            Parent individuals. shape = (2, dim)
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Offspring individuals. shape = (2, dim)
        """
        p1 = parent[0]
        p2 = parent[1]
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
        self, parent: np.ndarray, rng: np.random.Generator = np.random.default_rng()
    ) -> np.ndarray:
        """
        Execute two-point crossover.

        Parameters
        ----------
        parent : np.ndarray
            Parent individuals. shape = (2, dim)
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Offspring individuals. shape = (2, dim)
        """
        p1 = parent[0]
        p2 = parent[1]
        dim = len(p1)
        pts = np.sort(rng.choice(dim + 1, size=2, replace=False))
        pt1, pt2 = pts[0], pts[1]
        c1 = np.concatenate([p1[:pt1], p2[pt1:pt2], p1[pt2:]])
        c2 = np.concatenate([p2[:pt1], p1[pt1:pt2], p2[pt2:]])
        return np.array([c1, c2])


class CrossoverIntegerSBX(Crossover):
    """
    Simulated Binary Crossover (SBX) for integer-valued dimensions.

    Applies SBX then rounds offspring to the nearest integer.

    Attributes
    ----------
    crossover_rate : float
        Crossover rate.
    eta : float
        Distribution index. Larger values produce offspring closer to parents.
    """

    def __init__(self, crossover_rate: float, eta: float):
        """
        Initialize integer SBX crossover operator.

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
        self, parent: np.ndarray, rng: np.random.Generator = np.random.default_rng()
    ) -> np.ndarray:
        """
        Execute integer SBX crossover.

        Parameters
        ----------
        parent : np.ndarray
            Parent individuals. shape = (2, dim)
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Offspring individuals with integer values. shape = (2, dim)
        """
        p1 = parent[0]
        p2 = parent[1]
        dim = len(p1)
        u = rng.uniform(0.0, 1.0, size=dim)
        beta_q = np.where(
            u <= 0.5,
            (2.0 * u) ** (1.0 / (self.eta + 1)),
            (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta + 1)),
        )
        mid = 0.5 * (p1 + p2)
        half_diff = 0.5 * beta_q * (p2 - p1)
        c1 = np.round(mid - half_diff)
        c2 = np.round(mid + half_diff)
        return np.array([c1, c2])


class CrossoverCategorical(Crossover):
    """
    Uniform crossover for categorical dimensions.

    Each dimension independently inherits one parent's value with equal
    probability (50/50).  Offspring are always exact copies of a parent
    value, preserving the validity of categorical indices.

    Attributes
    ----------
    crossover_rate : float
        Crossover rate.
    """

    def __init__(self, crossover_rate: float):
        """
        Initialize categorical crossover operator.

        Parameters
        ----------
        crossover_rate : float
            Crossover rate.
        """
        super().__init__()
        self.crossover_rate = crossover_rate

    def crossover(
        self, parent: np.ndarray, rng: np.random.Generator = np.random.default_rng()
    ) -> np.ndarray:
        """
        Execute categorical crossover.

        Parameters
        ----------
        parent : np.ndarray
            Parent individuals. shape = (2, dim)
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Offspring individuals. shape = (2, dim)
        """
        p1 = parent[0]
        p2 = parent[1]
        mask = rng.random(len(p1)) < 0.5
        c1 = np.where(mask, p2, p1)
        c2 = np.where(mask, p1, p2)
        return np.array([c1, c2])
