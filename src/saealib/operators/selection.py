"""
Selection operators module.

This module defines selection operators for evolutionary algorithms.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING:
    from saealib.population import Population
    from saealib.optimizer import Optimizer


class ParentSelection(ABC):
    """
    Base class for parent selection operators.
    """
    @abstractmethod
    def select(self, opt: Optimizer, pop_x: np.ndarray, pop_f: np.ndarray, pop_cv: np.ndarray, n_pair: int, n_parents: int, rng=np.random.default_rng()) -> np.ndarray:
        pass


# TODO: check required
class TournamentSelection(ParentSelection):
    """
    Tournament selection operator.
    """
    def __init__(self, tournament_size: int):
        super().__init__()
        self.tournament_size = tournament_size

    def select(self, opt: Optimizer, pop_x: np.ndarray, pop_f: np.ndarray, pop_cv: np.ndarray, n_pair: int, n_parents: int, rng=np.random.default_rng()) -> np.ndarray:
        n_pop = len(pop_x)
        cmp = opt.problem.comparator
        selected_idx = np.zeros((n_pair, n_parents), dtype=int)
        for i in range(n_pair):
            for j in range(n_parents):
                tournament_idx = rng.choice(n_pop, size=self.tournament_size, replace=False)
                best_idx = tournament_idx[0]
                for idx in tournament_idx[1:]:
                    if cmp.compare(pop_f[idx:idx+1], pop_cv[idx], pop_f[best_idx:best_idx+1], pop_cv[best_idx]) < 0:
                        best_idx = idx
                selected_idx[i, j] = best_idx
        return selected_idx


class SequentialSelection(ParentSelection):
    """
    Sequential selection operator.
    """
    def __init__(self):
        """
        Initialize sequential selection operator.
        """
        super().__init__()

    def select(self, opt: Optimizer, pop_x: np.ndarray, pop_f: np.ndarray, pop_cv: np.ndarray, n_pair: int, n_parents: int, rng=np.random.default_rng()) -> np.ndarray:
        """
        Execute sequential selection.

        Parameters
        ----------
        opt : Optimizer
            The optimizer instance.
        pop_x : np.ndarray
            Population decision variables.
        pop_f : np.ndarray
            Population objective values.
        pop_cv : np.ndarray
            Population constraint violation values.
        n_pair : int
            Number of pairs to select.
        n_parents : int
            Number of parents per pair.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()
        
        Returns
        -------
        np.ndarray
            Selected parent indices. shape = (n_pair, n_parents)
        """
        n_pop = len(pop_x)
        selected_idx = np.zeros((n_pair, n_parents), dtype=int)
        i_grid, j_grid = np.meshgrid(np.arange(n_pair), np.arange(n_parents), indexing='ij')
        selected_idx = i_grid * n_parents + j_grid
        return selected_idx


class SurvivorSelection(ABC):
    """
    Base class for survivor selection operators.
    """
    def select(self, opt: Optimizer, pop_x: np.ndarray, pop_f: np.ndarray, pop_cv: np.ndarray, off_x: np.ndarray, off_f: np.ndarray, off_cv: np.ndarray, n_survivors: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute survivor selection.

        Create Pool from parents and offspring, then select survivors from the Pool.

        Parameters
        ----------
        opt : Optimizer
            The optimizer instance.
        pop_x : np.ndarray
            Parent population decision variables.
        pop_f : np.ndarray
            Parent population objective values.
        pop_cv : np.ndarray
            Parent population constraint violation values.
        off_x : np.ndarray
            Offspring decision variables.
        off_f : np.ndarray
            Offspring objective values.
        off_cv : np.ndarray
            Offspring constraint violation values.
        n_survivors : int
            Number of survivors to select.
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Selected survivors' decision variables, objective values, and constraint violation values.
        """
        pool_x, pool_f, pool_cv = self._create_pool(pop_x, pop_f, pop_cv, off_x, off_f, off_cv)
        survivor_idx = self._select_from_pool(opt, pool_x, pool_f, pool_cv, n_survivors)
        return pool_x[survivor_idx], pool_f[survivor_idx], pool_cv[survivor_idx]

    def _create_pool(self, pop_x: np.ndarray, pop_f: np.ndarray, pop_cv: np.ndarray, off_x: np.ndarray, off_f: np.ndarray, off_cv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Define how to create the selection pool from parents and offspring.

        Default implementation is (μ + λ) selection. Can be overridden in subclasses.

        Parameters
        ----------
        pop_x : np.ndarray
            Parent population decision variables.
        pop_f : np.ndarray
            Parent population objective values.
        pop_cv : np.ndarray
            Parent population constraint violation values.
        off_x : np.ndarray
            Offspring decision variables.
        off_f : np.ndarray
            Offspring objective values.
        off_cv : np.ndarray
            Offspring constraint violation values.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Selection pool decision variables, objective values, and constraint violation values.
        """
        pool_x = np.vstack((pop_x, off_x))
        pool_f = np.hstack((pop_f, off_f))
        pool_cv = np.hstack((pop_cv, off_cv))
        return pool_x, pool_f, pool_cv
    
    @abstractmethod
    def _select_from_pool(self, opt: Optimizer, pool_x: np.ndarray, pool_f: np.ndarray, pool_cv: np.ndarray, n_survivors: int) -> np.ndarray:
        """
        Select survivors from the selection pool.

        Parameters
        ----------
        opt : Optimizer
            The optimizer instance.
        pool_x : np.ndarray
            Selection pool decision variables.
        pool_f : np.ndarray
            Selection pool objective values.
        pool_cv : np.ndarray
            Selection pool constraint violation values.
        n_survivors : int
            Number of survivors to select.
        
        Returns
        -------
        np.ndarray
            Indices of selected survivors in the pool.
        """
        pass


class TruncationSelection(SurvivorSelection):
    """
    Truncation selection operator.
    """
    def __init__(self):
        super().__init__()

    def _select_from_pool(self, opt: Optimizer, pool_x: np.ndarray, pool_f: np.ndarray, pool_cv: np.ndarray, n_survivors: int) -> np.ndarray:
        cmp = opt.problem.comparator
        cand_idx = cmp.sort(pool_f, pool_cv)
        survivor_idx = cand_idx[:n_survivors]
        return survivor_idx
