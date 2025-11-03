from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from saealib.population import Population
    from saealib.optimizer import Optimizer


class ParentSelection:
    """
    Base class for parent selection operators.
    """
    def __init__(self):
        pass

    def select(self, population: Population) -> np.ndarray:
        pass


class TournamentSelection(ParentSelection):
    """
    Tournament selection operator.
    """
    def __init__(self, tournament_size: int):
        super().__init__()
        self.tournament_size = tournament_size

    def select(self, opt: "Optimizer", pop_x: np.ndarray, pop_f: np.ndarray, pop_cv: np.ndarray, n_pair: int, n_parents: int, rng=np.random.default_rng()) -> np.ndarray:
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
        super().__init__()

    def select(self, opt: "Optimizer", pop_x: np.ndarray, pop_f: np.ndarray, pop_cv: np.ndarray, n_pair: int, n_parents: int, rng=np.random.default_rng()) -> np.ndarray:
        n_pop = len(pop_x)
        selected_idx = np.zeros((n_pair, n_parents), dtype=int)
        i_grid, j_grid = np.meshgrid(np.arange(n_pair), np.arange(n_parents), indexing='ij')
        selected_idx = i_grid * n_parents + j_grid
        return selected_idx


class SurvivorSelection:
    """
    Base class for survivor selection operators.
    """
    def __init__(self):
        pass

    def select(self, opt: "Optimizer", pop_x: np.ndarray, pop_f: np.ndarray, pop_cv: np.ndarray, off_x: np.ndarray, off_f: np.ndarray, off_cv: np.ndarray, n_survivors: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pool_x, pool_f, pool_cv = self._create_pool(pop_x, pop_f, pop_cv, off_x, off_f, off_cv)
        survivor_idx = self._select_from_pool(opt, pool_x, pool_f, pool_cv, n_survivors)
        return pool_x[survivor_idx], pool_f[survivor_idx], pool_cv[survivor_idx]

    def _create_pool(self, pop_x: np.ndarray, pop_f: np.ndarray, pop_cv: np.ndarray, off_x: np.ndarray, off_f: np.ndarray, off_cv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # default: (μ + λ) selection
        # can be overridden in subclasses
        pool_x = np.vstack((pop_x, off_x))
        pool_f = np.hstack((pop_f, off_f))
        pool_cv = np.hstack((pop_cv, off_cv))
        return pool_x, pool_f, pool_cv
    
    def _select_from_pool(self, opt: "Optimizer", pool_x: np.ndarray, pool_f: np.ndarray, pool_cv: np.ndarray, n_survivors: int) -> np.ndarray:
        pass


class TruncationSelection(SurvivorSelection):
    """
    Truncation selection operator.
    """
    def __init__(self):
        super().__init__()

    def _select_from_pool(self, opt: "Optimizer", pool_x: np.ndarray, pool_f: np.ndarray, pool_cv: np.ndarray, n_survivors: int) -> np.ndarray:
        cmp = opt.problem.comparator
        cand_idx = cmp.sort(pool_f, pool_cv)
        survivor_idx = cand_idx[:n_survivors]
        return survivor_idx
