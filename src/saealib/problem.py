"""
Problem module.

Problem has definitions for optimization problems,
and includes classes that depend on problem.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from saealib.population import Population


class Problem:
    """
    Definition of optimization problem.

    Attributes
    ----------
    dim : int
        Dimension of the design variables.
    n_obj : int
        Number of objectives.
    weight : np.ndarray
        Weights for objectives. shape = (n_obj, )
    lb : np.ndarray
        Lower bounds for design variables. shape = (dim, )
    ub : np.ndarray
        Upper bounds for design variables. shape = (dim, )
    constraint_manager : ConstraintManager
        Constraint manager instance to handle constraints.
    comparator : Comparator
        Comparator instance to compare solutions.
    eps : float
        Epsilon value for comparison (Comparator use).
    func : callable -> float
        Objective function to evaluate solutions.
    """

    def __init__(
        self,
        func: callable,
        dim: int,
        n_obj: int,
        weight: np.ndarray,
        lb: list[float],
        ub: list[float],
        constraints: list[Constraint] | None = None,
        eps: float = 1e-6,
        comparator: Comparator | None = None,
    ):
        """
        Initialize Problem instance.

        Parameters
        ----------
        func : callable -> float
            Objective function to evaluate solutions.
        dim : int
            Dimension of the design variables.
        n_obj : int
            Number of objectives.
        weight : np.ndarray
            Weights for objectives. shape = (n_obj, )
            Used by SingleObjectiveComparator and WeightedSumComparator.
            Not used by NSGA2Comparator.
        lb : list[float]
            Lower bounds for design variables. length = dim
        ub : list[float]
            Upper bounds for design variables. length = dim
        constraints : list[Constraint], optional
            List of constraints, by default None
        eps : float, optional
            Epsilon value for comparison (Comparator use), by default 1e-6
        comparator : Comparator, optional
            Comparator instance to use. If None, auto-selected based on n_obj:
            n_obj == 1 -> SingleObjectiveComparator,
            n_obj >  1 -> NSGA2Comparator.
        """
        self.dim = dim
        self.n_obj = n_obj
        self.n_constraint = len(constraints) if constraints is not None else 0
        self.weight = weight
        self.eps = eps
        self.lb = np.asarray(lb)
        self.ub = np.asarray(ub)
        self.func = func
        constraints_list = [] if constraints is None else constraints

        for i in range(dim):
            constraints_list.append(
                Constraint(partial(ub_constraint, ub=self.ub), type=ConstraintType.INEQ)
            )
            constraints_list.append(
                Constraint(partial(lb_constraint, lb=self.lb), type=ConstraintType.INEQ)
            )

        self.constraint_manager = ConstraintManager(constraints=constraints_list)

        if comparator is not None:
            self.comparator = comparator
        elif n_obj == 1:
            self.comparator = SingleObjectiveComparator(weight=weight, eps=eps)
        else:
            self.comparator = NSGA2Comparator(weights=weight, eps=eps)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the objective function at given solution x.

        Parameters
        ----------
        x : np.ndarray
            The solution to evaluate.

        Returns
        -------
        np.ndarray
            The objective value(s) at solution x. shape = (n_obj, )
        """
        result = self.func(x)
        return np.atleast_1d(np.asarray(result, dtype=float))


class ConstraintType(Enum):
    """
    Constraint types.

    Attributes
    ----------
    EQ
        Equality constraint.
    INEQ
        Inequality constraint.
    """

    EQ = auto()
    INEQ = auto()


class Constraint:
    """
    Constraint class to handle single constraint.

    Attributes
    ----------
    type_constraint : ConstraintType
        Type of the constraint.
    func : callable -> float
        Returns constraint violation value.
    """

    def __init__(self, func, type: ConstraintType = ConstraintType.INEQ):
        """
        Initialize Constraint instance.

        Parameters
        ----------
        func : callable -> float
            Returns constraint violation value.
        type : ConstraintType, optional
            Type of the constraint, by default ConstraintType.INEQ
        """
        self.type_constraint = type
        self.func = func

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the constraint violation at given solution x.

        Parameters
        ----------
        x : np.ndarray
            The solution to evaluate.
        """
        v = self.func(x)
        if self.type_constraint == ConstraintType.INEQ:
            return max(0, v)
        elif self.type_constraint == ConstraintType.EQ:
            return abs(v)


def ub_constraint(x: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """
    Upper bound constraint function.

    Parameters
    ----------
    x : np.ndarray
        The solution to evaluate.
    ub : np.ndarray
        Upper bounds for design variables.

    Returns
    -------
    np.ndarray
        The upper bound constraint violation.
    """
    return x - ub


def lb_constraint(x: np.ndarray, lb: np.ndarray) -> np.ndarray:
    """
    Lower bound constraint function.

    Parameters
    ----------
    x : np.ndarray
        The solution to evaluate.
    lb : np.ndarray
        Lower bounds for design variables.

    Returns
    -------
    np.ndarray
        The lower bound constraint violation.
    """
    return lb - x


class ConstraintManager:
    """
    Constraint manager to handle multiple constraints.

    Attributes
    ----------
    constraints : list[Constraint]
        List of constraints.
    """

    def __init__(self, constraints: list[Constraint]):
        """
        Initialize ConstraintManager instance.

        Parameters
        ----------
        constraints : list[Constraint]
            List of constraints.
        """
        if constraints is None:
            self.constraints = []
        else:
            self.constraints = constraints

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the total constraint violation at given solution x.

        Parameters
        ----------
        x : np.ndarray
            The solution to evaluate.

        Returns
        -------
        float
            The total constraint violation at solution x.
        """
        if not self.constraints:
            return 0.0
        return sum(constraint.evaluate(x) for constraint in self.constraints)

    def evaluate_population(self, population: Population) -> np.ndarray:
        """
        Evaluate the total constraint violation for each individual in the population.

        Parameters
        ----------
        population : Population
            The population to evaluate.

        Returns
        -------
        np.ndarray
            The total constraint violation for each individual in the population.
        """
        if not self.constraints:
            return np.zeros(len(population))
        return np.array([self.evaluate(ind.get("x")) for ind in population])


class Comparator(ABC):
    """
    Base class for comparator.

    Attributes
    ----------
    weights : np.ndarray
        Weights for objectives. shape = (n_obj, )
    eps : float
        Epsilon value for comparison tolerance.
    """

    @abstractmethod
    def __init__(self, weights: np.ndarray, eps: float):
        """
        Initialize Comparator instance.

        Parameters
        ----------
        weights : np.ndarray
            Weights for objectives. shape = (n_obj, )
        eps : float
            Epsilon value for comparison tolerance.
        """
        self.weights = weights
        self.eps = eps

    @abstractmethod
    def sort_population(self, population: Population) -> np.ndarray:
        """
        Sort population based on their attributes.

        Parameters
        ----------
        population : Population
            The population to sort.

        Returns
        -------
        np.ndarray
            Sorted population indices.
        """
        pass

    @abstractmethod
    def compare_population(self, population: Population, idx_a: int, idx_b: int) -> int:
        """
        Compare individual a and b in population.

        Parameters
        ----------
        population : Population
            The population to compare.
        idx_a : int
            Index of the first individual.
        idx_b : int
            Index of the second individual.

        Returns
        -------
        int
            Comparison result.
            -1 if the individual a is better than the individual b.
            1 if the individual a is worse than the individual b.
            0 if the individual a is equal to the individual b.
        """
        pass


class SingleObjectiveComparator(Comparator):
    """Comparator for single-objective optimization."""

    def __init__(self, weight: float = 1.0, eps: float = 1e-6):
        super().__init__(np.array([weight]), eps)

    def sort_population(self, population: Population) -> np.ndarray:
        """
        Sort population based on their fitness and constraint violations.

        Parameters
        ----------
        population : Population
            The population to sort.

        Returns
        -------
        np.ndarray
            Sorted population indices.
        """
        f = population.get("f")
        cv = population.get("cv")
        return self._sort(f, cv)

    def compare_population(self, population: Population, idx_a: int, idx_b: int) -> int:
        """
        Compare individual a and b in population.

        Parameters
        ----------
        population : Population
            The population to compare.
        idx_a : int
            Index of the first individual.
        idx_b : int
            Index of the second individual.

        Returns
        -------
        int
            Comparison result.
            -1 if the individual a is better than the individual b.
            1 if the individual a is worse than the individual b.
            0 if the individual a is equal to the individual b.
        """
        f = population.get("f")
        cv = population.get("cv")
        return self._compare(f[idx_a], cv[idx_a], f[idx_b], cv[idx_b])

    def _compare(
        self, fitness_a: np.ndarray, cv_a: float, fitness_b: np.ndarray, cv_b: float
    ) -> int:
        """
        Compare two solutions.

        Parameters
        ----------
        fitness_a : np.ndarray
            Objective values of solution a. shape = (n_obj, )
        cv_a : float
            Constraint violation of solution a.
        fitness_b : np.ndarray
            Objective values of solution b. shape = (n_obj, )
        cv_b : float
            Constraint violation of solution b.

        Returns
        -------
        int
            Comparison result.
            -1 if a is better than b.
            1 if b is better than a.
            0 if a and b are equal.
        """
        if cv_a > self.eps and cv_b > self.eps:
            if cv_a < cv_b:
                return -1
            elif cv_a > cv_b:
                return 1
            else:
                return 0
        elif cv_a > self.eps and cv_b <= self.eps:
            return 1
        elif cv_a <= self.eps and cv_b > self.eps:
            return -1
        else:
            if fitness_a[0] < fitness_b[0] - self.eps:
                return -1
            elif fitness_a[0] > fitness_b[0] + self.eps:
                return 1
            else:
                return 0

    def _sort(self, fitness: np.ndarray, cv: np.ndarray) -> np.ndarray:
        """
        Sort solutions based on their fitness and constraint violations.

        Parameters
        ----------
        fitness : np.ndarray
            Objective values of solutions. shape = (n_individuals, n_obj)
        cv : np.ndarray
            Constraint violations of solutions. shape = (n_individuals, )

        Returns
        -------
        np.ndarray
            Sorted indices of the solutions.
        """
        cv_key = np.where(cv > self.eps, cv, 0)
        obj_key = fitness.flatten() * self.weights[0]
        return np.lexsort((-obj_key, cv_key))


class WeightedSumComparator(Comparator):
    """
    Comparator for multi-objective optimization via weighted scalarization.

    Aggregates multiple objectives into a single scalar using a weighted
    dot product: score = f @ weights. The sign convention for weights
    follows the same pattern as SingleObjectiveComparator: use negative
    weights for minimization (e.g., weights=np.array([-1.0, -1.0])).

    Also supports single-objective problems (n_obj == 1).

    Parameters
    ----------
    weights : np.ndarray
        Weights for objectives. shape = (n_obj,)
    eps : float
        Epsilon tolerance for constraint violation and fitness comparison.
    """

    def __init__(self, weights: np.ndarray, eps: float = 1e-6):
        super().__init__(np.asarray(weights, dtype=float), eps)

    def sort_population(self, population: Population) -> np.ndarray:
        f = population.get("f")    # (n_ind, n_obj)
        cv = population.get("cv")  # (n_ind,)
        scalar = f @ self.weights  # (n_ind,) weighted sum per individual
        cv_key = np.where(cv > self.eps, cv, 0)
        return np.lexsort((-scalar, cv_key))

    def compare_population(self, population: Population, idx_a: int, idx_b: int) -> int:
        f = population.get("f")
        cv = population.get("cv")
        return self._compare(f[idx_a], float(cv[idx_a]), f[idx_b], float(cv[idx_b]))

    def _compare(
        self, fa: np.ndarray, cv_a: float, fb: np.ndarray, cv_b: float
    ) -> int:
        if cv_a > self.eps and cv_b > self.eps:
            if cv_a < cv_b:
                return -1
            elif cv_a > cv_b:
                return 1
            else:
                return 0
        elif cv_a > self.eps:
            return 1
        elif cv_b > self.eps:
            return -1
        sa = float(np.dot(fa, self.weights))
        sb = float(np.dot(fb, self.weights))
        if sa > sb + self.eps:
            return -1
        elif sa < sb - self.eps:
            return 1
        return 0


# ---------------------------------------------------------------------------
# Non-dominated sorting utilities
#
# These functions implement the core NDS algorithm shared across NSGA-family
# comparators. Crowding distance is NSGA-II specific and kept here for now;
# it will be separated when NSGA-III (reference-point diversity) is added.
#
# TODO: move to saealib/moo/ when multiple NDS-based comparators exist.
# ---------------------------------------------------------------------------


def _pareto_dominates(fa: np.ndarray, fb: np.ndarray) -> bool:
    """
    Return True if fa Pareto-dominates fb (minimization).

    NaN values in fa are treated as non-dominating (returns False).

    Parameters
    ----------
    fa, fb : np.ndarray
        Objective vectors to compare.
    """
    fa = np.asarray(fa, float)
    fb = np.asarray(fb, float)
    if np.any(np.isnan(fa)):
        return False
    return bool(np.all(fa <= fb) and np.any(fa < fb))


def non_dominated_sort(
    f: np.ndarray,
) -> tuple[np.ndarray, list[list[int]]]:
    """
    O(MN^2) non-dominated sorting (Deb et al., 2002).

    NaN rows are treated as infinitely bad and placed in the last front.

    Parameters
    ----------
    f : np.ndarray
        Objective matrix. shape: (n, n_obj)

    Returns
    -------
    ranks : np.ndarray shape (n,)
        Pareto front index for each individual (0 = first/best front).
    fronts : list[list[int]]
        fronts[i] contains the local indices of individuals in front i.
    """
    n = len(f)
    nan_mask = np.any(np.isnan(f), axis=1)
    valid = np.where(~nan_mask)[0]

    dominated_by_count = np.zeros(n, int)
    dominates_set: list[list[int]] = [[] for _ in range(n)]
    ranks = np.full(n, -1, int)
    fronts: list[list[int]] = [[]]

    for i in valid:
        for j in valid:
            if i == j:
                continue
            if _pareto_dominates(f[i], f[j]):
                dominates_set[i].append(j)
                dominated_by_count[j] += 1

    for i in valid:
        if dominated_by_count[i] == 0:
            ranks[i] = 0
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front: list[int] = []
        for i in fronts[k]:
            for j in dominates_set[i]:
                dominated_by_count[j] -= 1
                if dominated_by_count[j] == 0:
                    ranks[j] = k + 1
                    next_front.append(j)
        fronts.append(next_front)
        k += 1
    fronts.pop()  # remove trailing empty front

    # Push NaN individuals to a final sentinel front
    last_rank = k
    for i in np.where(nan_mask)[0]:
        ranks[i] = last_rank
        fronts.append([int(i)])

    return ranks, fronts


def crowding_distance(f_front: np.ndarray) -> np.ndarray:
    """
    Compute crowding distance for a single Pareto front (NSGA-II).

    Boundary solutions (minimum and maximum per objective) are assigned
    infinite crowding distance.

    Parameters
    ----------
    f_front : np.ndarray
        Objective values of individuals in the front. shape: (n, n_obj)

    Returns
    -------
    np.ndarray
        Crowding distances. shape: (n,)
    """
    n, m = f_front.shape
    cd = np.zeros(n)
    if n <= 2:
        cd[:] = np.inf
        return cd
    for obj in range(m):
        order = np.argsort(f_front[:, obj])
        cd[order[0]] = np.inf
        cd[order[-1]] = np.inf
        f_range = f_front[order[-1], obj] - f_front[order[0], obj]
        if f_range < 1e-12:
            continue
        for k in range(1, n - 1):
            cd[order[k]] += (
                f_front[order[k + 1], obj] - f_front[order[k - 1], obj]
            ) / f_range
    return cd


def crowding_distance_all_fronts(
    f: np.ndarray,
    fronts: list[list[int]],
) -> np.ndarray:
    """
    Compute crowding distance for all fronts.

    Parameters
    ----------
    f : np.ndarray
        Objective matrix. shape: (n, n_obj)
    fronts : list[list[int]]
        Output of non_dominated_sort: fronts[i] = local indices in front i.

    Returns
    -------
    np.ndarray
        Crowding distances for all individuals. shape: (n,)
    """
    cd = np.zeros(len(f))
    for front in fronts:
        if not front:
            continue
        idx = np.array(front)
        cd[idx] = crowding_distance(f[idx])
    return cd


class NSGA2Comparator(Comparator):
    """
    Comparator for multi-objective optimization via NSGA-II style ranking.

    Implements NSGA-II style ranking:
    - sort_population: non-dominated sorting + crowding distance
    - compare_population: Pareto dominance (-1 = a dominates b, 1 = b
      dominates a, 0 = non-dominated)

    Infeasible individuals (cv > eps) are always ranked after feasible
    ones, ordered by ascending constraint violation.

    The sort result is cached in the Population cache and automatically
    invalidated when the population is modified.

    Parameters
    ----------
    weights : np.ndarray or None
        Stored for interface compatibility but not used in Pareto ranking.
    eps : float
        Epsilon tolerance for constraint violation.
    """

    def __init__(self, weights: np.ndarray | None = None, eps: float = 1e-6):
        w = np.asarray(weights, dtype=float) if weights is not None else np.empty(0)
        super().__init__(w, eps)

    def sort_population(self, population: Population) -> np.ndarray:
        """Sort by Pareto front rank then crowding distance (NSGA-II style)."""
        cached = population.get_cache("pareto_sort")
        if cached is not None:
            return cached

        f = population.get("f")    # (n, n_obj)
        cv = population.get("cv")  # (n,)
        feasible = np.where(cv <= self.eps)[0]
        infeasible = np.where(cv > self.eps)[0]

        sorted_feasible = np.empty(0, int)
        if len(feasible):
            ranks, fronts = non_dominated_sort(f[feasible])
            cd = crowding_distance_all_fronts(f[feasible], fronts)
            order = np.lexsort((-cd, ranks))
            sorted_feasible = feasible[order]

        sorted_infeasible = np.empty(0, int)
        if len(infeasible):
            sorted_infeasible = infeasible[np.argsort(cv[infeasible])]

        result = np.concatenate([sorted_feasible, sorted_infeasible]).astype(int)
        population.set_cache("pareto_sort", result)
        return result

    def compare_population(self, population: Population, idx_a: int, idx_b: int) -> int:
        """Compare via Pareto dominance; -1=a dominates, 1=b dominates, 0=non-dominated."""
        f = population.get("f")
        cv = population.get("cv")
        cv_a, cv_b = float(cv[idx_a]), float(cv[idx_b])

        if cv_a > self.eps and cv_b > self.eps:
            if cv_a < cv_b:
                return -1
            elif cv_a > cv_b:
                return 1
            return 0
        elif cv_a > self.eps:
            return 1
        elif cv_b > self.eps:
            return -1

        if _pareto_dominates(f[idx_a], f[idx_b]):
            return -1
        if _pareto_dominates(f[idx_b], f[idx_a]):
            return 1
        return 0
