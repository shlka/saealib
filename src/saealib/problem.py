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
        lb : list[float]
            Lower bounds for design variables. length = dim
        ub : list[float]
            Upper bounds for design variables. length = dim
        constraints : list[Constraint], optional
            List of constraints, by default None
        eps : float, optional
            Epsilon value for comparison (Comparator use), by default 1e-6
        """
        self.dim = dim
        self.n_obj = n_obj
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

        # TODO: multiple objective support
        self.comparator = SingleObjectiveComparator(weight=weight, eps=eps)

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the objective function at given solution x.

        Parameters
        ----------
        x : np.ndarray
            The solution to evaluate.

        Returns
        -------
        float
            The objective value at solution x.
        """
        return self.func(x)


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
    def compare(
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
        pass

    @abstractmethod
    def sort(self, fitness: np.ndarray, cv: np.ndarray) -> np.ndarray:
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
        pass


class SingleObjectiveComparator(Comparator):
    """Comparator for single-objective optimization."""

    def __init__(self, weight: float = 1.0, eps: float = 1e-6):
        super().__init__(np.array([weight]), eps)

    def compare(
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

    def sort(self, fitness: np.ndarray, cv: np.ndarray) -> np.ndarray:
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
