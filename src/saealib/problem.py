"""
Problem module.

This module defines the optimization problem and constraint classes.
Comparator classes and Pareto-related utilities are in saealib.comparators.
"""

from __future__ import annotations

import numpy as np

from saealib.comparators import (
    Comparator,
    NSGA2Comparator,
    SingleObjectiveComparator,
)


class Constraint:
    """
    A single inequality constraint: g(x) <= threshold.

    Parameters
    ----------
    func : callable
        Constraint function g(x). Should return a scalar float.
    threshold : float, optional
        Right-hand side of the constraint g(x) <= threshold. Default: 0.0.
    """

    def __init__(self, func: callable, threshold: float = 0.0):
        self.func = func
        self.threshold = threshold

    def evaluate(self, x: np.ndarray) -> float:
        """Return the raw constraint value g(x)."""
        return float(self.func(x))

    def violation(self, x: np.ndarray) -> float:
        """Return constraint violation: max(0, g(x) - threshold)."""
        return max(0.0, self.evaluate(x) - self.threshold)


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
    comparator : Comparator
        Comparator instance to compare solutions.
    eps : float
        Epsilon value for comparison (Comparator use).
    func : callable -> float
        Objective function to evaluate solutions.
    constraints : list[Constraint]
        List of inequality constraint definitions.
    """

    def __init__(
        self,
        func: callable,
        dim: int,
        n_obj: int,
        weight: np.ndarray,
        lb: list[float],
        ub: list[float],
        eps: float = 1e-6,
        comparator: Comparator | None = None,
        constraints: list[Constraint] | None = None,
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
        eps : float, optional
            Epsilon value for comparison (Comparator use), by default 1e-6
        comparator : Comparator, optional
            Comparator instance to use. If None, auto-selected based on n_obj:
            n_obj == 1 -> SingleObjectiveComparator,
            n_obj >  1 -> NSGA2Comparator.
        constraints : list[Constraint], optional
            List of inequality constraint definitions. Default: empty list.
        """
        self.dim = dim
        self.n_obj = n_obj
        self.weight = weight
        self.eps = eps
        self.lb = np.asarray(lb)
        self.ub = np.asarray(ub)
        self.func = func
        self.constraints = constraints if constraints is not None else []

        if comparator is not None:
            self.comparator = comparator
        elif n_obj == 1:
            self.comparator = SingleObjectiveComparator(weight=weight, eps=eps)
        else:
            self.comparator = NSGA2Comparator(weights=weight, eps=eps)

    @property
    def n_constraints(self) -> int:
        """Number of constraint functions."""
        return len(self.constraints)

    def evaluate_constraints(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Evaluate all constraint functions at x.

        Parameters
        ----------
        x : np.ndarray
            The solution to evaluate. shape = (dim, )

        Returns
        -------
        g : np.ndarray
            Raw constraint values. shape = (n_constraints, )
            Empty array when no constraints are defined.
        cv : float
            Aggregate constraint violation = sum(max(0, g_i - threshold_i)).
            0.0 when no constraints are defined.
        """
        if not self.constraints:
            return np.empty(0, dtype=float), 0.0
        g = np.array([c.evaluate(x) for c in self.constraints], dtype=float)
        cv = float(sum(c.violation(x) for c in self.constraints))
        return g, cv

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
