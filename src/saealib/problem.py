"""
Problem module.

This module defines the optimization problem and constraint classes.
Comparator classes and Pareto-related utilities are in saealib.comparators.
"""

from __future__ import annotations

import warnings

import numpy as np

from saealib.comparators import (
    Comparator,
    NSGA2Comparator,
    SingleObjectiveComparator,
)


class InequalityConstraint:
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

    def evaluate_with_violation(self, x: np.ndarray) -> tuple[float, float]:
        """
        Return the raw value and violation with a single function evaluation.

        Returns
        -------
        g : float
            Raw constraint value g(x).
        cv : float
            Constraint violation: max(0, g(x) - threshold).
        """
        g = self.evaluate(x)
        return g, max(0.0, g - self.threshold)

    def gradient(self, x: np.ndarray) -> np.ndarray | None:
        """
        Return the gradient of g(x) with respect to x, if available.

        Override this in a subclass to enable gradient-based constraint
        handlers (e.g. gradient-based repair). The default returns ``None``,
        signalling that no analytical Jacobian is provided.

        Parameters
        ----------
        x : np.ndarray
            The solution at which to evaluate the gradient. shape = (dim, )

        Returns
        -------
        np.ndarray or None
            Gradient vector with shape = (dim, ), or ``None`` when no
            gradient is available.
        """
        return None


class Constraint(InequalityConstraint):
    """
    Deprecated alias of :class:`InequalityConstraint`.

    .. deprecated::
        Use :class:`InequalityConstraint`. ``Constraint`` will be removed
        in a future release.
    """

    def __init__(self, func: callable, threshold: float = 0.0):
        warnings.warn(
            "Constraint is deprecated and will be removed in a future release. "
            "Use InequalityConstraint instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(func, threshold)


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
    eps_cv : float
        Epsilon for constraint violation feasibility threshold.
    eps_obj : float
        Epsilon for objective value equality comparison.
    func : callable -> float
        Objective function to evaluate solutions.
    constraints : list[InequalityConstraint]
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
        eps: float | None = None,
        comparator: Comparator | None = None,
        constraints: list[InequalityConstraint] | None = None,
        *,
        eps_cv: float = 1e-6,
        eps_obj: float = 1e-6,
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
            Deprecated. Use eps_cv and eps_obj. Will be removed in 0.1.0.
        comparator : Comparator, optional
            Comparator instance to use. If None, auto-selected based on n_obj:
            n_obj == 1 -> SingleObjectiveComparator,
            n_obj >  1 -> NSGA2Comparator.
        constraints : list[InequalityConstraint], optional
            List of inequality constraint definitions. Default: empty list.
        eps_cv : float, optional
            Epsilon for constraint violation feasibility threshold. Default: 1e-6.
        eps_obj : float, optional
            Epsilon for objective value equality comparison. Default: 1e-6.
        """
        if eps is not None:
            warnings.warn(
                "Problem(eps=...) is deprecated and will be removed in 0.1.0. "
                "Use eps_cv and eps_obj.",
                DeprecationWarning,
                stacklevel=2,
            )
            eps_cv = eps_obj = eps
        self.dim = dim
        self.n_obj = n_obj
        self.weight = weight
        self.eps_cv = eps_cv
        self.eps_obj = eps_obj
        self.lb = np.asarray(lb)
        self.ub = np.asarray(ub)
        self.func = func
        self.constraints = constraints if constraints is not None else []

        if comparator is not None:
            self.comparator = comparator
        elif n_obj == 1:
            self.comparator = SingleObjectiveComparator(
                weight=weight, eps_cv=eps_cv, eps_obj=eps_obj
            )
        else:
            self.comparator = NSGA2Comparator(
                weights=weight, eps_cv=eps_cv, eps_obj=eps_obj
            )

    @property
    def eps(self) -> float:
        """Deprecated. Use eps_cv or eps_obj."""
        warnings.warn(
            "Problem.eps is deprecated and will be removed in 0.1.0. "
            "Use eps_cv or eps_obj.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.eps_cv

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
        g = np.empty(len(self.constraints), dtype=float)
        cv = 0.0
        for i, c in enumerate(self.constraints):
            g[i], v = c.evaluate_with_violation(x)
            cv += v
        return g, float(cv)

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
