"""Problem class for optimization."""

from __future__ import annotations

import numpy as np

from saealib._deprecated import warn_deprecated
from saealib.comparators import (
    Comparator,
    NSGA2Comparator,
    SingleObjectiveComparator,
)
from saealib.exceptions import ValidationError
from saealib.problem.constraint import (
    ConstraintHandler,
    InequalityConstraint,
    StaticToleranceHandler,
)


class Problem:
    """
    Definition of optimization problem.

    Attributes
    ----------
    dim : int
        Dimension of the design variables.
    n_obj : int
        Number of objectives.
    direction : np.ndarray
        Optimization direction per objective. shape = (n_obj, )
        Each element must be +1 (maximize) or -1 (minimize).
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
    handler : ConstraintHandler
        Constraint-handling strategy used to aggregate violations and augment
        objectives.
    """

    def __init__(
        self,
        func: callable,
        dim: int,
        n_obj: int,
        direction: np.ndarray,
        lb: list[float],
        ub: list[float],
        eps: float | None = None,
        comparator: Comparator | None = None,
        constraints: list[InequalityConstraint] | None = None,
        *,
        eps_cv: float = 1e-6,
        eps_obj: float = 1e-6,
        handler: ConstraintHandler | None = None,
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
        direction : np.ndarray
            Optimization direction per objective. shape = (n_obj, )
            Each element must be +1 (maximize) or -1 (minimize).
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
        handler : ConstraintHandler, optional
            Constraint-handling strategy. If None, a StaticToleranceHandler
            (sum-of-violations, fixed eps_cv) is used, reproducing the default
            behavior.
        """
        if eps is not None:
            warn_deprecated("eps", "eps_cv and eps_obj", "0.1.0")
            eps_cv = eps_obj = eps
        direction = np.asarray(direction, dtype=float)
        if not np.all(np.abs(direction) == 1):
            raise ValidationError("direction elements must be +1 or -1")
        self.dim = dim
        self.n_obj = n_obj
        self.direction = direction
        self.eps_cv = eps_cv
        self.eps_obj = eps_obj
        self.lb = np.asarray(lb)
        self.ub = np.asarray(ub)
        self.func = func
        self.constraints = constraints if constraints is not None else []
        self.handler = (
            handler if handler is not None else StaticToleranceHandler(eps_cv=eps_cv)
        )

        if comparator is not None:
            self.comparator = comparator
        elif n_obj == 1:
            self.comparator = SingleObjectiveComparator(
                direction=direction[0], eps_cv=eps_cv, eps_obj=eps_obj
            )
        else:
            self.comparator = NSGA2Comparator(
                direction=direction, eps_cv=eps_cv, eps_obj=eps_obj
            )

    @property
    def eps(self) -> float:
        """Deprecated. Use eps_cv or eps_obj."""
        warn_deprecated("Problem.eps", "eps_cv or eps_obj", "0.1.0")
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
            Aggregate constraint violation as computed by ``handler.compute_cv``.
            0.0 when no constraints are defined.
        """
        if not self.constraints:
            return np.empty(0, dtype=float), 0.0
        g = np.empty(len(self.constraints), dtype=float)
        for i, c in enumerate(self.constraints):
            g[i] = c.evaluate(x)
        cv = self.handler.compute_cv(self.constraints, x, g)
        return g, float(cv)

    def evaluate(self, x: np.ndarray, g: np.ndarray | None = None) -> np.ndarray:
        """
        Evaluate the objective function at given solution x.

        After computing the raw objective, ``handler.augment_objective`` is
        applied so that penalty-based or augmented-Lagrangian handlers can
        transform the objective using constraint information. The default
        ``StaticToleranceHandler`` leaves the objective unchanged.

        Parameters
        ----------
        x : np.ndarray
            The solution to evaluate.
        g : np.ndarray, optional
            Pre-computed raw constraint values g(x), shape = (n_constraints, ).
            When None, constraints are evaluated internally if any are defined.
            Pass this to avoid re-evaluating constraints when ``g`` is already
            available (e.g. from :meth:`evaluate_constraints`).

        Returns
        -------
        np.ndarray
            The objective value(s) at solution x. shape = (n_obj, )
        """
        result = self.func(x)
        f = np.atleast_1d(np.asarray(result, dtype=float))
        if g is None:
            g, _ = self.evaluate_constraints(x)
        return self.handler.augment_objective(f, self.constraints, x, g)
