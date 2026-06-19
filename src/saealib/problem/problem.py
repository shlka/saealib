"""Problem class for optimization."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
from saealib.variables import (
    CategoricalVariable,
    ContinuousVariable,
    IntegerVariable,
    Variable,
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
    variables : list[Variable]
        Per-dimension variable definitions.
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
        func: Callable[..., Any],
        dim: int,
        n_obj: int,
        direction: np.ndarray,
        lb: list[float] | None = None,
        ub: list[float] | None = None,
        eps: float | None = None,
        comparator: Comparator | None = None,
        constraints: list[InequalityConstraint] | None = None,
        *,
        eps_cv: float = 1e-6,
        eps_obj: float = 1e-6,
        handler: ConstraintHandler | None = None,
        variables: list[Variable] | None = None,
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
        lb : list[float], optional
            Lower bounds for design variables. length = dim.
            Required when *variables* is not provided.
        ub : list[float], optional
            Upper bounds for design variables. length = dim.
            Required when *variables* is not provided.
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
        variables : list[Variable], optional
            Per-dimension variable definitions.  When provided, *lb* and *ub*
            are derived from the variable bounds.  ``len(variables)`` must equal
            *dim*.
        """
        if eps is not None:
            warn_deprecated("eps", "eps_cv and eps_obj", "0.1.0")
            eps_cv = eps_obj = eps
        direction = np.asarray(direction, dtype=float)
        if not np.all(np.abs(direction) == 1):
            raise ValidationError("direction elements must be +1 or -1")

        if variables is not None:
            if len(variables) != dim:
                raise ValidationError(
                    f"len(variables)={len(variables)} does not match dim={dim}"
                )
            self.variables: list[Variable] = list(variables)
            self.lb = np.array([v.lb for v in self.variables], dtype=float)
            self.ub = np.array([v.ub for v in self.variables], dtype=float)
        else:
            if lb is None or ub is None:
                raise ValidationError(
                    "lb and ub are required when variables is not provided"
                )
            self.lb = np.asarray(lb, dtype=float)
            self.ub = np.asarray(ub, dtype=float)
            self.variables = [
                ContinuousVariable(float(self.lb[i]), float(self.ub[i]))
                for i in range(dim)
            ]

        # Cache type masks (computed once).
        self._integer_mask = np.array(
            [isinstance(v, IntegerVariable) for v in self.variables]
        )
        self._categorical_mask = np.array(
            [isinstance(v, CategoricalVariable) for v in self.variables]
        )
        self._continuous_mask = ~(self._integer_mask | self._categorical_mask)
        self._n_categories = np.array(
            [
                v.n_categories if isinstance(v, CategoricalVariable) else 0
                for v in self.variables
            ],
            dtype=int,
        )

        self.dim = dim
        self.n_obj = n_obj
        self.direction = direction
        self.eps_cv = eps_cv
        self.eps_obj = eps_obj
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

    @property
    def continuous_mask(self) -> np.ndarray:
        """Boolean mask of continuous dimensions. shape = (dim,)."""
        return self._continuous_mask

    @property
    def integer_mask(self) -> np.ndarray:
        """Boolean mask of integer dimensions. shape = (dim,)."""
        return self._integer_mask

    @property
    def categorical_mask(self) -> np.ndarray:
        """Boolean mask of categorical dimensions. shape = (dim,)."""
        return self._categorical_mask

    @property
    def n_categories(self) -> np.ndarray:
        """Category counts per dimension (0 for non-categorical). shape = (dim,)."""
        return self._n_categories

    def repair(self, x: np.ndarray) -> np.ndarray:
        """Project *x* onto valid variable domains.

        Rounds integer dimensions to the nearest integer and clips to bounds.
        Rounds categorical dimensions to the nearest valid index.
        Clips continuous dimensions to ``[lb, ub]``.

        Parameters
        ----------
        x : np.ndarray
            Design variable array. shape = ``(dim,)`` or ``(n, dim)``.

        Returns
        -------
        np.ndarray
            Repaired array, same shape as *x*.
        """
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 1
        if scalar:
            x = x[np.newaxis, :]
        result = x.copy()
        for i, v in enumerate(self.variables):
            result[:, i] = v.repair(result[:, i])
        return result[0] if scalar else result

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
