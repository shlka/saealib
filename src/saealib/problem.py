"""
Problem module.

This module defines the optimization problem and constraint classes.
Comparator classes and Pareto-related utilities are in saealib.comparators.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from saealib.comparators import (
    Comparator,
    NSGA2Comparator,
    SingleObjectiveComparator,
)

if TYPE_CHECKING:
    from saealib.population import Population


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

    def violation_from_value(self, g: float) -> float:
        """
        Return the violation for an already-evaluated raw value g(x).

        This is the single source of truth for the per-constraint violation
        formula: :meth:`violation`, :meth:`evaluate_with_violation`, and
        :class:`StaticToleranceHandler` all delegate to it. Subclasses (e.g.
        :class:`EqualityConstraint`) override this one method to change how a
        raw value maps to a violation, without re-evaluating ``func``.

        Parameters
        ----------
        g : float
            Raw constraint value g(x).

        Returns
        -------
        float
            Constraint violation: max(0, g - threshold).
        """
        return max(0.0, g - self.threshold)

    def violation(self, x: np.ndarray) -> float:
        """Return constraint violation: max(0, g(x) - threshold)."""
        return self.violation_from_value(self.evaluate(x))

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
        return g, self.violation_from_value(g)

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


class EqualityConstraint(InequalityConstraint):
    """
    A single equality constraint: h(x) = 0, satisfied within a tolerance.

    The violation is ``max(0, |h(x)| - tolerance)``, reusing the inequality
    violation path so that mixed inequality/equality problems aggregate through
    the same :class:`ConstraintHandler`. Only :meth:`violation_from_value` is
    overridden; :meth:`violation` and :meth:`evaluate_with_violation` inherit
    the equality behavior automatically.

    Parameters
    ----------
    func : callable
        Constraint function h(x). Should return a scalar float.
    tolerance : float, optional
        Feasibility tolerance for ``|h(x)| <= tolerance``. Default: 1e-6.
        Set ``tolerance=0.0`` to defer the feasibility threshold to a handler
        such as ``EpsilonConstraintHandler``, where the threshold is managed
        externally rather than baked into the constraint.

    Notes
    -----
    Provide an analytical Jacobian by overriding :meth:`gradient` to return
    ``∇h(x)``; this enables gradient-based constraint handlers (e.g. repair).
    """

    def __init__(self, func: callable, tolerance: float = 1e-6):
        super().__init__(func, threshold=0.0)
        self.tolerance = tolerance

    def violation_from_value(self, g: float) -> float:
        """Return equality violation: max(0, |h(x)| - tolerance)."""
        return max(0.0, abs(g) - self.tolerance)


class ConstraintHandler(ABC):
    """
    Pluggable constraint-processing strategy.

    A ``ConstraintHandler`` exposes the constraint-handling lifecycle as a set
    of overridable hooks, decoupling the per-constraint violation formula and
    the aggregation method from the core :class:`Problem`. This lets research
    code swap in alternative strategies (e.g. ε-constraint, penalty functions,
    gradient-based repair, augmented Lagrangian) without forking core classes.

    Lifecycle::

        Ask            -> [repair(x, constraints)]
                       -> evaluate f, g
                       -> [compute_cv(constraints, x, g)]        -> cv
                       -> [augment_objective(f, constraints, x, g)] -> f'
        Tell           -> Comparator(f', cv) with eps_cv = feasibility_threshold
        Generation end -> [on_generation_end(gen, population)]

    Only :meth:`compute_cv` is abstract; the remaining hooks default to no-ops
    so that subclasses implement just what they need.
    """

    def repair(
        self, x: np.ndarray, constraints: list[InequalityConstraint]
    ) -> np.ndarray:
        """
        Repair a design vector before evaluation.

        Parameters
        ----------
        x : np.ndarray
            The design vector to repair. shape = (dim, )
        constraints : list[InequalityConstraint]
            The problem's inequality constraints.

        Returns
        -------
        np.ndarray
            The (possibly) repaired design vector. The default returns ``x``
            unchanged.
        """
        return x

    @abstractmethod
    def compute_cv(
        self,
        constraints: list[InequalityConstraint],
        x: np.ndarray,
        g: np.ndarray,
    ) -> float:
        """
        Aggregate raw constraint values into a scalar constraint violation.

        Parameters
        ----------
        constraints : list[InequalityConstraint]
            The problem's inequality constraints, aligned with ``g``.
        x : np.ndarray
            The evaluated design vector. shape = (dim, )
        g : np.ndarray
            Raw constraint values g(x). shape = (n_constraints, )

        Returns
        -------
        float
            Aggregate constraint violation (``cv``); ``0.0`` means feasible.
        """
        ...

    def augment_objective(
        self,
        f: np.ndarray,
        constraints: list[InequalityConstraint],
        x: np.ndarray,
        g: np.ndarray,
    ) -> np.ndarray:
        """
        Transform objective values using constraint information.

        Used by penalty-based or augmented-Lagrangian strategies. The default
        is the identity (objectives are returned unchanged).

        Parameters
        ----------
        f : np.ndarray
            Raw objective values. shape = (n_obj, )
        constraints : list[InequalityConstraint]
            The problem's inequality constraints, aligned with ``g``.
        x : np.ndarray
            The evaluated design vector. shape = (dim, )
        g : np.ndarray
            Raw constraint values g(x). shape = (n_constraints, )

        Returns
        -------
        np.ndarray
            The (possibly) augmented objective values. shape = (n_obj, )
        """
        return f

    @property
    def feasibility_threshold(self) -> float:
        """Constraint-violation threshold below which a solution is feasible."""
        return 1e-6

    def on_generation_end(self, gen: int, population: Population) -> None:
        """
        Run end-of-generation bookkeeping.

        Used by adaptive strategies (e.g. ε-level control) that update their
        internal state between generations. The default is a no-op.

        Parameters
        ----------
        gen : int
            The generation index that just finished.
        population : Population
            The current population.
        """


class StaticToleranceHandler(ConstraintHandler):
    """
    Default constraint handler reproducing the static-tolerance behavior.

    The constraint violation is the sum of per-constraint violations
    ``sum(c_i.violation_from_value(g_i))`` and the feasibility threshold is a
    fixed ``eps_cv``. Each constraint maps its own raw value to a violation,
    so inequality (``max(0, g - threshold)``) and equality
    (``max(0, |h| - tolerance)``) constraints can be mixed freely. Objectives
    are not augmented.

    Parameters
    ----------
    eps_cv : float, optional
        Feasibility threshold returned by :attr:`feasibility_threshold`.
        Default: 1e-6.
    """

    def __init__(self, eps_cv: float = 1e-6):
        self._eps_cv = eps_cv

    def compute_cv(
        self,
        constraints: list[InequalityConstraint],
        x: np.ndarray,
        g: np.ndarray,
    ) -> float:
        """Return the sum of per-constraint ``c_i.violation_from_value(g_i)``."""
        cv = 0.0
        for gi, c in zip(g, constraints):
            cv += c.violation_from_value(float(gi))
        return cv

    @property
    def feasibility_threshold(self) -> float:
        """Fixed feasibility threshold ``eps_cv``."""
        return self._eps_cv


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
    handler : ConstraintHandler
        Constraint-handling strategy used to aggregate violations and augment
        objectives.
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
        handler : ConstraintHandler, optional
            Constraint-handling strategy. If None, a StaticToleranceHandler
            (sum-of-violations, fixed eps_cv) is used, reproducing the default
            behavior.
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
        self.handler = (
            handler if handler is not None else StaticToleranceHandler(eps_cv=eps_cv)
        )

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
