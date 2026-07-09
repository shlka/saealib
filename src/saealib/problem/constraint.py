"""Constraint classes and handlers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

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

    def __init__(self, func: Callable[..., Any], threshold: float = 0.0):
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

    def __init__(self, func: Callable[..., Any], tolerance: float = 1e-6):
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

        Ask            -> [repair(x, constraints, lb, ub)]
                       -> evaluate f, g
                       -> [compute_cv(constraints, x, g)]        -> cv
                       -> [augment_objective(f, constraints, x, g)] -> f'
        Tell           -> Comparator(f', cv) with eps_cv = feasibility_threshold
        Generation end -> [on_generation_end(gen, population)]

    Only :meth:`compute_cv` is abstract; the remaining hooks have sensible
    defaults so that subclasses implement just what they need.
    """

    def repair(
        self,
        x: np.ndarray,
        constraints: list[InequalityConstraint],
        lb: np.ndarray,
        ub: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Repair a design vector before evaluation.

        Called by the algorithm after each variation operator (crossover /
        mutation) and before objective evaluation. The default clips ``x`` to
        ``[lb, ub]``, reproducing the behaviour of ``repair_clipping``.

        Subclasses override this to add domain-constraint repair (e.g. a
        Newton step toward an equality-constraint manifold) on top of—or
        instead of—bounds clipping.

        Parameters
        ----------
        x : np.ndarray
            The design vector to repair. shape = (dim, )
        constraints : list[InequalityConstraint]
            The problem's domain constraints.
        lb : np.ndarray
            Lower bounds. shape = (dim, )
        ub : np.ndarray
            Upper bounds. shape = (dim, )
        **kwargs
            Reserved for future use (e.g. parent solution for reflection
            repair).

        Returns
        -------
        np.ndarray
            The repaired design vector. shape = (dim, )
        """
        return np.clip(x, lb, ub)

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


class EpsilonConstraintHandler(ConstraintHandler):
    """
    Dynamic ε-constraint handler.

    The feasibility threshold ε starts at ``schedule(0)`` and is updated
    to ``schedule(gen)`` at the end of each generation via
    :meth:`on_generation_end`. As ε decreases toward 0 the population is
    gradually driven into the feasible region.

    Each call to :meth:`on_generation_end` sets
    ``comparator.eps_cv = feasibility_threshold`` via the runner, so the
    comparator's feasibility criterion tightens in sync with ε.

    Parameters
    ----------
    schedule : callable
        Maps generation index (int) → current ε value (float).
        ``schedule(0)`` is the initial ε; it should decrease toward 0
        over generations.

    Notes
    -----
    :meth:`compute_cv` aggregates violations as:

    - *Equality* constraints: ``|h(x)|``  (raw absolute value; the
      ``EqualityConstraint.tolerance`` field is intentionally bypassed so
      that ε controls the feasibility threshold globally).
    - *Inequality* constraints: ``max(0, g(x) - threshold)``  (standard
      positive part; the constraint's ``threshold`` still applies).

    This is a simplified form of the sum-of-constraint-violation measure
    in Eq. (7) of Mezura-Montes & Coello Coello (2011) — inequality
    violations are not squared and values are not normalised.

    References
    ----------
    Mezura-Montes, E., & Coello Coello, C. A. (2011).
    *Constraint-handling in nature-inspired numerical optimization: Past,
    present and future.*
    Swarm and Evolutionary Computation, 1(4), 173-194.
    https://doi.org/10.1016/j.swevo.2011.10.001
    """

    def __init__(self, schedule: Callable[[int], float]):
        self._schedule = schedule
        self._eps: float = float(schedule(0))

    def compute_cv(
        self,
        constraints: list[InequalityConstraint],
        x: np.ndarray,
        g: np.ndarray,
    ) -> float:
        """Return ``|h|`` for equality and ``max(0, g-threshold)`` for inequality."""
        cv = 0.0
        for gi, c in zip(g, constraints):
            if isinstance(c, EqualityConstraint):
                cv += abs(float(gi))
            else:
                cv += max(0.0, float(gi) - c.threshold)
        return cv

    @property
    def feasibility_threshold(self) -> float:
        """Current ε; the comparator treats ``cv <= ε`` as feasible."""
        return self._eps

    def on_generation_end(self, gen: int, population: Population) -> None:
        """Update internal ε to ``schedule(gen)``."""
        self._eps = float(self._schedule(gen))


class GradientRepairHandler(ConstraintHandler):
    """
    Gradient-based constraint repair handler.

    For each :class:`EqualityConstraint` whose :meth:`~InequalityConstraint.gradient`
    returns a non-``None`` vector, applies one Newton-like step that projects
    the design vector toward the constraint manifold ``h(x) = 0``::

        x <- x - h(x) * grad_h(x) / (||grad_h(x)||^2 + ridge)

    The step is repeated ``max_iter`` times.  After all iterations the result
    is clipped to ``[lb, ub]`` so bounds feasibility is always guaranteed.

    :class:`InequalityConstraint` objects and any :class:`EqualityConstraint`
    whose ``gradient()`` returns ``None`` are skipped during repair; they still
    contribute to ``cv`` via :meth:`compute_cv`.

    Parameters
    ----------
    max_iter : int, optional
        Number of Newton steps per call. Default: 1.
    ridge : float, optional
        Regularisation term added to ``‖∇h‖²`` for numerical stability.
        Default: 1e-12.

    References
    ----------
    Chootinan, P., & Chen, A. (2006).
    *Constraint handling in genetic algorithms using a gradient-based repair
    method.*
    Computers & Operations Research, 33(8), 2263-2281.
    https://doi.org/10.1016/j.cor.2005.02.002
    """

    def __init__(self, max_iter: int = 1, ridge: float = 1e-12):
        self.max_iter = max_iter
        self.ridge = ridge

    def repair(
        self,
        x: np.ndarray,
        constraints: list[InequalityConstraint],
        lb: np.ndarray,
        ub: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Apply Newton steps toward equality-constraint manifolds, then clip."""
        x = x.copy()
        for _ in range(self.max_iter):
            for c in constraints:
                if not isinstance(c, EqualityConstraint):
                    continue
                grad = c.gradient(x)
                if grad is None:
                    continue
                h = c.evaluate(x)
                x = x - h * grad / (np.dot(grad, grad) + self.ridge)
        return np.clip(x, lb, ub)

    def compute_cv(
        self,
        constraints: list[InequalityConstraint],
        x: np.ndarray,
        g: np.ndarray,
    ) -> float:
        """Return sum of per-constraint violations (same as StaticToleranceHandler)."""
        cv = 0.0
        for gi, c in zip(g, constraints):
            cv += c.violation_from_value(float(gi))
        return cv


def linear_epsilon_schedule(eps0: float, n_gen: int) -> Callable[[int], float]:
    """
    Return a linear ε schedule: ``eps0`` at gen 0, ``0.0`` at gen ``n_gen``.

    Parameters
    ----------
    eps0 : float
        Initial ε value (gen 0).
    n_gen : int
        Generation at which ε reaches 0.

    Returns
    -------
    callable
        ``schedule(gen) -> float``
    """

    def _schedule(gen: int) -> float:
        return max(0.0, eps0 * (1.0 - gen / n_gen))

    return _schedule


def exponential_epsilon_schedule(eps0: float, decay: float) -> Callable[[int], float]:
    """
    Return an exponential ε schedule: ``eps0 * decay**gen``.

    Parameters
    ----------
    eps0 : float
        Initial ε value (gen 0).
    decay : float
        Multiplicative decay factor per generation (0 < decay < 1).

    Returns
    -------
    callable
        ``schedule(gen) -> float``
    """

    def _schedule(gen: int) -> float:
        return eps0 * (decay**gen)

    return _schedule
