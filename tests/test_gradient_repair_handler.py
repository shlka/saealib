"""Tests for GradientRepairHandler."""

import numpy as np
import pytest

from saealib import (
    ConstraintHandler,
    EqualityConstraint,
    GradientRepairHandler,
    InequalityConstraint,
    StaticToleranceHandler,
)

LB = np.zeros(2)
UB = np.ones(2)


def _linear_eq(gradient_vec):
    """EqualityConstraint h(x) = dot(gradient_vec, x) - 1 with analytic gradient."""

    class _LinearEq(EqualityConstraint):
        def gradient(self, x):
            return gradient_vec.copy()

    return _LinearEq(func=lambda x: float(np.dot(gradient_vec, x)) - 1.0, tolerance=0.0)


# ---------------------------------------------------------------------------
# Subclass / export checks
# ---------------------------------------------------------------------------


class TestGradientRepairHandlerType:
    def test_is_constraint_handler(self):
        assert isinstance(GradientRepairHandler(), ConstraintHandler)

    def test_default_params(self):
        h = GradientRepairHandler()
        assert h.max_iter == 1
        assert h.epsilon == pytest.approx(1e-6)

    def test_custom_params(self):
        h = GradientRepairHandler(max_iter=5, epsilon=1e-3)
        assert h.max_iter == 5
        assert h.epsilon == pytest.approx(1e-3)


# ---------------------------------------------------------------------------
# repair: Newton step reduces |h(x)|
# ---------------------------------------------------------------------------


class TestGradientRepairHandlerRepair:
    def test_newton_step_reduces_violation(self):
        # h(x) = x[0] + x[1] - 1, grad = [1, 1]
        # x = [0.8, 0.8] -> h = 0.6, after 1 step -> h ≈ 0
        c = _linear_eq(np.array([1.0, 1.0]))
        h = GradientRepairHandler()
        x = np.array([0.8, 0.8])
        x_rep = h.repair(x, [c], LB, UB)
        assert abs(c.evaluate(x_rep)) < abs(c.evaluate(x))

    def test_newton_step_exact_for_linear(self):
        # One Newton step is exact for linear constraints.
        # h(x) = x[0] + x[1] - 1, x = [0.8, 0.8] -> x_rep = [0.5, 0.5]
        c = _linear_eq(np.array([1.0, 1.0]))
        h = GradientRepairHandler()
        x_rep = h.repair(np.array([0.8, 0.8]), [c], LB, UB)
        assert abs(c.evaluate(x_rep)) == pytest.approx(0.0, abs=1e-10)

    def test_input_not_mutated(self):
        c = _linear_eq(np.array([1.0, 1.0]))
        h = GradientRepairHandler()
        x = np.array([0.8, 0.8])
        x_orig = x.copy()
        h.repair(x, [c], LB, UB)
        np.testing.assert_array_equal(x, x_orig)

    def test_clips_to_bounds_after_newton(self):
        # h(x) = x[0] - 3.0, grad = [1, 0]
        # x = [0.0, 0.5] -> Newton step moves x[0] to 3.0 (out of ub=1)
        # after clip: x[0] = 1.0
        class _OutOfBoundsEq(EqualityConstraint):
            def gradient(self, x):
                return np.array([1.0, 0.0])

        c = _OutOfBoundsEq(func=lambda x: float(x[0]) - 3.0, tolerance=0.0)
        h = GradientRepairHandler()
        x_rep = h.repair(np.array([0.0, 0.5]), [c], LB, UB)
        assert x_rep[0] == pytest.approx(1.0)
        assert x_rep[1] == pytest.approx(0.5)

    def test_max_iter_improves_convergence(self):
        # For a linear constraint one step is exact; verify iter=2 also works.
        c = _linear_eq(np.array([1.0, 1.0]))
        h = GradientRepairHandler(max_iter=2)
        x_rep = h.repair(np.array([0.8, 0.8]), [c], LB, UB)
        assert abs(c.evaluate(x_rep)) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# repair: violated constraints are now repaired regardless of type/gradient
# ---------------------------------------------------------------------------


def _total_violation(constraints, x):
    return sum(c.violation_from_value(c.evaluate(x)) for c in constraints)


class TestGradientRepairHandlerRepairsAllViolatedConstraints:
    def test_violated_inequality_constraint_is_repaired(self):
        # g(x) = x[0] - 0.5, threshold=0.0; x[0]=0.8 violates it.
        # gradient() is None -> falls back to the numerical approximation.
        c = InequalityConstraint(lambda x: float(x[0]) - 0.5)
        h = GradientRepairHandler()
        x = np.array([0.8, 0.3])
        x_rep = h.repair(x, [c], LB, UB)
        assert c.evaluate(x_rep) <= c.threshold + 1e-6
        assert x_rep[1] == pytest.approx(0.3)

    def test_satisfied_inequality_constraint_is_left_untouched(self):
        # g(x) = x[0] - 0.5, threshold=0.0; x[0]=0.3 already satisfies it.
        c = InequalityConstraint(lambda x: float(x[0]) - 0.5)
        h = GradientRepairHandler()
        x = np.array([0.3, 0.3])
        x_rep = h.repair(x, [c], LB, UB)
        np.testing.assert_array_equal(x_rep, x)

    def test_equality_without_gradient_uses_numerical_fallback(self):
        # EqualityConstraint with gradient()=None is now repaired via the
        # forward-difference approximation instead of being skipped.
        c = EqualityConstraint(func=lambda x: float(x[0]) - 0.5, tolerance=0.0)
        h = GradientRepairHandler()
        x = np.array([0.3, 0.3])
        x_rep = h.repair(x, [c], LB, UB)
        assert abs(c.evaluate(x_rep)) < abs(c.evaluate(x))

    def test_mixed_violated_constraints_reduce_total_violation(self):
        # Mix: inequality + equality without gradient + equality with
        # gradient, all violated -> jointly repaired via the stacked
        # pseudoinverse update (an overdetermined system, so exact
        # per-constraint satisfaction is not guaranteed, only a net
        # reduction in total violation).
        c_ineq = InequalityConstraint(lambda x: float(x[0]) - 0.5)
        c_eq_no_grad = EqualityConstraint(
            func=lambda x: float(x[1]) - 0.2, tolerance=0.0
        )
        c_eq_grad = _linear_eq(np.array([1.0, 0.0]))  # h = x[0] - 1

        constraints = [c_ineq, c_eq_no_grad, c_eq_grad]
        h = GradientRepairHandler()
        x = np.array([0.8, 0.5])
        x_rep = h.repair(x, constraints, LB, UB)
        assert _total_violation(constraints, x_rep) < _total_violation(constraints, x)

    def test_stops_once_all_constraints_satisfied(self):
        c = _linear_eq(np.array([1.0, 1.0]))
        h = GradientRepairHandler(max_iter=100)
        x_rep = h.repair(np.array([0.8, 0.8]), [c], LB, UB)
        # One step is exact for a linear constraint, so the second
        # iteration sees no violated constraints and breaks immediately.
        assert abs(c.evaluate(x_rep)) == pytest.approx(0.0, abs=1e-10)

    def test_epsilon_stops_before_convergence(self):
        # h(x) = x[0]**2 + x[1]**2 - 1, a nonlinear constraint so one step
        # does not reach the manifold: a large epsilon should stop the
        # iteration early, leaving more residual violation than a tiny
        # epsilon that keeps iterating toward the manifold.
        def _circle():
            return EqualityConstraint(
                func=lambda x: float(x[0] ** 2 + x[1] ** 2) - 1.0, tolerance=0.0
            )

        c_slow, c_full = _circle(), _circle()
        x = np.array([0.9, 0.9])
        h_slow = GradientRepairHandler(max_iter=10, epsilon=0.5)
        h_full = GradientRepairHandler(max_iter=10, epsilon=1e-12)
        slow = h_slow.repair(x, [c_slow], LB, UB)
        full = h_full.repair(x, [c_full], LB, UB)
        assert abs(c_full.evaluate(full)) < abs(c_slow.evaluate(slow))


# ---------------------------------------------------------------------------
# compute_cv: delegates to sum-of-violations
# ---------------------------------------------------------------------------


class TestGradientRepairHandlerComputeCV:
    def test_compute_cv_matches_static_tolerance(self):
        constraints = [
            InequalityConstraint(lambda x: x[0] - 0.3),
            EqualityConstraint(func=lambda x: float(x[1]) - 0.5, tolerance=0.0),
        ]
        x = np.array([0.5, 0.8])
        g = np.array([0.2, 0.3])
        cv_grad = GradientRepairHandler().compute_cv(constraints, x, g)
        cv_static = StaticToleranceHandler().compute_cv(constraints, x, g)
        assert cv_grad == pytest.approx(cv_static)

    def test_compute_cv_zero_when_feasible(self):
        c = _linear_eq(np.array([1.0, 1.0]))
        x = np.array([0.5, 0.5])
        g = np.array([c.evaluate(x)])
        cv = GradientRepairHandler().compute_cv([c], x, g)
        assert cv == pytest.approx(0.0)
