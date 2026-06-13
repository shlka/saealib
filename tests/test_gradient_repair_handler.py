"""Tests for GradientRepairHandler (Issue #110)."""

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
        assert h.ridge == pytest.approx(1e-12)

    def test_custom_params(self):
        h = GradientRepairHandler(max_iter=5, ridge=1e-6)
        assert h.max_iter == 5
        assert h.ridge == pytest.approx(1e-6)


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
# repair: skipping rules
# ---------------------------------------------------------------------------


class TestGradientRepairHandlerSkipping:
    def test_inequality_constraint_skipped(self):
        # InequalityConstraint should not be touched by repair.
        c = InequalityConstraint(lambda x: float(x[0]) - 0.5)
        h = GradientRepairHandler()
        x = np.array([0.3, 0.3])
        x_rep = h.repair(x, [c], LB, UB)
        # Only clip is applied; x is already in bounds so it is unchanged.
        np.testing.assert_array_equal(x_rep, x)

    def test_equality_without_gradient_skipped(self):
        # EqualityConstraint with gradient()=None is not repaired.
        c = EqualityConstraint(func=lambda x: float(x[0]) - 0.5, tolerance=0.0)
        h = GradientRepairHandler()
        x = np.array([0.3, 0.3])
        x_rep = h.repair(x, [c], LB, UB)
        np.testing.assert_array_equal(x_rep, x)

    def test_mixed_constraints_only_eq_with_gradient_repaired(self):
        # Mix: inequality (skip) + equality without gradient (skip)
        #      + equality with gradient (repair).
        c_ineq = InequalityConstraint(lambda x: float(x[0]) - 0.5)
        c_eq_no_grad = EqualityConstraint(
            func=lambda x: float(x[1]) - 0.2, tolerance=0.0
        )
        c_eq_grad = _linear_eq(np.array([1.0, 0.0]))  # h = x[0] - 1

        h = GradientRepairHandler()
        x = np.array([0.8, 0.5])
        x_rep = h.repair(x, [c_ineq, c_eq_no_grad, c_eq_grad], LB, UB)
        # c_eq_grad: x[0] -> 1.0 (then clipped to 1.0); x[1] unchanged
        assert x_rep[0] == pytest.approx(1.0)
        assert x_rep[1] == pytest.approx(0.5)


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
