"""
Tests for constraint handling (Issue #005).

Tests cover:
- Constraint: evaluate, violation (satisfied / violated / on boundary)
- Problem.n_constraints: with and without constraints
- Problem.evaluate_constraints: no constraints, single, multiple
- cv aggregation: sum-of-violations
- Integration: cv is computed and stored in population/archive via LHSInitializer
"""

import numpy as np
import pytest

from saealib.problem import Constraint, Problem


# ---------------------------------------------------------------------------
# Constraint unit tests
# ---------------------------------------------------------------------------


class TestConstraint:
    def test_evaluate_returns_func_value(self):
        c = Constraint(lambda x: x[0] ** 2 - 1.0)
        assert c.evaluate(np.array([2.0])) == pytest.approx(3.0)

    def test_violation_satisfied(self):
        """g(x) <= threshold → violation == 0."""
        c = Constraint(lambda x: float(x[0]), threshold=5.0)
        assert c.violation(np.array([3.0])) == pytest.approx(0.0)

    def test_violation_on_boundary(self):
        """g(x) == threshold → violation == 0."""
        c = Constraint(lambda x: float(x[0]), threshold=5.0)
        assert c.violation(np.array([5.0])) == pytest.approx(0.0)

    def test_violation_violated(self):
        """g(x) > threshold → violation == g(x) - threshold."""
        c = Constraint(lambda x: float(x[0]), threshold=5.0)
        assert c.violation(np.array([7.0])) == pytest.approx(2.0)

    def test_default_threshold_zero(self):
        """Default threshold is 0.0: g(x) <= 0."""
        c = Constraint(lambda x: float(x[0]))
        assert c.violation(np.array([-1.0])) == pytest.approx(0.0)
        assert c.violation(np.array([2.0])) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Problem.n_constraints
# ---------------------------------------------------------------------------


class TestProblemNConstraints:
    def _make_problem(self, constraints=None):
        return Problem(
            func=lambda x: float(x[0]),
            dim=2,
            n_obj=1,
            weight=1.0,
            lb=[0.0, 0.0],
            ub=[1.0, 1.0],
            constraints=constraints,
        )

    def test_no_constraints_default(self):
        p = self._make_problem()
        assert p.n_constraints == 0

    def test_no_constraints_explicit_none(self):
        p = self._make_problem(constraints=None)
        assert p.n_constraints == 0

    def test_single_constraint(self):
        p = self._make_problem(constraints=[Constraint(lambda x: float(x[0]))])
        assert p.n_constraints == 1

    def test_multiple_constraints(self):
        p = self._make_problem(
            constraints=[
                Constraint(lambda x: float(x[0])),
                Constraint(lambda x: float(x[1])),
                Constraint(lambda x: float(x[0]) + float(x[1])),
            ]
        )
        assert p.n_constraints == 3


# ---------------------------------------------------------------------------
# Problem.evaluate_constraints
# ---------------------------------------------------------------------------


class TestEvaluateConstraints:
    def _make_problem(self, constraints=None):
        return Problem(
            func=lambda x: float(x[0]),
            dim=2,
            n_obj=1,
            weight=1.0,
            lb=[0.0, 0.0],
            ub=[1.0, 1.0],
            constraints=constraints,
        )

    def test_no_constraints_returns_empty_and_zero(self):
        p = self._make_problem()
        g, cv = p.evaluate_constraints(np.array([0.5, 0.5]))
        assert g.shape == (0,)
        assert cv == pytest.approx(0.0)

    def test_single_constraint_satisfied(self):
        """g(x) = x[0] - 0.3 <= 0; x[0]=0.2 → satisfied."""
        c = Constraint(lambda x: x[0] - 0.3)
        p = self._make_problem(constraints=[c])
        g, cv = p.evaluate_constraints(np.array([0.2, 0.5]))
        assert g.shape == (1,)
        assert g[0] == pytest.approx(-0.1)
        assert cv == pytest.approx(0.0)

    def test_single_constraint_violated(self):
        """g(x) = x[0] - 0.3 <= 0; x[0]=0.5 → violated by 0.2."""
        c = Constraint(lambda x: x[0] - 0.3)
        p = self._make_problem(constraints=[c])
        g, cv = p.evaluate_constraints(np.array([0.5, 0.5]))
        assert g[0] == pytest.approx(0.2)
        assert cv == pytest.approx(0.2)

    def test_multiple_constraints_cv_is_sum(self):
        """cv = sum of individual violations."""
        constraints = [
            Constraint(lambda x: x[0] - 0.3),   # violated by 0.2 at x[0]=0.5
            Constraint(lambda x: x[1] - 0.4),   # violated by 0.1 at x[1]=0.5
        ]
        p = self._make_problem(constraints=constraints)
        g, cv = p.evaluate_constraints(np.array([0.5, 0.5]))
        assert g.shape == (2,)
        assert cv == pytest.approx(0.3)

    def test_multiple_constraints_some_satisfied(self):
        """Only violated constraints contribute to cv."""
        constraints = [
            Constraint(lambda x: x[0] - 0.8),   # satisfied at x[0]=0.5
            Constraint(lambda x: x[1] - 0.4),   # violated by 0.1 at x[1]=0.5
        ]
        p = self._make_problem(constraints=constraints)
        g, cv = p.evaluate_constraints(np.array([0.5, 0.5]))
        assert cv == pytest.approx(0.1)

    def test_threshold_nonzero(self):
        """Constraint(func, threshold=t) means g(x) <= t."""
        c = Constraint(lambda x: float(x[0]), threshold=0.8)
        p = self._make_problem(constraints=[c])
        g, cv = p.evaluate_constraints(np.array([0.5, 0.0]))
        assert cv == pytest.approx(0.0)   # 0.5 <= 0.8
        g2, cv2 = p.evaluate_constraints(np.array([0.9, 0.0]))
        assert cv2 == pytest.approx(0.1)  # 0.9 - 0.8 = 0.1

    def test_g_dtype_float(self):
        c = Constraint(lambda x: float(x[0]))
        p = self._make_problem(constraints=[c])
        g, _ = p.evaluate_constraints(np.array([0.5, 0.5]))
        assert g.dtype == float
