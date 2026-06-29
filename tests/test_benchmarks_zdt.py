"""Tests for saealib.benchmarks.zdt.

Structural checks (shape, bounds, direction) and Pareto-front spot checks
for ZDT1-4, ZDT6. Pareto-optimal solutions are identified by setting all
non-primary variables to 0, which yields g = 1 for ZDT1-4 and g = 1 for
ZDT6 when the rest-variables sum to 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from saealib.benchmarks.zdt import zdt1, zdt2, zdt3, zdt4, zdt5, zdt6
from saealib.problem.problem import Problem

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _pareto_x(problem: Problem, x1: float) -> np.ndarray:
    """Return x with x[0]=x1 and remaining variables = lower bound."""
    x = np.array(problem.lb, dtype=float)
    x[0] = x1
    return x


# ---------------------------------------------------------------------------
# structural tests (parametrised over all five functions)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,n_var",
    [
        (zdt1, 30),
        (zdt2, 30),
        (zdt3, 30),
        (zdt4, 10),
        (zdt6, 10),
    ],
)
class TestZDTStructure:
    def test_returns_problem(self, factory, n_var):
        assert isinstance(factory(), Problem)

    def test_default_n_var(self, factory, n_var):
        p = factory()
        assert p.dim == n_var

    def test_custom_n_var(self, factory, n_var):
        p = factory(n_var=5)
        assert p.dim == 5

    def test_n_obj(self, factory, n_var):
        assert factory().n_obj == 2

    def test_direction_minimize_both(self, factory, n_var):
        d = factory().direction
        np.testing.assert_array_equal(d, [-1.0, -1.0])

    def test_evaluate_shape(self, factory, n_var):
        p = factory()
        x = (np.array(p.lb) + np.array(p.ub)) / 2.0
        f = p.evaluate(x)
        assert f.shape == (2,)

    def test_bounds_length(self, factory, n_var):
        p = factory()
        assert len(p.lb) == p.dim
        assert len(p.ub) == p.dim

    def test_no_constraints(self, factory, n_var):
        assert factory().n_constraints == 0


# ---------------------------------------------------------------------------
# ZDT1 — Pareto front: f2 = 1 - sqrt(f1)  (Eq. 7)
# ---------------------------------------------------------------------------


class TestZDT1ParetoFront:
    def test_pareto_point_f1_half(self):
        p = zdt1()
        x = _pareto_x(p, 0.5)
        f = p.evaluate(x)
        assert f[0] == pytest.approx(0.5)
        assert f[1] == pytest.approx(1.0 - np.sqrt(0.5), rel=1e-9)

    def test_pareto_point_f1_zero(self):
        p = zdt1()
        x = _pareto_x(p, 0.0)
        f = p.evaluate(x)
        assert f[0] == pytest.approx(0.0)
        assert f[1] == pytest.approx(1.0)

    def test_off_pareto_increases_f2(self):
        p = zdt1()
        x_on = _pareto_x(p, 0.5)
        x_off = x_on.copy()
        x_off[1] = 0.5  # g > 1 → f2 increases
        assert p.evaluate(x_off)[1] > p.evaluate(x_on)[1]


# ---------------------------------------------------------------------------
# ZDT2 — Pareto front: f2 = 1 - f1**2  (Eq. 8)
# ---------------------------------------------------------------------------


class TestZDT2ParetoFront:
    def test_pareto_point_f1_half(self):
        p = zdt2()
        x = _pareto_x(p, 0.5)
        f = p.evaluate(x)
        assert f[0] == pytest.approx(0.5)
        assert f[1] == pytest.approx(1.0 - 0.5**2, rel=1e-9)


# ---------------------------------------------------------------------------
# ZDT3 — Pareto front: f2 = 1 - sqrt(f1) - f1*sin(10*pi*f1)  (Eq. 9)
# ---------------------------------------------------------------------------


class TestZDT3ParetoFront:
    def test_pareto_point_f1_half(self):
        p = zdt3()
        x = _pareto_x(p, 0.5)
        f = p.evaluate(x)
        expected_f2 = 1.0 - np.sqrt(0.5) - 0.5 * np.sin(10.0 * np.pi * 0.5)
        assert f[0] == pytest.approx(0.5)
        assert f[1] == pytest.approx(expected_f2, rel=1e-9)


# ---------------------------------------------------------------------------
# ZDT4 — mixed bounds; global front at g=1  (Eq. 10)
# ---------------------------------------------------------------------------


class TestZDT4:
    def test_bounds_x1(self):
        p = zdt4()
        assert p.lb[0] == pytest.approx(0.0)
        assert p.ub[0] == pytest.approx(1.0)

    def test_bounds_rest(self):
        p = zdt4()
        assert all(p.lb[i] == pytest.approx(-5.0) for i in range(1, p.dim))
        assert all(p.ub[i] == pytest.approx(5.0) for i in range(1, p.dim))

    def test_global_pareto_point(self):
        # x[1:] = 0 → g = 1 + 10*(m-1) + sum(0 - 10*cos(0)) = 1+90-90 = 1
        p = zdt4()
        x = _pareto_x(p, 0.5)  # lb for x[1:] is -5; need 0 explicitly
        x[1:] = 0.0
        f = p.evaluate(x)
        assert f[0] == pytest.approx(0.5)
        assert f[1] == pytest.approx(1.0 - np.sqrt(0.5), rel=1e-9)


# ---------------------------------------------------------------------------
# ZDT5 — deceptive binary problem  (Eq. 11)
# ---------------------------------------------------------------------------


class TestZDT5Structure:
    def test_returns_problem(self):
        assert isinstance(zdt5(), Problem)

    def test_default_n_var(self):
        assert zdt5().dim == 80  # 30 + 10*5

    def test_custom_params(self):
        p = zdt5(n_bits_b1=10, n_bits_rest=3, n_rest=4)
        assert p.dim == 10 + 4 * 3

    def test_n_obj(self):
        assert zdt5().n_obj == 2

    def test_direction_minimize_both(self):
        np.testing.assert_array_equal(zdt5().direction, [-1.0, -1.0])

    def test_evaluate_shape(self):
        p = zdt5()
        f = p.evaluate(np.zeros(p.dim))
        assert f.shape == (2,)

    def test_bounds_all_binary(self):
        p = zdt5()
        assert all(lb == pytest.approx(0.0) for lb in p.lb)
        assert all(ub == pytest.approx(1.0) for ub in p.ub)

    def test_no_constraints(self):
        assert zdt5().n_constraints == 0


class TestZDT5ParetoFront:
    def test_global_pareto_b1_all_zero(self):
        # u(b1)=0 → f1=1; all rest bits=1 → u_i=5 → v=1 → g=10; f2=10
        p = zdt5()
        x = np.zeros(p.dim)
        x[30:] = 1
        f = p.evaluate(x)
        assert f[0] == pytest.approx(1.0)
        assert f[1] == pytest.approx(10.0)

    def test_global_pareto_b1_all_one(self):
        # u(b1)=30 → f1=31; all rest bits=1 → g=10; f2=10/31
        p = zdt5()
        x = np.ones(p.dim)
        f = p.evaluate(x)
        assert f[0] == pytest.approx(31.0)
        assert f[1] == pytest.approx(10.0 / 31.0, rel=1e-9)

    def test_off_pareto_rest_all_zero(self):
        # all bits=0 → u_i=0 → v=2 for each → g=20; f2=20/1=20
        p = zdt5()
        x = np.zeros(p.dim)
        f = p.evaluate(x)
        assert f[0] == pytest.approx(1.0)
        assert f[1] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# ZDT6 — non-uniform distribution; front at g=1  (Eq. 12)
# ---------------------------------------------------------------------------


class TestZDT6ParetoFront:
    def test_pareto_point_x1_zero(self):
        # x1=0 → f1 = 1 - exp(0)*sin^6(0) = 1 - 0 = 1; g=1 → f2=0
        p = zdt6()
        x = _pareto_x(p, 0.0)
        f = p.evaluate(x)
        assert f[0] == pytest.approx(1.0)
        assert f[1] == pytest.approx(0.0, abs=1e-12)

    def test_pareto_point_nonzero(self):
        p = zdt6()
        x1 = 0.5
        x = _pareto_x(p, x1)
        f = p.evaluate(x)
        expected_f1 = 1.0 - np.exp(-4.0 * x1) * np.sin(6.0 * np.pi * x1) ** 6
        expected_f2 = 1.0 - (expected_f1 / 1.0) ** 2
        assert f[0] == pytest.approx(expected_f1, rel=1e-9)
        assert f[1] == pytest.approx(expected_f2, rel=1e-9)
