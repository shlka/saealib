"""Tests for saealib.benchmarks.dtlz.

Structural checks (shape, bounds, direction) and Pareto-front spot checks
for DTLZ1-7.
"""

from __future__ import annotations

import numpy as np
import pytest

from saealib.benchmarks.dtlz import dtlz1, dtlz2, dtlz3, dtlz4, dtlz5, dtlz6, dtlz7
from saealib.problem.problem import Problem

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_FACTORIES = [
    (dtlz1, 3, 5),  # n_obj, k defaults
    (dtlz2, 3, 10),
    (dtlz3, 3, 10),
    (dtlz4, 3, 10),
    (dtlz5, 3, 10),
    (dtlz6, 3, 10),
    (dtlz7, 3, 20),
]


def _pareto_x_dtlz12(p: Problem) -> np.ndarray:
    """Return x on global PF for DTLZ2/3/4: position=0.5, distance=0.5."""
    return np.full(p.dim, 0.5)


# ---------------------------------------------------------------------------
# structural tests (parametrised over all five functions)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory,n_obj,k", _FACTORIES)
class TestDTLZStructure:
    def test_returns_problem(self, factory, n_obj, k):
        assert isinstance(factory(), Problem)

    def test_default_n_var(self, factory, n_obj, k):
        assert factory().dim == n_obj - 1 + k

    def test_custom_params(self, factory, n_obj, k):
        p = factory(n_obj=4, k=3)
        assert p.dim == 4 - 1 + 3

    def test_n_obj(self, factory, n_obj, k):
        assert factory().n_obj == n_obj

    def test_direction_minimize_all(self, factory, n_obj, k):
        d = factory().direction
        np.testing.assert_array_equal(d, np.full(n_obj, -1.0))

    def test_evaluate_shape(self, factory, n_obj, k):
        p = factory()
        f = p.evaluate(np.full(p.dim, 0.5))
        assert f.shape == (n_obj,)

    def test_bounds_all_unit(self, factory, n_obj, k):
        p = factory()
        assert all(lb == pytest.approx(0.0) for lb in p.lb)
        assert all(ub == pytest.approx(1.0) for ub in p.ub)

    def test_no_constraints(self, factory, n_obj, k):
        assert factory().n_constraints == 0


# ---------------------------------------------------------------------------
# DTLZ1 -- linear hyperplane sum(f) = 0.5 when g = 0  (Eq. 6.18-6.19)
# ---------------------------------------------------------------------------


class TestDTLZ1ParetoFront:
    def test_g_zero_at_half(self):
        # x_M = 0.5 => (xi-0.5)^2 = 0, cos(0) = 1 => g = 100*(k-k) = 0
        p = dtlz1()
        x = np.full(p.dim, 0.5)
        f = p.evaluate(x)
        assert pytest.approx(sum(f), rel=1e-9) == 0.5

    def test_pareto_sum_arbitrary_position(self):
        # sum(f) = 0.5*(1+g) = 0.5 when g=0, for any position variables
        p = dtlz1()
        x = np.full(p.dim, 0.5)
        x[0] = 0.2
        x[1] = 0.7
        f = p.evaluate(x)
        assert pytest.approx(sum(f), rel=1e-9) == 0.5

    def test_off_pareto_g_positive(self):
        # x_M != 0.5 => g > 0 => sum(f) > 0.5
        p = dtlz1()
        x = np.full(p.dim, 0.5)
        x[p.n_obj - 1] = 0.0  # perturb one distance variable
        f = p.evaluate(x)
        assert sum(f) > 0.5


# ---------------------------------------------------------------------------
# DTLZ2 -- unit-sphere PF sum(f^2) = 1 when g = 0  (Eq. 6.20)
# ---------------------------------------------------------------------------


class TestDTLZ2ParetoFront:
    def test_pareto_norm(self):
        # x_M = 0.5 => g = 0 => sum(f^2) = 1
        p = dtlz2()
        f = p.evaluate(_pareto_x_dtlz12(p))
        assert pytest.approx(float(np.sum(f**2)), rel=1e-9) == 1.0

    def test_pareto_equal_angles(self):
        # x_pos = pi/4 in angle space => all f equal: (1/sqrt(2))^i * ...
        p = dtlz2()
        x = np.full(p.dim, 0.5)
        x[0] = 1.0 / 3.0  # theta0 = pi/6
        x[1] = 0.5  # theta1 = pi/4
        f = p.evaluate(x)
        assert pytest.approx(float(np.sum(f**2)), rel=1e-9) == 1.0

    def test_off_pareto_norm_greater(self):
        # g > 0 => sum(f^2) = (1+g)^2 > 1
        p = dtlz2()
        x = _pareto_x_dtlz12(p)
        x[p.n_obj - 1] = 0.0  # g = 0.25 > 0
        f = p.evaluate(x)
        assert float(np.sum(f**2)) > 1.0


# ---------------------------------------------------------------------------
# DTLZ3 -- same PF shape as DTLZ2; harder g  (Eq. 6.21)
# ---------------------------------------------------------------------------


class TestDTLZ3ParetoFront:
    def test_pareto_norm(self):
        # x_M = 0.5 => g = 0 => sum(f^2) = 1
        p = dtlz3()
        f = p.evaluate(_pareto_x_dtlz12(p))
        assert pytest.approx(float(np.sum(f**2)), rel=1e-9) == 1.0


# ---------------------------------------------------------------------------
# DTLZ4 -- biased density; same PF as DTLZ2  (Eq. 6.22)
# ---------------------------------------------------------------------------


class TestDTLZ4ParetoFront:
    def test_pareto_norm(self):
        p = dtlz4()
        f = p.evaluate(_pareto_x_dtlz12(p))
        assert pytest.approx(float(np.sum(f**2)), rel=1e-9) == 1.0

    def test_biased_vs_dtlz2(self):
        # Same x_pos gives different f distributions in DTLZ4 vs DTLZ2 on the PF.
        # x_pos=0.9: DTLZ2 has theta=81deg → f[0]<<1; DTLZ4 has 0.9^100*pi/2≈0 → f[0]≈1.
        x_pos = 0.9
        p2 = dtlz2()
        p4 = dtlz4(alpha=100.0)
        x2 = _pareto_x_dtlz12(p2)
        x2[0] = x_pos
        x2[1] = x_pos
        x4 = _pareto_x_dtlz12(p4)
        x4[0] = x_pos
        x4[1] = x_pos
        f2 = p2.evaluate(x2)
        f4 = p4.evaluate(x4)
        assert pytest.approx(float(np.sum(f2**2)), rel=1e-9) == 1.0
        assert pytest.approx(float(np.sum(f4**2)), rel=1e-9) == 1.0
        assert not np.allclose(f2, f4, atol=0.01)


# ---------------------------------------------------------------------------
# DTLZ5 -- degenerate 1-D PF curve on sphere  (Eq. 6.23)
# ---------------------------------------------------------------------------


class TestDTLZ5ParetoFront:
    def test_pareto_norm_on_pf(self):
        # x_M = 0.5 => g=0 => theta_i=pi/4 for i>=1 => sum(f^2) = 1
        p = dtlz5()
        x = np.full(p.dim, 0.5)
        f = p.evaluate(x)
        assert pytest.approx(float(np.sum(f**2)), rel=1e-9) == 1.0

    def test_pf_parameterized_by_x1(self):
        # Different x_1 values each give sum(f^2) = 1 on the PF
        p = dtlz5()
        for x1 in [0.0, 0.25, 0.5, 0.75, 1.0]:
            x = np.full(p.dim, 0.5)
            x[0] = x1
            f = p.evaluate(x)
            assert pytest.approx(float(np.sum(f**2)), rel=1e-9) == 1.0

    def test_off_pareto_norm_greater(self):
        p = dtlz5()
        x = np.full(p.dim, 0.5)
        x[p.n_obj - 1] = 0.0  # g > 0
        f = p.evaluate(x)
        assert float(np.sum(f**2)) > 1.0


# ---------------------------------------------------------------------------
# DTLZ6 -- harder DTLZ5; g=sum(x_i^0.1), PF at x_M=0  (Eq. 6.24)
# ---------------------------------------------------------------------------


class TestDTLZ6ParetoFront:
    def test_pareto_norm_on_pf(self):
        # x_M = 0 => g=0 => same structure as DTLZ5 => sum(f^2) = 1
        p = dtlz6()
        x = np.zeros(p.dim)
        x[0] = 0.5  # position variable
        f = p.evaluate(x)
        assert pytest.approx(float(np.sum(f**2)), rel=1e-9) == 1.0

    def test_g_at_zeros(self):
        # x_M = 0 => g = sum(0^0.1) = 0
        p = dtlz6()
        x = np.zeros(p.dim)
        x[0] = 0.3
        f_on = p.evaluate(x)
        # x_M != 0 => g > 0 => larger norm
        x_off = x.copy()
        x_off[p.n_obj - 1] = 0.5
        f_off = p.evaluate(x_off)
        assert float(np.sum(f_off**2)) > float(np.sum(f_on**2))


# ---------------------------------------------------------------------------
# DTLZ7 -- disconnected fronts  (Eq. 6.25)
# ---------------------------------------------------------------------------


class TestDTLZ7:
    def test_g_at_zeros(self):
        # x_M = 0 => g = 1 + 9/k * 0 = 1
        p = dtlz7()
        x = np.zeros(p.dim)
        f = p.evaluate(x)
        # f[:M-1] = 0, g = 1; h = M => f_M = (1+1)*M = 2*M
        assert f[p.n_obj - 1] == pytest.approx(2.0 * p.n_obj)

    def test_g_at_ones(self):
        # x_M = 1 => g = 1 + 9/k * k = 10
        p = dtlz7()
        x = np.ones(p.dim)
        f = p.evaluate(x)
        # f[:M-1] = 1, g=10; h = M - sum[1/11*(1+sin(3pi))] = M - 0 = M
        # sin(3*pi) = ~0, so h ≈ M - sum[1/11 * 1] = M - (M-1)/11
        expected_h = p.n_obj - np.sum(
            np.ones(p.n_obj - 1) / 11.0 * (1.0 + np.sin(3.0 * np.pi))
        )
        assert f[p.n_obj - 1] == pytest.approx(11.0 * expected_h, rel=1e-9)

    def test_f_position_equals_x(self):
        p = dtlz7()
        x = np.full(p.dim, 0.3)
        f = p.evaluate(x)
        np.testing.assert_allclose(f[: p.n_obj - 1], x[: p.n_obj - 1])
