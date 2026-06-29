"""Tests for saealib.benchmarks.singleobj.

Structural checks and global-minimum spot checks for Sphere, Rosenbrock,
Ackley, and Rastrigin.
"""

from __future__ import annotations

import numpy as np
import pytest

from saealib.benchmarks.singleobj import ackley, rastrigin, rosenbrock, sphere
from saealib.problem.problem import Problem

_FACTORIES = [
    (sphere, 10),
    (rosenbrock, 10),
    (ackley, 10),
    (rastrigin, 10),
]


# ---------------------------------------------------------------------------
# structural tests (parametrised)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory,n_var", _FACTORIES)
class TestSingleObjStructure:
    def test_returns_problem(self, factory, n_var):
        assert isinstance(factory(), Problem)

    def test_default_n_var(self, factory, n_var):
        assert factory().dim == n_var

    def test_custom_n_var(self, factory, n_var):
        p = factory(n_var=5)
        assert p.dim == 5

    def test_n_obj(self, factory, n_var):
        assert factory().n_obj == 1

    def test_direction_minimize(self, factory, n_var):
        np.testing.assert_array_equal(factory().direction, [-1.0])

    def test_evaluate_shape(self, factory, n_var):
        p = factory()
        f = p.evaluate(np.zeros(p.dim))
        assert f.shape == (1,)

    def test_bounds_length(self, factory, n_var):
        p = factory()
        assert len(p.lb) == p.dim
        assert len(p.ub) == p.dim

    def test_no_constraints(self, factory, n_var):
        assert factory().n_constraints == 0


# ---------------------------------------------------------------------------
# Sphere: f(x) = sum(xi^2)
# ---------------------------------------------------------------------------


class TestSphere:
    def test_global_min_at_origin(self):
        p = sphere()
        f = p.evaluate(np.zeros(p.dim))
        assert f[0] == pytest.approx(0.0)

    def test_positive_elsewhere(self):
        p = sphere()
        x = np.ones(p.dim)
        assert p.evaluate(x)[0] == pytest.approx(float(p.dim))

    def test_separable(self):
        p = sphere(n_var=3)
        x = np.array([1.0, 2.0, 3.0])
        assert p.evaluate(x)[0] == pytest.approx(1.0 + 4.0 + 9.0)


# ---------------------------------------------------------------------------
# Rosenbrock: f(x) = sum 100*(x_{i+1}-xi^2)^2 + (xi-1)^2
# ---------------------------------------------------------------------------


class TestRosenbrock:
    def test_global_min_at_ones(self):
        p = rosenbrock()
        f = p.evaluate(np.ones(p.dim))
        assert f[0] == pytest.approx(0.0)

    def test_two_var_spot(self):
        # n_var=2: f(1,1) = 100*(1-1)^2 + (1-1)^2 = 0
        p = rosenbrock(n_var=2)
        assert p.evaluate(np.array([1.0, 1.0]))[0] == pytest.approx(0.0)
        # f(0,0) = 100*(0-0)^2 + (0-1)^2 = 1
        assert p.evaluate(np.array([0.0, 0.0]))[0] == pytest.approx(1.0)

    def test_increases_away_from_ones(self):
        p = rosenbrock(n_var=5)
        assert p.evaluate(np.zeros(p.dim))[0] > 0


# ---------------------------------------------------------------------------
# Ackley: f(x) = -20*exp(-0.2*sqrt(mean(xi^2))) - exp(mean(cos(2pi*xi))) + 20 + e
# ---------------------------------------------------------------------------


class TestAckley:
    def test_global_min_at_origin(self):
        p = ackley()
        f = p.evaluate(np.zeros(p.dim))
        assert f[0] == pytest.approx(0.0, abs=1e-12)

    def test_positive_elsewhere(self):
        p = ackley(n_var=2)
        x = np.array([1.0, 1.0])
        assert p.evaluate(x)[0] > 0.0

    def test_bounded_above(self):
        # Max value ≈ 20 + e + 1 ~ 24; function is bounded
        p = ackley(n_var=5)
        x = np.array([35.0] * 5)  # near corner of domain
        assert p.evaluate(x)[0] < 25.0


# ---------------------------------------------------------------------------
# Rastrigin: f(x) = A*D + sum(xi^2 - A*cos(2pi*xi))
# ---------------------------------------------------------------------------


class TestRastrigin:
    def test_global_min_at_origin(self):
        p = rastrigin()
        f = p.evaluate(np.zeros(p.dim))
        assert f[0] == pytest.approx(0.0)

    def test_local_minima_at_integers(self):
        # f(1,0,...,0) = A*D + (1 - A*1) + 0*... = 10*D + 1 - 10 = 10*D - 9
        p = rastrigin(n_var=5)
        x = np.zeros(p.dim)
        x[0] = 1.0
        rest = (p.dim - 1) * (0.0 - 10.0)  # xi=0 terms: 0 - 10*cos(0)
        expected = 10.0 * p.dim + (1.0 - 10.0 * np.cos(0.0)) + rest
        assert p.evaluate(x)[0] == pytest.approx(expected, rel=1e-9)

    def test_custom_amplitude(self):
        p = rastrigin(n_var=3, amplitude=5.0)
        f = p.evaluate(np.zeros(p.dim))
        assert f[0] == pytest.approx(0.0)
        x = np.ones(p.dim)
        expected = 5.0 * 3 + np.sum(np.ones(3) - 5.0 * np.cos(2 * np.pi * np.ones(3)))
        assert p.evaluate(x)[0] == pytest.approx(float(expected), rel=1e-9)
