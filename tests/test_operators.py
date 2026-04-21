"""
Tests for evolutionary operators (Issue #010).

Tests cover:
- CrossoverBLXAlpha: alpha rename, output shape, determinism
- CrossoverSBX: output shape, center preservation, determinism
- CrossoverUniform: output shape, swap_rate=0/1 boundary cases
- CrossoverOnePoint: output shape, segment integrity
- CrossoverTwoPoint: output shape, segment integrity
- MutationPolynomial: output shape, within-bounds, zero-rate
- MutationGaussian: output shape, zero-rate, zero-sigma
- RouletteWheelSelection: output shape, probability bias
"""

import numpy as np
import pytest

from saealib.context import OptimizationContext
from saealib.operators.crossover import (
    CrossoverBLXAlpha,
    CrossoverOnePoint,
    CrossoverSBX,
    CrossoverTwoPoint,
    CrossoverUniform,
)
from saealib.operators.mutation import MutationGaussian, MutationPolynomial, MutationUniform
from saealib.operators.selection import RouletteWheelSelection
from saealib.population import Archive, Population, PopulationAttribute
from saealib.problem import Problem, SingleObjectiveComparator

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DIM = 6
_ATTRS = [
    PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
    PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
    PopulationAttribute(name="g", dtype=np.float64, shape=(0,)),
    PopulationAttribute(name="cv", dtype=np.float64, shape=()),
]


def _make_parents(rng: np.random.Generator) -> np.ndarray:
    """Two parent individuals, shape=(2, DIM)."""
    return rng.uniform(-1.0, 1.0, size=(2, DIM))


def _make_problem() -> Problem:
    return Problem(
        func=lambda x: np.array([np.sum(x**2)]),
        dim=DIM,
        n_obj=1,
        weight=np.array([-1.0]),
        lb=[-5.0] * DIM,
        ub=[5.0] * DIM,
        comparator=SingleObjectiveComparator(),
    )


def _make_ctx(n_pop: int = 10, rng_seed: int = 0) -> OptimizationContext:
    problem = _make_problem()
    rng = np.random.default_rng(rng_seed)
    pop = Population(_ATTRS, init_capacity=n_pop + 5)
    xs = rng.uniform(-3.0, 3.0, size=(n_pop, DIM))
    fs = np.array([[np.sum(x**2)] for x in xs])
    gs = np.zeros((n_pop, 0))
    cvs = np.zeros(n_pop)
    pop.extend({"x": xs, "f": fs, "g": gs, "cv": cvs})
    arc = Archive(_ATTRS, init_capacity=5)
    return OptimizationContext(
        problem=problem,
        population=pop,
        archive=arc,
        rng=np.random.default_rng(rng_seed),
    )


# ---------------------------------------------------------------------------
# CrossoverBLXAlpha
# ---------------------------------------------------------------------------


class TestCrossoverBLXAlpha:
    def test_alpha_attribute(self):
        op = CrossoverBLXAlpha(crossover_rate=0.7, alpha=0.4)
        assert op.alpha == pytest.approx(0.4)

    def test_output_shape(self):
        op = CrossoverBLXAlpha(crossover_rate=0.7, alpha=0.4)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        assert c.shape == (2, DIM)

    def test_deterministic_with_seed(self):
        op = CrossoverBLXAlpha(crossover_rate=0.7, alpha=0.4)
        p = _make_parents(np.random.default_rng(1))
        c1 = op.crossover(p, rng=np.random.default_rng(42))
        c2 = op.crossover(p, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(c1, c2)


# ---------------------------------------------------------------------------
# CrossoverSBX
# ---------------------------------------------------------------------------


class TestCrossoverSBX:
    def test_output_shape(self):
        op = CrossoverSBX(crossover_rate=0.9, eta=2.0)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        assert c.shape == (2, DIM)

    def test_center_preservation(self):
        """Mid-point of children equals mid-point of parents (SBX property)."""
        op = CrossoverSBX(crossover_rate=0.9, eta=2.0)
        rng = np.random.default_rng(5)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        np.testing.assert_allclose((c[0] + c[1]) / 2, (p[0] + p[1]) / 2, atol=1e-10)

    def test_deterministic_with_seed(self):
        op = CrossoverSBX(crossover_rate=0.9, eta=2.0)
        p = _make_parents(np.random.default_rng(1))
        c1 = op.crossover(p, rng=np.random.default_rng(42))
        c2 = op.crossover(p, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(c1, c2)


# ---------------------------------------------------------------------------
# CrossoverUniform
# ---------------------------------------------------------------------------


class TestCrossoverUniform:
    def test_output_shape(self):
        op = CrossoverUniform(crossover_rate=0.8)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        assert c.shape == (2, DIM)

    def test_swap_rate_zero(self):
        """swap_rate=0: c1==p1 and c2==p2."""
        op = CrossoverUniform(crossover_rate=0.8, swap_rate=0.0)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        np.testing.assert_array_equal(c[0], p[0])
        np.testing.assert_array_equal(c[1], p[1])

    def test_swap_rate_one(self):
        """swap_rate=1: c1==p2 and c2==p1."""
        op = CrossoverUniform(crossover_rate=0.8, swap_rate=1.0)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        np.testing.assert_array_equal(c[0], p[1])
        np.testing.assert_array_equal(c[1], p[0])


# ---------------------------------------------------------------------------
# CrossoverOnePoint
# ---------------------------------------------------------------------------


class TestCrossoverOnePoint:
    def test_output_shape(self):
        op = CrossoverOnePoint(crossover_rate=0.8)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        assert c.shape == (2, DIM)

    def test_segment_intact(self):
        """Before cut point, c1 == p1; from cut point onward, c1 == p2."""
        op = CrossoverOnePoint(crossover_rate=0.8)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        # Fix cut point by seeding: integers(1, DIM) with seed=7 gives a known value
        rng2 = np.random.default_rng(7)
        point = rng2.integers(1, DIM)
        rng3 = np.random.default_rng(7)
        c = op.crossover(p, rng=rng3)
        np.testing.assert_array_equal(c[0, :point], p[0, :point])
        np.testing.assert_array_equal(c[0, point:], p[1, point:])
        np.testing.assert_array_equal(c[1, :point], p[1, :point])
        np.testing.assert_array_equal(c[1, point:], p[0, point:])


# ---------------------------------------------------------------------------
# CrossoverTwoPoint
# ---------------------------------------------------------------------------


class TestCrossoverTwoPoint:
    def test_output_shape(self):
        op = CrossoverTwoPoint(crossover_rate=0.8)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        assert c.shape == (2, DIM)

    def test_segment_intact(self):
        """Between the two cut points, c1 == p2; outside, c1 == p1."""
        op = CrossoverTwoPoint(crossover_rate=0.8)
        rng2 = np.random.default_rng(3)
        pts = np.sort(rng2.choice(DIM + 1, size=2, replace=False))
        pt1, pt2 = pts[0], pts[1]
        rng3 = np.random.default_rng(3)
        p = _make_parents(np.random.default_rng(99))
        c = op.crossover(p, rng=rng3)
        np.testing.assert_array_equal(c[0, :pt1], p[0, :pt1])
        np.testing.assert_array_equal(c[0, pt1:pt2], p[1, pt1:pt2])
        np.testing.assert_array_equal(c[0, pt2:], p[0, pt2:])


# ---------------------------------------------------------------------------
# MutationPolynomial
# ---------------------------------------------------------------------------


class TestMutationPolynomial:
    def _range(self):
        lb = np.full(DIM, -5.0)
        ub = np.full(DIM, 5.0)
        return lb, ub

    def test_output_shape(self):
        op = MutationPolynomial(mutation_rate=0.5, eta=20.0)
        rng = np.random.default_rng(0)
        p = rng.uniform(-2.0, 2.0, size=DIM)
        c = op.mutate(p, self._range(), rng=rng)
        assert c.shape == (DIM,)

    def test_within_bounds(self):
        op = MutationPolynomial(mutation_rate=1.0, eta=20.0)
        lb, ub = self._range()
        rng = np.random.default_rng(0)
        for _ in range(20):
            p = rng.uniform(-4.0, 4.0, size=DIM)
            c = op.mutate(p, (lb, ub), rng=rng)
            assert np.all(c >= lb) and np.all(c <= ub)

    def test_zero_rate_no_change(self):
        op = MutationPolynomial(mutation_rate=0.0, eta=20.0)
        rng = np.random.default_rng(0)
        p = rng.uniform(-2.0, 2.0, size=DIM)
        c = op.mutate(p, self._range(), rng=rng)
        np.testing.assert_array_equal(c, p)


# ---------------------------------------------------------------------------
# MutationGaussian
# ---------------------------------------------------------------------------


class TestMutationGaussian:
    def _range(self):
        lb = np.full(DIM, -5.0)
        ub = np.full(DIM, 5.0)
        return lb, ub

    def test_output_shape(self):
        op = MutationGaussian(mutation_rate=0.5, sigma=0.1)
        rng = np.random.default_rng(0)
        p = rng.uniform(-2.0, 2.0, size=DIM)
        c = op.mutate(p, self._range(), rng=rng)
        assert c.shape == (DIM,)

    def test_zero_rate_no_change(self):
        op = MutationGaussian(mutation_rate=0.0, sigma=1.0)
        rng = np.random.default_rng(0)
        p = rng.uniform(-2.0, 2.0, size=DIM)
        c = op.mutate(p, self._range(), rng=rng)
        np.testing.assert_array_equal(c, p)

    def test_zero_sigma_no_change(self):
        op = MutationGaussian(mutation_rate=1.0, sigma=0.0)
        rng = np.random.default_rng(0)
        p = rng.uniform(-2.0, 2.0, size=DIM)
        c = op.mutate(p, self._range(), rng=rng)
        np.testing.assert_array_equal(c, p)


# ---------------------------------------------------------------------------
# RouletteWheelSelection
# ---------------------------------------------------------------------------


class TestRouletteWheelSelection:
    def test_output_shape(self):
        op = RouletteWheelSelection()
        ctx = _make_ctx(n_pop=10)
        rng = np.random.default_rng(0)
        idx = op.select(ctx, ctx.population, n_pair=4, n_parents=2, rng=rng)
        assert idx.shape == (4, 2)

    def test_indices_in_range(self):
        op = RouletteWheelSelection()
        ctx = _make_ctx(n_pop=10)
        rng = np.random.default_rng(0)
        idx = op.select(ctx, ctx.population, n_pair=5, n_parents=2, rng=rng)
        assert np.all(idx >= 0) and np.all(idx < len(ctx.population))

    def test_best_selected_more_often(self):
        """Best individual (lowest f) should appear more than worst over many trials."""
        op = RouletteWheelSelection()
        n_pop = 10
        ctx = _make_ctx(n_pop=n_pop, rng_seed=0)
        # Identify best and worst indices via comparator
        sorted_idx = ctx.comparator.sort_population(ctx.population)
        best_idx = sorted_idx[0]
        worst_idx = sorted_idx[-1]

        rng = np.random.default_rng(1)
        counts = np.zeros(n_pop, dtype=int)
        for _ in range(500):
            idx = op.select(ctx, ctx.population, n_pair=10, n_parents=1, rng=rng)
            for i in idx.flatten():
                counts[i] += 1

        assert counts[best_idx] > counts[worst_idx]
