"""
Tests for evolutionary operators (Issue #010).

Tests cover:
- CrossoverBLXAlpha: alpha rename, output shape, determinism
- CrossoverSBX: output shape, center preservation, determinism
- CrossoverUniform: output shape, swap_rate=0/1 boundary cases
- CrossoverOnePoint: output shape, segment integrity
- CrossoverTwoPoint: output shape, segment integrity
- Crossover (base): n_children default and consistency with output shape
- MutationPolynomial: output shape, within-bounds, zero-rate
- MutationGaussian: output shape, zero-rate, zero-sigma
- RouletteWheelSelection: output shape, probability bias
"""

import numpy as np
import pytest

from saealib import GA, SequentialSelection, TruncationSelection, minimize
from saealib.comparators import NSGA2Comparator, SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.operators.crossover import (
    Crossover,
    CrossoverBLXAlpha,
    CrossoverCategorical,
    CrossoverIntegerSBX,
    CrossoverOnePoint,
    CrossoverSBX,
    CrossoverTwoPoint,
    CrossoverUniform,
)
from saealib.operators.mutation import (
    MutationCategorical,
    MutationGaussian,
    MutationIntegerUniform,
    MutationPolynomial,
    MutationUniform,
)
from saealib.operators.selection import RouletteWheelSelection, TournamentSelection
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem

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
        direction=np.array([-1.0]),
        lb=[-5.0] * DIM,
        ub=[5.0] * DIM,
        comparator=SingleObjectiveComparator(),
    )


def _make_ctx(n_pop: int = 10, rng_seed: int = 0) -> OptimizationState:
    problem = _make_problem()
    rng = np.random.default_rng(rng_seed)
    pop = Population(_ATTRS, init_capacity=n_pop + 5)
    xs = rng.uniform(-3.0, 3.0, size=(n_pop, DIM))
    fs = np.array([[np.sum(x**2)] for x in xs])
    gs = np.zeros((n_pop, 0))
    cvs = np.zeros(n_pop)
    pop.extend({"x": xs, "f": fs, "g": gs, "cv": cvs})
    arc = Archive(_ATTRS, init_capacity=5)
    pareto_arc = ParetoArchive(_ATTRS, init_capacity=5, direction=np.array([-1.0]))
    return OptimizationState(
        problem=problem,
        population=pop,
        archive=arc,
        pareto_archive=pareto_arc,
        rng=np.random.default_rng(rng_seed),
    )


# ---------------------------------------------------------------------------
# CrossoverBLXAlpha
# ---------------------------------------------------------------------------


class TestCrossoverBLXAlpha:
    def test_alpha_attribute(self):
        op = CrossoverBLXAlpha(prob=0.7, alpha=0.4)
        assert op.alpha == pytest.approx(0.4)

    def test_output_shape(self):
        op = CrossoverBLXAlpha(prob=0.7, alpha=0.4)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        assert c.shape == (2, DIM)

    def test_deterministic_with_seed(self):
        op = CrossoverBLXAlpha(prob=0.7, alpha=0.4)
        p = _make_parents(np.random.default_rng(1))
        c1 = op.crossover(p, rng=np.random.default_rng(42))
        c2 = op.crossover(p, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(c1, c2)


# ---------------------------------------------------------------------------
# CrossoverSBX
# ---------------------------------------------------------------------------


class TestCrossoverSBX:
    def test_output_shape(self):
        op = CrossoverSBX(prob=0.9, eta=2.0)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        assert c.shape == (2, DIM)

    def test_center_preservation(self):
        """Mid-point of children equals mid-point of parents (SBX property)."""
        op = CrossoverSBX(prob=0.9, eta=2.0)
        rng = np.random.default_rng(5)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        np.testing.assert_allclose((c[0] + c[1]) / 2, (p[0] + p[1]) / 2, atol=1e-10)

    def test_deterministic_with_seed(self):
        op = CrossoverSBX(prob=0.9, eta=2.0)
        p = _make_parents(np.random.default_rng(1))
        c1 = op.crossover(p, rng=np.random.default_rng(42))
        c2 = op.crossover(p, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(c1, c2)

    def test_crossover_batch_output_shape(self):
        op = CrossoverSBX(prob=0.9, eta=2.0)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        c = op.crossover_batch(parents_batch, rng=rng)
        assert c is not None
        assert c.shape == (n_pair, 2, DIM)

    def test_crossover_batch_matches_single_at_n_pair_one(self):
        """Unbounded branch: draws are u, then do_cross (2 phases)."""
        op = CrossoverSBX(prob=0.9, eta=2.0)
        p = _make_parents(np.random.default_rng(1))
        parents_batch = p[np.newaxis, :, :]
        c_batch = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        assert c_batch is not None
        c_single = op.crossover(p, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(c_batch[0], c_single)

    def test_crossover_batch_matches_single_at_n_pair_one_bounded(self):
        """Bounded branch: draws are u, then swap, then do_cross (3 phases)
        -- the branch GA actually exercises, since GA always supplies bounds."""
        op = CrossoverSBX(prob=1.0, eta=20.0, prob_var=1.0)
        lb = np.zeros(DIM)
        ub = np.ones(DIM)
        p = np.random.default_rng(1).uniform(0.0, 1.0, size=(2, DIM))
        parents_batch = p[np.newaxis, :, :]
        c_batch = op.crossover_batch(
            parents_batch, (lb, ub), rng=np.random.default_rng(42)
        )
        assert c_batch is not None
        c_single = op.crossover(p, (lb, ub), rng=np.random.default_rng(42))
        np.testing.assert_array_equal(c_batch[0], c_single)

    def test_crossover_batch_deterministic_with_seed(self):
        op = CrossoverSBX(prob=0.9, eta=2.0)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        c1 = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        c2 = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(c1, c2)

    def test_crossover_batch_respects_bounds(self):
        op = CrossoverSBX(prob=1.0, eta=20.0, prob_var=1.0)
        lb = np.zeros(DIM)
        ub = np.ones(DIM)
        rng = np.random.default_rng(0)
        n_pair = 8
        for _ in range(20):
            parents_batch = rng.uniform(0.0, 1.0, size=(n_pair, 2, DIM))
            c = op.crossover_batch(parents_batch, (lb, ub), rng=rng)
            assert c is not None
            assert np.all(c >= lb) and np.all(c <= ub)

    def test_crossover_batch_mixed_identical_and_separated_rows(self):
        """A batch with one row of identical parents and one row of
        separated parents: only the identical row must come back unchanged
        (batch mirror of TestCrossoverSBXBounded.test_identical_parents_unchanged
        -- confirms `separated` masks per row, not collapsed across the batch)."""
        op = CrossoverSBX(prob=1.0, eta=20.0, prob_var=1.0)
        lb = np.zeros(3)
        ub = np.ones(3)
        identical = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        separated = np.array([[0.1, 0.2, 0.3], [0.8, 0.7, 0.9]])
        parents_batch = np.stack([identical, separated], axis=0)
        c = op.crossover_batch(parents_batch, (lb, ub), rng=np.random.default_rng(0))
        assert c is not None
        np.testing.assert_array_equal(c[0, 0], identical[0])
        np.testing.assert_array_equal(c[0, 1], identical[1])
        assert np.all(c[1] >= lb) and np.all(c[1] <= ub)
        # discriminates per-row masking from a collapsed `separated` mask:
        # if row 1's separated-ness were incorrectly suppressed by row 0's
        # identical-ness, c[1] would also come back as unmodified parents
        assert not np.array_equal(c[1, 0], separated[0])


# ---------------------------------------------------------------------------
# CrossoverUniform
# ---------------------------------------------------------------------------


class TestCrossoverUniform:
    def test_output_shape(self):
        op = CrossoverUniform(prob=0.8)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        assert c.shape == (2, DIM)

    def test_swap_rate_zero(self):
        """swap_rate=0: c1==p1 and c2==p2."""
        op = CrossoverUniform(prob=0.8, swap_rate=0.0)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        np.testing.assert_array_equal(c[0], p[0])
        np.testing.assert_array_equal(c[1], p[1])

    def test_swap_rate_one(self):
        """swap_rate=1: c1==p2 and c2==p1."""
        op = CrossoverUniform(prob=0.8, swap_rate=1.0)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        np.testing.assert_array_equal(c[0], p[1])
        np.testing.assert_array_equal(c[1], p[0])

    def test_crossover_batch_output_shape(self):
        op = CrossoverUniform(prob=0.8)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        c = op.crossover_batch(parents_batch, rng=rng)
        assert c is not None
        assert c.shape == (n_pair, 2, DIM)

    def test_crossover_batch_matches_single_at_n_pair_one(self):
        op = CrossoverUniform(prob=0.8)
        p = _make_parents(np.random.default_rng(1))
        parents_batch = p[np.newaxis, :, :]
        c_batch = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        assert c_batch is not None
        c_single = op.crossover(p, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(c_batch[0], c_single)

    def test_crossover_batch_deterministic_with_seed(self):
        op = CrossoverUniform(prob=0.8)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        c1 = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        c2 = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(c1, c2)

    def test_crossover_batch_swap_rate_zero(self):
        """swap_rate=0: c1==p1 and c2==p2 across every row."""
        op = CrossoverUniform(prob=0.8, swap_rate=0.0)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        c = op.crossover_batch(parents_batch, rng=rng)
        assert c is not None
        np.testing.assert_array_equal(c[:, 0, :], parents_batch[:, 0, :])
        np.testing.assert_array_equal(c[:, 1, :], parents_batch[:, 1, :])

    def test_crossover_batch_swap_rate_one(self):
        """swap_rate=1: c1==p2 and c2==p1 across every row."""
        op = CrossoverUniform(prob=0.8, swap_rate=1.0)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        c = op.crossover_batch(parents_batch, rng=rng)
        assert c is not None
        np.testing.assert_array_equal(c[:, 0, :], parents_batch[:, 1, :])
        np.testing.assert_array_equal(c[:, 1, :], parents_batch[:, 0, :])


# ---------------------------------------------------------------------------
# CrossoverOnePoint
# ---------------------------------------------------------------------------


class TestCrossoverOnePoint:
    def test_output_shape(self):
        op = CrossoverOnePoint(prob=0.8)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        assert c.shape == (2, DIM)

    def test_segment_intact(self):
        """Before cut point, c1 == p1; from cut point onward, c1 == p2."""
        op = CrossoverOnePoint(prob=0.8)
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
        op = CrossoverTwoPoint(prob=0.8)
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        c = op.crossover(p, rng=rng)
        assert c.shape == (2, DIM)

    def test_segment_intact(self):
        """Between the two cut points, c1 == p2; outside, c1 == p1."""
        op = CrossoverTwoPoint(prob=0.8)
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
# Crossover base: n_children
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op",
    [
        CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        CrossoverSBX(prob=0.9, eta=2.0),
        CrossoverUniform(prob=0.9),
        CrossoverOnePoint(prob=0.9),
        CrossoverTwoPoint(prob=0.9),
    ],
)
class TestCrossoverNChildren:
    def test_n_children_default(self, op):
        assert op.n_children == 2

    def test_output_rows_match_n_children(self, op):
        rng = np.random.default_rng(0)
        c = op.crossover(_make_parents(rng), rng=rng)
        assert c.shape[0] == op.n_children


# ---------------------------------------------------------------------------
# Crossover base: crossover_batch default
# ---------------------------------------------------------------------------


class TestCrossoverBatchDefault:
    """CrossoverBLXAlpha/CrossoverOnePoint remain unbatched (commit 8, out of
    scope): CrossoverSBX/CrossoverUniform/CrossoverCategorical/
    CrossoverIntegerSBX now override crossover_batch (see TestCrossoverSBX
    etc.), so they are no longer suitable subjects for this "still returns
    None" check.
    """

    def test_default_returns_none_blx_alpha(self):
        op = CrossoverBLXAlpha(prob=0.9, alpha=0.4)
        rng = np.random.default_rng(0)
        n_pair = 3
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        result = op.crossover_batch(parents_batch, rng=rng)
        assert result is None

    def test_default_returns_none_one_point(self):
        op = CrossoverOnePoint(prob=0.8)
        rng = np.random.default_rng(0)
        n_pair = 3
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        result = op.crossover_batch(parents_batch, rng=rng)
        assert result is None


class TestCrossoverBatchOverridden:
    """Positive mirror of TestCrossoverBatchDefault: confirms the four
    classes actually satisfy GA._crossover_pairs's dispatch criterion
    (``type(op).crossover_batch is not Crossover.crossover_batch``), i.e.
    that this commit's new methods are picked up by GA's existing batch-vs-
    loop routing rather than silently shadowed by some other override.
    """

    @pytest.mark.parametrize(
        "op",
        [
            CrossoverSBX(prob=0.9, eta=2.0),
            CrossoverUniform(prob=0.8),
            CrossoverCategorical(prob=1.0),
            CrossoverIntegerSBX(prob=1.0, eta=2.0),
        ],
    )
    def test_overrides_crossover_batch(self, op):
        assert type(op).crossover_batch is not Crossover.crossover_batch


# ---------------------------------------------------------------------------
# MutationPolynomial
# ---------------------------------------------------------------------------


class TestMutationPolynomial:
    def _range(self):
        lb = np.full(DIM, -5.0)
        ub = np.full(DIM, 5.0)
        return lb, ub

    def test_output_shape(self):
        op = MutationPolynomial(prob_var=0.5, eta=20.0)
        rng = np.random.default_rng(0)
        p = rng.uniform(-2.0, 2.0, size=DIM)
        c = op.mutate(p, self._range(), rng=rng)
        assert c.shape == (DIM,)

    def test_within_bounds(self):
        op = MutationPolynomial(prob_var=1.0, eta=20.0)
        lb, ub = self._range()
        rng = np.random.default_rng(0)
        for _ in range(20):
            p = rng.uniform(-4.0, 4.0, size=DIM)
            c = op.mutate(p, (lb, ub), rng=rng)
            assert np.all(c >= lb) and np.all(c <= ub)

    def test_zero_rate_no_change(self):
        op = MutationPolynomial(prob_var=0.0, eta=20.0)
        rng = np.random.default_rng(0)
        p = rng.uniform(-2.0, 2.0, size=DIM)
        c = op.mutate(p, self._range(), rng=rng)
        np.testing.assert_array_equal(c, p)

    def test_asymmetric_delta_allows_larger_boundary_excursion(self):
        # p near lb, ub far away: the old shared min(delta1, delta2)
        # formula reused the (tiny) lower-bound delta for the upward
        # (u>0.5) branch too, capping upward excursions near zero. The
        # asymmetric formula uses delta2 (upper-bound distance) for that
        # branch instead, matching nsga2-gnuplot-v1.1.6 / pymoo / DEAP.
        op = MutationPolynomial(prob_var=1.0, eta=20.0)
        lb = np.array([0.0])
        ub = np.array([100.0])
        p = np.array([1.0])
        rng = np.random.default_rng(0)
        max_excursion = 0.0
        for _ in range(50):
            c = op.mutate(p, (lb, ub), rng=rng)
            max_excursion = max(max_excursion, float(c[0] - p[0]))
        assert max_excursion > 5.0


# ---------------------------------------------------------------------------
# MutationGaussian
# ---------------------------------------------------------------------------


class TestMutationGaussian:
    def _range(self):
        lb = np.full(DIM, -5.0)
        ub = np.full(DIM, 5.0)
        return lb, ub

    def test_output_shape(self):
        op = MutationGaussian(prob_var=0.5, sigma=0.1)
        rng = np.random.default_rng(0)
        p = rng.uniform(-2.0, 2.0, size=DIM)
        c = op.mutate(p, self._range(), rng=rng)
        assert c.shape == (DIM,)

    def test_zero_rate_no_change(self):
        op = MutationGaussian(prob_var=0.0, sigma=1.0)
        rng = np.random.default_rng(0)
        p = rng.uniform(-2.0, 2.0, size=DIM)
        c = op.mutate(p, self._range(), rng=rng)
        np.testing.assert_array_equal(c, p)

    def test_zero_sigma_no_change(self):
        op = MutationGaussian(prob_var=1.0, sigma=0.0)
        rng = np.random.default_rng(0)
        p = rng.uniform(-2.0, 2.0, size=DIM)
        c = op.mutate(p, self._range(), rng=rng)
        np.testing.assert_array_equal(c, p)


# ---------------------------------------------------------------------------
# Mutation base: mutate_batch default
# ---------------------------------------------------------------------------


class TestMutationBatchDefault:
    def _range(self):
        lb = np.full(DIM, -5.0)
        ub = np.full(DIM, 5.0)
        return lb, ub

    def test_default_returns_none_polynomial(self):
        op = MutationPolynomial(prob_var=0.5, eta=20.0)
        rng = np.random.default_rng(0)
        candidates_batch = rng.uniform(-2.0, 2.0, size=(5, DIM))
        result = op.mutate_batch(candidates_batch, self._range(), rng=rng)
        assert result is None

    def test_default_returns_none_uniform(self):
        op = MutationUniform(prob_var=0.5)
        rng = np.random.default_rng(0)
        candidates_batch = rng.uniform(-2.0, 2.0, size=(5, DIM))
        result = op.mutate_batch(candidates_batch, self._range(), rng=rng)
        assert result is None


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


# ---------------------------------------------------------------------------
# TruncationSelection
# ---------------------------------------------------------------------------


def _make_moo_ctx(f_values: np.ndarray, rng_seed: int = 0) -> OptimizationState:
    """MOO context with an NSGA2Comparator and explicit objective values."""
    n, n_obj = f_values.shape
    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(n_obj,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(n_obj,)),
        PopulationAttribute(name="cv", dtype=np.float64, shape=(), default=0.0),
    ]
    pop = Population(attrs, init_capacity=n + 1)
    for row in f_values:
        pop.append(x=row, f=row, cv=0.0)
    direction = np.full(n_obj, -1.0)
    comparator = NSGA2Comparator(direction=direction)
    problem = Problem(
        func=lambda x: x,
        dim=n_obj,
        n_obj=n_obj,
        direction=direction,
        lb=[-10.0] * n_obj,
        ub=[10.0] * n_obj,
        comparator=comparator,
    )
    arc = Archive(attrs, init_capacity=5)
    pareto_arc = ParetoArchive(attrs, init_capacity=5, direction=direction)
    return OptimizationState(
        problem=problem,
        population=pop,
        archive=arc,
        pareto_archive=pareto_arc,
        rng=np.random.default_rng(rng_seed),
    )


class TestTruncationSelection:
    # Front with a genuine crowding-distance tie between rows 2 and 4
    # (both f=[0.5, 0.5], rank=0, cd=1.0), plus a dominated point (row 5).
    # Verified via direct inspection of NSGA2Comparator's cached rank/cd:
    # order=[0,1,2,4,3,5], rank=[0,0,0,0,0,1], cd=[inf,inf,1,0,1,inf].
    _F_WITH_TIE = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [2.0, 2.0],
        ]
    )
    # Unequal spacing along the tradeoff line so crowding distances differ
    # (evenly-spaced points give symmetric, tied cd values by construction).
    # Verified: order=[0,5,4,3,2,1], rank=all 0, cd=[inf,0.7,0.8,0.9,1.0,inf].
    _F_NO_TIE = np.array(
        [
            [0.0, 1.0],
            [0.1, 0.9],
            [0.35, 0.65],
            [0.5, 0.5],
            [0.8, 0.2],
            [1.0, 0.0],
        ]
    )

    def test_default_matches_plain_sort(self):
        op = TruncationSelection()
        assert op.randomize_ties is False
        ctx = _make_moo_ctx(self._F_WITH_TIE)
        expected = ctx.comparator.sort_population(ctx.population)[:3]
        result = op.select(ctx, ctx.population, 3)
        np.testing.assert_array_equal(result, expected)

    def test_randomize_ties_no_ties_matches_plain_sort(self):
        op = TruncationSelection(randomize_ties=True)
        ctx = _make_moo_ctx(self._F_NO_TIE)
        expected = ctx.comparator.sort_population(ctx.population)[:3]
        result = op.select(ctx, ctx.population, 3)
        np.testing.assert_array_equal(result, expected)

    def test_randomize_ties_preserves_non_tied_boundary(self):
        op = TruncationSelection(randomize_ties=True)
        for seed in range(10):
            ctx = _make_moo_ctx(self._F_WITH_TIE, rng_seed=seed)
            result = op.select(ctx, ctx.population, 3)
            assert result.shape == (3,)
            assert 0 in result and 1 in result  # cd=inf, always kept
            assert 5 not in result  # dominated, never kept
            assert set(result.tolist()) <= {0, 1, 2, 4}

    def test_randomize_ties_varies_across_seeds(self):
        op = TruncationSelection(randomize_ties=True)
        outcomes = set()
        for seed in range(20):
            ctx = _make_moo_ctx(self._F_WITH_TIE, rng_seed=seed)
            result = op.select(ctx, ctx.population, 3)
            outcomes.add(frozenset(result.tolist()))
        assert outcomes == {frozenset({0, 1, 2}), frozenset({0, 1, 4})}

    def test_n_survivors_exceeds_pool(self):
        op = TruncationSelection(randomize_ties=True)
        ctx = _make_moo_ctx(self._F_WITH_TIE)
        result = op.select(ctx, ctx.population, 10)
        assert set(result.tolist()) == set(range(6))


# ---------------------------------------------------------------------------
# Crossover lifecycle hooks
# ---------------------------------------------------------------------------


class TestCrossoverHooks:
    def _op(self):
        return CrossoverBLXAlpha(prob=0.9, alpha=0.4)

    def _parents(self, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        return rng.uniform(-1.0, 1.0, size=(2, DIM))

    def test_post_crossover_default_is_noop(self):
        op = self._op()
        rng = np.random.default_rng(0)
        parents = self._parents(rng)
        offspring = op.crossover(parents, rng=rng)
        result = op.post_crossover(offspring.copy(), parents, rng)
        np.testing.assert_array_equal(result, offspring)

    def test_with_post_transforms_offspring(self):
        op = self._op().with_post(lambda o, p, rng, ctx: np.zeros_like(o))
        rng = np.random.default_rng(0)
        parents = self._parents(rng)
        offspring = op.crossover(parents, rng=rng)
        result = op.post_crossover(offspring, parents, rng)
        np.testing.assert_array_equal(result, np.zeros_like(offspring))

    def test_with_post_fn_receives_correct_args(self):
        received = {}

        def hook(offspring, parents, rng, ctx):
            received["offspring_shape"] = offspring.shape
            received["parents_shape"] = parents.shape
            received["ctx"] = ctx
            return offspring

        ctx = _make_ctx()
        op = self._op().with_post(hook)
        rng = np.random.default_rng(0)
        parents = self._parents(rng)
        offspring = op.crossover(parents, rng=rng)
        op.post_crossover(offspring, parents, rng, ctx)
        assert received["offspring_shape"] == (2, DIM)
        assert received["parents_shape"] == (2, DIM)
        assert received["ctx"] is ctx

    def test_with_post_chains_in_order(self):
        log = []
        op = (
            self._op()
            .with_post(lambda o, p, rng, ctx: (log.append(1), o)[1])
            .with_post(lambda o, p, rng, ctx: (log.append(2), o)[1])
        )
        rng = np.random.default_rng(0)
        parents = self._parents(rng)
        offspring = op.crossover(parents, rng=rng)
        op.post_crossover(offspring, parents, rng)
        assert log == [1, 2]

    def test_with_post_does_not_mutate_original(self):
        op = self._op()
        _ = op.with_post(lambda o, p, rng, ctx: np.zeros_like(o))
        rng = np.random.default_rng(0)
        parents = self._parents(rng)
        offspring = op.crossover(parents, rng=rng)
        result = op.post_crossover(offspring.copy(), parents, rng)
        np.testing.assert_array_equal(result, offspring)


# ---------------------------------------------------------------------------
# Mutation lifecycle hooks
# ---------------------------------------------------------------------------


class TestMutationHooks:
    def _op(self):
        return MutationPolynomial(prob_var=1.0, eta=20.0)

    def _individual(self, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        return rng.uniform(-3.0, 3.0, size=DIM)

    def _range(self):
        return np.full(DIM, -5.0), np.full(DIM, 5.0)

    def test_post_mutation_default_is_noop(self):
        op = self._op()
        rng = np.random.default_rng(0)
        p = self._individual(rng)
        mutated = op.mutate(p, self._range(), rng=rng)
        result = op.post_mutation(mutated.copy(), self._range(), rng)
        np.testing.assert_array_equal(result, mutated)

    def test_with_post_transforms_individual(self):
        op = self._op().with_post(lambda o, mr, rng, ctx: np.zeros_like(o))
        rng = np.random.default_rng(0)
        p = self._individual(rng)
        mutated = op.mutate(p, self._range(), rng=rng)
        result = op.post_mutation(mutated, self._range(), rng)
        np.testing.assert_array_equal(result, np.zeros(DIM))

    def test_with_post_fn_receives_correct_args(self):
        received = {}

        def hook(offspring, mutate_range, rng, ctx):
            received["shape"] = offspring.shape
            received["ctx"] = ctx
            return offspring

        ctx = _make_ctx()
        op = self._op().with_post(hook)
        rng = np.random.default_rng(0)
        p = self._individual(rng)
        mutated = op.mutate(p, self._range(), rng=rng)
        op.post_mutation(mutated, self._range(), rng, ctx)
        assert received["shape"] == (DIM,)
        assert received["ctx"] is ctx

    def test_with_post_chains_in_order(self):
        log = []
        op = (
            self._op()
            .with_post(lambda o, mr, rng, ctx: (log.append(1), o)[1])
            .with_post(lambda o, mr, rng, ctx: (log.append(2), o)[1])
        )
        rng = np.random.default_rng(0)
        p = self._individual(rng)
        mutated = op.mutate(p, self._range(), rng=rng)
        op.post_mutation(mutated, self._range(), rng)
        assert log == [1, 2]

    def test_with_post_does_not_mutate_original(self):
        op = self._op()
        _ = op.with_post(lambda o, mr, rng, ctx: np.zeros_like(o))
        rng = np.random.default_rng(0)
        p = self._individual(rng)
        mutated = op.mutate(p, self._range(), rng=rng)
        result = op.post_mutation(mutated.copy(), self._range(), rng)
        np.testing.assert_array_equal(result, mutated)


# ---------------------------------------------------------------------------
# GA hook invocation (integration)
# ---------------------------------------------------------------------------


class _DispatchOnlyProvider:
    def dispatch(self, event):
        pass


class TestGAHookInvocation:
    """Verify that GA.ask() calls post_crossover and post_mutation hooks."""

    def _make_ctx(self):
        return _make_ctx()

    def test_post_crossover_called_during_ask(self):
        call_count = [0]

        def hook(offspring, parents, rng, ctx):
            call_count[0] += 1
            return offspring

        crossover = CrossoverBLXAlpha(prob=1.0, alpha=0.4).with_post(hook)
        ga = GA(
            crossover=crossover,
            mutation=MutationUniform(prob_var=0.0),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        )
        provider = _DispatchOnlyProvider()
        ctx = _make_ctx()
        ga.ask(ctx, provider)
        assert call_count[0] > 0

    def test_post_mutation_called_once_per_offspring(self):
        call_count = [0]

        def hook(offspring, mutate_range, rng, ctx):
            call_count[0] += 1
            return offspring

        mutation = MutationUniform(prob_var=0.0).with_post(hook)
        ga = GA(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
            mutation=mutation,
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        )
        provider = _DispatchOnlyProvider()
        ctx = _make_ctx()
        candidates = ga.ask(ctx, provider)
        assert call_count[0] == len(candidates)


# ---------------------------------------------------------------------------
# CrossoverIntegerSBX
# ---------------------------------------------------------------------------


class TestCrossoverIntegerSBX:
    _rng = np.random.default_rng(0)
    _parents = np.array([[1.0, 3.0, 5.0], [3.0, 7.0, 9.0]])

    def test_output_shape(self):
        op = CrossoverIntegerSBX(prob=1.0, eta=2.0)
        c = op.crossover(self._parents, rng=np.random.default_rng(0))
        assert c.shape == (2, 3)

    def test_offspring_are_integers(self):
        op = CrossoverIntegerSBX(prob=1.0, eta=2.0)
        rng = np.random.default_rng(42)
        for _ in range(20):
            c = op.crossover(self._parents, rng=rng)
            assert np.all(c == np.round(c)), "offspring must be integers"

    def test_crossover_rate_attribute(self):
        op = CrossoverIntegerSBX(prob=0.8, eta=5.0)
        assert op.prob == 0.8

    def test_n_children(self):
        assert CrossoverIntegerSBX(prob=1.0, eta=2.0).n_children == 2

    def test_determinism(self):
        op = CrossoverIntegerSBX(prob=1.0, eta=2.0)
        c1 = op.crossover(self._parents, rng=np.random.default_rng(7))
        c2 = op.crossover(self._parents, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(c1, c2)

    def test_crossover_batch_output_shape(self):
        op = CrossoverIntegerSBX(prob=1.0, eta=2.0)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = rng.integers(0, 10, size=(n_pair, 2, 3)).astype(float)
        c = op.crossover_batch(parents_batch, rng=rng)
        assert c is not None
        assert c.shape == (n_pair, 2, 3)

    def test_crossover_batch_matches_single_at_n_pair_one(self):
        """Unbounded branch: draws are u, then do_cross (2 phases)."""
        op = CrossoverIntegerSBX(prob=1.0, eta=2.0)
        parents_batch = self._parents[np.newaxis, :, :]
        c_batch = op.crossover_batch(parents_batch, rng=np.random.default_rng(7))
        assert c_batch is not None
        c_single = op.crossover(self._parents, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(c_batch[0], c_single)

    def test_crossover_batch_matches_single_at_n_pair_one_bounded(self):
        """Bounded branch: draws are u, then swap, then do_cross (3 phases)
        -- the branch GA actually exercises, since GA always supplies bounds."""
        op = CrossoverIntegerSBX(prob=1.0, eta=20.0, prob_var=1.0)
        lb = np.zeros(3)
        ub = np.full(3, 9.0)
        parents_batch = self._parents[np.newaxis, :, :]
        c_batch = op.crossover_batch(
            parents_batch, (lb, ub), rng=np.random.default_rng(7)
        )
        assert c_batch is not None
        c_single = op.crossover(self._parents, (lb, ub), rng=np.random.default_rng(7))
        np.testing.assert_array_equal(c_batch[0], c_single)

    def test_crossover_batch_deterministic_with_seed(self):
        op = CrossoverIntegerSBX(prob=1.0, eta=2.0)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = rng.integers(0, 10, size=(n_pair, 2, 3)).astype(float)
        c1 = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        c2 = op.crossover_batch(parents_batch, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(c1, c2)

    def test_crossover_batch_offspring_are_integers(self):
        op = CrossoverIntegerSBX(prob=1.0, eta=2.0)
        rng = np.random.default_rng(42)
        n_pair = 8
        for _ in range(20):
            parents_batch = rng.integers(0, 10, size=(n_pair, 2, 3)).astype(float)
            c = op.crossover_batch(parents_batch, rng=rng)
            assert c is not None
            assert np.all(c == np.round(c)), "offspring must be integers"

    def test_crossover_batch_respects_bounds(self):
        op = CrossoverIntegerSBX(prob=1.0, eta=20.0, prob_var=1.0)
        lb = np.zeros(3)
        ub = np.full(3, 9.0)
        rng = np.random.default_rng(0)
        n_pair = 8
        for _ in range(20):
            parents_batch = rng.integers(0, 10, size=(n_pair, 2, 3)).astype(float)
            c = op.crossover_batch(parents_batch, (lb, ub), rng=rng)
            assert c is not None
            assert np.all(c >= lb) and np.all(c <= ub)


# ---------------------------------------------------------------------------
# CrossoverCategorical
# ---------------------------------------------------------------------------


class TestCrossoverCategorical:
    _parents = np.array([[0.0, 2.0, 1.0], [2.0, 0.0, 2.0]])

    def test_output_shape(self):
        op = CrossoverCategorical(prob=1.0)
        c = op.crossover(self._parents, rng=np.random.default_rng(0))
        assert c.shape == (2, 3)

    def test_offspring_values_from_parents(self):
        op = CrossoverCategorical(prob=1.0)
        rng = np.random.default_rng(0)
        for _ in range(50):
            c = op.crossover(self._parents, rng=rng)
            for dim in range(self._parents.shape[1]):
                valid = {self._parents[0, dim], self._parents[1, dim]}
                assert c[0, dim] in valid
                assert c[1, dim] in valid

    def test_complementary_swap(self):
        op = CrossoverCategorical(prob=1.0)
        rng = np.random.default_rng(0)
        for _ in range(50):
            c = op.crossover(self._parents, rng=rng)
            for dim in range(self._parents.shape[1]):
                # if c1 took p2's value, c2 must have taken p1's
                p_sum = self._parents[0, dim] + self._parents[1, dim]
                assert c[0, dim] + c[1, dim] == p_sum

    def test_crossover_rate_attribute(self):
        assert CrossoverCategorical(prob=0.9).prob == 0.9

    def test_n_children(self):
        assert CrossoverCategorical(prob=1.0).n_children == 2

    def test_determinism(self):
        op = CrossoverCategorical(prob=1.0)
        c1 = op.crossover(self._parents, rng=np.random.default_rng(3))
        c2 = op.crossover(self._parents, rng=np.random.default_rng(3))
        np.testing.assert_array_equal(c1, c2)

    def test_crossover_batch_output_shape(self):
        op = CrossoverCategorical(prob=1.0)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = np.tile(self._parents, (n_pair, 1, 1))
        c = op.crossover_batch(parents_batch, rng=rng)
        assert c is not None
        assert c.shape == (n_pair, 2, 3)

    def test_crossover_batch_matches_single_at_n_pair_one(self):
        op = CrossoverCategorical(prob=1.0)
        parents_batch = self._parents[np.newaxis, :, :]
        c_batch = op.crossover_batch(parents_batch, rng=np.random.default_rng(3))
        assert c_batch is not None
        c_single = op.crossover(self._parents, rng=np.random.default_rng(3))
        np.testing.assert_array_equal(c_batch[0], c_single)

    def test_crossover_batch_deterministic_with_seed(self):
        op = CrossoverCategorical(prob=1.0)
        n_pair = 5
        parents_batch = np.tile(self._parents, (n_pair, 1, 1))
        c1 = op.crossover_batch(parents_batch, rng=np.random.default_rng(3))
        c2 = op.crossover_batch(parents_batch, rng=np.random.default_rng(3))
        np.testing.assert_array_equal(c1, c2)

    def test_crossover_batch_complementary_swap(self):
        op = CrossoverCategorical(prob=1.0)
        rng = np.random.default_rng(0)
        n_pair = 5
        parents_batch = np.tile(self._parents, (n_pair, 1, 1))
        for _ in range(50):
            c = op.crossover_batch(parents_batch, rng=rng)
            assert c is not None
            # if c1 took p2's value, c2 must have taken p1's, for every row
            p_sum = parents_batch[:, 0, :] + parents_batch[:, 1, :]
            np.testing.assert_array_equal(c[:, 0, :] + c[:, 1, :], p_sum)


# ---------------------------------------------------------------------------
# MutationIntegerUniform
# ---------------------------------------------------------------------------


class TestMutationIntegerUniform:
    _lb = np.array([0.0, 1.0, 3.0])
    _ub = np.array([5.0, 4.0, 8.0])

    def test_output_shape(self):
        op = MutationIntegerUniform(prob_var=1.0)
        p = np.array([2.0, 2.0, 5.0])
        c = op.mutate(p, (self._lb, self._ub), rng=np.random.default_rng(0))
        assert c.shape == (3,)

    def test_offspring_are_integers(self):
        op = MutationIntegerUniform(prob_var=1.0)
        rng = np.random.default_rng(0)
        p = np.array([2.0, 2.0, 5.0])
        for _ in range(30):
            c = op.mutate(p, (self._lb, self._ub), rng=rng)
            assert np.all(c == np.round(c))

    def test_values_within_bounds(self):
        op = MutationIntegerUniform(prob_var=1.0)
        rng = np.random.default_rng(1)
        p = np.array([2.0, 2.0, 5.0])
        for _ in range(50):
            c = op.mutate(p, (self._lb, self._ub), rng=rng)
            assert np.all(c >= self._lb)
            assert np.all(c <= self._ub)

    def test_zero_rate_unchanged(self):
        op = MutationIntegerUniform(prob_var=0.0)
        p = np.array([2.0, 3.0, 6.0])
        c = op.mutate(p, (self._lb, self._ub), rng=np.random.default_rng(0))
        np.testing.assert_array_equal(c, p)

    def test_determinism(self):
        op = MutationIntegerUniform(prob_var=0.5)
        p = np.array([2.0, 2.0, 5.0])
        c1 = op.mutate(p, (self._lb, self._ub), rng=np.random.default_rng(5))
        c2 = op.mutate(p, (self._lb, self._ub), rng=np.random.default_rng(5))
        np.testing.assert_array_equal(c1, c2)


# ---------------------------------------------------------------------------
# MutationCategorical
# ---------------------------------------------------------------------------


class TestMutationCategorical:
    # 3 categorical dims with 3, 2, 4 categories respectively
    _lb = np.array([0.0, 0.0, 0.0])
    _ub = np.array([2.0, 1.0, 3.0])

    def test_output_shape(self):
        op = MutationCategorical(prob_var=1.0)
        p = np.array([1.0, 0.0, 2.0])
        c = op.mutate(p, (self._lb, self._ub), rng=np.random.default_rng(0))
        assert c.shape == (3,)

    def test_offspring_are_valid_indices(self):
        op = MutationCategorical(prob_var=1.0)
        rng = np.random.default_rng(0)
        p = np.array([1.0, 0.0, 2.0])
        for _ in range(50):
            c = op.mutate(p, (self._lb, self._ub), rng=rng)
            assert np.all(c >= self._lb)
            assert np.all(c <= self._ub)
            assert np.all(c == np.round(c))

    def test_uniform_distribution(self):
        op = MutationCategorical(prob_var=1.0)
        rng = np.random.default_rng(0)
        p = np.array([0.0])
        lb = np.array([0.0])
        ub = np.array([3.0])  # 4 categories
        counts = np.zeros(4)
        for _ in range(4000):
            c = op.mutate(p, (lb, ub), rng=rng)
            counts[int(c[0])] += 1
        # Each category should appear roughly 25% of the time
        assert np.all(counts > 800), f"distribution not uniform: {counts}"

    def test_zero_rate_unchanged(self):
        op = MutationCategorical(prob_var=0.0)
        p = np.array([1.0, 0.0, 2.0])
        c = op.mutate(p, (self._lb, self._ub), rng=np.random.default_rng(0))
        np.testing.assert_array_equal(c, p)

    def test_determinism(self):
        op = MutationCategorical(prob_var=0.5)
        p = np.array([1.0, 0.0, 2.0])
        c1 = op.mutate(p, (self._lb, self._ub), rng=np.random.default_rng(9))
        c2 = op.mutate(p, (self._lb, self._ub), rng=np.random.default_rng(9))
        np.testing.assert_array_equal(c1, c2)


# ---------------------------------------------------------------------------
# CrossoverSBX: bounded variant and prob_var
# ---------------------------------------------------------------------------


class TestCrossoverSBXBounded:
    def _parents(self):
        return np.array([[0.1, 0.2, 0.5], [0.8, 0.9, 0.6]])

    def _bounds(self):
        lb = np.zeros(3)
        ub = np.ones(3)
        return lb, ub

    def test_bounded_offspring_within_bounds(self):
        op = CrossoverSBX(prob=1.0, eta=20.0, prob_var=1.0)
        lb, ub = self._bounds()
        rng = np.random.default_rng(0)
        for _ in range(50):
            c = op.crossover(self._parents(), (lb, ub), rng=rng)
            assert np.all(c >= lb) and np.all(c <= ub)

    def test_unbounded_fallback_when_bounds_none(self):
        op = CrossoverSBX(prob=1.0, eta=2.0, prob_var=1.0)
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        p = self._parents()
        c_none = op.crossover(p, None, rng=rng1)
        c_default = op.crossover(p, rng=rng2)
        np.testing.assert_array_equal(c_none, c_default)

    def test_prob_var_zero_returns_parents(self):
        op = CrossoverSBX(prob=1.0, eta=20.0, prob_var=0.0)
        lb, ub = self._bounds()
        p = self._parents()
        c = op.crossover(p, (lb, ub), rng=np.random.default_rng(0))
        np.testing.assert_array_equal(c[0], p[0])
        np.testing.assert_array_equal(c[1], p[1])

    def test_symmetric_margins_preserve_center(self):
        # Equal margins to lb/ub make beta_q identical for both children,
        # so the offspring center matches the parent center (special case
        # of the asymmetric formula, not a general guarantee).
        op = CrossoverSBX(prob=1.0, eta=20.0, prob_var=1.0)
        rng = np.random.default_rng(3)
        for _ in range(30):
            p = rng.uniform(0.4, 0.6, size=(2, 3))
            margin = rng.uniform(0.05, 0.3, size=3)
            y1 = np.minimum(p[0], p[1])
            y2 = np.maximum(p[0], p[1])
            lb = y1 - margin
            ub = y2 + margin
            c = op.crossover(p, (lb, ub), rng=rng)
            mid_p = 0.5 * (p[0] + p[1])
            mid_c = 0.5 * (c[0] + c[1])
            np.testing.assert_allclose(mid_c, mid_p, atol=1e-9)

    def test_asymmetric_bounds_produce_unequal_offsets(self):
        # With lb close to the parents and ub far away, beta_q must differ
        # between c1 (constrained by lb) and c2 (constrained by ub), so the
        # offspring are not symmetric about the parent center.
        op = CrossoverSBX(prob=1.0, eta=20.0, prob_var=1.0)
        lb = np.array([0.0])
        ub = np.array([1.0])
        p = np.array([[0.01], [0.5]])
        rng = np.random.default_rng(3)
        max_gap = 0.0
        for _ in range(30):
            c = op.crossover(p, (lb, ub), rng=rng)
            mid = 0.5 * (p[0] + p[1])
            offset1 = mid - c[0]
            offset2 = c[1] - mid
            max_gap = max(max_gap, np.abs(offset1 - offset2).max())
        assert max_gap > 1e-6

    def test_identical_parents_unchanged(self):
        op = CrossoverSBX(prob=1.0, eta=20.0, prob_var=1.0)
        lb, ub = self._bounds()
        p = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        c = op.crossover(p, (lb, ub), rng=np.random.default_rng(0))
        np.testing.assert_array_equal(c[0], p[0])
        np.testing.assert_array_equal(c[1], p[1])


# ---------------------------------------------------------------------------
# Mutation: individual-level prob gate
# ---------------------------------------------------------------------------


class TestMutationProbGate:
    def _range(self):
        lb = np.zeros(5)
        ub = np.ones(5)
        return lb, ub

    def test_prob_zero_returns_parent_unchanged(self):
        op = MutationPolynomial(prob=0.0, eta=20.0, prob_var=1.0)
        rng = np.random.default_rng(0)
        p = rng.uniform(0.1, 0.9, size=5)
        for _ in range(20):
            c = op.mutate(p, self._range(), rng=rng)
            np.testing.assert_array_equal(c, p)

    def test_prob_zero_uniform(self):
        op = MutationUniform(prob=0.0, prob_var=1.0)
        rng = np.random.default_rng(0)
        p = rng.uniform(0.1, 0.9, size=5)
        for _ in range(20):
            c = op.mutate(p, self._range(), rng=rng)
            np.testing.assert_array_equal(c, p)

    def test_prob_var_none_uses_adaptive_default(self):
        dim = 10
        op = MutationPolynomial(prob=1.0, eta=20.0, prob_var=None)
        lb = np.zeros(dim)
        ub = np.ones(dim)
        p = np.full(dim, 0.5)
        changed = np.zeros(dim)
        rng = np.random.default_rng(1)
        n_trials = 500
        for _ in range(n_trials):
            c = op.mutate(p, (lb, ub), rng=rng)
            changed += (c != p).astype(float)
        expected_rate = min(0.5, 1.0 / dim)
        observed_rate = changed / n_trials
        np.testing.assert_allclose(observed_rate, expected_rate, atol=0.05)


# ---------------------------------------------------------------------------
# DuplicateElimination
# ---------------------------------------------------------------------------


class TestDuplicateElimination:
    def test_exact_duplicate_detected(self):
        from saealib.operators.dedup import DuplicateElimination

        de = DuplicateElimination(atol=1e-16, rtol=0.0)
        pop = np.array([[0.1, 0.2], [0.3, 0.4]])
        off = np.array([[0.1, 0.2], [0.5, 0.6]])
        mask = de.find_duplicates(off, pop)
        assert mask[0]
        assert not mask[1]

    def test_within_tolerance_detected(self):
        from saealib.operators.dedup import DuplicateElimination

        de = DuplicateElimination(atol=1e-6, rtol=0.0)
        pop = np.array([[0.1, 0.2]])
        off = np.array([[0.1 + 1e-7, 0.2]])
        mask = de.find_duplicates(off, pop)
        assert mask[0]

    def test_outside_tolerance_not_detected(self):
        from saealib.operators.dedup import DuplicateElimination

        de = DuplicateElimination(atol=1e-10, rtol=0.0)
        pop = np.array([[0.1, 0.2]])
        off = np.array([[0.1 + 1e-9, 0.2]])
        mask = de.find_duplicates(off, pop)
        assert not mask[0]

    def test_empty_offspring_returns_empty_mask(self):
        from saealib.operators.dedup import DuplicateElimination

        de = DuplicateElimination()
        pop = np.array([[0.1, 0.2]])
        off = np.empty((0, 2))
        mask = de.find_duplicates(off, pop)
        assert mask.shape == (0,)
        assert mask.dtype == bool

    def test_no_duplicates_all_false(self):
        from saealib.operators.dedup import DuplicateElimination

        de = DuplicateElimination()
        pop = np.array([[0.1, 0.2], [0.3, 0.4]])
        off = np.array([[0.5, 0.6], [0.7, 0.8]])
        mask = de.find_duplicates(off, pop)
        assert not mask.any()

    def test_ga_stores_duplicate_elimination_attribute(self):
        from saealib.operators.dedup import DuplicateElimination

        de = DuplicateElimination(atol=1e-10, rtol=0.0, max_retries=5)
        ga = GA(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
            mutation=MutationPolynomial(prob=1.0, eta=20.0, prob_var=1.0),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
            duplicate_elimination=de,
        )
        assert ga.duplicate_elimination is de

    def test_ga_with_dedup_returns_correct_count(self):
        """GA with duplicate_elimination set produces expected offspring count."""
        from saealib.operators.dedup import DuplicateElimination

        de = DuplicateElimination()
        ga = GA(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
            mutation=MutationPolynomial(prob=1.0, eta=20.0, prob_var=1.0),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
            duplicate_elimination=de,
        )
        provider = _DispatchOnlyProvider()
        ctx = _make_ctx(n_pop=10)
        candidates = ga.ask(ctx, provider)
        assert len(candidates) == 10

    def test_ga_dedup_retries_when_offspring_are_copies(self):
        """With no-op crossover and no-op mutation, retries exhaust without crash."""
        from saealib.operators.dedup import DuplicateElimination

        # prob=0.0 crossover → offspring = parent copies (always duplicates)
        # prob_var=0.0 mutation → no variables mutated → still copies after each retry
        # dedup exhausts max_retries without fixing any, but must not raise
        de = DuplicateElimination(atol=1e-14, rtol=0.0, max_retries=3)
        ga = GA(
            crossover=CrossoverBLXAlpha(prob=0.0, alpha=0.4),
            mutation=MutationUniform(prob_var=0.0),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
            duplicate_elimination=de,
        )
        provider = _DispatchOnlyProvider()
        ctx = _make_ctx(n_pop=10)
        candidates = ga.ask(ctx, provider)
        assert len(candidates) == 10

    def test_ga_none_dedup_preserves_behavior(self):
        """GA with duplicate_elimination=None must not raise."""
        ga = GA(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
            mutation=MutationPolynomial(prob=1.0, eta=20.0, prob_var=1.0),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
            duplicate_elimination=None,
        )
        provider = _DispatchOnlyProvider()
        ctx = _make_ctx(n_pop=10)
        candidates = ga.ask(ctx, provider)
        assert len(candidates) == 10


# ---------------------------------------------------------------------------
# GA + batched default crossover operators: end-to-end smoke test
# (Issue #224, commit 7 — CrossoverSBX now dispatches through
# crossover_batch by default on continuous-only problems.)
# ---------------------------------------------------------------------------


class TestGABatchedCrossoverSmoke:
    def test_ga_with_sbx_runs_several_generations_on_sphere(self):
        """Coarse sanity net for the new default crossover_batch dispatch
        path: GA with CrossoverSBX (now batch-capable) on a continuous-only
        problem must run several generations without error and produce
        finite, in-bounds candidates. Not a strict numeric-equality check."""
        dim = 5
        lb = [-5.0] * dim
        ub = [5.0] * dim
        result = minimize(
            lambda x: np.sum(x**2),
            dim=dim,
            lb=lb,
            ub=ub,
            algorithm=GA(
                crossover=CrossoverSBX(prob=0.9, eta=15.0),
                mutation=MutationPolynomial(prob=1.0, eta=20.0, prob_var=0.2),
                parent_selection=TournamentSelection(2),
                survivor_selection=TruncationSelection(),
            ),
            surrogate="rbf",
            max_fe=150,
            pop_size=10,
            seed=0,
            verbose=False,
        )
        assert result.fe > 0
        assert np.isfinite(result.f).all()
        assert np.all(result.x >= lb) and np.all(result.x <= ub)
