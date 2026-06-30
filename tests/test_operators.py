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

from saealib import GA, SequentialSelection, TruncationSelection
from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.operators.crossover import (
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
from saealib.operators.selection import RouletteWheelSelection
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
        return MutationPolynomial(mutation_rate=1.0, eta=20.0)

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
            mutation=MutationUniform(mutation_rate=0.0),
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

        mutation = MutationUniform(mutation_rate=0.0).with_post(hook)
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


# ---------------------------------------------------------------------------
# MutationIntegerUniform
# ---------------------------------------------------------------------------


class TestMutationIntegerUniform:
    _lb = np.array([0.0, 1.0, 3.0])
    _ub = np.array([5.0, 4.0, 8.0])

    def test_output_shape(self):
        op = MutationIntegerUniform(mutation_rate=1.0)
        p = np.array([2.0, 2.0, 5.0])
        c = op.mutate(p, (self._lb, self._ub), rng=np.random.default_rng(0))
        assert c.shape == (3,)

    def test_offspring_are_integers(self):
        op = MutationIntegerUniform(mutation_rate=1.0)
        rng = np.random.default_rng(0)
        p = np.array([2.0, 2.0, 5.0])
        for _ in range(30):
            c = op.mutate(p, (self._lb, self._ub), rng=rng)
            assert np.all(c == np.round(c))

    def test_values_within_bounds(self):
        op = MutationIntegerUniform(mutation_rate=1.0)
        rng = np.random.default_rng(1)
        p = np.array([2.0, 2.0, 5.0])
        for _ in range(50):
            c = op.mutate(p, (self._lb, self._ub), rng=rng)
            assert np.all(c >= self._lb)
            assert np.all(c <= self._ub)

    def test_zero_rate_unchanged(self):
        op = MutationIntegerUniform(mutation_rate=0.0)
        p = np.array([2.0, 3.0, 6.0])
        c = op.mutate(p, (self._lb, self._ub), rng=np.random.default_rng(0))
        np.testing.assert_array_equal(c, p)

    def test_determinism(self):
        op = MutationIntegerUniform(mutation_rate=0.5)
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
        op = MutationCategorical(mutation_rate=1.0)
        p = np.array([1.0, 0.0, 2.0])
        c = op.mutate(p, (self._lb, self._ub), rng=np.random.default_rng(0))
        assert c.shape == (3,)

    def test_offspring_are_valid_indices(self):
        op = MutationCategorical(mutation_rate=1.0)
        rng = np.random.default_rng(0)
        p = np.array([1.0, 0.0, 2.0])
        for _ in range(50):
            c = op.mutate(p, (self._lb, self._ub), rng=rng)
            assert np.all(c >= self._lb)
            assert np.all(c <= self._ub)
            assert np.all(c == np.round(c))

    def test_uniform_distribution(self):
        op = MutationCategorical(mutation_rate=1.0)
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
        op = MutationCategorical(mutation_rate=0.0)
        p = np.array([1.0, 0.0, 2.0])
        c = op.mutate(p, (self._lb, self._ub), rng=np.random.default_rng(0))
        np.testing.assert_array_equal(c, p)

    def test_determinism(self):
        op = MutationCategorical(mutation_rate=0.5)
        p = np.array([1.0, 0.0, 2.0])
        c1 = op.mutate(p, (self._lb, self._ub), rng=np.random.default_rng(9))
        c2 = op.mutate(p, (self._lb, self._ub), rng=np.random.default_rng(9))
        np.testing.assert_array_equal(c1, c2)
