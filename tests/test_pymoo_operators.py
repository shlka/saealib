"""Tests for PymooCrossover / PymooMutation against the real pymoo library."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from saealib import (
    GA,
    CategoricalVariable,
    ContinuousVariable,
    IntegerVariable,
    TournamentSelection,
    TruncationSelection,
    minimize,
)
from saealib._dispatch import batch_override_is_consistent
from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.operators import PymooCrossover, PymooMutation
from saealib.operators.dedup import DuplicateElimination
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem

DIM = 6


def _make_parents(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=(2, DIM))


class _CountedSBX(SBX):
    """SBX subclass counting real ``_do()`` invocations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_do_calls = 0

    def _do(self, *args, **kwargs):
        self.n_do_calls += 1
        return super()._do(*args, **kwargs)


class _CountedPM(PM):
    """PM subclass counting real ``_do()`` invocations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_do_calls = 0

    def _do(self, *args, **kwargs):
        self.n_do_calls += 1
        return super()._do(*args, **kwargs)


# ---------------------------------------------------------------------------
# PymooCrossover
# ---------------------------------------------------------------------------


class TestPymooCrossover:
    def test_output_shape(self):
        op = PymooCrossover(SBX(eta=15))
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        c = op.crossover(p, bounds=(lb, ub), rng=rng)
        assert c.shape == (2, DIM)

    def test_mirrors_pymoo_n_parents_n_children(self):
        pymoo_op = SBX()
        op = PymooCrossover(pymoo_op)
        assert op.n_parents == pymoo_op.n_parents
        assert op.n_children == pymoo_op.n_offsprings

    def test_explicit_n_parents_n_children_override(self):
        op = PymooCrossover(SBX(), n_parents=3, n_children=1)
        assert op.n_parents == 3
        assert op.n_children == 1

    def test_prob_mirrors_pymoo_operator(self):
        pymoo_op = SBX(prob=0.75)
        op = PymooCrossover(pymoo_op)
        assert op.prob == pytest.approx(pymoo_op.prob.value)

    def test_prob_override(self):
        op = PymooCrossover(SBX(), prob=0.42)
        assert op.prob == pytest.approx(0.42)

    def test_respects_bounds(self):
        op = PymooCrossover(SBX(eta=5))
        rng = np.random.default_rng(1)
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        for _ in range(20):
            p = rng.uniform(-1.0, 1.0, size=(2, DIM))
            c = op.crossover(p, bounds=(lb, ub), rng=rng)
            assert np.all(c >= lb) and np.all(c <= ub)

    def test_offspring_differ_from_parents_over_repeated_calls(self):
        op = PymooCrossover(SBX(eta=5))
        rng = np.random.default_rng(2)
        p = _make_parents(rng)
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        any_diff = False
        for _ in range(20):
            c = op.crossover(p, bounds=(lb, ub), rng=rng)
            if not np.array_equal(c, p):
                any_diff = True
        assert any_diff

    def test_unbounded_call_raises(self):
        """pymoo operators such as SBX unconditionally read problem.xl/xu, so
        calling without bounds fails inside pymoo itself. saealib's own GA
        always supplies bounds (ga.py's _route_crossover); this documents the
        requirement for direct callers instead of adding a defensive guard."""
        op = PymooCrossover(SBX(eta=5))
        rng = np.random.default_rng(3)
        p = _make_parents(rng)
        with pytest.raises(TypeError):
            op.crossover(p, rng=rng)

    def test_crossover_batch_calls_do_once(self):
        n_pair = 5
        counted = _CountedSBX(eta=15)
        op = PymooCrossover(counted)
        rng = np.random.default_rng(5)
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)

        result = op.crossover_batch(parents_batch, bounds=(lb, ub), rng=rng)
        assert counted.n_do_calls == 1
        assert result.shape == (n_pair, 2, DIM)

        counted.n_do_calls = 0
        for k in range(n_pair):
            op.crossover(parents_batch[k], bounds=(lb, ub), rng=rng)
        assert counted.n_do_calls == n_pair

    def test_crossover_batch_matches_single_crossover_at_n_pair_one(self):
        op = PymooCrossover(SBX(eta=15))
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        p = np.random.default_rng(6).uniform(-1.0, 1.0, size=(2, DIM))

        rng_batch = np.random.default_rng(7)
        batch_result = op.crossover_batch(
            p[np.newaxis, :, :], bounds=(lb, ub), rng=rng_batch
        )
        batch_result = batch_result[0]

        rng_single = np.random.default_rng(7)
        single_result = op.crossover(p, bounds=(lb, ub), rng=rng_single)

        np.testing.assert_allclose(batch_result, single_result)

    def test_crossover_batch_output_shape(self):
        op = PymooCrossover(SBX(eta=15))
        rng = np.random.default_rng(8)
        n_pair = 3
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        c = op.crossover_batch(parents_batch, bounds=(lb, ub), rng=rng)
        assert c.shape == (n_pair, 2, DIM)

    def test_subdimensional_call_rebuilds_shim_problem(self):
        """A smaller dim slice (e.g. GA's per-type variable routing) works and
        does not reuse the wrong cached pymoo Problem shim."""
        op = PymooCrossover(SBX(eta=5))
        rng = np.random.default_rng(3)
        p_full = rng.uniform(-1.0, 1.0, size=(2, DIM))
        lb_full = np.full(DIM, -1.0)
        ub_full = np.full(DIM, 1.0)
        op.crossover(p_full, bounds=(lb_full, ub_full), rng=rng)
        p_small = p_full[:, :2]
        c_small = op.crossover(p_small, bounds=(lb_full[:2], ub_full[:2]), rng=rng)
        assert c_small.shape == (2, 2)


# ---------------------------------------------------------------------------
# PymooMutation
# ---------------------------------------------------------------------------


class TestPymooMutation:
    def test_output_shape(self):
        op = PymooMutation(PM(eta=20))
        rng = np.random.default_rng(0)
        p = rng.uniform(-1.0, 1.0, size=DIM)
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        m = op.mutate(p, (lb, ub), rng=rng)
        assert m.shape == (DIM,)

    def test_prob_zero_returns_unchanged(self):
        op = PymooMutation(PM(eta=20), prob=0.0)
        rng = np.random.default_rng(0)
        p = rng.uniform(-1.0, 1.0, size=DIM)
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        m = op.mutate(p, (lb, ub), rng=rng)
        np.testing.assert_array_equal(m, p)

    def test_prob_one_changes_over_repeated_calls(self):
        op = PymooMutation(PM(eta=5), prob=1.0)
        rng = np.random.default_rng(1)
        p = rng.uniform(-1.0, 1.0, size=DIM)
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        any_diff = False
        for _ in range(20):
            m = op.mutate(p, (lb, ub), rng=rng)
            if not np.array_equal(m, p):
                any_diff = True
        assert any_diff

    def test_prob_var_is_not_mirrored(self):
        """Regression guard: prob_var must stay None so GA's mixed-variable
        routing falls back to its own default instead of a foreign, possibly
        non-float pymoo.core.variable.Real value."""
        op = PymooMutation(PM(eta=20, prob_var=0.3))
        assert op.prob_var is None

    def test_respects_bounds(self):
        op = PymooMutation(PM(eta=5), prob=1.0)
        rng = np.random.default_rng(4)
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        for _ in range(20):
            p = rng.uniform(-1.0, 1.0, size=DIM)
            m = op.mutate(p, (lb, ub), rng=rng)
            assert np.all(m >= lb) and np.all(m <= ub)

    def test_mutate_batch_calls_do_once(self):
        n = 6
        counted = _CountedPM(eta=20, prob_var=1.0)
        op = PymooMutation(counted, prob=1.0)
        rng = np.random.default_rng(9)
        candidates_batch = rng.uniform(-1.0, 1.0, size=(n, DIM))
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)

        result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
        assert result is not None
        assert counted.n_do_calls == 1
        assert result.shape == (n, DIM)

        counted.n_do_calls = 0
        for k in range(n):
            op.mutate(candidates_batch[k], (lb, ub), rng=rng)
        assert counted.n_do_calls == n

    def test_mutate_batch_matches_single_mutate_at_n_one(self):
        op = PymooMutation(PM(eta=20, prob_var=1.0), prob=1.0)
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        p = np.random.default_rng(10).uniform(-1.0, 1.0, size=DIM)

        rng_batch = np.random.default_rng(11)
        batch_result = op.mutate_batch(p[np.newaxis, :], (lb, ub), rng=rng_batch)
        assert batch_result is not None
        batch_result = batch_result[0]

        rng_single = np.random.default_rng(11)
        single_result = op.mutate(p, (lb, ub), rng=rng_single)

        np.testing.assert_allclose(batch_result, single_result)

    def test_mutate_batch_prob_zero_returns_unchanged_without_do_call(self):
        counted = _CountedPM(eta=20)
        op = PymooMutation(counted, prob=0.0)
        rng = np.random.default_rng(12)
        n = 5
        candidates_batch = rng.uniform(-1.0, 1.0, size=(n, DIM))
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)

        result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
        np.testing.assert_array_equal(result, candidates_batch)
        assert counted.n_do_calls == 0

    def test_mutate_batch_prob_one_mutates_all_rows_with_single_do_call(self):
        counted = _CountedPM(eta=20, prob_var=1.0)
        op = PymooMutation(counted, prob=1.0)
        rng = np.random.default_rng(13)
        n = 5
        candidates_batch = rng.uniform(-1.0, 1.0, size=(n, DIM))
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)

        result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
        assert result is not None
        assert counted.n_do_calls == 1
        assert not np.array_equal(result, candidates_batch)

    def test_mutate_batch_fractional_prob_gates_rows_exactly(self):
        seed = 14
        counted = _CountedPM(eta=20, prob_var=1.0)
        op = PymooMutation(counted, prob=0.5)
        n = 8
        candidates_batch = np.random.default_rng(seed + 1).uniform(
            -1.0, 1.0, size=(n, DIM)
        )
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)

        # mutate_batch's gate draw is the first thing to touch the rng
        # stream, so a parallel, identically-seeded generator predicts it.
        expected_gate = np.random.default_rng(seed).random(n) < op.prob

        result = op.mutate_batch(
            candidates_batch, (lb, ub), rng=np.random.default_rng(seed)
        )
        assert result is not None

        np.testing.assert_array_equal(
            result[~expected_gate], candidates_batch[~expected_gate]
        )
        assert counted.n_do_calls == 1

    def test_mutate_batch_output_shape(self):
        op = PymooMutation(PM(eta=20), prob=1.0)
        rng = np.random.default_rng(15)
        n = 4
        candidates_batch = rng.uniform(-1.0, 1.0, size=(n, DIM))
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        m = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
        assert m is not None
        assert m.shape == (n, DIM)


# ---------------------------------------------------------------------------
# End-to-end: pymoo operators driving saealib's own GA
# ---------------------------------------------------------------------------


class TestPymooOperatorsEndToEnd:
    def test_ga_with_pymoo_operators_improves_sphere(self):
        rng_seed = 0
        result = minimize(
            lambda x: np.sum(x**2),
            dim=5,
            lb=[-5.0] * 5,
            ub=[5.0] * 5,
            algorithm=GA(
                crossover=PymooCrossover(SBX(eta=15)),
                mutation=PymooMutation(PM(eta=20)),
                parent_selection=TournamentSelection(2),
                survivor_selection=TruncationSelection(),
            ),
            surrogate="rbf",
            max_fe=150,
            pop_size=10,
            seed=rng_seed,
            verbose=False,
        )
        assert result.fe > 0
        assert np.isfinite(result.f).all()


# ---------------------------------------------------------------------------
# GA.ask() batch-path dispatch (Issue #224, commit 6)
# ---------------------------------------------------------------------------


class _NoopProvider:
    """Minimal provider that silently discards dispatched events."""

    def dispatch(self, event):
        pass


def _make_continuous_ctx(n_pop=10, seed=0, identical=False):
    problem = Problem(
        func=lambda x: np.array([np.sum(x**2)]),
        dim=DIM,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0] * DIM,
        ub=[1.0] * DIM,
        comparator=SingleObjectiveComparator(),
    )
    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
        PopulationAttribute(name="g", dtype=np.float64, shape=(0,)),
        PopulationAttribute(name="cv", dtype=np.float64, shape=()),
    ]
    rng = np.random.default_rng(seed)
    pop = Population(attrs, init_capacity=n_pop + 5)
    if identical:
        xs = np.tile(rng.uniform(-1.0, 1.0, size=DIM), (n_pop, 1))
    else:
        xs = rng.uniform(-1.0, 1.0, size=(n_pop, DIM))
    fs = np.array([[np.sum(x**2)] for x in xs])
    pop.extend({"x": xs, "f": fs, "g": np.zeros((n_pop, 0)), "cv": np.zeros(n_pop)})
    arc = Archive(attrs, init_capacity=5)
    pareto_arc = ParetoArchive(attrs, init_capacity=5, direction=np.array([-1.0]))
    return OptimizationState(
        problem=problem,
        population=pop,
        archive=arc,
        pareto_archive=pareto_arc,
        rng=np.random.default_rng(seed),
    )


def _make_mixed_problem():
    variables = [
        ContinuousVariable(-1.0, 1.0),
        ContinuousVariable(-1.0, 1.0),
        IntegerVariable(0, 9),
        CategoricalVariable(["a", "b", "c"]),
    ]
    return Problem(
        func=lambda x: np.array([x[0]]),
        dim=4,
        n_obj=1,
        direction=np.array([-1.0]),
        variables=variables,
    )


def _make_mixed_ctx(n_pop=8, seed=42):
    problem = _make_mixed_problem()
    dim = problem.dim
    n_obj = problem.n_obj
    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(dim,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(n_obj,)),
        PopulationAttribute(name="g", dtype=np.float64, shape=(0,)),
        PopulationAttribute(name="cv", dtype=np.float64, shape=()),
    ]
    rng = np.random.default_rng(seed)
    pop = Population(attrs, init_capacity=n_pop + 2)
    arc = Archive(attrs, init_capacity=n_pop + 2)
    pareto_arc = ParetoArchive(
        attrs, init_capacity=n_pop + 2, direction=problem.direction
    )
    xs = problem.repair(rng.uniform(problem.lb, problem.ub, size=(n_pop, dim)))
    fs = np.zeros((n_pop, n_obj))
    pop.extend({"x": xs, "f": fs, "g": np.zeros((n_pop, 0)), "cv": np.zeros(n_pop)})
    arc.extend({"x": xs, "f": fs, "g": np.zeros((n_pop, 0)), "cv": np.zeros(n_pop)})
    return OptimizationState(
        problem=problem,
        population=pop,
        archive=arc,
        pareto_archive=pareto_arc,
        rng=np.random.default_rng(seed + 1),
    )


class TestPymooOperatorsDispatchConsistency:
    """PymooCrossover inherits crossover from its crossover_batch primitive.

    A plain, unsubclassed instance must remain "consistent" under
    batch_override_is_consistent (Issue #224 follow-up fix) exactly like every
    built-in operator. PymooMutation continues to define mutate/mutate_batch
    together. This is what lets GA still engage the batch path for
    pymoo-wrapped operators (see
    TestGABatchDispatch.test_continuous_problem_calls_do_once_each below,
    which is the end-to-end proof).
    """

    def test_pymoo_crossover_consistent(self):
        op = PymooCrossover(SBX(eta=15))
        assert batch_override_is_consistent(op, "crossover_batch", "crossover") is True

    def test_pymoo_mutation_consistent(self):
        op = PymooMutation(PM(eta=20))
        assert batch_override_is_consistent(op, "mutate_batch", "mutate") is True


class TestGABatchDispatch:
    """Verify GA.ask() engages crossover_batch/mutate_batch for continuous-only
    problems with batch-capable operators, and correctly falls back to the
    per-pair/per-individual loop on mixed-variable problems (Issue #224,
    commit 6 — the batch-vs-loop routing added to ga.py)."""

    def test_continuous_problem_calls_do_once_each(self):
        """A single ga.ask() call with n_pair > 1 must call the wrapped
        pymoo crossover's/mutation's _do() exactly once total, not once per
        pair/individual — the actual proof the batch path is engaged."""
        counted_cx = _CountedSBX(eta=15)
        counted_mut = _CountedPM(eta=20)
        ga = GA(
            crossover=PymooCrossover(counted_cx, prob=1.0),
            mutation=PymooMutation(counted_mut, prob=1.0),
            parent_selection=TournamentSelection(2),
            survivor_selection=TruncationSelection(),
        )
        ctx = _make_continuous_ctx(n_pop=10)
        offspring = ga.ask(ctx, _NoopProvider(), n_offspring=13)
        assert len(offspring) == 13
        assert counted_cx.n_do_calls == 1
        assert counted_mut.n_do_calls == 1

    def test_mixed_problem_calls_do_once_per_pair_and_individual(self):
        """Safety-net for the `mixed` short-circuit: on a mixed-variable
        problem, _do() must be called once per pair/individual, i.e. the
        batch path must NOT be engaged even though the continuous
        crossover/mutation operators support it."""
        counted_cx = _CountedSBX(eta=15)
        counted_mut = _CountedPM(eta=20)
        ga = GA(
            crossover=PymooCrossover(counted_cx, prob=1.0),
            mutation=PymooMutation(counted_mut, prob=1.0),
            parent_selection=TournamentSelection(2),
            survivor_selection=TruncationSelection(),
        )
        ctx = _make_mixed_ctx(n_pop=8)
        n_offspring = 10
        offspring = ga.ask(ctx, _NoopProvider(), n_offspring=n_offspring)
        n_children = ga.crossover.n_children
        n_pair = math.ceil(n_offspring / n_children)
        assert len(offspring) == n_offspring
        assert counted_cx.n_do_calls == n_pair
        assert counted_mut.n_do_calls == n_pair * n_children

    def test_with_post_wrapped_batch_operator_hooks_fire_once_per_unit(self):
        """Extends TestGAHookInvocation (tests/test_operators.py) to a
        batch-capable, with_post-wrapped operator: post_crossover/
        post_mutation must still fire exactly once per pair/individual."""
        cx_calls = [0]
        mut_calls = [0]

        def cx_hook(offspring, parents, rng, ctx):
            cx_calls[0] += 1
            return offspring

        def mut_hook(offspring, mutate_range, rng, ctx):
            mut_calls[0] += 1
            return offspring

        crossover = PymooCrossover(SBX(eta=15), prob=1.0).with_post(cx_hook)
        mutation = PymooMutation(PM(eta=20), prob=1.0).with_post(mut_hook)
        ga = GA(
            crossover=crossover,
            mutation=mutation,
            parent_selection=TournamentSelection(2),
            survivor_selection=TruncationSelection(),
        )
        ctx = _make_continuous_ctx(n_pop=10)
        ga.ask(ctx, _NoopProvider(), n_offspring=13)
        n_children = ga.crossover.n_children
        n_pair = math.ceil(13 / n_children)
        assert cx_calls[0] == n_pair
        # post_mutation fires once per pre-truncation candidate (n_pair *
        # n_children rows), not once per final n_offspring=13 individuals.
        assert mut_calls[0] == n_pair * n_children

    def test_duplicate_elimination_empty_gate_branch(self):
        """prob=0.0 on both operators forces every offspring to be an exact
        parent copy (guaranteed duplicate) while also driving crossover_batch's
        gate.any() == False, exercising the empty-batch skip inside
        _make_offspring's retry path."""
        de = DuplicateElimination(atol=1e-10, rtol=0.0, max_retries=3)
        ga = GA(
            crossover=PymooCrossover(SBX(eta=15), prob=0.0),
            mutation=PymooMutation(PM(eta=20), prob=0.0),
            parent_selection=TournamentSelection(2),
            survivor_selection=TruncationSelection(),
            duplicate_elimination=de,
        )
        ctx = _make_continuous_ctx(n_pop=10)
        offspring = ga.ask(ctx, _NoopProvider(), n_offspring=10)
        assert len(offspring) == 10

    def test_duplicate_elimination_populated_gate_branch(self):
        """An all-identical population + prob=1.0 crossover (SBX degenerates
        to returning the parents unchanged when they are identical) forces
        duplicates while keeping crossover_batch's gate.any() == True,
        exercising the populated-batch path inside _make_offspring's retry."""
        de = DuplicateElimination(atol=1e-10, rtol=0.0, max_retries=3)
        ga = GA(
            crossover=PymooCrossover(SBX(eta=15), prob=1.0),
            mutation=PymooMutation(PM(eta=20), prob=0.0),
            parent_selection=TournamentSelection(2),
            survivor_selection=TruncationSelection(),
            duplicate_elimination=de,
        )
        ctx = _make_continuous_ctx(n_pop=10, identical=True)
        offspring = ga.ask(ctx, _NoopProvider(), n_offspring=10)
        assert len(offspring) == 10
