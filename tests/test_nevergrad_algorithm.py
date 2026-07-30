"""Tests for NevergradAlgorithm against the real nevergrad library."""

from __future__ import annotations

import nevergrad as ng
import numpy as np
import pytest

from saealib import minimize
from saealib.algorithms.nevergrad_algorithm import NevergradAlgorithm
from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.exceptions import ConfigurationError
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.problem.constraint import InequalityConstraint
from saealib.strategies.direct import DirectStrategy
from saealib.strategies.ps import PreSelectionStrategy

DIM = 5
N_POP = 10

# Three registry families with genuinely distinct internal candidate-tracking
# behavior (see NevergradAlgorithm's docstring): CMA batches its engine
# update per popsize-worth of tells, DE tracks outstanding candidates via an
# internal UID queue, OnePlusOne uses a simpler incumbent-based update.
FAMILIES = ["CMA", "OnePlusOne", "DE"]


class _DummyProvider:
    def dispatch(self, event):
        pass


def _make_problem(dim: int = DIM, constrained: bool = False, n_obj: int = 1) -> Problem:
    direction = np.full(n_obj, -1.0)
    comparator = (
        SingleObjectiveComparator(direction=direction[0]) if n_obj == 1 else None
    )

    def func(x: np.ndarray) -> np.ndarray:
        return np.array([np.sum(x**2)] * n_obj)

    constraints = (
        [InequalityConstraint(lambda x: x[0] - 3.0, threshold=0.0)]
        if constrained
        else None
    )

    return Problem(
        func=func,
        dim=dim,
        n_obj=n_obj,
        direction=direction,
        lb=[-5.0] * dim,
        ub=[5.0] * dim,
        comparator=comparator,
        constraints=constraints,
    )


def _make_ctx(
    algo: NevergradAlgorithm,
    problem: Problem,
    n_pop: int = N_POP,
    rng_seed: int = 0,
) -> OptimizationState:
    rng = np.random.default_rng(rng_seed)
    attrs = [
        PopulationAttribute("x", float, (problem.dim,), default=np.nan),
        PopulationAttribute("f", float, (problem.n_obj,), default=np.nan),
        PopulationAttribute("g", float, (problem.n_constraints,), default=0.0),
        PopulationAttribute("cv", float, (), default=0.0),
        *algo.get_required_attrs(problem),
    ]
    pop = Population(attrs, init_capacity=n_pop + 5)
    xs = rng.uniform(problem.lb, problem.ub, size=(n_pop, problem.dim))
    fs = np.array([problem.func(x) for x in xs])
    gs = np.zeros((n_pop, problem.n_constraints))
    cvs = np.zeros(n_pop)
    pop.extend({"x": xs, "f": fs, "g": gs, "cv": cvs})
    arc = Archive(attrs, init_capacity=n_pop + 5)
    pareto_arc = ParetoArchive(
        attrs, init_capacity=n_pop + 5, direction=problem.direction
    )
    return OptimizationState(
        problem=problem,
        population=pop,
        archive=arc,
        pareto_archive=pareto_arc,
        rng=np.random.default_rng(rng_seed),
    )


def _make_optimizer(
    family: str = "CMA", dim: int = DIM, budget: int | None = 1000
) -> ng.optimization.base.Optimizer:
    param = ng.p.Array(shape=(dim,)).set_bounds(-5.0, 5.0)
    return ng.optimizers.registry[family](parametrization=param, budget=budget)


# ---------------------------------------------------------------------------
# ask / tell round trip (whitebox, default family: CMA)
# ---------------------------------------------------------------------------


class TestNevergradAlgorithmAskTell:
    def test_ask_produces_pop_size_candidates_by_default(self):
        problem = _make_problem()
        algo = NevergradAlgorithm(_make_optimizer())
        ctx = _make_ctx(algo, problem)
        cand = algo.ask(ctx, _DummyProvider())
        assert len(cand) == N_POP
        assert np.all(cand.x >= problem.lb) and np.all(cand.x <= problem.ub)
        np.testing.assert_array_equal(
            np.sort(cand.get_array("nevergrad_idx")), np.arange(N_POP)
        )

    def test_n_offspring_is_honored(self):
        problem = _make_problem()
        algo = NevergradAlgorithm(_make_optimizer())
        ctx = _make_ctx(algo, problem)

        cand_small = algo.ask(ctx, _DummyProvider(), n_offspring=3)
        assert len(cand_small) == 3

        cand_large = algo.ask(ctx, _DummyProvider(), n_offspring=17)
        assert len(cand_large) == 17

    def test_ask_tell_round_trip_updates_population(self):
        problem = _make_problem()
        algo = NevergradAlgorithm(_make_optimizer())
        ctx = _make_ctx(algo, problem)
        cand = algo.ask(ctx, _DummyProvider())
        cand.update_array("f", np.array([problem.func(x) for x in cand.x]))
        algo.tell(ctx, _DummyProvider(), cand)
        assert len(ctx.population) == N_POP
        assert np.isfinite(ctx.population.get_array("f")).all()

    def test_tell_with_reordered_offspring_still_aligns(self):
        """Offspring reordered before tell() (as SortByScoreStage would do).

        The nevergrad_idx column must recover the correct candidate<->loss
        mapping so that tell() attaches each loss to the Parameter object it
        actually belongs to, not a mismatched one.
        """
        problem = _make_problem()
        algo = NevergradAlgorithm(_make_optimizer())
        ctx = _make_ctx(algo, problem)
        cand = algo.ask(ctx, _DummyProvider())

        f_correct = np.array([problem.func(x) for x in cand.x])
        cand.update_array("f", f_correct)

        perm = np.random.default_rng(1).permutation(N_POP)
        reordered = cand.extract(perm)
        assert not np.array_equal(
            reordered.get_array("nevergrad_idx"), cand.get_array("nevergrad_idx")
        )

        algo.tell(ctx, _DummyProvider(), reordered)  # should not raise
        assert len(ctx.population) == N_POP

    def test_constrained_problem_rejected(self):
        problem = _make_problem(constrained=True)
        algo = NevergradAlgorithm(_make_optimizer())
        with pytest.raises(ConfigurationError):
            algo.get_required_attrs(problem)

    def test_multi_objective_problem_rejected(self):
        problem = _make_problem(n_obj=2)
        algo = NevergradAlgorithm(_make_optimizer())
        with pytest.raises(ConfigurationError):
            algo.get_required_attrs(problem)

    def test_dimension_mismatch_rejected(self):
        problem = _make_problem(dim=DIM)
        algo = NevergradAlgorithm(_make_optimizer(dim=DIM + 1))
        with pytest.raises(ConfigurationError):
            algo.get_required_attrs(problem)

    def test_non_array_parametrization_rejected(self):
        problem = _make_problem(dim=DIM)
        param = ng.p.Instrumentation(*[ng.p.Scalar() for _ in range(DIM)])
        optimizer = ng.optimizers.registry["CMA"](parametrization=param, budget=1000)
        algo = NevergradAlgorithm(optimizer)
        with pytest.raises(ConfigurationError):
            algo.get_required_attrs(problem)

    def test_non_flat_array_parametrization_rejected(self):
        problem = _make_problem(dim=6)
        param = ng.p.Array(shape=(2, 3)).set_bounds(-5.0, 5.0)  # dimension=6, not flat
        optimizer = ng.optimizers.registry["CMA"](parametrization=param, budget=1000)
        algo = NevergradAlgorithm(optimizer)
        with pytest.raises(ConfigurationError):
            algo.get_required_attrs(problem)

    def test_ask_repairs_and_writes_back_out_of_bounds_samples(self):
        """A parametrization wider than the problem's own bounds must have
        its samples clipped by ask()'s repair chain, and the repaired
        coordinates must be written back into the tracked candidate object
        (verified directly here) -- otherwise tell() would teach the
        optimizer the loss at the wrong (unrepaired) point."""
        problem = _make_problem(dim=DIM)
        param = ng.p.Array(shape=(DIM,)).set_bounds(-50.0, 50.0)
        optimizer = ng.optimizers.registry["CMA"](parametrization=param, budget=1000)
        algo = NevergradAlgorithm(optimizer)
        ctx = _make_ctx(algo, problem)
        for _ in range(5):
            cand = algo.ask(ctx, _DummyProvider())
            assert np.all(cand.x >= problem.lb) and np.all(cand.x <= problem.ub)
            for row, tracked in enumerate(algo._asked):
                np.testing.assert_allclose(
                    np.asarray(tracked.value, dtype=float), cand.x[row]
                )
            cand.update_array("f", np.array([problem.func(x) for x in cand.x]))
            algo.tell(ctx, _DummyProvider(), cand)


# ---------------------------------------------------------------------------
# Partial tell
# ---------------------------------------------------------------------------


class TestNevergradAlgorithmPartialTell:
    def test_partial_tell_default_raises(self):
        """Concretely verified that untold candidates are not harmless: CMA
        fires its own 'orphanated injected solution' warning and only
        advances its engine once a full popsize-worth of tells has arrived,
        and DE-family optimizers leave a dangling UID-queue entry."""
        problem = _make_problem()
        algo = NevergradAlgorithm(_make_optimizer())
        ctx = _make_ctx(algo, problem)
        cand = algo.ask(ctx, _DummyProvider())
        cand.update_array("f", np.array([problem.func(x) for x in cand.x]))
        truncated = cand.extract(np.arange(N_POP - 4))
        with pytest.raises(ConfigurationError):
            algo.tell(ctx, _DummyProvider(), truncated)

    def test_partial_tell_opt_in_runs(self):
        problem = _make_problem()
        algo = NevergradAlgorithm(_make_optimizer(), allow_partial_tell=True)
        ctx = _make_ctx(algo, problem)
        cand = algo.ask(ctx, _DummyProvider())
        cand.update_array("f", np.array([problem.func(x) for x in cand.x]))
        truncated = cand.extract(np.arange(N_POP - 4))
        algo.tell(ctx, _DummyProvider(), truncated)  # should not raise
        assert len(ctx.population) == N_POP - 4

    def test_preselection_strategy_requires_opt_in(self):
        with pytest.raises(ConfigurationError):
            minimize(
                lambda x: np.sum(x**2),
                dim=DIM,
                lb=[-5.0] * DIM,
                ub=[5.0] * DIM,
                algorithm=NevergradAlgorithm(_make_optimizer(dim=DIM)),
                surrogate="rbf",
                strategy=PreSelectionStrategy(n_candidates=N_POP, n_select=N_POP - 3),
                max_fe=100,
                pop_size=N_POP,
                seed=0,
                verbose=False,
            )

    def test_preselection_strategy_opt_in_end_to_end(self):
        result = minimize(
            lambda x: np.sum(x**2),
            dim=DIM,
            lb=[-5.0] * DIM,
            ub=[5.0] * DIM,
            algorithm=NevergradAlgorithm(
                _make_optimizer(dim=DIM), allow_partial_tell=True
            ),
            surrogate="rbf",
            strategy=PreSelectionStrategy(n_candidates=N_POP, n_select=N_POP - 3),
            max_fe=100,
            pop_size=N_POP,
            seed=0,
            verbose=False,
        )
        assert result.fe > 0
        assert np.isfinite(result.f).all()


# ---------------------------------------------------------------------------
# Freshness, budget, no_parallelization guards
# ---------------------------------------------------------------------------


class TestNevergradAlgorithmGuards:
    def test_already_used_optimizer_rejected(self):
        problem = _make_problem()
        optimizer = _make_optimizer()
        # Pre-use the optimizer before wrapping it.
        c = optimizer.ask()
        optimizer.tell(c, 1.0)
        algo = NevergradAlgorithm(optimizer)
        ctx = _make_ctx(algo, problem)
        with pytest.raises(ConfigurationError):
            algo.ask(ctx, _DummyProvider())

    def test_budget_exhaustion_rejected(self):
        problem = _make_problem()
        algo = NevergradAlgorithm(_make_optimizer(budget=5))
        ctx = _make_ctx(algo, problem)
        with pytest.raises(ConfigurationError):
            algo.ask(ctx, _DummyProvider(), n_offspring=6)

    def test_budget_respecting_asks_succeed(self):
        problem = _make_problem()
        algo = NevergradAlgorithm(_make_optimizer(budget=10))
        ctx = _make_ctx(algo, problem)
        cand = algo.ask(ctx, _DummyProvider(), n_offspring=5)
        cand.update_array("f", np.array([problem.func(x) for x in cand.x]))
        algo.tell(ctx, _DummyProvider(), cand)
        cand = algo.ask(ctx, _DummyProvider(), n_offspring=5)  # exactly at budget
        assert len(cand) == 5

    def test_no_parallelization_optimizer_rejects_batched_ask(self):
        problem = _make_problem()
        optimizer = _make_optimizer()
        optimizer.no_parallelization = True  # simulate a sequential-only family
        algo = NevergradAlgorithm(optimizer)
        ctx = _make_ctx(algo, problem)
        with pytest.raises(ConfigurationError):
            algo.ask(ctx, _DummyProvider(), n_offspring=2)

    def test_no_parallelization_optimizer_allows_single_ask(self):
        problem = _make_problem()
        optimizer = _make_optimizer()
        optimizer.no_parallelization = True
        algo = NevergradAlgorithm(optimizer)
        ctx = _make_ctx(algo, problem)
        cand = algo.ask(ctx, _DummyProvider(), n_offspring=1)
        assert len(cand) == 1


# ---------------------------------------------------------------------------
# RNG seeding
# ---------------------------------------------------------------------------


class TestNevergradAlgorithmRng:
    @pytest.mark.parametrize("family", FAMILIES)
    def test_two_runs_from_the_same_rng_seed_reproduce(self, family):
        problem_a = _make_problem()
        problem_b = _make_problem()
        algo_a = NevergradAlgorithm(_make_optimizer(family))
        algo_b = NevergradAlgorithm(_make_optimizer(family))
        ctx_a = _make_ctx(algo_a, problem_a, rng_seed=7)
        ctx_b = _make_ctx(algo_b, problem_b, rng_seed=7)

        for _ in range(5):
            cand_a = algo_a.ask(ctx_a, _DummyProvider())
            cand_b = algo_b.ask(ctx_b, _DummyProvider())
            np.testing.assert_allclose(cand_a.x, cand_b.x)

            cand_a.update_array("f", np.array([problem_a.func(x) for x in cand_a.x]))
            cand_b.update_array("f", np.array([problem_b.func(x) for x in cand_b.x]))
            algo_a.tell(ctx_a, _DummyProvider(), cand_a)
            algo_b.tell(ctx_b, _DummyProvider(), cand_b)

    def test_rng_is_seeded_only_once_not_per_ask(self):
        """Reseeding on every ask() (copying DEAP's per-call pattern) would
        consume a fresh ctx.rng draw, and reset the optimizer's random walk,
        on every single call. Confirm neither happens after the first
        ask(): ctx.rng is left untouched by this adapter from the second
        ask() onward, and the optimizer keeps evolving its own random_state
        naturally (consecutive asks differ, they are not repeatedly reset
        back to the same seeded state)."""
        problem = _make_problem()
        algo = NevergradAlgorithm(_make_optimizer())
        ctx = _make_ctx(algo, problem, rng_seed=3)

        cand_1 = algo.ask(ctx, _DummyProvider())
        assert algo._rng_seeded is True
        state_after_first_ask = ctx.rng.bit_generator.state

        cand_2 = algo.ask(ctx, _DummyProvider())
        # ctx.rng consumed exactly one draw total, on the first ask() only;
        # a second call must not consume any further draws from it.
        assert ctx.rng.bit_generator.state == state_after_first_ask
        # The optimizer's own random_state nonetheless keeps evolving
        # naturally between calls (not reset back to the same seeded draw).
        assert not np.allclose(cand_1.x, cand_2.x)


# ---------------------------------------------------------------------------
# End-to-end, across the three tested families
# ---------------------------------------------------------------------------


class TestNevergradAlgorithmEndToEnd:
    @pytest.mark.parametrize("family", FAMILIES)
    def test_ask_tell_improves_sphere(self, family):
        problem = _make_problem(dim=DIM)
        algo = NevergradAlgorithm(_make_optimizer(family, dim=DIM))
        ctx = _make_ctx(algo, problem, n_pop=20)

        best = np.inf
        for _ in range(30):
            cand = algo.ask(ctx, _DummyProvider(), n_offspring=20)
            f = np.array([problem.func(x) for x in cand.x])
            cand.update_array("f", f)
            algo.tell(ctx, _DummyProvider(), cand)
            best = min(best, float(f.min()))

        assert best < 5.0

    def test_single_objective_direct_strategy_end_to_end(self):
        result = minimize(
            lambda x: np.sum(x**2),
            dim=DIM,
            lb=[-5.0] * DIM,
            ub=[5.0] * DIM,
            algorithm=NevergradAlgorithm(_make_optimizer(dim=DIM)),
            surrogate="rbf",
            strategy=DirectStrategy(),
            max_fe=200,
            pop_size=N_POP,
            seed=0,
            verbose=False,
        )
        assert result.fe > 0
        assert np.isfinite(result.f).all()
