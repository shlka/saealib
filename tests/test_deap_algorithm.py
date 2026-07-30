"""Tests for DeapGenerateUpdateAlgorithm against the real DEAP library."""

from __future__ import annotations

import numpy as np
import pytest
from deap import cma

from saealib import minimize
from saealib.algorithms.deap_algorithm import DeapGenerateUpdateAlgorithm
from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.exceptions import ConfigurationError
from saealib.operators._deap_rng import seeded_global_numpy_random
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.problem.constraint import InequalityConstraint
from saealib.strategies.direct import DirectStrategy
from saealib.strategies.ps import PreSelectionStrategy

DIM = 5
N_POP = 10


class _DummyProvider:
    def dispatch(self, event):
        pass


def _make_problem(dim: int = DIM, n_obj: int = 1, constrained: bool = False) -> Problem:
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
    algo: DeapGenerateUpdateAlgorithm,
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


def _make_strategy(dim: int = DIM, lambda_: int = N_POP) -> cma.Strategy:
    return cma.Strategy(centroid=[0.0] * dim, sigma=1.0, lambda_=lambda_)


# ---------------------------------------------------------------------------
# ask / tell round trip (whitebox)
# ---------------------------------------------------------------------------


class TestDeapGenerateUpdateAlgorithmAskTell:
    def test_ask_produces_lambda_candidates(self):
        problem = _make_problem()
        algo = DeapGenerateUpdateAlgorithm(_make_strategy())
        ctx = _make_ctx(algo, problem)
        cand = algo.ask(ctx, _DummyProvider())
        assert len(cand) == N_POP
        assert np.all(cand.x >= problem.lb) and np.all(cand.x <= problem.ub)
        np.testing.assert_array_equal(
            np.sort(cand.get_array("deap_idx")), np.arange(N_POP)
        )

    def test_n_offspring_is_ignored(self):
        problem = _make_problem()
        algo = DeapGenerateUpdateAlgorithm(_make_strategy())
        ctx = _make_ctx(algo, problem)
        cand = algo.ask(ctx, _DummyProvider(), n_offspring=N_POP * 3)
        assert len(cand) == N_POP  # fixed by strategy.lambda_, not n_offspring

    def test_ask_tell_round_trip_updates_population(self):
        problem = _make_problem()
        algo = DeapGenerateUpdateAlgorithm(_make_strategy())
        ctx = _make_ctx(algo, problem)
        cand = algo.ask(ctx, _DummyProvider())
        cand.update_array("f", np.array([problem.func(x) for x in cand.x]))
        algo.tell(ctx, _DummyProvider(), cand)
        assert len(ctx.population) == N_POP
        assert np.isfinite(ctx.population.get_array("f")).all()

    def test_tell_with_reordered_offspring_still_aligns(self):
        """Offspring reordered before tell() (as SortByScoreStage would do).

        The deap_idx column must recover the correct X<->F mapping so that
        strategy.update() attaches each fitness value to the individual it
        actually belongs to, not a mismatched one.
        """
        problem = _make_problem()
        algo = DeapGenerateUpdateAlgorithm(_make_strategy())
        ctx = _make_ctx(algo, problem)
        cand = algo.ask(ctx, _DummyProvider())

        f_correct = np.array([problem.func(x) for x in cand.x])
        cand.update_array("f", f_correct)

        perm = np.random.default_rng(1).permutation(N_POP)
        reordered = cand.extract(perm)
        assert not np.array_equal(
            reordered.get_array("deap_idx"), cand.get_array("deap_idx")
        )

        algo.tell(ctx, _DummyProvider(), reordered)

        for x, f in zip(ctx.population.x, ctx.population.f, strict=True):
            np.testing.assert_allclose(f, problem.func(x), atol=1e-8)

    def test_constrained_problem_rejected(self):
        problem = _make_problem(constrained=True)
        algo = DeapGenerateUpdateAlgorithm(_make_strategy())
        with pytest.raises(ConfigurationError):
            algo.get_required_attrs(problem)

    def test_multi_objective_problem_rejected(self):
        problem = _make_problem(n_obj=2)
        algo = DeapGenerateUpdateAlgorithm(_make_strategy())
        with pytest.raises(ConfigurationError):
            algo.get_required_attrs(problem)

    def test_partial_tell_default_raises(self):
        problem = _make_problem()
        algo = DeapGenerateUpdateAlgorithm(_make_strategy())
        ctx = _make_ctx(algo, problem)
        cand = algo.ask(ctx, _DummyProvider())
        cand.update_array("f", np.array([problem.func(x) for x in cand.x]))
        truncated = cand.extract(np.arange(N_POP - 2))
        with pytest.raises(ConfigurationError):
            algo.tell(ctx, _DummyProvider(), truncated)

    def test_partial_tell_opt_in_runs_above_mu(self):
        """Above strategy.mu, update() completes (silently statistically weak)."""
        problem = _make_problem()
        strategy = _make_strategy()
        assert strategy.mu < N_POP - 2  # keep the subset above mu for this case
        algo = DeapGenerateUpdateAlgorithm(strategy, allow_partial_tell=True)
        ctx = _make_ctx(algo, problem)
        cand = algo.ask(ctx, _DummyProvider())
        cand.update_array("f", np.array([problem.func(x) for x in cand.x]))
        truncated = cand.extract(np.arange(N_POP - 2))
        algo.tell(ctx, _DummyProvider(), truncated)  # should not raise
        assert len(ctx.population) == N_POP - 2

    def test_partial_tell_opt_in_below_mu_raises_from_wrapped_strategy(self):
        """Below strategy.mu, opting in does not save you: cma.Strategy's own
        update() raises a bare shape-mismatch error, not a saealib
        ConfigurationError -- see the allow_partial_tell docstring caveat."""
        problem = _make_problem()
        strategy = _make_strategy()
        assert strategy.mu > 2  # keep the subset below mu for this case
        algo = DeapGenerateUpdateAlgorithm(strategy, allow_partial_tell=True)
        ctx = _make_ctx(algo, problem)
        cand = algo.ask(ctx, _DummyProvider())
        cand.update_array("f", np.array([problem.func(x) for x in cand.x]))
        truncated = cand.extract(np.arange(2))
        with pytest.raises(Exception):  # bare DEAP-internal error, not saealib's own
            algo.tell(ctx, _DummyProvider(), truncated)

    def test_ask_repairs_out_of_bounds_samples(self):
        """A large sigma pushes samples outside [lb, ub]; ask() must clip them."""
        problem = _make_problem()
        strategy = cma.Strategy(centroid=[0.0] * DIM, sigma=50.0, lambda_=N_POP)
        algo = DeapGenerateUpdateAlgorithm(strategy)
        ctx = _make_ctx(algo, problem)
        cand = algo.ask(ctx, _DummyProvider())
        assert np.all(cand.x >= problem.lb) and np.all(cand.x <= problem.ub)


# ---------------------------------------------------------------------------
# numpy global RNG bridging
# ---------------------------------------------------------------------------


def _numpy_states_equal(a: tuple, b: tuple) -> bool:
    """Compare two ``numpy.random.get_state()`` tuples (array-valued element)."""
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and tuple(a[2:]) == tuple(b[2:])


class TestSeededGlobalNumpyRandom:
    def test_rng_state_restored_after_normal_call(self):
        # Deliberate legacy numpy global RNG usage: exercising the very
        # state this context manager snapshots/restores.
        np.random.seed(1234)  # noqa: NPY002
        pre_state = np.random.get_state()  # noqa: NPY002
        rng = np.random.default_rng(11)
        with seeded_global_numpy_random(rng):
            np.random.standard_normal(5)  # noqa: NPY002
        assert _numpy_states_equal(np.random.get_state(), pre_state)  # noqa: NPY002

    def test_rng_state_restored_after_exception(self):
        np.random.seed(5678)  # noqa: NPY002
        pre_state = np.random.get_state()  # noqa: NPY002
        rng = np.random.default_rng(12)
        with (
            pytest.raises(RuntimeError, match="boom"),
            seeded_global_numpy_random(rng),
        ):
            np.random.standard_normal(5)  # noqa: NPY002
            raise RuntimeError("boom")
        assert _numpy_states_equal(np.random.get_state(), pre_state)  # noqa: NPY002

    def test_generate_actually_uses_the_seeded_state(self):
        """Regression guard: two calls seeded from the same rng draw agree."""
        strategy_a = _make_strategy()
        strategy_b = _make_strategy()

        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)

        with seeded_global_numpy_random(rng_a):
            pop_a = strategy_a.generate(lambda row: np.asarray(row))
        with seeded_global_numpy_random(rng_b):
            pop_b = strategy_b.generate(lambda row: np.asarray(row))

        np.testing.assert_allclose(np.stack(pop_a), np.stack(pop_b))


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


class TestDeapGenerateUpdateAlgorithmEndToEnd:
    def test_cma_strategy_ask_tell_improves_sphere(self):
        """Whitebox generation loop: repeated ask/tell should drive the
        centroid toward the sphere's optimum at the origin."""
        problem = _make_problem(dim=DIM)
        algo = DeapGenerateUpdateAlgorithm(_make_strategy(dim=DIM, lambda_=20))
        ctx = _make_ctx(algo, problem, n_pop=20)

        best = np.inf
        for _ in range(30):
            cand = algo.ask(ctx, _DummyProvider())
            f = np.array([problem.func(x) for x in cand.x])
            cand.update_array("f", f)
            algo.tell(ctx, _DummyProvider(), cand)
            best = min(best, float(f.min()))

        assert best < 1.0

    def test_lambda_mismatched_with_initial_pop_size_resizes_cleanly(self):
        """strategy.lambda_ != Initializer's n_pop: ctx.population's size
        tracks lambda_ from the first tell() onward. Repeated clear()/extend()
        across many generations must not leave stale rows or corrupt data,
        even though the population starts smaller than it ends up (capacity
        must grow) -- see the "No population mirroring" docstring note."""
        problem = _make_problem(dim=DIM)
        algo = DeapGenerateUpdateAlgorithm(_make_strategy(dim=DIM, lambda_=20))
        ctx = _make_ctx(algo, problem, n_pop=10)  # smaller than lambda_=20

        for _ in range(10):
            cand = algo.ask(ctx, _DummyProvider())
            f = np.array([problem.func(x) for x in cand.x])
            cand.update_array("f", f)
            algo.tell(ctx, _DummyProvider(), cand)
            assert len(ctx.population) == 20
            assert np.isfinite(ctx.population.get_array("f")).all()
            assert np.isfinite(ctx.population.get_array("x")).all()

    def test_single_objective_direct_strategy_end_to_end(self):
        result = minimize(
            lambda x: np.sum(x**2),
            dim=DIM,
            lb=[-5.0] * DIM,
            ub=[5.0] * DIM,
            algorithm=DeapGenerateUpdateAlgorithm(_make_strategy(dim=DIM)),
            surrogate="rbf",
            strategy=DirectStrategy(),
            max_fe=200,
            pop_size=N_POP,
            seed=0,
            verbose=False,
        )
        assert result.fe > 0
        assert np.isfinite(result.f).all()

    def test_preselection_partial_tell_raises_configuration_error(self):
        with pytest.raises(ConfigurationError):
            minimize(
                lambda x: np.sum(x**2),
                dim=DIM,
                lb=[-5.0] * DIM,
                ub=[5.0] * DIM,
                algorithm=DeapGenerateUpdateAlgorithm(_make_strategy(dim=DIM)),
                surrogate="rbf",
                strategy=PreSelectionStrategy(n_candidates=N_POP, n_select=N_POP - 3),
                max_fe=100,
                pop_size=N_POP,
                seed=0,
                verbose=False,
            )
