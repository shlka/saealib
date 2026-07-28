"""Tests for PymooAlgorithm against the real pymoo library."""

from __future__ import annotations

import numpy as np
import pytest
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA as PymooGA  # noqa: N811

from saealib import minimize
from saealib.algorithms.pymoo_algorithm import PymooAlgorithm
from saealib.comparators import NSGA2Comparator, SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.exceptions import ConfigurationError
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


def _make_problem(
    n_obj: int = 1, direction: float = -1.0, constrained: bool = False
) -> Problem:
    if n_obj == 1:
        comparator = SingleObjectiveComparator(direction=direction)

        def func(x: np.ndarray) -> np.ndarray:
            return np.array([np.sum(x**2)])
    else:
        comparator = NSGA2Comparator(direction=np.full(n_obj, direction))

        def func(x: np.ndarray) -> np.ndarray:
            f1 = x[0]
            g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
            f2 = g * (1.0 - np.sqrt(max(f1 / g, 0.0)))
            return np.array([f1, f2])

    constraints = (
        [InequalityConstraint(lambda x: x[0] - 3.0, threshold=0.0)]
        if constrained
        else None
    )

    return Problem(
        func=func,
        dim=DIM,
        n_obj=n_obj,
        direction=np.full(n_obj, direction),
        lb=[0.0] * DIM if n_obj > 1 else [-5.0] * DIM,
        ub=[1.0] * DIM if n_obj > 1 else [5.0] * DIM,
        comparator=comparator,
        constraints=constraints,
    )


def _make_ctx(
    algo: PymooAlgorithm,
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


# ---------------------------------------------------------------------------
# ask / tell round trip (whitebox)
# ---------------------------------------------------------------------------


class TestPymooAlgorithmAskTell:
    def test_ask_produces_popsize_candidates(self):
        problem = _make_problem()
        algo = PymooAlgorithm(PymooGA(pop_size=N_POP))
        ctx = _make_ctx(algo, problem)
        cand = algo.ask(ctx, _DummyProvider())
        assert len(cand) == N_POP
        assert np.all(cand.x >= problem.lb) and np.all(cand.x <= problem.ub)
        np.testing.assert_array_equal(
            np.sort(cand.get_array("pymoo_idx")), np.arange(N_POP)
        )

    def test_ask_tell_round_trip_updates_population(self):
        problem = _make_problem()
        algo = PymooAlgorithm(PymooGA(pop_size=N_POP))
        ctx = _make_ctx(algo, problem)
        cand = algo.ask(ctx, _DummyProvider())
        cand.update_array("f", np.array([problem.func(x) for x in cand.x]))
        algo.tell(ctx, _DummyProvider(), cand)
        assert len(ctx.population) == N_POP
        assert np.isfinite(ctx.population.get_array("f")).all()

    def test_tell_with_reordered_offspring_still_aligns(self):
        """Simulates SortByScoreStage: offspring reordered before tell().

        The pymoo_idx column must recover the correct X<->F mapping even
        after a permutation, otherwise pymoo's internal survival compares
        the wrong F against the wrong X.
        """
        problem = _make_problem()
        algo = PymooAlgorithm(PymooGA(pop_size=N_POP))
        ctx = _make_ctx(algo, problem)
        cand = algo.ask(ctx, _DummyProvider())

        f_correct = np.array([problem.func(x) for x in cand.x])
        cand.update_array("f", f_correct)

        perm = np.random.default_rng(1).permutation(N_POP)
        reordered = cand.extract(perm)
        # sanity: the permutation actually changed the order
        assert not np.array_equal(
            reordered.get_array("pymoo_idx"), cand.get_array("pymoo_idx")
        )

        algo.tell(ctx, _DummyProvider(), reordered)

        # every surviving x must match its own true f, not a mismatched one
        for x, f in zip(ctx.population.x, ctx.population.f, strict=True):
            np.testing.assert_allclose(f, problem.func(x), atol=1e-8)

    def test_maximize_sign_conversion(self):
        """F must be negated for saealib direction=+1 (maximize) and passed
        through unmodified for direction=-1 (minimize)."""
        for direction, expect_negated in [(-1.0, False), (1.0, True)]:
            problem = _make_problem(direction=direction)
            algo = PymooAlgorithm(PymooGA(pop_size=N_POP))
            ctx = _make_ctx(algo, problem)
            cand = algo.ask(ctx, _DummyProvider())
            f = np.array([problem.func(x) for x in cand.x])
            cand.update_array("f", f)
            algo.tell(ctx, _DummyProvider(), cand)

            alg_pop = algo.pymoo_algorithm.pop
            assert alg_pop is not None
            pymoo_f = np.asarray(alg_pop.get("F"), dtype=float)
            saealib_f = ctx.population.get_array("f")
            # pymoo always stores minimization-form F; saealib_f is the
            # direction-aware value, so they should differ in sign iff
            # direction == +1.
            if expect_negated:
                np.testing.assert_allclose(pymoo_f, -saealib_f, atol=1e-8)
            else:
                np.testing.assert_allclose(pymoo_f, saealib_f, atol=1e-8)

    def test_multi_objective_ask_tell(self):
        problem = _make_problem(n_obj=2)
        algo = PymooAlgorithm(NSGA2(pop_size=20))
        ctx = _make_ctx(algo, problem, n_pop=20)
        cand = algo.ask(ctx, _DummyProvider())
        assert len(cand) == 20
        f = np.array([problem.func(x) for x in cand.x])
        cand.update_array("f", f)
        algo.tell(ctx, _DummyProvider(), cand)
        assert len(ctx.population) == 20
        assert ctx.population.get_array("f").shape == (20, 2)

    def test_constrained_g_round_trips(self):
        problem = _make_problem(constrained=True)
        algo = PymooAlgorithm(PymooGA(pop_size=N_POP))
        ctx = _make_ctx(algo, problem)
        cand = algo.ask(ctx, _DummyProvider())
        g = np.array([problem.evaluate_constraints(x)[0] for x in cand.x])
        f = np.array([problem.func(x) for x in cand.x])
        cand.update_array("f", f)
        cand.update_array("g", g)
        algo.tell(ctx, _DummyProvider(), cand)
        assert ctx.population.get_array("g").shape == (N_POP, 1)
        assert np.isfinite(ctx.population.get_array("cv")).all()

    def test_popsize_mismatch_raises_configuration_error(self):
        problem = _make_problem()
        algo = PymooAlgorithm(PymooGA(pop_size=N_POP + 3))
        ctx = _make_ctx(algo, problem, n_pop=N_POP)
        with pytest.raises(ConfigurationError):
            algo.ask(ctx, _DummyProvider())

    def test_partial_tell_default_raises(self):
        problem = _make_problem()
        algo = PymooAlgorithm(PymooGA(pop_size=N_POP))
        ctx = _make_ctx(algo, problem)
        cand = algo.ask(ctx, _DummyProvider())
        cand.update_array("f", np.array([problem.func(x) for x in cand.x]))
        truncated = cand.extract(np.arange(N_POP - 2))
        with pytest.raises(ConfigurationError):
            algo.tell(ctx, _DummyProvider(), truncated)

    def test_partial_tell_opt_in_runs(self):
        problem = _make_problem()
        algo = PymooAlgorithm(PymooGA(pop_size=N_POP), allow_partial_tell=True)
        ctx = _make_ctx(algo, problem)
        cand = algo.ask(ctx, _DummyProvider())
        cand.update_array("f", np.array([problem.func(x) for x in cand.x]))
        truncated = cand.extract(np.arange(N_POP - 2))
        algo.tell(ctx, _DummyProvider(), truncated)  # should not raise
        assert len(ctx.population) == N_POP


# ---------------------------------------------------------------------------
# End-to-end via minimize()
# ---------------------------------------------------------------------------


class TestPymooAlgorithmEndToEnd:
    def test_single_objective_ga_direct_strategy(self):
        result = minimize(
            lambda x: np.sum(x**2),
            dim=DIM,
            lb=[-5.0] * DIM,
            ub=[5.0] * DIM,
            algorithm=PymooAlgorithm(PymooGA(pop_size=N_POP)),
            surrogate="rbf",
            strategy=DirectStrategy(),
            max_fe=100,
            pop_size=N_POP,
            seed=0,
            verbose=False,
        )
        assert result.fe > 0
        assert np.isfinite(result.f).all()

    def test_multi_objective_nsga2(self):
        def zdt1(x):
            f1 = x[0]
            g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
            f2 = g * (1.0 - np.sqrt(f1 / g))
            return np.array([f1, f2])

        result = minimize(
            zdt1,
            dim=10,
            lb=[0.0] * 10,
            ub=[1.0] * 10,
            n_obj=2,
            algorithm=PymooAlgorithm(NSGA2(pop_size=20)),
            surrogate="rbf",
            strategy=DirectStrategy(),
            max_fe=200,
            pop_size=20,
            seed=0,
            verbose=False,
        )
        assert result.fe > 0
        assert result.x.shape[-1] == 10

    def test_preselection_partial_tell_raises_configuration_error(self):
        with pytest.raises(ConfigurationError):
            minimize(
                lambda x: np.sum(x**2),
                dim=DIM,
                lb=[-5.0] * DIM,
                ub=[5.0] * DIM,
                algorithm=PymooAlgorithm(PymooGA(pop_size=N_POP)),
                surrogate="rbf",
                strategy=PreSelectionStrategy(n_candidates=N_POP, n_select=N_POP - 3),
                max_fe=100,
                pop_size=N_POP,
                seed=0,
                verbose=False,
            )
