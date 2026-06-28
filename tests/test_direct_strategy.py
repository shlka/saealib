"""Tests for DirectStrategy."""

from __future__ import annotations

import numpy as np

from saealib import (
    GA,
    CrossoverBLXAlpha,
    DirectStrategy,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    SequentialSelection,
    Termination,
    TruncationSelection,
    max_gen,
)
from saealib.callback import CallbackManager
from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.execution.evaluator import SerialEvaluator
from saealib.pipeline import Pipeline
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.strategies.base import OptimizationStrategy

DIM = 4
N_OBJ = 1
N_POP = 8

_ATTRS = [
    PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
    PopulationAttribute(name="f", dtype=np.float64, shape=(N_OBJ,)),
    PopulationAttribute(name="g", dtype=np.float64, shape=(0,)),
    PopulationAttribute(name="cv", dtype=np.float64, shape=()),
]


def _make_problem() -> Problem:
    return Problem(
        func=lambda x: np.array([np.sum(x**2)]),
        dim=DIM,
        n_obj=N_OBJ,
        direction=np.array([-1.0]),
        lb=[-5.0] * DIM,
        ub=[5.0] * DIM,
        comparator=SingleObjectiveComparator(),
    )


def _make_ctx(rng_seed: int = 0) -> OptimizationState:
    problem = _make_problem()
    rng = np.random.default_rng(rng_seed)
    pop = Population(_ATTRS, init_capacity=N_POP + 5)
    xs = rng.uniform(-3.0, 3.0, size=(N_POP, DIM))
    fs = np.array([[np.sum(x**2)] for x in xs])
    pop.extend({"x": xs, "f": fs, "g": np.zeros((N_POP, 0)), "cv": np.zeros(N_POP)})
    arc = Archive(_ATTRS, init_capacity=N_POP + 5)
    arc.extend({"x": xs, "f": fs, "g": np.zeros((N_POP, 0)), "cv": np.zeros(N_POP)})
    pareto_arc = ParetoArchive(
        _ATTRS, init_capacity=N_POP + 5, direction=np.array([-1.0])
    )
    return OptimizationState(
        problem=problem,
        population=pop,
        archive=arc,
        pareto_archive=pareto_arc,
        rng=np.random.default_rng(rng_seed + 1),
    )


def _make_ga() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(crossover_rate=0.9, alpha=0.4),
        mutation=MutationUniform(mutation_rate=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


class _MockProvider:
    seed: int | None = None

    def __init__(self):
        self.algorithm = _make_ga()
        self.evaluator = SerialEvaluator()
        self.cbmanager = CallbackManager()

    def dispatch(self, event):
        self.cbmanager.dispatch(event)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestDirectStrategy:
    def test_requires_surrogate_is_false(self):
        assert DirectStrategy.requires_surrogate is False

    def test_is_optimization_strategy_subclass(self):
        assert isinstance(DirectStrategy(), OptimizationStrategy)

    def test_pipeline_none_before_step(self):
        assert DirectStrategy().pipeline is None

    def test_pipeline_is_pipeline_after_step(self):
        strategy = DirectStrategy()
        ctx = _make_ctx()
        provider = _MockProvider()
        strategy.step(ctx, provider)
        assert isinstance(strategy.pipeline, Pipeline)

    def test_pipeline_reused_across_steps(self):
        strategy = DirectStrategy()
        ctx = _make_ctx()
        provider = _MockProvider()
        strategy.step(ctx, provider)
        first = strategy.pipeline
        strategy.step(ctx, provider)
        assert strategy.pipeline is first

    def test_generation_incremented(self):
        strategy = DirectStrategy()
        ctx = _make_ctx()
        provider = _MockProvider()
        ctx = strategy.step(ctx, provider)
        assert ctx.gen == 1

    def test_fe_equals_offspring_count(self):
        strategy = DirectStrategy()
        ctx = _make_ctx()
        provider = _MockProvider()
        ctx = strategy.step(ctx, provider)
        assert ctx.fe == N_POP

    def test_archive_grows_by_offspring_count(self):
        strategy = DirectStrategy()
        ctx = _make_ctx()
        provider = _MockProvider()
        before = len(ctx.archive)
        ctx = strategy.step(ctx, provider)
        assert len(ctx.archive) == before + N_POP


# ---------------------------------------------------------------------------
# Integration: minimize without surrogate manager
# ---------------------------------------------------------------------------


def test_minimize_without_surrogate_manager():
    problem = _make_problem()
    ctx = (
        Optimizer(problem)
        .set_initializer(LHSInitializer(n_init_archive=10, n_init_population=8, seed=0))
        .set_algorithm(_make_ga())
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_gen(5)))
        .run()
    )
    assert ctx is not None
    assert ctx.gen == 5
    assert ctx.fe > 0


def test_importable_from_top_level():
    import saealib

    assert saealib.DirectStrategy is DirectStrategy


def test_importable_from_strategies():
    import saealib.strategies as m

    assert m.DirectStrategy is DirectStrategy
