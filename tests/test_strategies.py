"""
Tests for optimization strategies (Issue #009).

Tests cover:
- GenerationBasedStrategy: generation count, fe count, archive growth
- PreSelectionStrategy: fe count equals n_select, archive growth, n_select cap
"""

import numpy as np
import pytest

from saealib import (
    GA,
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
)
from saealib.context import OptimizationContext
from saealib.population import Archive, Population, PopulationAttribute
from saealib.problem import Problem, SingleObjectiveComparator
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ps import PreSelectionStrategy
from saealib.surrogate.prediction import SurrogatePrediction

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DIM = 6
N_OBJ = 1
N_POP = 10

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
        weight=np.array([-1.0]),
        lb=[-5.0] * DIM,
        ub=[5.0] * DIM,
        comparator=SingleObjectiveComparator(),
    )


def _make_ctx(n_pop: int = N_POP, rng_seed: int = 0) -> OptimizationContext:
    problem = _make_problem()
    rng = np.random.default_rng(rng_seed)

    pop = Population(_ATTRS, init_capacity=n_pop + 5)
    xs = rng.uniform(-3.0, 3.0, size=(n_pop, DIM))
    fs = np.array([[np.sum(x**2)] for x in xs])
    pop.extend({"x": xs, "f": fs, "g": np.zeros((n_pop, 0)), "cv": np.zeros(n_pop)})

    arc = Archive(_ATTRS, init_capacity=n_pop + 5)
    arc.extend({"x": xs, "f": fs, "g": np.zeros((n_pop, 0)), "cv": np.zeros(n_pop)})

    return OptimizationContext(
        problem=problem,
        population=pop,
        archive=arc,
        rng=np.random.default_rng(rng_seed + 1),
    )


def _make_ga() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(crossover_rate=0.9, alpha=0.4),
        mutation=MutationUniform(mutation_rate=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


class _MockSurrogateManager:
    """Returns uniform scores and constant predictions."""

    def score_candidates(self, candidates_x, archive, reference, provider, ctx):
        n = len(candidates_x)
        scores = np.linspace(1.0, 0.0, n)
        predictions = [
            SurrogatePrediction(
                mean=np.array([[1.0]]),
                std=None,
                label=None,
                metadata={},
            )
            for _ in range(n)
        ]
        return scores, predictions


class _MockProvider:
    def __init__(self, algorithm, surrogate_manager):
        self.algorithm = algorithm
        self.surrogate_manager = surrogate_manager

    def dispatch(self, event):
        pass


# ---------------------------------------------------------------------------
# GenerationBasedStrategy
# ---------------------------------------------------------------------------


class TestGenerationBasedStrategy:
    def _setup(self, gen_ctrl: int = 3):
        ctx = _make_ctx()
        provider = _MockProvider(_make_ga(), _MockSurrogateManager())
        strategy = GenerationBasedStrategy(gen_ctrl=gen_ctrl)
        return ctx, provider, strategy

    def test_gen_ctrl_stored(self):
        strategy = GenerationBasedStrategy(gen_ctrl=5)
        assert strategy.gen_ctrl == 5

    def test_generation_count_per_step(self):
        gen_ctrl = 3
        ctx, provider, strategy = self._setup(gen_ctrl=gen_ctrl)
        strategy.step(ctx, provider)
        assert ctx.gen == gen_ctrl + 1

    def test_fe_count_is_offspring_count(self):
        ctx, provider, strategy = self._setup(gen_ctrl=2)
        strategy.step(ctx, provider)
        # Only the final real-evaluation generation counts fe
        n_offspring = len(ctx.population)
        assert ctx.fe == n_offspring

    def test_archive_grows_by_offspring_count(self):
        ctx, provider, strategy = self._setup(gen_ctrl=2)
        before = len(ctx.archive)
        strategy.step(ctx, provider)
        n_offspring = len(ctx.population)
        assert len(ctx.archive) == before + n_offspring

    def test_surrogate_only_gens_do_not_increment_fe(self):
        gen_ctrl = 4
        ctx, provider, strategy = self._setup(gen_ctrl=gen_ctrl)
        strategy.step(ctx, provider)
        # fe should only reflect the single real-evaluation generation
        assert ctx.fe < len(ctx.population) * (gen_ctrl + 1)

    def test_gen_ctrl_zero_runs_one_real_generation(self):
        ctx, provider, strategy = self._setup(gen_ctrl=0)
        strategy.step(ctx, provider)
        assert ctx.gen == 1
        assert ctx.fe == len(ctx.population)


# ---------------------------------------------------------------------------
# PreSelectionStrategy
# ---------------------------------------------------------------------------


class TestPreSelectionStrategy:
    def _setup(self, n_candidates: int = 20, n_select: int = 5):
        ctx = _make_ctx()
        provider = _MockProvider(_make_ga(), _MockSurrogateManager())
        strategy = PreSelectionStrategy(n_candidates=n_candidates, n_select=n_select)
        return ctx, provider, strategy

    def test_parameters_stored(self):
        strategy = PreSelectionStrategy(n_candidates=20, n_select=5)
        assert strategy.n_candidates == 20
        assert strategy.n_select == 5

    def test_fe_equals_n_select(self):
        ctx, provider, strategy = self._setup(n_candidates=20, n_select=5)
        strategy.step(ctx, provider)
        assert ctx.fe == 5

    def test_archive_grows_by_n_select(self):
        ctx, provider, strategy = self._setup(n_candidates=20, n_select=5)
        before = len(ctx.archive)
        strategy.step(ctx, provider)
        assert len(ctx.archive) == before + 5

    def test_generation_count_incremented_once(self):
        ctx, provider, strategy = self._setup()
        strategy.step(ctx, provider)
        assert ctx.gen == 1

    def test_n_select_capped_by_n_candidates(self):
        # When n_select > n_candidates, evaluate all n_candidates
        n_candidates = 6
        ctx, provider, strategy = self._setup(
            n_candidates=n_candidates, n_select=n_candidates + 10
        )
        strategy.step(ctx, provider)
        assert ctx.fe == n_candidates

    def test_fe_not_equal_n_candidates(self):
        # Surrogate screening saves real evaluations: fe << n_candidates
        ctx, provider, strategy = self._setup(n_candidates=30, n_select=3)
        strategy.step(ctx, provider)
        assert ctx.fe == 3
        assert ctx.fe < 30
