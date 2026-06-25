"""
Tests for optimization strategies (Issue #009).

Tests cover:
- GenerationBasedStrategy: generation count, fe count, archive growth
- PreSelectionStrategy: fe count equals n_select, archive growth, n_select cap
"""

import operator

import numpy as np

from saealib import (
    GA,
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
)
from saealib.acquisition import MeanPrediction
from saealib.callback import CallbackManager
from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.execution.evaluator import SerialEvaluator
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.strategies.ps import PreSelectionStrategy
from saealib.surrogate.archive_manager import DensityManager, NoveltyManager
from saealib.surrogate.manager import (
    CompositeSurrogateManager,
    GlobalSurrogateManager,
    LocalSurrogateManager,
    rank_weighted_combine,
)
from saealib.surrogate.prediction import SurrogatePrediction
from saealib.surrogate.rbf import RBFSurrogate, gaussian_kernel
from saealib.termination import Termination, max_gen

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
        direction=np.array([-1.0]),
        lb=[-5.0] * DIM,
        ub=[5.0] * DIM,
        comparator=SingleObjectiveComparator(),
    )


def _make_ctx(n_pop: int = N_POP, rng_seed: int = 0) -> OptimizationState:
    problem = _make_problem()
    rng = np.random.default_rng(rng_seed)

    pop = Population(_ATTRS, init_capacity=n_pop + 5)
    xs = rng.uniform(-3.0, 3.0, size=(n_pop, DIM))
    fs = np.array([[np.sum(x**2)] for x in xs])
    pop.extend({"x": xs, "f": fs, "g": np.zeros((n_pop, 0)), "cv": np.zeros(n_pop)})

    arc = Archive(_ATTRS, init_capacity=n_pop + 5)
    arc.extend({"x": xs, "f": fs, "g": np.zeros((n_pop, 0)), "cv": np.zeros(n_pop)})

    pareto_arc = ParetoArchive(
        _ATTRS, init_capacity=n_pop + 5, direction=np.array([-1.0])
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


class _MockSurrogateManager:
    """Returns uniform scores and constant predictions."""

    def fit(self, archive, ctx=None):
        pass

    def score_candidates(self, candidates_x, archive, ctx=None, *, refit=True):
        n = len(candidates_x)
        scores = np.linspace(1.0, 0.0, n)
        predictions = [
            SurrogatePrediction(
                value=np.array([[1.0]]),
                std=None,
                label=None,
                metadata={},
            )
            for _ in range(n)
        ]
        return scores, predictions


class _MockProvider:
    seed: int | None = None
    strategy = IndividualBasedStrategy(evaluation_ratio=0.1)
    termination: Termination = Termination(max_gen(100_000))

    def __init__(self, algorithm, surrogate_manager):
        self.algorithm = algorithm
        self.surrogate_manager = surrogate_manager
        self.evaluator = SerialEvaluator()
        self.cbmanager = CallbackManager()

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
        ctx = strategy.step(ctx, provider)
        assert ctx.gen == gen_ctrl + 1

    def test_fe_count_is_offspring_count(self):
        ctx, provider, strategy = self._setup(gen_ctrl=2)
        ctx = strategy.step(ctx, provider)
        # Only the final real-evaluation generation counts fe
        n_offspring = len(ctx.population)
        assert ctx.fe == n_offspring

    def test_archive_grows_by_offspring_count(self):
        ctx, provider, strategy = self._setup(gen_ctrl=2)
        before = len(ctx.archive)
        ctx = strategy.step(ctx, provider)
        n_offspring = len(ctx.population)
        assert len(ctx.archive) == before + n_offspring

    def test_surrogate_only_gens_do_not_increment_fe(self):
        gen_ctrl = 4
        ctx, provider, strategy = self._setup(gen_ctrl=gen_ctrl)
        ctx = strategy.step(ctx, provider)
        # fe should only reflect the single real-evaluation generation
        assert ctx.fe < len(ctx.population) * (gen_ctrl + 1)

    def test_gen_ctrl_zero_runs_one_real_generation(self):
        ctx, provider, strategy = self._setup(gen_ctrl=0)
        ctx = strategy.step(ctx, provider)
        assert ctx.gen == 1
        assert ctx.fe == len(ctx.population)

    def test_pipeline_cached_after_first_step(self):
        ctx, provider, strategy = self._setup(gen_ctrl=3)
        ctx = strategy.step(ctx, provider)
        pipeline_ref = strategy.pipeline
        strategy.step(ctx, provider)
        assert strategy.pipeline is pipeline_ref

    def test_surrogate_fit_called_once_per_step(self):
        """fit() must be called exactly once per step(), not once per surrogate gen."""
        fit_count = [0]

        ctx = _make_ctx()
        surrogate = RBFSurrogate(gaussian_kernel, DIM).with_post_fit(
            lambda tx, ty, c: operator.setitem(fit_count, 0, fit_count[0] + 1)
        )
        manager = GlobalSurrogateManager(surrogate, MeanPrediction())
        provider = _MockProvider(_make_ga(), manager)
        strategy = GenerationBasedStrategy(gen_ctrl=3)

        strategy.step(ctx, provider)

        assert fit_count[0] == 1

    def test_local_surrogate_refit_false_ignored(self):
        """LocalSurrogateManager always fits per candidate; refit=False is a no-op."""
        fit_count = [0]

        ctx = _make_ctx()
        surrogate = RBFSurrogate(gaussian_kernel, DIM).with_post_fit(
            lambda tx, ty, c: operator.setitem(fit_count, 0, fit_count[0] + 1)
        )
        manager = LocalSurrogateManager(surrogate, MeanPrediction())
        provider = _MockProvider(_make_ga(), manager)
        strategy = GenerationBasedStrategy(gen_ctrl=1)

        strategy.step(ctx, provider)

        # LocalSurrogateManager always fits per candidate; refit=False is ignored
        n_offspring = len(ctx.population)
        assert fit_count[0] == n_offspring


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
        ctx = strategy.step(ctx, provider)
        assert ctx.fe == 5

    def test_archive_grows_by_n_select(self):
        ctx, provider, strategy = self._setup(n_candidates=20, n_select=5)
        before = len(ctx.archive)
        ctx = strategy.step(ctx, provider)
        assert len(ctx.archive) == before + 5

    def test_generation_count_incremented_once(self):
        ctx, provider, strategy = self._setup()
        ctx = strategy.step(ctx, provider)
        assert ctx.gen == 1

    def test_n_select_capped_by_n_candidates(self):
        # When n_select > n_candidates, evaluate all n_candidates
        n_candidates = 6
        ctx, provider, strategy = self._setup(
            n_candidates=n_candidates, n_select=n_candidates + 10
        )
        ctx = strategy.step(ctx, provider)
        assert ctx.fe == n_candidates

    def test_fe_not_equal_n_candidates(self):
        # Surrogate screening saves real evaluations: fe << n_candidates
        ctx, provider, strategy = self._setup(n_candidates=30, n_select=3)
        ctx = strategy.step(ctx, provider)
        assert ctx.fe == 3
        assert ctx.fe < 30


# ---------------------------------------------------------------------------
# IndividualBasedStrategy + NoveltyManager (via CompositeSurrogateManager)
# ---------------------------------------------------------------------------


def _make_ensemble_novelty() -> CompositeSurrogateManager:
    """Regression surrogate first (finite tell_f) + NoveltyManager second."""
    return CompositeSurrogateManager(
        [
            LocalSurrogateManager(RBFSurrogate(gaussian_kernel, DIM), MeanPrediction()),
            NoveltyManager(k=3),
        ],
        combine_fn=rank_weighted_combine(np.array([0.7, 0.3])),
    )


def _make_ensemble_density() -> CompositeSurrogateManager:
    """Regression surrogate first (finite tell_f) + DensityManager second."""
    return CompositeSurrogateManager(
        [
            LocalSurrogateManager(RBFSurrogate(gaussian_kernel, DIM), MeanPrediction()),
            DensityManager(eps=1.0),
        ],
        combine_fn=rank_weighted_combine(np.array([0.7, 0.3])),
    )


class TestIndividualBasedStrategyWithNoveltyManager:
    """End-to-end: IndividualBasedStrategy + regression + NoveltyManager ensemble."""

    def _setup(self, evaluation_ratio: float = 0.5):
        ctx = _make_ctx()
        provider = _MockProvider(_make_ga(), _make_ensemble_novelty())
        strategy = IndividualBasedStrategy(evaluation_ratio=evaluation_ratio)
        return ctx, provider, strategy

    def test_step_runs_without_error(self):
        ctx, provider, strategy = self._setup()
        strategy.step(ctx, provider)

    def test_fe_equals_evaluation_ratio_times_offspring(self):
        evaluation_ratio = 0.5
        ctx, provider, strategy = self._setup(evaluation_ratio=evaluation_ratio)
        ctx = strategy.step(ctx, provider)
        n_offspring = len(ctx.population)
        expected_fe = max(1, int(evaluation_ratio * n_offspring))
        assert ctx.fe == expected_fe

    def test_archive_grows_by_n_eval(self):
        evaluation_ratio = 0.5
        ctx, provider, strategy = self._setup(evaluation_ratio=evaluation_ratio)
        before = len(ctx.archive)
        ctx = strategy.step(ctx, provider)
        n_eval = max(1, int(evaluation_ratio * len(ctx.population)))
        assert len(ctx.archive) == before + n_eval

    def test_generation_count_incremented(self):
        ctx, provider, strategy = self._setup()
        ctx = strategy.step(ctx, provider)
        assert ctx.gen == 1


class TestPreSelectionStrategyWithDensityManager:
    """End-to-end: PreSelectionStrategy + regression + DensityManager ensemble."""

    def _setup(self, n_candidates: int = 20, n_select: int = 5):
        ctx = _make_ctx()
        provider = _MockProvider(_make_ga(), _make_ensemble_density())
        strategy = PreSelectionStrategy(n_candidates=n_candidates, n_select=n_select)
        return ctx, provider, strategy

    def test_step_runs_without_error(self):
        ctx, provider, strategy = self._setup()
        ctx = strategy.step(ctx, provider)

    def test_fe_equals_n_select(self):
        ctx, provider, strategy = self._setup(n_candidates=20, n_select=5)
        ctx = strategy.step(ctx, provider)
        assert ctx.fe == 5

    def test_archive_grows_after_step(self):
        """Archive grows by at most n_select (duplicates may be rejected)."""
        n_select = 5
        ctx, provider, strategy = self._setup(n_candidates=20, n_select=n_select)
        before = len(ctx.archive)
        ctx = strategy.step(ctx, provider)
        assert before < len(ctx.archive) <= before + n_select

    def test_generation_count_incremented(self):
        ctx, provider, strategy = self._setup()
        ctx = strategy.step(ctx, provider)
        assert ctx.gen == 1


# ---------------------------------------------------------------------------
# Strategy.pipeline caching
# ---------------------------------------------------------------------------


from saealib.pipeline import Pipeline  # noqa: E402


class TestStrategyPipelineAttribute:
    """Strategy.pipeline is None before first step, Pipeline after."""

    def _providers_and_strategies(self):
        ga = _make_ga()
        sm = _MockSurrogateManager()
        provider = _MockProvider(ga, sm)
        return provider, {
            "ps": PreSelectionStrategy(n_candidates=20, n_select=5),
            "ib": IndividualBasedStrategy(evaluation_ratio=0.5),
            "gb": GenerationBasedStrategy(gen_ctrl=2),
        }

    def test_ps_pipeline_none_before_step(self):
        _, strategies = self._providers_and_strategies()
        assert strategies["ps"].pipeline is None

    def test_ib_pipeline_none_before_step(self):
        _, strategies = self._providers_and_strategies()
        assert strategies["ib"].pipeline is None

    def test_gb_pipeline_none_before_step(self):
        _, strategies = self._providers_and_strategies()
        assert strategies["gb"].pipeline is None

    def test_ps_pipeline_is_pipeline_after_step(self):
        provider, strategies = self._providers_and_strategies()
        ctx = _make_ctx()
        strategies["ps"].step(ctx, provider)
        assert isinstance(strategies["ps"].pipeline, Pipeline)

    def test_ib_pipeline_is_pipeline_after_step(self):
        provider, strategies = self._providers_and_strategies()
        ctx = _make_ctx()
        strategies["ib"].step(ctx, provider)
        assert isinstance(strategies["ib"].pipeline, Pipeline)

    def test_gb_pipeline_is_pipeline_after_step(self):
        provider, strategies = self._providers_and_strategies()
        ctx = _make_ctx()
        strategies["gb"].step(ctx, provider)
        assert isinstance(strategies["gb"].pipeline, Pipeline)

    def test_ps_pipeline_reused_across_steps(self):
        provider, strategies = self._providers_and_strategies()
        ctx = _make_ctx()
        s = strategies["ps"]
        s.step(ctx, provider)
        first = s.pipeline
        s.step(ctx, provider)
        assert s.pipeline is first

    def test_ps_pipeline_rebuilds_when_reset(self):
        provider, strategies = self._providers_and_strategies()
        ctx = _make_ctx()
        s = strategies["ps"]
        s.step(ctx, provider)
        first = s.pipeline
        s.pipeline = None
        s.step(ctx, provider)
        assert s.pipeline is not first
