"""Tests for concrete Stage execute() logic in saealib.stages."""

import numpy as np

from saealib import (
    GA,
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
)
from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.execution.evaluator import SerialEvaluator
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.stages import (
    AskStage,
    SurrogateFitStage,
    SurrogateScoreStage,
    TrueEvaluationStage,
)
from saealib.surrogate.prediction import SurrogatePrediction

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DIM = 4
N_POP = 6
N_OBJ = 1

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


def _make_state() -> OptimizationState:
    problem = _make_problem()
    rng = np.random.default_rng(0)
    xs = rng.uniform(-3.0, 3.0, size=(N_POP, DIM))
    fs = np.array([[np.sum(x**2)] for x in xs])

    pop = Population(_ATTRS, init_capacity=N_POP + 5)
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
        rng=np.random.default_rng(1),
    )


def _make_ga() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


class _MockSurrogateManager:
    def fit(self, archive, ctx=None):
        pass

    def score_candidates(self, candidates_x, archive, ctx=None, *, refit=True):
        n = len(candidates_x)
        scores = np.linspace(1.0, 0.0, n)
        predictions = [
            SurrogatePrediction(
                value=np.array([[1.0]]), std=None, label=None, metadata={}
            )
            for _ in range(n)
        ]
        return scores, predictions


# ---------------------------------------------------------------------------
# AskStage — _DispatchProxy.dispatch when cbmanager is None
# ---------------------------------------------------------------------------


class TestAskStageNoCbmanager:
    """AskStage with cbmanager=None exercises _DispatchProxy.dispatch(None path)."""

    def test_execute_sets_offspring(self):
        state = _make_state()
        stage = AskStage(_make_ga(), cbmanager=None)
        new_state = stage.execute(state)
        assert new_state.offspring is not None

    def test_execute_offspring_count_matches_population(self):
        state = _make_state()
        stage = AskStage(_make_ga(), cbmanager=None)
        new_state = stage.execute(state)
        assert new_state.offspring is not None
        assert len(new_state.offspring) == len(state.population)


# ---------------------------------------------------------------------------
# SurrogateFitStage
# ---------------------------------------------------------------------------


class TestSurrogateFitStage:
    def test_execute_calls_fit(self):
        fit_called = [False]

        class _TrackedSM(_MockSurrogateManager):
            def fit(self, archive, ctx=None):
                fit_called[0] = True

        state = _make_state()
        SurrogateFitStage(_TrackedSM()).execute(state)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        assert fit_called[0]

    def test_execute_returns_same_state_object(self):
        state = _make_state()
        result = SurrogateFitStage(_MockSurrogateManager()).execute(state)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        assert result is state


# ---------------------------------------------------------------------------
# SurrogateScoreStage — cbmanager=None branches
# ---------------------------------------------------------------------------


class TestSurrogateScoreStageNoCbmanager:
    """cbmanager=None skips SurrogateStartEvent / SurrogateEndEvent dispatch."""

    def test_execute_sets_scores_and_predictions(self):
        state = _make_state()
        state = AskStage(_make_ga(), cbmanager=None).execute(state)

        stage = SurrogateScoreStage(_MockSurrogateManager(), cbmanager=None)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        new_state = stage.execute(state)
        assert new_state.scores is not None
        assert new_state.predictions is not None

    def test_scores_length_matches_offspring(self):
        state = _make_state()
        state = AskStage(_make_ga(), cbmanager=None).execute(state)
        assert state.offspring is not None
        n = len(state.offspring)

        stage = SurrogateScoreStage(_MockSurrogateManager(), cbmanager=None)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        new_state = stage.execute(state)
        assert new_state.scores is not None
        assert len(new_state.scores) == n


# ---------------------------------------------------------------------------
# TrueEvaluationStage — n_eval as int and cbmanager=None
# ---------------------------------------------------------------------------


class TestTrueEvaluationStage:
    def _state_with_offspring(self):
        state = _make_state()
        return AskStage(_make_ga(), cbmanager=None).execute(state)

    def test_n_eval_as_int_limits_evaluation(self):
        state = self._state_with_offspring()
        n_eval = 2
        stage = TrueEvaluationStage(SerialEvaluator(), n_eval=n_eval)
        new_state = stage.execute(state)
        assert new_state.fe == n_eval
        assert new_state.evaluated_offspring is not None
        assert len(new_state.evaluated_offspring) == n_eval

    def test_n_eval_as_int_without_cbmanager(self):
        state = self._state_with_offspring()
        stage = TrueEvaluationStage(SerialEvaluator(), cbmanager=None, n_eval=2)
        new_state = stage.execute(state)
        assert new_state.fe == 2

    def test_n_eval_as_int_capped_by_offspring_size(self):
        state = self._state_with_offspring()
        assert state.offspring is not None
        n_offspring = len(state.offspring)
        stage = TrueEvaluationStage(SerialEvaluator(), n_eval=n_offspring + 100)
        new_state = stage.execute(state)
        assert new_state.fe == n_offspring
