"""Tests for concrete Stage execute() logic in saealib.stages."""

import numpy as np

from saealib import (
    GA,
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
)
from saealib.callback import CallbackManager, PostEvaluationEvent
from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.execution.evaluator import EvaluationResult, Evaluator, SerialEvaluator
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


class _StubEvaluator(Evaluator):
    """Evaluator returning known, distinguishable f/g/cv values (no computation)."""

    def __init__(self, f: np.ndarray, g: np.ndarray, cv: np.ndarray) -> None:
        self._f = f
        self._g = g
        self._cv = cv

    def evaluate_batch(self, x, problem):
        n = len(x)
        return EvaluationResult(f=self._f[:n], g=self._g[:n], cv=self._cv[:n])


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

    def test_bulk_write_matches_evaluator_result(self):
        """f/g/cv written by the bulk assignment exactly match the evaluator output."""
        state = self._state_with_offspring()
        assert state.offspring is not None
        n_offspring = len(state.offspring)
        n_eval = n_offspring - 1

        f = np.arange(n_offspring, dtype=float).reshape(n_offspring, N_OBJ)
        g = np.zeros((n_offspring, 0))
        cv = np.arange(100.0, 100.0 + n_offspring)

        stage = TrueEvaluationStage(_StubEvaluator(f, g, cv), n_eval=n_eval)
        new_state = stage.execute(state)

        assert new_state.evaluated_offspring is not None
        np.testing.assert_array_equal(new_state.evaluated_offspring.f, f[:n_eval])
        np.testing.assert_array_equal(new_state.evaluated_offspring.g, g[:n_eval])
        np.testing.assert_array_equal(new_state.evaluated_offspring.cv, cv[:n_eval])

        assert new_state.offspring is not None
        np.testing.assert_array_equal(new_state.offspring.f[:n_eval], f[:n_eval])
        np.testing.assert_array_equal(new_state.offspring.g[:n_eval], g[:n_eval])
        np.testing.assert_array_equal(new_state.offspring.cv[:n_eval], cv[:n_eval])

    def test_candidates_beyond_n_are_untouched(self):
        """Only the first n candidates are written; the rest keep their prior values."""
        state = self._state_with_offspring()
        assert state.offspring is not None
        n_offspring = len(state.offspring)
        n_eval = n_offspring - 1

        before_f = np.array(state.offspring.f, copy=True)
        before_g = np.array(state.offspring.g, copy=True)
        before_cv = np.array(state.offspring.cv, copy=True)

        f = np.arange(n_offspring, dtype=float).reshape(n_offspring, N_OBJ)
        g = np.zeros((n_offspring, 0))
        cv = np.arange(100.0, 100.0 + n_offspring)

        stage = TrueEvaluationStage(_StubEvaluator(f, g, cv), n_eval=n_eval)
        new_state = stage.execute(state)

        assert new_state.offspring is not None
        np.testing.assert_array_equal(new_state.offspring.f[n_eval:], before_f[n_eval:])
        np.testing.assert_array_equal(new_state.offspring.g[n_eval:], before_g[n_eval:])
        np.testing.assert_array_equal(
            new_state.offspring.cv[n_eval:], before_cv[n_eval:]
        )

    def test_bulk_write_aligns_multi_column_arrays(self):
        """With n_obj > 1 / n_constraints > 1, columns must not be transposed/mixed."""
        n = 4
        attrs_mo = [
            PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
            PopulationAttribute(name="f", dtype=np.float64, shape=(2,)),
            PopulationAttribute(name="g", dtype=np.float64, shape=(3,)),
            PopulationAttribute(name="cv", dtype=np.float64, shape=()),
        ]
        pop = Population(attrs_mo, init_capacity=n + 2)
        pop.extend({"x": np.zeros((n, DIM))})  # f/g/cv fall back to the NaN default

        state = self._state_with_offspring()
        state = state.replace(offspring=pop)

        # _StubEvaluator ignores `problem` entirely, so the state's n_obj=1
        # problem (from _make_state) is harmless here.
        f = np.arange(n * 2, dtype=float).reshape(n, 2)
        g = np.arange(100.0, 100.0 + n * 3).reshape(n, 3)
        cv = np.arange(200.0, 200.0 + n)

        stage = TrueEvaluationStage(_StubEvaluator(f, g, cv))
        new_state = stage.execute(state)

        assert new_state.evaluated_offspring is not None
        np.testing.assert_array_equal(new_state.evaluated_offspring.f, f)
        np.testing.assert_array_equal(new_state.evaluated_offspring.g, g)
        np.testing.assert_array_equal(new_state.evaluated_offspring.cv, cv)

    def test_zero_constraints_does_not_crash_g_assignment(self):
        """g has shape (n, 0) when the problem defines no constraints."""
        state = self._state_with_offspring()
        assert state.offspring is not None
        n_offspring = len(state.offspring)

        f = np.zeros((n_offspring, N_OBJ))
        g = np.zeros((n_offspring, 0))
        cv = np.zeros(n_offspring)

        stage = TrueEvaluationStage(_StubEvaluator(f, g, cv))
        new_state = stage.execute(state)
        assert new_state.evaluated_offspring is not None
        assert new_state.evaluated_offspring.g.shape == (n_offspring, 0)

    def test_value_version_bumped_exactly_once(self):
        """A single mod_value() call still signals the population as changed."""
        state = self._state_with_offspring()
        assert state.offspring is not None
        v0 = state.offspring._value_version

        stage = TrueEvaluationStage(SerialEvaluator(), n_eval=2)
        new_state = stage.execute(state)

        assert new_state.offspring is not None
        assert new_state.offspring._value_version == v0 + 1

    def test_post_evaluation_event_receives_written_values(self):
        """PostEvaluationEvent.offspring carries the bulk-written f/g/cv values."""
        state = self._state_with_offspring()
        assert state.offspring is not None
        n_offspring = len(state.offspring)
        n_eval = n_offspring - 1

        f = np.arange(n_offspring, dtype=float).reshape(n_offspring, N_OBJ)
        g = np.zeros((n_offspring, 0))
        cv = np.arange(100.0, 100.0 + n_offspring)

        received = []
        cbmanager = CallbackManager()
        cbmanager.register(
            PostEvaluationEvent, lambda event: received.append(event.offspring)
        )

        stage = TrueEvaluationStage(
            _StubEvaluator(f, g, cv), cbmanager=cbmanager, n_eval=n_eval
        )
        stage.execute(state)

        assert len(received) == 1
        np.testing.assert_array_equal(received[0].f, f[:n_eval])
        np.testing.assert_array_equal(received[0].cv, cv[:n_eval])
