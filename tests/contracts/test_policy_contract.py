import numpy as np
import pytest

from saealib.acquisition import AcquisitionResult
from saealib.context import OptimizationState
from saealib.exceptions import ValidationError
from saealib.execution.evaluator import EvaluationResult
from saealib.policies import (
    ComparatorWorstFallback,
    EvaluateAll,
    MixedFeedback,
    NoFeedback,
    PredictedFeedback,
    TopKEvaluation,
    TrueOnlyFeedback,
    select_ratio,
    select_top_k,
)
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.strategies.direct import DirectStrategy
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.strategies.ps import PreSelectionStrategy
from saealib.surrogate import PredictionChannel, SurrogatePrediction


def _state() -> OptimizationState:
    problem = Problem(
        func=lambda x: np.array([x[0]]),
        dim=1,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0],
        ub=[1.0],
    )
    attrs = [
        PopulationAttribute("id", np.int64, (), default=-1),
        PopulationAttribute("x", np.float64, (1,)),
        PopulationAttribute("f", np.float64, (1,)),
        PopulationAttribute("g", np.float64, (0,)),
        PopulationAttribute("cv", np.float64, ()),
    ]
    pop = Population(attrs, 2)
    pop._extend_internal(
        {
            "id": np.array([10, 11], dtype=np.int64),
            "x": np.array([[0.1], [0.2]], dtype=np.float64),
            "f": np.full((2, 1), np.nan),
            "g": np.empty((2, 0), dtype=np.float64),
            "cv": np.zeros(2),
        },
        preserve_ids=True,
    )
    return OptimizationState(
        problem=problem,
        population=pop,
        archive=Archive(attrs, 2),
        pareto_archive=ParetoArchive(attrs, 2, direction=np.array([-1.0])),
        rng=np.random.default_rng(0),
        offspring=pop,
    )


def test_selection_kernels_are_stable_and_non_mutating():
    scores = np.array([2.0, 2.0, 1.0], dtype=np.float64)
    original = scores.copy()
    np.testing.assert_array_equal(select_top_k(scores, 2), [0, 1])
    np.testing.assert_array_equal(select_ratio(scores, 2 / 3), [0, 1])
    np.testing.assert_array_equal(scores, original)


def test_selection_boundaries_and_empty():
    scores = np.empty(0, dtype=np.float64)
    np.testing.assert_array_equal(select_top_k(scores, 0), np.empty(0, dtype=np.intp))
    np.testing.assert_array_equal(select_ratio(scores, 0.1), np.empty(0, dtype=np.intp))
    with pytest.raises(ValidationError):
        select_top_k(np.ones(2), 3)
    with pytest.raises(ValidationError):
        select_ratio(np.ones(2), 1.1)
    with pytest.raises(ValidationError):
        select_top_k(np.array([1.0, np.nan]), 1)
    assert TopKEvaluation(1, sanitize_nonfinite=True).plan(
        _state().offspring,
        AcquisitionResult(np.array([1.0, np.nan], dtype=np.float64)),
        _state(),
    ).candidate_ids.tolist() == [10]


def test_evaluation_policies_allocate_requests():
    state = _state()
    acquisition = AcquisitionResult(np.array([1.0, 0.0], dtype=np.float64))
    request = TopKEvaluation(1).plan(state.offspring, acquisition, state)
    assert request.request_id == 0
    np.testing.assert_array_equal(request.candidate_ids, [10])
    assert not request.x.flags.writeable
    request_all = EvaluateAll().plan(state.offspring, None, state)
    assert request_all.request_id == 1
    np.testing.assert_array_equal(request_all.candidate_ids, [10, 11])


def test_feedback_sources_and_objective_channel_requirement():
    state = _state()
    evaluation = EvaluationResult(
        np.array([[4.0]], dtype=np.float64),
        np.empty((1, 0), dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        candidate_ids=np.array([10], dtype=np.int64),
    )
    prediction = SurrogatePrediction(
        {"objective": PredictionChannel(np.array([[5.0], [6.0]]))}
    )
    mixed = MixedFeedback().build(
        state.offspring, prediction, evaluation, np.array([10]), state
    )
    np.testing.assert_array_equal(mixed.candidate_ids, [10, 11])
    np.testing.assert_array_equal(mixed.f, [[4.0], [6.0]])
    np.testing.assert_array_equal(mixed.source, [0, 1])
    true_only = TrueOnlyFeedback().build(
        state.offspring, prediction, evaluation, np.array([10]), state
    )
    np.testing.assert_array_equal(true_only.candidate_ids, [10])
    assert (
        len(NoFeedback().build(state.offspring, prediction, evaluation, [], state).f)
        == 0
    )
    with pytest.raises(ValidationError):
        PredictedFeedback().build(
            state.offspring,
            SurrogatePrediction({"win_rate": PredictionChannel(np.ones((2, 1)))}),
            None,
            [],
            state,
        )


def test_mixed_missing_rows_are_available_to_explicit_fallback():
    state = _state()
    state.population.update_rows(
        np.array([0, 1]), {"f": np.array([[1.0], [3.0]], dtype=np.float64)}
    )
    evaluation = EvaluationResult(
        np.array([[4.0]], dtype=np.float64),
        np.empty((1, 0), dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        candidate_ids=np.array([10], dtype=np.int64),
    )
    result = ComparatorWorstFallback().build(
        state.offspring,
        SurrogatePrediction({"win_rate": PredictionChannel(np.ones((2, 1)))}),
        evaluation,
        np.array([10]),
        state,
    )
    np.testing.assert_array_equal(result.f, [[4.0], [3.0]])
    np.testing.assert_array_equal(result.source, [0, 2])


def test_feedback_result_rejects_misaligned_and_object_arrays():
    with pytest.raises(ValidationError):
        from saealib.policies import FeedbackResult

        FeedbackResult(
            np.array([1, 2], dtype=np.int64),
            np.ones((2, 1), dtype=np.float64),
            None,
            None,
            np.ones(2, dtype=bool),
            np.ones(2, dtype=np.uint8),
            {"artifact": np.ones(1, dtype=np.float64)},
        )
    with pytest.raises(ValidationError):
        from saealib.policies import FeedbackResult

        FeedbackResult(
            np.array([1, 1], dtype=np.int64),
            np.ones((2, 1), dtype=np.float64),
            None,
            None,
            np.ones(2, dtype=bool),
            np.ones(2, dtype=np.uint8),
        )
    with pytest.raises(ValidationError):
        from saealib.policies import FeedbackResult

        FeedbackResult(
            np.array([1], dtype=np.int64),
            np.array([["x"]], dtype=object),
            None,
            None,
            np.ones(1, dtype=bool),
            np.ones(1, dtype=np.uint8),
        )


def test_builtin_strategy_policy_compositions():
    assert isinstance(DirectStrategy().feedback_policy, TrueOnlyFeedback)
    assert isinstance(IndividualBasedStrategy().feedback_policy, MixedFeedback)
    assert isinstance(PreSelectionStrategy(8, 2).feedback_policy, TrueOnlyFeedback)
    assert isinstance(GenerationBasedStrategy(1).feedback_policy, MixedFeedback)
