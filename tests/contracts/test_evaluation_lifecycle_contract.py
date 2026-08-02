import dataclasses
import pickle

import numpy as np
import pytest

from saealib.callback import CallbackManager, PostEvaluationEvent
from saealib.context import OptimizationState
from saealib.exceptions import (
    ConfigurationError,
    EvaluationProtocolError,
    ValidationError,
)
from saealib.execution.evaluator import (
    EvaluationHandle,
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
    EvaluationUpdate,
    Evaluator,
)
from saealib.optimizer import Optimizer
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.stages import (
    EvaluationAcknowledgeStage,
    EvaluationApplyStage,
    EvaluationCollectStage,
    EvaluationPlanStage,
    EvaluationSubmitStage,
)


class _Evaluator(Evaluator):
    def evaluate_batch(self, x, problem):
        n = len(x)
        return EvaluationResult(
            f=np.arange(n, dtype=np.float64).reshape(n, 1),
            g=np.empty((n, 0), dtype=np.float64),
            cv=np.zeros(n, dtype=np.float64),
        )


class _PartialOverrideEvaluator(_Evaluator):
    def submit(self, request, problem):
        return super().submit(request, problem)


class _MultiEvaluator(_Evaluator):
    def submit(self, request, problem):
        return EvaluationHandle(request.request_id, EvaluationStatus.RUNNING)

    def collect(self, handle, *, wait=True):
        return [
            EvaluationUpdate(
                handle.request_id,
                EvaluationStatus.PARTIAL,
                np.array([10], dtype=np.int64),
                EvaluationResult(
                    np.array([[1.0]]),
                    np.empty((1, 0)),
                    np.zeros(1),
                    np.array([10], dtype=np.int64),
                ),
                sequence=0,
            ),
            EvaluationUpdate(
                handle.request_id,
                EvaluationStatus.COMPLETED,
                np.array([11], dtype=np.int64),
                EvaluationResult(
                    np.array([[2.0]]),
                    np.empty((1, 0)),
                    np.zeros(1),
                    np.array([11], dtype=np.int64),
                ),
                sequence=1,
            ),
        ]

    def acknowledge(self, handle, sequence):
        if sequence != handle._acknowledged_sequence + 1:
            raise EvaluationProtocolError("non-contiguous acknowledgement")
        handle._acknowledged_sequence = sequence


class _GapEvaluator(_MultiEvaluator):
    def collect(self, handle, *, wait=True):
        updates = super().collect(handle, wait=wait)
        return [dataclasses.replace(updates[0], sequence=1), updates[1]]


class _OutOfOrderEvaluator(_MultiEvaluator):
    def collect(self, handle, *, wait=True):
        updates = super().collect(handle, wait=wait)
        return [updates[1], updates[0]]


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
    archive = Archive(attrs, 2)
    pareto = ParetoArchive(attrs, 2, direction=np.array([-1.0]))
    return OptimizationState(
        problem=problem,
        population=pop,
        archive=archive,
        pareto_archive=pareto,
        rng=np.random.default_rng(0),
        offspring=pop,
    )


def test_request_owns_read_only_snapshot():
    ids = np.array([1, 2], dtype=np.int64)
    x = np.ones((2, 1), dtype=np.float64)
    request = EvaluationRequest(np.int64(3), ids, x)
    ids[0] = 9
    x[0, 0] = 9
    assert request.candidate_ids[0] == 1
    assert request.x[0, 0] == 1
    assert not request.x.flags.writeable


def test_result_rejects_wrong_dtype_and_shape():
    with pytest.raises(ValidationError):
        EvaluationResult(
            f=np.ones((2, 1), dtype=np.float32),
            g=np.empty((2, 0), dtype=np.float64),
            cv=np.zeros(2),
        )


def test_result_channels_have_exact_shapes_and_owned_arrays():
    result = EvaluationResult(
        np.ones((2, 2)),
        np.empty((2, 0)),
        np.zeros(2),
        np.array([1, 2], dtype=np.int64),
        np.ones(2),
        np.ones((2, 2)),
        {"raw": np.ones((2, 3))},
    )
    assert result.noise is not None and result.noise.shape == (2, 2)
    assert result.cost is not None and result.cost.shape == (2,)
    assert not result.noise.flags.writeable
    with pytest.raises(ValidationError):
        EvaluationResult(
            np.ones((2, 2)), np.empty((2, 0)), np.zeros(2), noise=np.ones(2)
        )


def test_sync_adapter_has_one_contiguous_update():
    state = _state()
    assert state.offspring is not None
    request = EvaluationRequest(
        np.int64(0),
        np.array([10, 11], dtype=np.int64),
        state.offspring.x.copy(),
    )
    evaluator = _Evaluator()
    handle = evaluator.submit(request, state.problem)
    updates = evaluator.collect(handle)
    assert handle.status is EvaluationStatus.COMPLETED
    assert len(updates) == 1
    assert updates[0].sequence == 0
    evaluator.acknowledge(handle, 0)
    assert evaluator.collect(handle) == []
    with pytest.raises(EvaluationProtocolError):
        evaluator.acknowledge(handle, 2)


def test_lifecycle_applies_rows_counts_fe_and_removes_terminal_pending():
    state = _state()
    evaluator = _Evaluator()
    for stage in (
        EvaluationPlanStage(),
        EvaluationSubmitStage(evaluator),
        EvaluationCollectStage(evaluator),
        EvaluationApplyStage(),
        EvaluationAcknowledgeStage(evaluator),
    ):
        state = stage.execute(state)
    assert state.fe == 2
    assert state.offspring is not None
    assert np.allclose(state.offspring.get_array("f"), [[0.0], [1.0]])
    assert state.pending_evaluations == {}


def test_pending_records_are_serializable_without_handles():
    state = _state()
    state = EvaluationPlanStage().execute(state)
    payload = pickle.dumps(state.pending_evaluations)
    assert payload
    assert state.evaluation_handles == {}


def test_multi_update_lifecycle_is_contiguous_and_exactly_once():
    state = _state()
    evaluator = _MultiEvaluator()
    callback = CallbackManager()
    events = []
    callback.register(PostEvaluationEvent, events.append)
    for stage in (
        EvaluationPlanStage(),
        EvaluationSubmitStage(evaluator),
        EvaluationCollectStage(evaluator),
        EvaluationApplyStage(),
        EvaluationAcknowledgeStage(evaluator, callback),
    ):
        state = stage.execute(state)
    assert state.fe == 2
    assert state.pending_evaluations == {}
    assert state.evaluation_handles == {}
    assert [event.candidate_ids.tolist() for event in events] == [[10], [11]]
    assert [
        event.offspring.id.tolist() for event in events if event.offspring is not None
    ] == [
        [10],
        [11],
    ]
    assert [event.status for event in events] == [
        EvaluationStatus.PARTIAL,
        EvaluationStatus.COMPLETED,
    ]
    assert all(event.ctx.fe == i for i, event in enumerate(events, 1))


@pytest.mark.parametrize("evaluator", [_GapEvaluator(), _OutOfOrderEvaluator()])
def test_collect_rejects_gap_and_out_of_order(evaluator):
    state = EvaluationPlanStage().execute(_state())
    state = EvaluationSubmitStage(evaluator).execute(state)
    with pytest.raises(EvaluationProtocolError):
        EvaluationCollectStage(evaluator).execute(state)


def test_apply_rejects_duplicate_result_ids_and_collect_redelivery():
    evaluator = _MultiEvaluator()
    state = EvaluationPlanStage().execute(_state())
    state = EvaluationSubmitStage(evaluator).execute(state)
    state = EvaluationCollectStage(evaluator).execute(state)
    updates = list(state.evaluation_updates)
    assert updates[1].result is not None
    updates[1] = dataclasses.replace(
        updates[1],
        candidate_ids=np.array([10], dtype=np.int64),
        result=dataclasses.replace(
            updates[1].result, candidate_ids=np.array([10], dtype=np.int64)
        ),
    )
    with pytest.raises(EvaluationProtocolError):
        EvaluationApplyStage().execute(state.replace(evaluation_updates=updates))
    with pytest.raises(EvaluationProtocolError):
        EvaluationCollectStage(evaluator).execute(state)


def test_partial_lifecycle_override_is_rejected():
    with pytest.raises(ConfigurationError):
        Optimizer(_state().problem).set_evaluator(_PartialOverrideEvaluator())


def test_checkpoint_rejects_pending_and_accepts_completed_state(tmp_path):
    state = EvaluationPlanStage().execute(_state())
    with pytest.raises(ValidationError):
        state.save(tmp_path / "pending.npz")
    completed = _state()
    path = tmp_path / "complete.npz"
    completed.save(path)
    loaded = OptimizationState.load(path, completed.problem)
    assert loaded.pending_evaluations == {}
    assert loaded.evaluation_handles == {}
