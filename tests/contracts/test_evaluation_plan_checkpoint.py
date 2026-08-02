import json

import numpy as np
import pytest
from test_named_state_checkpoint import _problem, _state

from saealib.context import EvaluationPlanState
from saealib.exceptions import CheckpointError, ValidationError
from saealib.execution.evaluator import EvaluationRequest
from saealib.policies.evaluation import EvaluationPlan


def test_multi_request_plan_round_trip(tmp_path):
    state = _state()
    requests = tuple(
        EvaluationRequest(
            np.int64(index),
            np.array([7, 8], dtype=np.int64),
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
            metadata={"replicate": index, "plan_id": 1},
        )
        for index in (1, 2, 3)
    )
    state = state.replace(
        evaluation_plan=EvaluationPlan(
            requests,
            completion_rule="all_requests_completed",
            artifacts={"replicates": 3},
        ),
        evaluation_plan_state=EvaluationPlanState(
            submitted=(1, 2), deferred=(3,), continuation=False, feedback=True
        ),
    )
    path = tmp_path / "plan.npz"
    state.save(path)
    restored = type(state).load(path, _problem())
    assert restored.evaluation_plan is not None
    assert [int(item.request_id) for item in restored.evaluation_plan.requests] == [
        1,
        2,
        3,
    ]
    assert restored.evaluation_plan.requests[1].metadata["replicate"] == 2
    assert restored.evaluation_plan_state.deferred == (3,)
    assert restored.evaluation_plan_state.feedback is True


def test_plan_state_rejects_invalid_relationships_and_update_ids():
    with pytest.raises(ValidationError, match="completed requests"):
        EvaluationPlanState(completed=(1,))
    with pytest.raises(ValidationError, match="acknowledged requests"):
        EvaluationPlanState(acknowledged=(1,))
    state = _state()
    request = EvaluationRequest(
        np.int64(1), np.array([7, 8], dtype=np.int64), state.population.x.copy()
    )
    with pytest.raises(ValidationError, match="unknown request"):
        state.replace(
            evaluation_plan=EvaluationPlan((request,)),
            evaluation_plan_state=EvaluationPlanState(deferred=(1,)),
            evaluation_plan_updates={2: []},
        )


def _planned_state():
    state = _state()
    requests = tuple(
        EvaluationRequest(
            np.int64(request_id),
            np.array([7], dtype=np.int64),
            np.array([[0.1]], dtype=np.float64),
        )
        for request_id in (1, 2)
    )
    return state.replace(
        evaluation_plan=EvaluationPlan(requests),
        evaluation_plan_state=EvaluationPlanState(deferred=(1, 2)),
    )


def _rewrite_payload(path, destination, key, value):
    payload = dict(np.load(path, allow_pickle=False).items())
    payload[key] = np.frombuffer(json.dumps(value).encode(), dtype=np.uint8)
    np.savez(destination, **payload)


def test_checkpoint_rejects_malformed_evaluation_plan_payload(tmp_path):
    state = _planned_state()
    source = tmp_path / "valid.npz"
    state.save(source)
    bad = tmp_path / "bad-plan.npz"
    _rewrite_payload(source, bad, "_evaluation_plan", {"requests": [{}]})

    with pytest.raises(CheckpointError, match="evaluation plan is malformed"):
        type(state).load(bad, _problem())


def test_checkpoint_rejects_plan_state_referencing_unknown_request(tmp_path):
    state = _planned_state()
    source = tmp_path / "valid.npz"
    state.save(source)
    bad = tmp_path / "bad-state.npz"
    _rewrite_payload(
        source,
        bad,
        "_evaluation_plan_state",
        {"deferred": [99]},
    )

    with pytest.raises(CheckpointError, match="unknown request"):
        type(state).load(bad, _problem())


def test_checkpoint_rejects_updates_for_unknown_request(tmp_path):
    state = _planned_state()
    source = tmp_path / "valid.npz"
    state.save(source)
    bad = tmp_path / "bad-updates.npz"
    _rewrite_payload(source, bad, "_evaluation_plan_updates", {"99": []})

    with pytest.raises(CheckpointError, match="updates reference an unknown request"):
        type(state).load(bad, _problem())
