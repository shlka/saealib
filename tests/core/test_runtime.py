"""Contract tests for the Phase 6 L1 plan/runtime vocabulary."""

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from saealib.callback.events import Event
from saealib.context import OptimizationState
from saealib.core.compiler.compiler import ExecutablePlan
from saealib.core.compiler.graph import ComponentGraph
from saealib.core.runtime import (
    IssueCandidateIds,
    NodeResult,
    NodeStatus,
    RequestCheckpoint,
    RequestRecompile,
    RequestTermination,
    RuntimeCommand,
    RuntimeSession,
    RuntimeStep,
    SequentialPlan,
)
from saealib.core.state.patch import StatePatch
from saealib.exceptions import ValidationError


def _plan(*capabilities: str) -> ExecutablePlan:
    return ExecutablePlan(
        graph=ComponentGraph(nodes=()),
        diagnostics=(),
        required_runtime_capabilities=frozenset(capabilities),
        active_rule_namespaces=frozenset(),
        active_rule_names=(),
    )


def _state() -> OptimizationState:
    # The vocabulary tests only validate identity and threading.  Avoid
    # constructing a problem/population fixture unrelated to this boundary.
    return object.__new__(OptimizationState)


def _result(status: NodeStatus, *commands: RuntimeCommand) -> NodeResult:
    state = _state()
    return NodeResult(
        patch=StatePatch(writes={}),
        events=(Event(ctx=state),),
        commands=commands,
        status=status,
    )


@pytest.mark.parametrize("status", NodeStatus)
def test_node_result_accepts_each_status(status: NodeStatus) -> None:
    assert _result(status).status is status


def test_recompile_status_and_request_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="cannot be combined"):
        _result(NodeStatus.RECOMPILE_REQUIRED, RequestRecompile())


def test_runtime_step_rejects_status_command_overlap_after_node_guard_bypass() -> None:
    result = object.__new__(NodeResult)
    object.__setattr__(result, "patch", StatePatch(writes={}))
    object.__setattr__(result, "events", ())
    object.__setattr__(result, "commands", (RequestRecompile(),))
    object.__setattr__(result, "status", NodeStatus.RECOMPILE_REQUIRED)
    with pytest.raises(ValidationError, match="cannot be combined"):
        RuntimeStep(state=_state(), node_results=(result,))


def test_runtime_command_refusal_is_a_normal_step_outcome() -> None:
    command = RequestTermination(reason="policy")
    step = RuntimeStep(state=_state(), refused_commands=(command,))
    assert step.refused_commands == (command,)


def test_runtime_step_trace_is_immutable_vocabulary() -> None:
    step = RuntimeStep(state=_state(), executed_node_ids=("a", "b"))
    assert step.executed_node_ids == ("a", "b")
    with pytest.raises(ValidationError, match="executed_node_ids"):
        RuntimeStep(
            state=_state(),
            executed_node_ids=cast(tuple[str, ...], ("a", 1)),
        )


def test_core_command_variants_do_not_carry_evaluation_payloads() -> None:
    assert IssueCandidateIds(count=3).count == 3
    assert isinstance(RequestCheckpoint(), RequestCheckpoint)
    with pytest.raises(ValidationError):
        IssueCandidateIds(count=0)


def test_sequential_plan_is_immutable_and_matches_capabilities_by_subset() -> None:
    plan = SequentialPlan(plan=_plan("partial_feedback"), nodes=())
    assert plan.accepts({"partial_feedback", "extra"})
    assert not plan.accepts(set())
    with pytest.raises(FrozenInstanceError):
        setattr(plan, "nodes", ())


def test_runtime_session_threads_existing_optimization_state() -> None:
    state = _state()
    session = RuntimeSession(plan=_plan(), state=state)
    assert session.state is state
    assert not session.finished
