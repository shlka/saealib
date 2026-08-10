"""Focused contracts for the asynchronous runtime boundary."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from saealib.context import OptimizationState
from saealib.core.compiler import Compiler
from saealib.core.graph_builder import build_component_graph
from saealib.core.runtime import RuntimeSession, SequentialPlan
from saealib.exceptions import EvaluationFatalError
from saealib.execution.runtime import (
    AsyncPipelineRuntime,
    _OptimizerEnvironment,
    create_runtime,
)
from saealib.pipeline import Pipeline, Stage


class _Stage(Stage):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)

    def execute(self, state: OptimizationState) -> OptimizationState:
        return state


def _plan() -> SequentialPlan:
    plan = Compiler().compile(build_component_graph(Pipeline([_Stage("stage")])))
    return SequentialPlan.from_executable_plan(plan)


def _state(*, pending: bool = True) -> OptimizationState:
    state = object.__new__(OptimizationState)
    object.__setattr__(
        state, "pending_evaluations", {"request": object()} if pending else {}
    )
    object.__setattr__(state, "evaluation_handles", {})
    object.__setattr__(state, "async_fatal", None)
    return state


class _Environment:
    capabilities = frozenset()

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.terminated = False
        self.keep_pending_after_poll = False
        self.keep_pending_after_execute = False

    def execute(self, plan, state):
        self.calls.append("execute")
        return state

    def execute_async(self, plan, state):
        self.calls.append("execute_async")
        if not self.keep_pending_after_execute:
            object.__setattr__(state, "pending_evaluations", {})
        return state

    def reattach(self, state):
        self.calls.append("reattach")
        object.__setattr__(state, "evaluation_handles", dict(state.pending_evaluations))
        return state

    def poll(self, state):
        self.calls.append("poll")
        if not self.keep_pending_after_poll:
            object.__setattr__(state, "pending_evaluations", {})
        return state

    def is_terminated(self, state):
        self.calls.append("termination")
        return self.terminated

    def can_refill(self, state):
        self.calls.append("capacity")
        return True

    def dispatch(self, event):
        self.calls.append(type(event).__name__)

    def finish_generation(self, state):
        self.calls.append("finish")

    def fatal(self, state):
        self.calls.append("fatal")


def test_optimizer_environment_polls_scheduler_without_waiting() -> None:
    calls: list[bool] = []

    class Scheduler:
        def poll(self, state, *, wait):
            calls.append(wait)
            return state

    optimizer = SimpleNamespace(
        async_evaluation_scheduler=Scheduler(),
        algorithm=SimpleNamespace(allow_partial_tell=False),
        strategy=SimpleNamespace(),
    )
    environment = _OptimizerEnvironment(optimizer, _plan())

    environment.poll(_state())

    assert calls == [False]


def test_optimizer_environment_reaches_inserted_feedback_accumulator_seam():
    # The adapter is intentionally represented by insertion metadata here:
    # StageNodeAdapter self-loop execution is outside this boundary.
    enabled: list[bool] = []

    class Scheduler:
        def enable_feedback_accumulator(self):
            enabled.append(True)

    compiled = replace(
        _plan().plan,
        inserted_adapters=(SimpleNamespace(adapter_name="feedback_accumulator"),),
    )
    optimizer = SimpleNamespace(
        async_evaluation_scheduler=Scheduler(),
        algorithm=SimpleNamespace(allow_partial_tell=False),
        strategy=SimpleNamespace(),
    )

    _OptimizerEnvironment(optimizer, SequentialPlan.from_executable_plan(compiled))

    assert enabled == [True]


def test_async_environment_executes_compiled_plan_without_strategy_step() -> None:
    executed: list[OptimizationState] = []

    class AsyncStage(_Stage):
        def execute(self, state: OptimizationState) -> OptimizationState:
            executed.append(state)
            return state

    plan = Compiler().compile(
        build_component_graph(Pipeline([AsyncStage("async_stage")]))
    )
    sequential = SequentialPlan.from_executable_plan(plan)

    class PoisonStrategy:
        def step(self, state: OptimizationState, provider: object) -> OptimizationState:
            raise AssertionError("async runtime must not call strategy.step")

    optimizer = SimpleNamespace(
        async_evaluation_scheduler=object(),
        algorithm=SimpleNamespace(allow_partial_tell=False),
        strategy=PoisonStrategy(),
    )
    state = _state(pending=False)

    result = _OptimizerEnvironment(optimizer, sequential).execute_async(
        sequential, state
    )

    assert result is state
    assert executed == [state]


def test_factory_selects_async_runtime_when_scheduler_is_present() -> None:
    class Strategy:
        def build_graph(self, provider):
            raise AssertionError("existing executable plan should be reused")

        def step(self, state, provider):
            raise AssertionError("async runtime tests use the environment seam")

    optimizer = SimpleNamespace(
        executable_plan=_plan().plan,
        async_evaluation_scheduler=object(),
        strategy=Strategy(),
    )

    assert isinstance(create_runtime(optimizer), AsyncPipelineRuntime)


def test_async_runtime_reattaches_polls_and_finishes_one_generation() -> None:
    environment = _Environment()
    runtime = AsyncPipelineRuntime(environment)
    session = RuntimeSession(plan=_plan(), state=_state(), generation_open=True)

    step = runtime.advance(session)

    assert environment.calls == ["fatal", "reattach", "poll", "finish"]
    assert step.observable
    assert step.session is not None and not step.session.generation_open


def test_async_runtime_refills_pending_capacity_without_observing_step() -> None:
    environment = _Environment()
    environment.keep_pending_after_poll = True
    environment.keep_pending_after_execute = True
    runtime = AsyncPipelineRuntime(environment)
    state = _state()
    object.__setattr__(state, "evaluation_handles", dict(state.pending_evaluations))
    session = RuntimeSession(plan=_plan(), state=state, generation_open=True)

    step = runtime.advance(session)

    assert environment.calls == [
        "fatal",
        "poll",
        "termination",
        "capacity",
        "execute_async",
    ]
    assert not step.observable
    assert step.session is not None and step.session.generation_open


def test_async_runtime_drains_pending_after_termination_without_refill() -> None:
    environment = _Environment()
    environment.keep_pending_after_poll = True
    environment.terminated = True
    runtime = AsyncPipelineRuntime(environment)
    state = _state()
    object.__setattr__(state, "evaluation_handles", dict(state.pending_evaluations))
    session = RuntimeSession(plan=_plan(), state=state, generation_open=True)

    step = runtime.advance(session)

    assert environment.calls == ["fatal", "poll", "termination"]
    assert not step.observable
    assert step.session is not None and not step.session.finished


def test_async_runtime_preserves_fatal_boundary() -> None:
    class FatalEnvironment(_Environment):
        def fatal(self, state):
            self.calls.append("fatal")
            raise EvaluationFatalError("delivery failed", state)

    runtime = AsyncPipelineRuntime(FatalEnvironment())
    with pytest.raises(EvaluationFatalError, match="delivery failed"):
        runtime.advance(RuntimeSession(plan=_plan(), state=_state()))
