from __future__ import annotations

import numpy as np
import pytest

from saealib.callback import GenerationEndEvent, GenerationStartEvent, RunStartEvent
from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.core import (
    BranchRegion,
    ComponentContract,
    Condition,
    LoopRegion,
    RepeatRegion,
    StateContract,
    StatePatch,
    lower_pipeline,
)
from saealib.core.compiler import CompileContext, Compiler
from saealib.core.runtime import NodeResult, NodeStatus
from saealib.core.state import OPTIMIZATION_STATE_INITIAL_KEYS, USER_DATA
from saealib.exceptions import ValidationError
from saealib.execution.runtime import AsyncPipelineRuntime, PipelineRuntime
from saealib.pipeline import Branch, Loop, Pipeline, Repeat, Stage
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem


def _state() -> OptimizationState:
    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(1,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
        PopulationAttribute(name="g", dtype=np.float64, shape=(0,)),
        PopulationAttribute(name="cv", dtype=np.float64, shape=()),
    ]
    problem = Problem(
        func=lambda x: np.array([x[0] ** 2]),
        dim=1,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0],
        ub=[1.0],
        comparator=SingleObjectiveComparator(),
    )
    population = Population(attrs, init_capacity=1)
    archive = Archive(attrs, init_capacity=1)
    pareto_archive = ParetoArchive(attrs, init_capacity=1, direction=np.array([-1.0]))
    state = OptimizationState(
        problem=problem,
        population=population,
        archive=archive,
        pareto_archive=pareto_archive,
        data={"marker": 1},
    )
    state.set_state(USER_DATA, 0)
    return state


def _compile(pipeline: Pipeline):
    return Compiler().compile_pipeline(
        pipeline,
        CompileContext(
            initial_state_keys=frozenset((*OPTIMIZATION_STATE_INITIAL_KEYS, USER_DATA))
        ),
    )


class _Increment:
    name = "increment"

    def contract(self) -> ComponentContract:
        return ComponentContract(
            state=StateContract(reads=(USER_DATA,), writes=(USER_DATA,))
        )

    def execute(self, view):
        return StatePatch(writes={USER_DATA: view.get(USER_DATA) + 1})


class _UntilFive(Condition):
    def contract(self) -> StateContract:
        return StateContract(reads=(USER_DATA,))

    def evaluate(self, view) -> bool:
        return view.get(USER_DATA) >= 5


class _StateCondition(Condition):
    def __init__(self, value: bool) -> None:
        self.value = value

    def contract(self) -> StateContract:
        return StateContract()

    def evaluate(self, view) -> bool:
        del view
        return self.value


class _BlockOnce(_Increment):
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, view):
        self.calls += 1
        patch = super().execute(view)
        if self.calls == 1:
            return NodeResult(patch=patch, status=NodeStatus.BLOCKED)
        return patch


class _StructuredEnvironment:
    capabilities = frozenset()

    def __init__(self) -> None:
        self.dispatched: list[object] = []
        self.finished_generations = 0

    def dispatch(self, event) -> None:
        self.dispatched.append(event)

    def finish_generation(self, state) -> None:
        self.finished_generations += 1
        self.dispatch(GenerationEndEvent(ctx=state))

    def is_terminated(self, state) -> bool:
        del state
        return False

    def fatal(self, state) -> None:
        del state


class _StatusOnce(_Increment):
    def __init__(self, status: NodeStatus) -> None:
        self.status = status
        self.calls = 0

    def execute(self, view):
        self.calls += 1
        patch = super().execute(view)
        status = self.status if self.calls == 1 else NodeStatus.COMPLETED
        return NodeResult(patch=patch, status=status)


class _AsyncBlockOnce(_Increment):
    def __init__(self) -> None:
        self.calls = 0

    def execute_async(self, view):
        self.calls += 1
        patch = super().execute(view)
        status = NodeStatus.BLOCKED if self.calls == 1 else NodeStatus.COMPLETED
        return NodeResult(patch=patch, status=status)


class _AsyncRequiredWithoutDriver(_Increment):
    requires_async_execution = True


class _AsyncBranchLeaf(_Increment):
    def __init__(self) -> None:
        self.calls = 0

    def execute_async(self, view):
        self.calls += 1
        return super().execute(view)


def test_structured_plan_executes_repeat_loop_and_branch() -> None:
    state = _state()
    runtime = PipelineRuntime()
    pipeline = Pipeline(
        [
            Repeat(_Increment(), 3, name="repeat"),
            Loop(_Increment(), until=_UntilFive(), name="loop"),
            Branch(
                _StateCondition(True),
                then=_Increment(),
                else_=_Increment(),
                name="branch",
            ),
        ]
    )
    graph = lower_pipeline(pipeline)
    assert [type(region.region) for region in graph.regions] == [
        RepeatRegion,
        LoopRegion,
        BranchRegion,
    ]

    session = runtime.initialize(_compile(pipeline), state)
    step = runtime.advance(session)

    assert state.get_state(USER_DATA) == 6
    assert step.finished
    assert len(step.node_results) == 6


def test_structured_plan_resumes_blocked_leaf_from_saved_frame() -> None:
    component = _BlockOnce()
    runtime = PipelineRuntime()
    session = runtime.initialize(_compile(Pipeline([component])), _state())

    first = runtime.advance(session)
    assert not first.finished
    assert first.node_results[0].status is NodeStatus.BLOCKED
    assert first.session is not None and first.session.frames

    second = runtime.advance(first.session)
    assert second.finished
    assert component.calls == 2
    assert second.state.get_state(USER_DATA) == 2


def test_async_structured_repeat_resumes_the_same_iteration() -> None:
    component = _AsyncBlockOnce()
    runtime = AsyncPipelineRuntime()
    session = runtime.initialize(
        _compile(Pipeline([Repeat(component, count=2)])), _state()
    )

    first = runtime.advance(session)
    assert first.node_results[0].status is NodeStatus.BLOCKED
    assert first.session is not None
    assert first.session.frames[-1].operation_index == 0

    second = runtime.advance(first.session)

    assert second.finished
    assert component.calls == 3
    assert second.state.get_state(USER_DATA) == 3


def test_async_structured_branch_executes_only_the_selected_leaf() -> None:
    selected = _AsyncBranchLeaf()
    skipped = _AsyncBranchLeaf()
    runtime = AsyncPipelineRuntime()
    session = runtime.initialize(
        _compile(
            Pipeline(
                [
                    Branch(
                        _StateCondition(True),
                        then=selected,
                        else_=skipped,
                    )
                ]
            )
        ),
        _state(),
    )

    step = runtime.advance(session)

    assert step.finished
    assert selected.calls == 1
    assert skipped.calls == 0


def test_async_structured_requires_a_driver_without_sync_fallback() -> None:
    runtime = AsyncPipelineRuntime()
    session = runtime.initialize(
        _compile(Pipeline([_AsyncRequiredWithoutDriver()])), _state()
    )

    with pytest.raises(ValidationError, match="async execution driver"):
        runtime.advance(session)


def test_structured_external_environment_closes_generation_not_runtime_session() -> (
    None
):
    environment = _StructuredEnvironment()
    runtime = PipelineRuntime(environment=environment)
    session = runtime.initialize(_compile(Pipeline([_Increment()])), _state())

    step = runtime.advance(session)

    assert step.finished is False
    assert step.observable is True
    assert step.session is not None
    assert step.session.finished is False
    assert step.session.generation_open is False
    assert environment.finished_generations == 1
    assert [type(event) for event in environment.dispatched] == [
        RunStartEvent,
        GenerationStartEvent,
        GenerationEndEvent,
    ]


@pytest.mark.parametrize("status", [NodeStatus.BLOCKED, NodeStatus.RUNNING])
def test_structured_external_environment_keeps_generation_open_for_resume(
    status: NodeStatus,
) -> None:
    component = _StatusOnce(status)
    environment = _StructuredEnvironment()
    runtime = PipelineRuntime(environment=environment)
    session = runtime.initialize(_compile(Pipeline([component])), _state())

    first = runtime.advance(session)

    assert first.finished is False
    assert first.observable is False
    assert first.session is not None
    assert first.session.finished is False
    assert first.session.generation_open is True
    assert environment.finished_generations == 0
    assert first.node_results[0].status is status

    second = runtime.advance(first.session)

    assert second.finished is False
    assert second.observable is True
    assert second.session is not None
    assert second.session.finished is False
    assert second.session.generation_open is False
    assert environment.finished_generations == 1
    assert component.calls == 2


def test_structured_external_environment_rejects_failed_node() -> None:
    environment = _StructuredEnvironment()
    runtime = PipelineRuntime(environment=environment)
    session = runtime.initialize(
        _compile(Pipeline([_StatusOnce(NodeStatus.FAILED)])), _state()
    )

    with pytest.raises(
        ValidationError, match="structured runtime node reported FAILED"
    ):
        runtime.advance(session)

    assert environment.finished_generations == 0


def test_structured_runtime_rejects_recompile_required() -> None:
    plan = _compile(Pipeline([_StatusOnce(NodeStatus.RECOMPILE_REQUIRED)]))

    with pytest.raises(ValidationError, match="structured plan"):
        PipelineRuntime().advance(PipelineRuntime().initialize(plan, _state()))


def test_async_runtime_accepts_structured_plan_at_initialization() -> None:
    plan = _compile(Pipeline([_Increment()]))
    runtime = AsyncPipelineRuntime()

    session = runtime.initialize(plan, _state())

    step = runtime.advance(session)

    assert step.finished
    assert step.state.get_state(USER_DATA) == 1


class _LegacyStage(Stage):
    name = "legacy"

    def execute(self, state):
        return state


def test_structured_plan_rejects_optimization_state_stage_boundary() -> None:
    with pytest.raises(ValidationError, match="stage_component"):
        _compile(Pipeline([_LegacyStage()]))
