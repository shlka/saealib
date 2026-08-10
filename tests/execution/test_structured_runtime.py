from __future__ import annotations

import numpy as np

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
from saealib.execution.runtime import PipelineRuntime
from saealib.pipeline import Branch, Loop, Pipeline, Repeat
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
    graph = lower_pipeline(pipeline)
    return Compiler().compile(
        graph,
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
