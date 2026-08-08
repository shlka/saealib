"""Focused tests for the Phase 6 synchronous plan bridge."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from saealib import (
    GA,
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
)
from saealib.context import OptimizationState
from saealib.core.compiler import Compiler, ControlEdge, DataEdge
from saealib.core.compiler.adapters import Adapter, AdapterComponent
from saealib.core.compiler.graph import NodeRef
from saealib.core.contracts import DataSpec
from saealib.core.graph_builder import StageNodeAdapter, build_component_graph
from saealib.core.runtime import NodeStatus, SequentialPlan
from saealib.exceptions import ValidationError
from saealib.execution.evaluator import SerialEvaluator
from saealib.execution.runtime import PipelineRuntime, _OptimizerEnvironment
from saealib.optimizer import ComponentProvider
from saealib.pipeline import Pipeline, Stage
from saealib.strategies.direct import DirectStrategy


class _Stage(Stage):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.seen: list[OptimizationState] = []

    def execute(self, state: OptimizationState) -> OptimizationState:
        self.seen.append(state)
        next_state = object.__new__(OptimizationState)
        object.__setattr__(
            next_state, "marker", f"{self.name}({getattr(state, 'marker', '')})"
        )
        return next_state


def _state() -> OptimizationState:
    state = object.__new__(OptimizationState)
    object.__setattr__(state, "marker", "start")
    return state


def _compiled(stages: list[_Stage]):
    graph = build_component_graph(Pipeline(cast(list[Stage], stages)))
    return Compiler().compile(graph)


def test_real_default_strategy_graph_compiles_to_stage_order() -> None:
    class Provider:
        algorithm = GA(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
            mutation=MutationUniform(prob_var=0.1),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        )
        evaluator = SerialEvaluator()
        cbmanager = None
        async_evaluation_scheduler = None
        evaluation_planner = None
        feedback_builder = None
        feedback_builder_explicit = False

    plan = Compiler().compile(
        DirectStrategy().build_graph(cast(ComponentProvider, Provider()))
    )
    ordered = SequentialPlan.from_executable_plan(plan)

    assert ordered.nodes
    assert all(isinstance(node.component, StageNodeAdapter) for node in ordered.nodes)
    assert len(ordered.nodes) == sum(
        isinstance(node.component, StageNodeAdapter) for node in plan.graph.nodes
    )
    assert [node.component_id for node in ordered.execution_nodes] == [
        "count_generation",
        "ask",
        "evaluation_plan",
        "evaluation_submit",
        "evaluation_collect",
        "evaluation_apply",
        "archive_update",
        "feedback",
        "tell",
        "evaluation_acknowledge",
    ]


def test_runtime_threads_stage_return_values_and_reports_order_without_events() -> None:
    stages = [_Stage("a"), _Stage("b"), _Stage("c")]
    session = PipelineRuntime().initialize(_compiled(stages), _state())

    step = PipelineRuntime().advance(session)

    assert step.observable
    assert not step.finished
    assert [result.status for result in step.node_results] == [NodeStatus.COMPLETED] * 3
    assert step.executed_node_ids == ("a", "b", "c")
    assert all(result.events == () for result in step.node_results)
    assert stages[0].seen[0].marker == "start"
    assert stages[1].seen[0].marker == "a(start)"
    assert stages[2].seen[0].marker == "b(a(start))"
    assert step.session is not None and step.session.state is step.state


def test_sync_environment_executes_compiled_plan_without_strategy_step() -> None:
    stages = [_Stage("a"), _Stage("b")]
    plan = _compiled(stages)

    class PoisonStrategy:
        def step(self, state: OptimizationState, provider: object) -> OptimizationState:
            raise AssertionError("sync runtime must not call strategy.step")

    class Optimizer:
        strategy = PoisonStrategy()
        async_evaluation_scheduler = None

    environment = _OptimizerEnvironment(
        Optimizer(), SequentialPlan.from_executable_plan(plan)
    )
    result = environment.execute(SequentialPlan.from_executable_plan(plan), _state())

    assert result.marker == "b(a(start))"

    def poison(_: OptimizationState) -> OptimizationState:
        raise AssertionError("compiled Stage execution was bypassed")

    setattr(stages[0], "execute", poison)
    with pytest.raises(AssertionError, match="compiled Stage execution"):
        environment.execute(SequentialPlan.from_executable_plan(plan), _state())


def test_order_view_ignores_data_only_synthetic_node_and_stage_self_loop() -> None:
    stages = [_Stage("a"), _Stage("b")]
    plan = _compiled(stages)
    synthetic = AdapterComponent(
        adapter=Adapter(
            name="identity_test",
            source=DataSpec(kind="Population"),
            target=DataSpec(kind="Population"),
            lossless=True,
            auto_insertable=True,
        )
    )
    graph = replace(
        plan.graph,
        nodes=(
            *plan.graph.nodes,
            type(plan.graph.nodes[0])(component_id="adapter", component=synthetic),
        ),
        data_edges=(
            *plan.graph.data_edges,
            DataEdge(
                source=NodeRef(component_id="b"),
                target=NodeRef(component_id="b"),
                source_port="x",
                target_port="x",
            ),
        ),
    )
    ordered = SequentialPlan.from_executable_plan(replace(plan, graph=graph))
    assert [node.component_id for node in ordered.nodes] == ["a", "b"]


@pytest.mark.parametrize("kind", ["cycle", "ambiguous", "missing_entry"])
def test_invalid_control_order_is_rejected(kind: str) -> None:
    stages = [_Stage("a"), _Stage("b"), _Stage("c")]
    plan = _compiled(stages)
    if kind == "cycle":
        edges = (
            ControlEdge(
                source=NodeRef(component_id="a"), target=NodeRef(component_id="b")
            ),
            ControlEdge(
                source=NodeRef(component_id="b"), target=NodeRef(component_id="a")
            ),
        )
        graph = replace(plan.graph, control_edges=edges)
    elif kind == "ambiguous":
        edges = (
            ControlEdge(
                source=NodeRef(component_id="a"), target=NodeRef(component_id="c")
            ),
            ControlEdge(
                source=NodeRef(component_id="b"), target=NodeRef(component_id="c")
            ),
        )
        graph = replace(
            plan.graph,
            control_edges=edges,
            entry_points=(NodeRef(component_id="a"), NodeRef(component_id="b")),
        )
    else:
        graph = replace(plan.graph, entry_points=())
    with pytest.raises(ValidationError):
        SequentialPlan.from_executable_plan(replace(plan, graph=graph))


def test_runtime_rejects_missing_capability() -> None:
    plan = replace(
        _compiled([_Stage("a")]),
        required_runtime_capabilities=frozenset({"partial_feedback"}),
    )
    with pytest.raises(ValidationError, match="capabilities"):
        PipelineRuntime().initialize(plan, _state())
