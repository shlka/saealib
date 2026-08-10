"""Focused tests for the Phase 6 synchronous plan bridge."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

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
from saealib.core.compiler.graph import ComponentNode, NodeRef
from saealib.core.compiler.schema_rules import _FreshenedComponent
from saealib.core.contracts import DataSpec
from saealib.core.graph_builder import (
    StageContractNodeAdapter,
    StageNodeAdapter,
    build_component_graph,
    cached_execution_target,
)
from saealib.core.runtime import NodeStatus, SequentialPlan
from saealib.exceptions import ValidationError
from saealib.execution.evaluator import SerialEvaluator
from saealib.execution.runtime import PipelineRuntime, _OptimizerEnvironment
from saealib.optimizer import ComponentProvider
from saealib.pipeline import Pipeline, Stage
from saealib.policies.evaluation import EvaluateAll
from saealib.stages import AsyncEvaluationSubmitStage
from saealib.strategies.base import OptimizationStrategy, build_runtime_neutral_graph
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
    assert (
        sum(
            getattr(target, "__func__", None) is StageNodeAdapter.execute
            for target in ordered._execute_targets
        )
        == 0
    )


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

    adapter = plan.graph.nodes[0].component

    def poison(_: OptimizationState) -> OptimizationState:
        raise AssertionError("compiled Stage execution was bypassed")

    setattr(adapter, "execute", poison)
    assert environment.execute(
        SequentialPlan.from_executable_plan(plan), _state()
    ).marker == ("b(a(start))")


def test_sequential_plan_caches_bound_execute_targets() -> None:
    stages = [_Stage("a"), _Stage("b")]
    plan = SequentialPlan.from_executable_plan(_compiled(stages))

    assert [cast(Any, target).__self__ for target in plan._execute_targets] == [
        cast(Any, node.component).stage for node in plan.execution_nodes
    ]
    assert [cast(Any, target).__func__ for target in plan._execute_targets] == [
        type(stage).execute for stage in stages
    ]


def test_stage_adapter_binds_stage_execute_target_at_construction() -> None:
    stage = _Stage("a")
    adapter = StageNodeAdapter(stage)

    target = cast(Any, adapter._execute_target)
    assert target.__self__ is stage
    assert target.__func__ is type(stage).execute

    def poison(_: OptimizationState) -> OptimizationState:
        raise AssertionError("adapter wrapper was called")

    cast(Any, adapter).execute = poison
    assert target(_state()).marker == "a(start)"


def test_freshened_stage_adapter_uses_nested_cached_execute_target() -> None:
    stage = _Stage("a")
    adapter = StageContractNodeAdapter(stage)
    freshened = _FreshenedComponent(adapter, adapter.contract())

    target = cached_execution_target(freshened)
    assert cast(Any, target).__self__ is stage
    assert cast(Any, target).__func__ is type(stage).execute

    def poison(_: OptimizationState) -> OptimizationState:
        raise AssertionError("freshened adapter wrapper was called")

    setattr(adapter, "execute", poison)
    assert cast(Any, target)(_state()).marker == "a(start)"


def test_stage_adapter_rejects_non_callable_execute_target() -> None:
    class NonExecutableStage(Stage):
        execute = None  # type: ignore[assignment]

    with pytest.raises(ValidationError, match="stage must be executable"):
        StageNodeAdapter(NonExecutableStage())


def test_manually_constructed_plan_rejects_non_executable_node() -> None:
    compiled = _compiled([_Stage("a")])

    class NonExecutable:
        def contract(self):
            return compiled.graph.nodes[0].contract

    node = ComponentNode(component_id="a", component=NonExecutable())
    malformed = replace(
        compiled,
        graph=replace(compiled.graph, nodes=(node,)),
    )

    with pytest.raises(ValidationError, match="node 'a' is not executable"):
        SequentialPlan(plan=malformed, nodes=(node,), execution_nodes=(node,))


def test_graph_only_strategy_is_the_runtime_neutral_source() -> None:
    class MismatchedStrategy(OptimizationStrategy):
        def build_graph(self, provider: ComponentProvider):
            return build_component_graph(
                Pipeline(
                    [
                        _Stage("prefix"),
                        _Stage("async_only"),
                        AsyncEvaluationSubmitStage(object(), EvaluateAll()),
                    ]
                )
            )

    class Provider:
        async_evaluation_scheduler = object()

    graph = build_runtime_neutral_graph(
        MismatchedStrategy(), cast(ComponentProvider, Provider())
    )
    assert tuple(node.component_id for node in graph.nodes[:3]) == (
        "prefix",
        "async_only",
        "async_evaluation_submit",
    )


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
