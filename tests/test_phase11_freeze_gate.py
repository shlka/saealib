"""Drift guards for the Phase 11 framework-extension freeze surface."""

from __future__ import annotations

import inspect
from dataclasses import fields
from typing import Any, cast, get_type_hints

import saealib
import saealib.core.compiler as compiler_api
from saealib.context import OptimizationState
from saealib.core.compiler import (
    CompileContext,
    Compiler,
    ComponentGraph,
    ComponentNode,
    GraphTemplate,
    NodeRef,
)
from saealib.core.contracts import ComponentContract, StateContract
from saealib.core.state import USER_DATA, StateStore, StateView
from saealib.core.state.patch import StatePatch
from saealib.execution.runtime import PipelineRuntime
from saealib.strategies.base import OptimizationStrategy


def test_tier_three_shapes_and_compiler_surface_are_frozen() -> None:
    assert tuple(field.name for field in fields(ComponentNode)) == (
        "component_id",
        "component",
        "role",
        "resolved_services",
        "contract",
    )
    assert tuple(field.name for field in fields(ComponentGraph)) == (
        "nodes",
        "data_edges",
        "control_edges",
        "state_bindings",
        "entry_points",
    )
    assert tuple(inspect.signature(GraphTemplate.build_graph).parameters) == (
        "self",
        "bindings",
    )
    assert get_type_hints(GraphTemplate.build_graph)["return"] is ComponentGraph

    assert compiler_api.__all__ == [
        "DIAGNOSTIC_CODES",
        "CompilationRule",
        "CompileContext",
        "ComponentBindings",
        "ComponentGraph",
        "ComponentId",
        "ComponentNode",
        "ContractPath",
        "ControlEdge",
        "DataEdge",
        "Diagnostic",
        "DiagnosticBag",
        "DiagnosticCodeVocabulary",
        "ExecutablePlan",
        "GraphTemplate",
        "NodeRef",
        "Severity",
        "StateBinding",
    ]
    assert "Compiler" not in compiler_api.__all__
    assert "RuleContext" not in compiler_api.__all__
    assert not hasattr(saealib, "_TIER2_MAP")


class _CountingComponent:
    def __init__(self) -> None:
        self.contract_calls = 0

    def contract(self) -> ComponentContract:
        self.contract_calls += 1
        return ComponentContract(
            state=StateContract(reads=(USER_DATA,), writes=(USER_DATA,))
        )

    def execute(self, state: StateView) -> StatePatch:
        return StatePatch(writes={USER_DATA: cast(int, state.get(USER_DATA)) + 1})


def _graph(component: object) -> ComponentGraph:
    return ComponentGraph(
        nodes=(ComponentNode(component_id="component", component=component),),
        entry_points=(NodeRef(component_id="component"),),
    )


def test_compilation_reads_each_node_contract_once_and_runtime_revalidates_once() -> (
    None
):
    component = _CountingComponent()
    plan = Compiler().compile(
        _graph(component),
        CompileContext(initial_state_keys=frozenset({USER_DATA})),
    )
    assert component.contract_calls == 1

    state = object.__new__(OptimizationState)
    object.__setattr__(state, "_store", StateStore({USER_DATA: 0}))
    session = PipelineRuntime().initialize(plan, state)
    assert component.contract_calls == 2
    step = PipelineRuntime().advance(session)
    assert step.state.get_state(USER_DATA) == 1
    assert component.contract_calls == 2


def test_graph_only_strategy_uses_graph_and_recovers_only_a_compatibility_facade() -> (
    None
):
    component = _CountingComponent()

    class GraphOnlyStrategy(OptimizationStrategy):
        def build_graph(self, provider: Any) -> ComponentGraph:
            del provider
            return _graph(component)

    strategy = GraphOnlyStrategy()
    graph = strategy.build_graph(cast(Any, object()))
    pipeline = strategy.build_pipeline(cast(Any, object()))

    assert graph.node_by_id("component").component is component
    assert tuple(pipeline.stages) == ()
