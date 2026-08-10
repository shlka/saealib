"""Drift guards for the Phase 11 framework-extension freeze surface."""

from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path
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
from saealib.core.contracts import (
    MANY,
    AssumptionSet,
    ComponentContract,
    DataSpec,
    PortContract,
    PortDirection,
    PortSpec,
    ServiceRequirement,
    StateContract,
)
from saealib.core.state import USER_DATA, StateStore, StateView
from saealib.core.state.patch import StatePatch
from saealib.execution.runtime import PipelineRuntime
from saealib.space import VectorSpace
from saealib.strategies.base import OptimizationStrategy


def test_tier_three_shapes_and_compiler_surface_are_frozen() -> None:
    assert tuple(field.name for field in fields(ComponentNode)) == (
        "component_id",
        "component",
        "role",
        "resolved_services",
        "_contract_snapshot",
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


class _MutableContractComponent(_CountingComponent):
    def __init__(self) -> None:
        super().__init__()
        self.fixed_size = False

    def contract(self) -> ComponentContract:
        self.contract_calls += 1
        return ComponentContract(
            state=StateContract(reads=(USER_DATA,), writes=(USER_DATA,)),
            assumptions=AssumptionSet({"population.fixed_size": self.fixed_size}),
        )


class _ServiceCountingComponent(_CountingComponent):
    def contract(self) -> ComponentContract:
        self.contract_calls += 1
        return ComponentContract(
            ports={
                "provider": PortContract(
                    outputs=(
                        PortSpec(
                            name="value",
                            direction=PortDirection.OUTPUT,
                            data=DataSpec(kind="Population"),
                            cardinality=MANY,
                            required_services=(
                                ServiceRequirement(name="BoundsService"),
                            ),
                        ),
                    ),
                ),
            },
        )


def _graph(component: object) -> ComponentGraph:
    return ComponentGraph(
        nodes=(ComponentNode(component_id="component", component=component),),
        entry_points=(NodeRef(component_id="component"),),
    )


def test_compilation_reads_each_node_contract_once_and_runtime_revalidates_once() -> (
    None
):
    component = _CountingComponent()
    graph = _graph(component)
    assert component.contract_calls == 0
    plan = Compiler().compile(
        graph,
        CompileContext(initial_state_keys=frozenset({USER_DATA})),
    )
    assert component.contract_calls == 1
    assert plan.contract_snapshots == (("component", plan.graph.nodes[0].contract),)

    state = object.__new__(OptimizationState)
    object.__setattr__(state, "_store", StateStore({USER_DATA: 0}))
    session = PipelineRuntime().initialize(plan, state)
    assert component.contract_calls == 2
    step = PipelineRuntime().advance(session)
    assert step.state.get_state(USER_DATA) == 1
    assert component.contract_calls == 2


def test_graph_mutation_is_reflected_in_each_compile_snapshot() -> None:
    component = _MutableContractComponent()
    graph = _graph(component)
    assert component.contract_calls == 0

    component.fixed_size = True
    first = Compiler().compile(
        graph,
        CompileContext(initial_state_keys=frozenset({USER_DATA})),
    )
    assert component.contract_calls == 1
    assert first.contract_snapshots[0][1].assumptions["population.fixed_size"]

    component.fixed_size = False
    second = Compiler().compile(
        first.graph,
        CompileContext(initial_state_keys=frozenset({USER_DATA})),
    )
    assert component.contract_calls == 2
    assert not second.contract_snapshots[0][1].assumptions["population.fixed_size"]


def test_service_resolution_preserves_the_compile_snapshot() -> None:
    component = _ServiceCountingComponent()
    plan = Compiler().compile(
        _graph(component),
        CompileContext(space=VectorSpace(dim=1, lb=[0.0], ub=[1.0])),
    )

    assert component.contract_calls == 1
    assert plan.graph.node_by_id("component").resolved_services["BoundsService"]
    assert plan.contract_snapshots[0][1] is plan.graph.node_by_id("component").contract


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


def test_phase11_package_boundaries_match_the_adopted_tree() -> None:
    package_root = Path(saealib.__file__).parent
    for relative in (
        "core/runtime.py",
        "core/graph_builder.py",
        "core/compiler/graph.py",
        "profiles/vector/__init__.py",
        "algorithms",
        "operators",
        "acquisition",
        "surrogate",
        "execution",
        "space",
        "strategies",
    ):
        assert (package_root / relative).exists(), relative

    import saealib.profiles.vector as vector_profile

    assert vector_profile.__all__ == ["activate"]
    assert callable(vector_profile.activate)
