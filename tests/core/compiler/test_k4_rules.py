"""Tests for the K4 service-resolution and port-compatibility rules."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from saealib.algorithms.ga import GA
from saealib.core.compiler import (
    CompileContext,
    Compiler,
    ComponentGraph,
    ComponentNode,
    DataEdge,
    NodeRef,
    PortCompatibilityRule,
    ResolutionResult,
    RuleContext,
    ServiceResolutionRule,
)
from saealib.core.compiler.diagnostics import DiagnosticBag
from saealib.core.contracts import (
    MANY,
    ONE,
    ComponentContract,
    DataSpec,
    Fixed,
    ParameterSpec,
    PortContract,
    PortDirection,
    PortSpec,
    RepresentationSpec,
    StateContract,
)
from saealib.core.state import OPTIMIZATION_STATE_INITIAL_KEYS, SURROGATES_DEFAULT
from saealib.execution.evaluator import SerialEvaluator
from saealib.operators.crossover import CrossoverBLXAlpha
from saealib.operators.mutation import MutationUniform
from saealib.operators.selection import (
    SequentialSelection,
    SurvivorSelection,
    TruncationSelection,
)
from saealib.problem.problem import Problem
from saealib.space import ObjectSpace, VectorSpace
from saealib.strategies.direct import DirectStrategy


def _ga() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(prob=1.0, alpha=0.5),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=_SequentialSurvivors(),
    )


class _SequentialSurvivors(SurvivorSelection):
    """A service-free survivor operator for graph contract tests."""

    def select(self, ctx, pool, n_survivors) -> np.ndarray:
        return np.arange(n_survivors, dtype=int)


class _ContractComponent:
    """A minimal held component for an actual strategy graph probe."""

    def __init__(self, *, exports: tuple = ()) -> None:
        self._contract = ComponentContract(state=StateContract(exports=exports))

    def contract(self) -> ComponentContract:
        return self._contract


class _GraphProvider:
    """The provider shape consumed by DirectStrategy.build_graph()."""

    def __init__(self) -> None:
        self.algorithm = GA(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
            mutation=MutationUniform(prob_var=0.1),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        )
        self.evaluator = SerialEvaluator()
        self.surrogate_manager = _ContractComponent(exports=(SURROGATES_DEFAULT,))
        self.acquisition = _ContractComponent()
        self.cbmanager = None
        self.async_evaluation_scheduler = None
        self.evaluation_planner = None
        self.feedback_builder = None
        self.feedback_builder_explicit = False


def _ga_graph() -> ComponentGraph:
    return ComponentGraph(
        nodes=(ComponentNode(component_id="ga", component=_ga()),),
        entry_points=(NodeRef(component_id="ga"),),
    )


def _object_space() -> ObjectSpace:
    return ObjectSpace(
        RepresentationSpec(
            kind="vector",
            parameters=(ParameterSpec(name="dim", value=Fixed(value=1)),),
        )
    )


def test_missing_bounds_service_is_rejected_at_compile_time() -> None:
    plan = Compiler().compile(
        _ga_graph(),
        CompileContext(space=_object_space()),
    )

    findings = [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.code == "unresolved_service"
    ]
    assert findings
    assert all("ga" in str(diagnostic.path) for diagnostic in findings)
    assert all("BoundsService" in diagnostic.message for diagnostic in findings)
    assert all(diagnostic.resolutions for diagnostic in findings)
    assert plan.graph.node_by_id("ga").resolved_services == {}


def test_resolved_service_is_a_direct_reference_on_the_node() -> None:
    space = VectorSpace(dim=1, lb=[0.0], ub=[1.0])
    bounds = space.services.require("BoundsService")
    plan = Compiler().compile(
        _ga_graph(),
        CompileContext(space=space),
    )

    node = plan.graph.node_by_id("ga")
    assert node.resolved_services["BoundsService"] is bounds
    assert not any(
        diagnostic.code in {"unresolved_service", "unknown_service"}
        for diagnostic in plan.diagnostics
    )


def test_vector_space_and_normal_ga_have_no_k4_errors() -> None:
    space = VectorSpace(dim=1, lb=[0.0], ub=[1.0])
    plan = Compiler().compile(
        _ga_graph(),
        CompileContext(
            space=space,
            initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
        ),
    )

    assert not [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.severity.value == "error"
    ]


@dataclass(frozen=True)
class _Endpoint:
    data_kind: str
    cardinality: str
    direction: PortDirection

    def contract(self) -> ComponentContract:
        port = PortSpec(
            name="value",
            direction=self.direction,
            data=DataSpec(kind=self.data_kind),
            cardinality=self.cardinality,
        )
        role = (
            PortContract(outputs=(port,))
            if self.direction is PortDirection.OUTPUT
            else PortContract(inputs=(port,))
        )
        return ComponentContract(ports={"role": role})


@dataclass(frozen=True)
class _AmbiguousEndpoint:
    """Expose one directional port under two roles."""

    data_kind: str
    cardinality: str

    def contract(self) -> ComponentContract:
        port = PortSpec(
            name="value",
            direction=PortDirection.OUTPUT,
            data=DataSpec(kind=self.data_kind),
            cardinality=self.cardinality,
        )
        return ComponentContract(
            ports={
                "first": PortContract(outputs=(port,)),
                "second": PortContract(outputs=(port,)),
            }
        )


def _edge_graph(
    producer_kind: str,
    consumer_kind: str,
    *,
    source_port: str = "value",
    target_port: str = "value",
    source_role: str | None = "role",
    target_role: str | None = "role",
) -> ComponentGraph:
    return ComponentGraph(
        nodes=(
            ComponentNode(
                component_id="producer",
                component=_Endpoint(
                    data_kind=producer_kind,
                    cardinality=ONE,
                    direction=PortDirection.OUTPUT,
                ),
            ),
            ComponentNode(
                component_id="consumer",
                component=_Endpoint(
                    data_kind=consumer_kind,
                    cardinality=MANY,
                    direction=PortDirection.INPUT,
                ),
            ),
        ),
        data_edges=(
            DataEdge(
                source=NodeRef(component_id="producer", role=source_role),
                target=NodeRef(component_id="consumer", role=target_role),
                source_port=source_port,
                target_port=target_port,
            ),
        ),
        entry_points=(NodeRef(component_id="producer", role=source_role),),
    )


def _ambiguous_edge_graph(source_role: str | None = None) -> ComponentGraph:
    return ComponentGraph(
        nodes=(
            ComponentNode(
                component_id="producer",
                component=_AmbiguousEndpoint(
                    data_kind="Population",
                    cardinality=ONE,
                ),
            ),
            ComponentNode(
                component_id="consumer",
                component=_Endpoint(
                    data_kind="Population",
                    cardinality=MANY,
                    direction=PortDirection.INPUT,
                ),
            ),
        ),
        data_edges=(
            DataEdge(
                source=NodeRef(component_id="producer", role=source_role),
                target=NodeRef(component_id="consumer", role="role"),
                source_port="value",
                target_port="value",
            ),
        ),
        entry_points=(NodeRef(component_id="producer", role=source_role),),
    )


def _k4_signature(
    graph: ComponentGraph,
    order: tuple[ServiceResolutionRule | PortCompatibilityRule, ...],
):
    current = graph
    diagnostics = DiagnosticBag()
    for rule in order:
        context = RuleContext(
            graph=current,
            compile_context=CompileContext(),
            diagnostics=diagnostics,
        )
        result = rule.apply(context)
        if isinstance(result, ResolutionResult):
            current = result.graph
            diagnostics.extend(result.diagnostics)
        else:
            diagnostics.extend(result.diagnostics)
    return (
        current,
        tuple(
            (diagnostic.code, str(diagnostic.path), diagnostic.message)
            for diagnostic in diagnostics
        ),
    )


def test_incompatible_edge_has_both_connection_endpoints_in_diagnostic() -> None:
    graph = _edge_graph("Population", "GenomeBatch")
    plan = Compiler().compile(graph)

    findings = [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.code == "incompatible_port"
    ]
    assert len(findings) == 1
    finding = findings[0]
    assert "producer[role].value" in finding.message
    assert "consumer[role].value" in finding.message
    assert len(finding.related) == 1
    assert str(finding.related[0]) == "consumer[role].value"


def test_unknown_port_name_has_a_distinct_diagnostic() -> None:
    plan = Compiler().compile(
        _edge_graph("Population", "Population", source_port="missing")
    )

    findings = [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.code == "unknown_port"
    ]
    assert len(findings) == 1
    assert "missing" in findings[0].message
    assert findings[0].resolutions == (
        "Correct the edge port name or declare that directional port in the "
        "endpoint contract.",
    )


def test_unqualified_duplicate_port_name_has_an_ambiguous_diagnostic() -> None:
    plan = Compiler().compile(_ambiguous_edge_graph())

    findings = [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.code == "ambiguous_port"
    ]
    assert len(findings) == 1
    assert "multiple roles" in findings[0].message
    assert findings[0].resolutions == (
        "Specify the NodeRef role for the ambiguous endpoint.",
    )


def test_explicit_role_resolves_a_duplicate_port_name() -> None:
    plan = Compiler().compile(_ambiguous_edge_graph(source_role="first"))

    assert not any(
        diagnostic.code in {"ambiguous_port", "unknown_port", "incompatible_port"}
        for diagnostic in plan.diagnostics
    )


def test_compatible_edge_has_no_port_diagnostics() -> None:
    plan = Compiler().compile(_edge_graph("Population", "Population"))

    assert not any(
        diagnostic.code == "incompatible_port" for diagnostic in plan.diagnostics
    )


def test_k4_rules_have_the_same_result_when_enumeration_is_shuffled() -> None:
    graph = _edge_graph("Population", "GenomeBatch")
    rules = (ServiceResolutionRule(), PortCompatibilityRule())
    expected = None
    for seed in range(8):
        shuffled = list(rules)
        random.Random(seed).shuffle(shuffled)
        signature = _k4_signature(graph, tuple(shuffled))
        if expected is None:
            expected = signature
        else:
            assert signature == expected

    assert expected is not None


def test_actual_strategy_build_graph_is_compiled_and_reports_code_counts(
    capsys,
) -> None:
    provider: Any = _GraphProvider()
    problem = Problem(
        func=lambda x: np.array([x[0]]),
        dim=1,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[0.0],
        ub=[1.0],
    )
    graph = DirectStrategy().build_graph(provider)
    plan = Compiler().compile(
        graph,
        CompileContext(space=problem.space, problem=problem),
    )
    counts = Counter(diagnostic.code for diagnostic in plan.diagnostics)
    print(f"actual build_graph diagnostics: {dict(counts)}")

    assert graph.nodes
    assert not any(
        diagnostic.code == "incompatible_port"
        and "cannot be resolved" in diagnostic.message
        for diagnostic in plan.diagnostics
    )
    assert capsys.readouterr().out.startswith("actual build_graph diagnostics:")
