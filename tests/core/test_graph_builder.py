"""Tests for the Phase 5 stage-to-graph bridge."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from saealib import (
    GA,
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
)
from saealib.core.compiler import CompileContext, Compiler, RuleContext
from saealib.core.compiler.diagnostics import DiagnosticBag
from saealib.core.compiler.graph import IdentityRule, ReachabilityRule
from saealib.core.contracts import (
    MANY,
    ComponentContract,
    DataSpec,
    PartSpec,
    PortContract,
    PortDirection,
    PortSpec,
    StateContract,
)
from saealib.core.graph_builder import (
    StageNodeAdapter,
    build_component_graph,
    build_decomposed_component_graph,
)
from saealib.core.state import SURROGATES_DEFAULT
from saealib.execution.evaluator import SerialEvaluator
from saealib.optimizer import Optimizer
from saealib.pipeline import Pipeline, Stage
from saealib.problem import Problem
from saealib.stages import SurrogatePredictStage
from saealib.strategies.direct import DirectStrategy, SteadyStateStrategy
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.strategies.ps import PreSelectionStrategy


class _ContractComponent:
    """A no-op component sufficient for graph construction."""

    def __init__(self, *, exports: tuple = (), ports: dict | None = None) -> None:
        self._contract = ComponentContract(
            ports=ports or {}, state=StateContract(exports=exports)
        )

    def contract(self) -> ComponentContract:
        return self._contract


def _port(name: str, direction: PortDirection) -> PortSpec:
    return PortSpec(
        name=name,
        direction=direction,
        data=DataSpec(kind="Population"),
        cardinality=MANY,
    )


class _Provider:
    def __init__(self) -> None:
        self.algorithm = GA(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
            mutation=MutationUniform(prob_var=0.1),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        )
        self.evaluator = SerialEvaluator()
        self.surrogate_manager = _ContractComponent(
            exports=(SURROGATES_DEFAULT,),
            ports={
                "predictor": PortContract(
                    inputs=(_port("candidates", PortDirection.INPUT),),
                    outputs=(_port("prediction", PortDirection.OUTPUT),),
                )
            },
        )
        self.acquisition = _ContractComponent(
            ports={
                "acquisition": PortContract(
                    inputs=(_port("prediction", PortDirection.INPUT),),
                    outputs=(_port("scores", PortDirection.OUTPUT),),
                )
            }
        )
        self.cbmanager = None
        self.async_evaluation_scheduler = None
        self.evaluation_planner = None
        self.feedback_builder = None
        self.feedback_builder_explicit = False


def _strategies():
    return (
        DirectStrategy(),
        SteadyStateStrategy(),
        GenerationBasedStrategy(gen_ctrl=2),
        IndividualBasedStrategy(evaluation_ratio=0.5),
        PreSelectionStrategy(n_candidates=8, n_select=2),
    )


def _rule_diagnostics(rule, graph):
    result = rule.apply(
        RuleContext(
            graph=graph,
            compile_context=CompileContext(),
            diagnostics=DiagnosticBag(),
        )
    )
    return result.diagnostics


def test_each_strategy_builds_a_reachable_well_formed_graph():
    provider: Any = _Provider()

    for strategy in _strategies():
        graph = strategy.build_graph(provider)

        assert len(graph.well_formedness()) == 0
        assert len(_rule_diagnostics(IdentityRule(), graph)) == 0
        assert len(_rule_diagnostics(ReachabilityRule(), graph)) == 0
        assert len(graph.entry_points) == 1


def test_graph_nodes_match_actual_pipeline_and_preselection_has_no_top_k_stage():
    provider: Any = _Provider()

    for strategy in _strategies():
        pipeline = strategy.build_pipeline(provider)
        graph = strategy.build_graph(provider)

        assert [node.component.stage.name for node in graph.nodes] == [
            stage.name for stage in pipeline.stages
        ]
        assert all(
            node.component.stage.__class__.__name__ != "TopKSelectionStage"
            for node in graph.nodes
        )


def test_data_and_control_edges_are_distinct_and_archive_feedback_is_control_only():
    provider: Any = _Provider()
    graph = DirectStrategy().build_graph(provider)

    assert graph.data_edges
    assert graph.control_edges
    assert all(hasattr(edge, "source_port") for edge in graph.data_edges)
    assert not any(
        edge.source.component_id == "archive_update"
        and edge.target.component_id == "feedback"
        for edge in graph.data_edges
    )


def _resolve_edge_port(graph, endpoint, port_name, direction):
    node = graph.node_by_id(endpoint.component_id)
    contracts = (
        ((endpoint.role, node.contract.ports[endpoint.role]),)
        if endpoint.role is not None and endpoint.role in node.contract.ports
        else ()
        if endpoint.role is not None
        else tuple(node.contract.ports.items())
    )
    matches = [
        port
        for _, contract in contracts
        for port in (*contract.inputs, *contract.outputs)
        if port.name == port_name and port.direction is direction
    ]
    assert len(matches) == 1
    return matches[0]


def test_all_strategy_data_edges_resolve_to_declared_directional_ports():
    provider: Any = _Provider()

    common = {
        (
            "ask",
            "proposer",
            "genomes",
            "evaluation_plan",
            "evaluation_planner",
            "candidates",
        ),
        (
            "feedback",
            "feedback_builder",
            "feedback",
            "tell",
            "feedback_consumer",
            "offspring",
        ),
    }
    surrogate = {
        ("ask", "proposer", "genomes", "surrogate_predict", "predictor", "candidates"),
        (
            "surrogate_predict",
            "predictor",
            "prediction",
            "acquisition",
            "acquisition",
            "prediction",
        ),
        (
            "acquisition",
            "acquisition",
            "scores",
            "evaluation_plan",
            "evaluation_planner",
            "acquisition",
        ),
    }

    for strategy in _strategies():
        graph = strategy.build_graph(provider)
        expected = common | (
            surrogate
            if isinstance(strategy, (IndividualBasedStrategy, PreSelectionStrategy))
            else set()
        )
        assert {
            (
                edge.source.component_id,
                edge.source.role,
                edge.source_port,
                edge.target.component_id,
                edge.target.role,
                edge.target_port,
            )
            for edge in graph.data_edges
        } == expected
        for edge in graph.data_edges:
            assert edge.source.role is not None
            assert edge.target.role is not None
            _resolve_edge_port(
                graph, edge.source, edge.source_port, PortDirection.OUTPUT
            )
            _resolve_edge_port(
                graph, edge.target, edge.target_port, PortDirection.INPUT
            )


def test_control_edges_preserve_reachability_without_data_edges():
    provider: Any = _Provider()

    for strategy in _strategies():
        graph = strategy.build_graph(provider)
        control_only = graph.__class__(
            nodes=graph.nodes,
            control_edges=graph.control_edges,
            state_bindings=graph.state_bindings,
            entry_points=graph.entry_points,
        )
        assert len(_rule_diagnostics(ReachabilityRule(), control_only)) == 0


def test_adapter_composes_held_contract_without_declaring_stage_state():
    class _Stage(Stage):
        name = "custom"

        def __init__(self) -> None:
            super().__init__()
            self.component = _ContractComponent(exports=(SURROGATES_DEFAULT,))

        def execute(self, state):
            return state

    contract = StageNodeAdapter(_Stage()).contract()

    assert contract.state.reads == ()
    assert contract.state.writes == ()
    assert contract.state.exports == (SURROGATES_DEFAULT,)


def test_named_surrogates_get_distinct_node_qualified_state_bindings():
    class _NamedManager:
        def __init__(self) -> None:
            self.managers = {
                "cheap": _ContractComponent(exports=(SURROGATES_DEFAULT,)),
                "rich": _ContractComponent(exports=(SURROGATES_DEFAULT,)),
            }

        def contract(self) -> ComponentContract:
            return ComponentContract()

    stage = SurrogatePredictStage(cast(Any, _NamedManager()))
    graph = build_component_graph(Pipeline([stage]))
    names = {binding.state_key.name for binding in graph.state_bindings}

    assert names == {"surrogate_predict:cheap", "surrogate_predict:rich"}
    assert (
        SURROGATES_DEFAULT
        in _ContractComponent(exports=(SURROGATES_DEFAULT,)).contract().state.exports
    )


def test_multiple_surrogate_attributes_also_get_distinct_bindings():
    class _Manager:
        def __init__(self) -> None:
            self.cheap_surrogate = _ContractComponent(exports=(SURROGATES_DEFAULT,))
            self.rich_surrogate = _ContractComponent(exports=(SURROGATES_DEFAULT,))

        def contract(self) -> ComponentContract:
            return ComponentContract()

    graph = build_component_graph(
        Pipeline([SurrogatePredictStage(cast(Any, _Manager()))])
    )

    assert {binding.state_key.name for binding in graph.state_bindings} == {
        "surrogate_predict:cheap_surrogate",
        "surrogate_predict:rich_surrogate",
    }


def test_decomposed_surrogate_bindings_remain_node_qualified():
    class _NamedManager:
        def __init__(self) -> None:
            self.managers = {
                "cheap": _ContractComponent(exports=(SURROGATES_DEFAULT,)),
                "rich": _ContractComponent(exports=(SURROGATES_DEFAULT,)),
            }

        def contract(self) -> ComponentContract:
            return ComponentContract()

    graph = build_decomposed_component_graph(
        Pipeline([SurrogatePredictStage(cast(Any, _NamedManager()))])
    )

    assert {binding.state_key.name for binding in graph.state_bindings} == {
        "surrogate_predict:cheap",
        "surrogate_predict:rich",
    }
    assert all(
        binding.node.component_id != "surrogate_predict"
        for binding in graph.state_bindings
    )


def test_u2_decomposition_exposes_stage_contract_and_held_parts():
    class _Stage(Stage):
        name = "u2_stage"

        def __init__(self) -> None:
            super().__init__()
            self._component = _ContractComponent(exports=(SURROGATES_DEFAULT,))

        def contract(self) -> ComponentContract:
            return ComponentContract(
                state=StateContract(reads=(SURROGATES_DEFAULT,)),
                parts=(
                    # This is intentionally a direct Stage contract assertion;
                    # the part is still independently represented below.
                    PartSpec(name="component", contract=self._component.contract()),
                ),
            )

        def execute(self, state):
            return state

    graph = build_decomposed_component_graph(Pipeline([_Stage()]))
    stage = graph.node_by_id("u2_stage")
    part = graph.node_by_id("u2_stage__component")

    assert stage.contract.state.reads == (SURROGATES_DEFAULT,)
    assert part.contract.state.exports == (SURROGATES_DEFAULT,)
    assert type(stage.component).__name__ == "StageContractNodeAdapter"
    assert type(part.component).__name__ == "StagePartNodeAdapter"
    assert stage.component_id != part.component_id


def test_u2_feedback_is_cross_node_and_control_is_not_data():
    graph = build_decomposed_component_graph(
        Pipeline(list(DirectStrategy().build_pipeline(cast(Any, _Provider())).stages))
    )

    feedback_edges = [
        edge
        for edge in graph.data_edges
        if edge.source_port == "feedback" and edge.target_port == "offspring"
    ]
    assert len(feedback_edges) == 1
    edge = feedback_edges[0]
    assert edge.source.component_id != edge.target.component_id
    assert not any(
        item.source.component_id == item.target.component_id
        for item in graph.data_edges
    )

    control_only = graph.__class__(
        nodes=graph.nodes,
        control_edges=graph.control_edges,
        state_bindings=graph.state_bindings,
        entry_points=graph.entry_points,
    )
    assert len(_rule_diagnostics(ReachabilityRule(), control_only)) == 0

    partial_plan = Compiler().compile(
        graph,
        CompileContext(offered_runtime_capabilities=frozenset({"partial_feedback"})),
    )
    accumulator_edges = [
        item
        for item in partial_plan.graph.data_edges
        if item.source.component_id.startswith("__adapter_feedback_accumulator_")
        or item.target.component_id.startswith("__adapter_feedback_accumulator_")
    ]
    assert len(accumulator_edges) == 2
    assert (
        accumulator_edges[0].source.component_id
        != accumulator_edges[0].target.component_id
    )
    assert (
        accumulator_edges[1].source.component_id
        != accumulator_edges[1].target.component_id
    )
    assert {item.source.component_id for item in accumulator_edges} != {
        item.target.component_id for item in accumulator_edges
    }


def test_u2_five_strategy_graphs_compile_without_errors_or_self_loops():
    problem = Problem(
        func=lambda x: float(np.sum(np.asarray(x) ** 2)),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )

    for strategy in _strategies():
        optimizer = Optimizer(problem).set_strategy(strategy)
        optimizer._resolve_defaults()
        graph = build_decomposed_component_graph(strategy.build_pipeline(optimizer))
        plan = Compiler().compile(
            graph, CompileContext(space=problem.space, problem=problem)
        )
        errors = [item for item in plan.diagnostics if item.severity.value == "error"]
        assert errors == []
        assert all(
            edge.source.component_id != edge.target.component_id
            for edge in plan.graph.data_edges
        )
