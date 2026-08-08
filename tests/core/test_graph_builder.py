"""Tests for the Phase 5 stage-to-graph bridge."""

from __future__ import annotations

from typing import Any, cast

from saealib import (
    GA,
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
)
from saealib.core.compiler import CompileContext, RuleContext
from saealib.core.compiler.diagnostics import DiagnosticBag
from saealib.core.compiler.graph import IdentityRule, ReachabilityRule
from saealib.core.contracts import ComponentContract, StateContract
from saealib.core.graph_builder import StageNodeAdapter, build_component_graph
from saealib.core.state import SURROGATES_DEFAULT
from saealib.execution.evaluator import SerialEvaluator
from saealib.pipeline import Pipeline, Stage
from saealib.stages import SurrogatePredictStage
from saealib.strategies.direct import DirectStrategy, SteadyStateStrategy
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.strategies.ps import PreSelectionStrategy


class _ContractComponent:
    """A no-op component sufficient for graph construction."""

    def __init__(self, *, exports: tuple = ()) -> None:
        self._contract = ComponentContract(state=StateContract(exports=exports))

    def contract(self) -> ComponentContract:
        return self._contract


class _Provider:
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
