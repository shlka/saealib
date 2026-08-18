"""Tests for lifecycle verification and compile-time accumulation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA

from saealib.algorithms.pymoo_algorithm import PymooAlgorithm
from saealib.core.compiler import (
    CompileContext,
    Compiler,
    ComponentGraph,
    ComponentNode,
    DataEdge,
    DiagnosticBag,
    FeedbackAccumulatorRule,
    NodeRef,
    ResolutionResult,
    RuleContext,
)
from saealib.core.compiler.adapters import (
    DEFAULT_ADAPTER_REGISTRY,
    LosslessAdapterRule,
)
from saealib.core.contracts import (
    MANY,
    ComponentContract,
    DataSpec,
    FeedbackContract,
    LifecycleContract,
    PortContract,
    PortDirection,
    PortSpec,
)
from saealib.core.contracts.feedback import COMPLETE_BATCH, PARTIAL_ALLOWED
from saealib.core.state import OPTIMIZATION_STATE_INITIAL_KEYS
from saealib.execution.evaluator import SerialEvaluator
from saealib.execution.runtime import default_runtime_registry
from saealib.execution.scheduler import AsyncEvaluationScheduler
from saealib.islands import IslandModel
from saealib.optimizer import Optimizer
from saealib.problem import Problem
from saealib.strategies.direct import DirectStrategy
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel


@dataclass(frozen=True)
class _Endpoint:
    data: DataSpec
    direction: PortDirection
    completion: str | None = None

    def contract(self) -> ComponentContract:
        port = PortSpec(
            name="value",
            direction=self.direction,
            data=self.data,
            cardinality=MANY,
        )
        role = (
            PortContract(outputs=(port,))
            if self.direction is PortDirection.OUTPUT
            else PortContract(inputs=(port,))
        )
        feedback = (
            None
            if self.completion is None
            else FeedbackContract(
                accepted_channels=frozenset({"true"}),
                completion=self.completion,
            )
        )
        return ComponentContract(
            ports={"role": role},
            lifecycle=LifecycleContract(feedback=feedback),
        )


def _problem() -> Problem:
    return Problem(
        func=lambda x: float(np.sum(np.asarray(x) ** 2)),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )


def _feedback_graph(completion: str) -> ComponentGraph:
    return ComponentGraph(
        nodes=(
            ComponentNode(
                component_id="producer",
                component=_Endpoint(
                    data=DataSpec(kind="FeedbackBatch"),
                    direction=PortDirection.OUTPUT,
                ),
            ),
            ComponentNode(
                component_id="consumer",
                component=_Endpoint(
                    data=DataSpec(kind="FeedbackBatch"),
                    direction=PortDirection.INPUT,
                    completion=completion,
                ),
            ),
        ),
        data_edges=(
            DataEdge(
                source=NodeRef(component_id="producer", role="role"),
                target=NodeRef(component_id="consumer", role="role"),
                source_port="value",
                target_port="value",
            ),
        ),
        entry_points=(NodeRef(component_id="producer"),),
    )


def _compile_actual(optimizer: Optimizer):
    optimizer._resolve_defaults()
    graph = cast(Any, optimizer.strategy).build_graph(optimizer)
    plan = Compiler().compile(
        graph,
        CompileContext(
            space=optimizer.problem.space,
            problem=optimizer.problem,
            offered_runtime_capabilities=default_runtime_registry.offered_capabilities(
                optimizer
            ),
            initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
        ),
    )
    return graph, plan


def test_async_graph_inserts_accumulator_and_describes_it() -> None:
    optimizer = Optimizer(_problem()).set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(SerialEvaluator())
    )

    graph, plan = _compile_actual(optimizer)

    assert len(graph.data_edges) == 5
    assert not [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.severity.value == "error"
    ]
    assert [getattr(item, "adapter_name", None) for item in plan.inserted_adapters] == [
        "dense_numeric_view",
        "feedback_accumulator",
    ]
    assert "feedback_accumulator" in plan.describe()
    assert any(
        node.component_id.startswith("__adapter_feedback_accumulator_")
        for node in plan.graph.nodes
    )


def test_stage_internal_feedback_edges_are_claimed() -> None:
    optimizer = Optimizer(_problem()).set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(SerialEvaluator())
    )
    optimizer._resolve_defaults()
    graph = cast(Any, optimizer.strategy).build_graph(optimizer)
    result = FeedbackAccumulatorRule().apply(
        RuleContext(
            graph=graph,
            compile_context=CompileContext(
                space=optimizer.problem.space,
                problem=optimizer.problem,
                offered_runtime_capabilities=frozenset({"partial_feedback"}),
            ),
            diagnostics=DiagnosticBag(),
        )
    )

    new_edges = tuple(
        edge for edge in result.graph.data_edges if edge not in graph.data_edges
    )
    assert new_edges
    claimed_edges = {claim.key for claim in result.claims if claim.kind == "data_edge"}

    def edge_key(edge: DataEdge) -> str:
        def reference(ref: NodeRef) -> str:
            return ref.component_id + (f"[{ref.role}]" if ref.role else "")

        return (
            f"{reference(edge.source)}.{edge.source_port}->"
            f"{reference(edge.target)}.{edge.target_port}"
        )

    assert all(edge_key(edge) in claimed_edges for edge in new_edges)


def test_partial_allowed_consumer_does_not_get_accumulator() -> None:
    plan = Compiler().compile(
        _feedback_graph(PARTIAL_ALLOWED),
        CompileContext(offered_runtime_capabilities=frozenset({"partial_feedback"})),
    )

    assert not plan.inserted_adapters
    assert not any(
        node.component_id.startswith("__adapter_feedback_accumulator_")
        for node in plan.graph.nodes
    )


def test_accumulator_registry_matcher_requires_both_runtime_and_consumer() -> None:
    graph = _feedback_graph(COMPLETE_BATCH)
    producer = graph.node_by_id("producer")
    consumer = graph.node_by_id("consumer")
    source_port = producer.contract.ports["role"].outputs[0]
    target_port = consumer.contract.ports["role"].inputs[0]

    partial = DEFAULT_ADAPTER_REGISTRY.candidates(
        source_port.data,
        target_port.data,
        compile_context=CompileContext(
            offered_runtime_capabilities=frozenset({"partial_feedback"})
        ),
        source_node=producer,
        target_node=consumer,
        source_port=source_port,
        target_port=target_port,
        graph=graph,
    )
    assert tuple(item.name for item in partial) == ("feedback_accumulator",)

    plan = Compiler().compile(
        graph,
        CompileContext(offered_runtime_capabilities=frozenset({"partial_feedback"})),
    )
    assert [getattr(item, "adapter_name", None) for item in plan.inserted_adapters] == [
        "feedback_accumulator"
    ]
    assert not [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.severity.value == "error"
    ]

    allowed = _feedback_graph(PARTIAL_ALLOWED)
    partial_target = allowed.node_by_id("consumer")
    partial_candidates = DEFAULT_ADAPTER_REGISTRY.candidates(
        source_port.data,
        partial_target.contract.ports["role"].inputs[0].data,
        compile_context=CompileContext(
            offered_runtime_capabilities=frozenset({"partial_feedback"})
        ),
        source_node=allowed.node_by_id("producer"),
        target_node=partial_target,
        source_port=source_port,
        target_port=partial_target.contract.ports["role"].inputs[0],
        graph=allowed,
    )
    assert not partial_candidates


def test_pymoo_complete_batch_partial_runtime_keeps_j8_diagnostic() -> None:
    optimizer = Optimizer(_problem()).set_algorithm(
        PymooAlgorithm(GA(pop_size=4), allow_partial_tell=False)
    )
    optimizer.set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(SerialEvaluator())
    )
    optimizer._resolve_defaults()

    assert [diagnostic.code for diagnostic in optimizer.contract_diagnostics()] == [
        "pymoo_partial_feedback_unsupported"
    ]


def test_sync_graph_does_not_insert_accumulator() -> None:
    graph, plan = _compile_actual(Optimizer(_problem()))

    assert all(
        "partial_feedback" not in node.contract.execution.offered_runtime_capabilities
        for node in graph.nodes
    )
    assert not any(
        getattr(item, "adapter_name", None) == "feedback_accumulator"
        for item in plan.inserted_adapters
    )


def test_resolution_rule_order_is_confluent_for_lifecycle_and_adapters() -> None:
    optimizer = Optimizer(_problem()).set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(SerialEvaluator())
    )
    optimizer._resolve_defaults()
    graph = cast(Any, optimizer.strategy).build_graph(optimizer)
    context = CompileContext(
        space=optimizer.problem.space,
        problem=optimizer.problem,
        initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
    )

    signatures: list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = []
    for seed in range(8):
        rules = [FeedbackAccumulatorRule(), LosslessAdapterRule()]
        random.Random(seed).shuffle(rules)
        current = graph
        diagnostics = DiagnosticBag()
        for rule in rules:
            result = rule.apply(
                RuleContext(
                    graph=current,
                    compile_context=context,
                    diagnostics=diagnostics,
                )
            )
            assert isinstance(result, ResolutionResult)
            current = result.graph
            diagnostics.extend(result.diagnostics)
        signatures.append(
            (
                tuple(sorted(node.component_id for node in current.nodes)),
                tuple(sorted(str(edge) for edge in current.data_edges)),
                tuple(sorted(diagnostic.code for diagnostic in diagnostics)),
            )
        )
    assert all(signature == signatures[0] for signature in signatures)


def test_five_actual_configurations_have_zero_compile_errors() -> None:
    problem = _problem()
    configurations: dict[str, list[Optimizer]] = {
        "default": [Optimizer(problem)],
        "surrogate": [
            Optimizer(problem).set_surrogate(
                RBFSurrogate(GaussianKernel(), problem.dim)
            )
        ],
        "async": [
            Optimizer(problem).set_async_evaluation_scheduler(
                AsyncEvaluationScheduler(SerialEvaluator())
            )
        ],
        "island": [
            Optimizer(_problem()).set_strategy(DirectStrategy()),
            Optimizer(_problem()).set_strategy(DirectStrategy()),
        ],
        "pymoo": [
            Optimizer(problem)
            .set_algorithm(PymooAlgorithm(GA(pop_size=4)))
            .set_strategy(DirectStrategy())
        ],
    }
    IslandModel(configurations["island"])

    for optimizers in configurations.values():
        for optimizer in optimizers:
            _, plan = _compile_actual(optimizer)
            assert not [
                diagnostic
                for diagnostic in plan.diagnostics
                if diagnostic.severity.value == "error"
            ]
