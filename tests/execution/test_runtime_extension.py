"""Focused tests for the Phase 6 runtime extension boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest
from pymoo.algorithms.soo.nonconvex.ga import GA as PymooGA  # noqa: N811

from saealib.algorithms.pymoo_algorithm import PymooAlgorithm
from saealib.context import OptimizationState
from saealib.core.compiler.compiler import CompileContext, Compiler, ExecutablePlan
from saealib.core.compiler.graph import ComponentGraph, ComponentNode
from saealib.core.contracts import ComponentContract, ExecutionContract
from saealib.core.runtime import (
    NodeResult,
    NodeStatus,
    RequestRecompile,
    RuntimeStep,
)
from saealib.core.state import OPTIMIZATION_STATE_INITIAL_KEYS
from saealib.core.state.patch import StatePatch
from saealib.exceptions import ConfigurationError, StalePlanError, ValidationError
from saealib.execution import AsyncEvaluationScheduler, SerialEvaluator
from saealib.execution.runtime import (
    AsyncPipelineRuntime,
    PipelineRuntime,
    RuntimeRegistration,
    RuntimeRegistry,
    create_runtime,
)
from saealib.islands import IslandModel
from saealib.optimizer import Optimizer
from saealib.problem import Problem
from saealib.stages import EvaluationPlanStage
from saealib.strategies.direct import DirectStrategy, SteadyStateStrategy
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.strategies.ps import PreSelectionStrategy
from saealib.surrogate.rbf import RBFSurrogate, gaussian_kernel


def _state() -> OptimizationState:
    return object.__new__(OptimizationState)


def _problem() -> Problem:
    return Problem(
        func=lambda x: np.sum(np.asarray(x) ** 2, axis=-1, keepdims=True),
        dim=3,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0] * 3,
        ub=[1.0] * 3,
    )


@dataclass
class _Node:
    capabilities: tuple[str, ...] = ()
    reads: int = 0

    def contract(self) -> ComponentContract:
        self.reads += 1
        return ComponentContract(
            execution=ExecutionContract(required_runtime_capabilities=self.capabilities)
        )


def _plan(*nodes: ComponentNode) -> ExecutablePlan:
    return ExecutablePlan(
        graph=ComponentGraph(nodes=nodes),
        diagnostics=(),
        required_runtime_capabilities=frozenset(
            capability
            for node in nodes
            for capability in node.contract.execution.required_runtime_capabilities
        ),
        active_rule_namespaces=frozenset(),
        active_rule_names=(),
        contract_snapshots=tuple((node.component_id, node.contract) for node in nodes),
    )


def test_runtime_registry_selects_an_added_provider_without_consumer_changes() -> None:
    registry = RuntimeRegistry()
    selected = object()
    registry.register(
        RuntimeRegistration(
            name="test",
            matches=lambda provider: provider == "test",
            factory=cast(Any, lambda provider, plan: selected),
        )
    )

    assert registry.create("test", cast(Any, object())) is selected
    with pytest.raises(
        ConfigurationError, match=r"no registered runtime.*registered runtime names"
    ):
        registry.create("other", cast(Any, object()))


def test_runtime_registry_rejects_multiple_matching_providers() -> None:
    registry = RuntimeRegistry(
        (
            RuntimeRegistration(
                name="first",
                matches=lambda provider: provider == "shared",
                factory=cast(Any, lambda provider, plan: object()),
            ),
            RuntimeRegistration(
                name="second",
                matches=lambda provider: provider == "shared",
                factory=cast(Any, lambda provider, plan: object()),
            ),
        )
    )

    with pytest.raises(
        ConfigurationError,
        match=r"multiple registered runtimes.*'first'.*'second'",
    ):
        registry.create("shared", cast(Any, object()))


def test_runtime_registry_replace_leaves_exactly_one_match() -> None:
    remaining = object()
    registry = RuntimeRegistry(
        (
            RuntimeRegistration(
                name="first",
                matches=lambda provider: provider == "shared",
                factory=cast(Any, lambda provider, plan: remaining),
            ),
            RuntimeRegistration(
                name="second",
                matches=lambda provider: provider == "shared",
                factory=cast(Any, lambda provider, plan: object()),
            ),
        )
    )
    registry.replace(
        "second",
        RuntimeRegistration(
            name="second",
            matches=lambda provider: False,
            factory=cast(Any, lambda provider, plan: object()),
        ),
    )

    assert registry.create("shared", cast(Any, object())) is remaining


def test_default_runtime_capability_offers_are_owned_by_runtime_providers() -> None:
    from saealib.execution.runtime import default_runtime_registry

    async_optimizer = Optimizer(_problem()).set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(SerialEvaluator())
    )
    sync_optimizer = Optimizer(_problem())

    assert AsyncPipelineRuntime.capabilities == frozenset({"partial_feedback"})
    assert PipelineRuntime.capabilities == frozenset()
    assert default_runtime_registry.offered_capabilities(async_optimizer) == frozenset(
        {"partial_feedback"}
    )
    assert default_runtime_registry.offered_capabilities(sync_optimizer) == frozenset()

    partial_sync = Optimizer(_problem()).set_algorithm(
        PymooAlgorithm(PymooGA(pop_size=4), allow_partial_tell=True)
    )
    assert default_runtime_registry.offered_capabilities(partial_sync) == frozenset(
        {"partial_feedback"}
    )
    partial_sync.set_strategy(DirectStrategy(n_offspring=4))
    partial_sync._resolve_defaults()
    plan = partial_sync._compile_plan()
    assert plan is not None
    runtime = create_runtime(partial_sync)
    initializer = partial_sync.initializer
    assert initializer is not None
    state = initializer.initialize(partial_sync, partial_sync.problem)
    runtime.initialize(plan, state)


def test_runtime_registration_rejects_non_callable_capability_provider() -> None:
    with pytest.raises(ValidationError, match="capability_provider"):
        RuntimeRegistration(
            name="invalid",
            matches=lambda optimizer: True,
            factory=cast(Any, lambda optimizer, plan: object()),
            capability_provider=cast(Any, object()),
        )


def test_real_strategy_graphs_compile_and_initialize_selected_runtimes() -> None:
    island_optimizers = [
        Optimizer(_problem()).set_strategy(DirectStrategy(n_offspring=4)),
        Optimizer(_problem()).set_strategy(DirectStrategy(n_offspring=4)),
    ]
    for optimizer in island_optimizers:
        optimizer._resolve_defaults()
    island_model = IslandModel(island_optimizers, migration_interval=0)
    configurations = {
        "default": [Optimizer(_problem())],
        "surrogate": [
            Optimizer(_problem()).set_surrogate(RBFSurrogate(gaussian_kernel, dim=3))
        ],
        "async": [
            Optimizer(_problem()).set_async_evaluation_scheduler(
                AsyncEvaluationScheduler(SerialEvaluator(), max_pending=2)
            )
        ],
        # Construct the real model so this coverage includes its compatibility
        # and optimizer ownership wiring, then compile each owned strategy graph.
        "island": island_model.optimizers,
        "pymoo": [
            Optimizer(_problem())
            .set_algorithm(PymooAlgorithm(PymooGA(pop_size=4)))
            .set_strategy(DirectStrategy(n_offspring=4))
        ],
    }
    expected_plan_shapes = {"island": (34, 3), "pymoo": (18, 3)}

    for name, optimizers in configurations.items():
        for optimizer in optimizers:
            optimizer._resolve_defaults()
            plan = optimizer._compile_plan()
            assert plan is not None, name
            assert not [
                diagnostic
                for diagnostic in plan.diagnostics
                if diagnostic.severity.name == "ERROR"
            ], name
            if name in expected_plan_shapes:
                assert (
                    len(plan.graph.nodes),
                    len(plan.graph.data_edges),
                ) == expected_plan_shapes[name]
            runtime = create_runtime(optimizer)
            initializer = optimizer.initializer
            assert initializer is not None
            state = initializer.initialize(optimizer, optimizer.problem)
            session = runtime.initialize(plan, state)
            assert session.state is state, name
            if name == "async":
                assert isinstance(runtime, AsyncPipelineRuntime)
            else:
                assert isinstance(runtime, PipelineRuntime)


def _graph_signature(optimizer: Optimizer) -> tuple[object, ...]:
    graph = cast(Any, optimizer.strategy).build_graph(optimizer)
    return (
        tuple(node.component_id for node in graph.nodes),
        tuple(
            (
                edge.source.component_id,
                edge.source.role,
                edge.source_port,
                edge.target.component_id,
                edge.target.role,
                edge.target_port,
            )
            for edge in graph.data_edges
        ),
        tuple(
            (edge.source.component_id, edge.target.component_id)
            for edge in graph.control_edges
        ),
    )


def test_strategy_graph_signatures_are_runtime_neutral() -> None:
    strategy_factories = (
        lambda: DirectStrategy(n_offspring=4),
        SteadyStateStrategy,
        lambda: GenerationBasedStrategy(gen_ctrl=2),
        lambda: IndividualBasedStrategy(evaluation_ratio=0.5),
        lambda: PreSelectionStrategy(n_candidates=8, n_select=2),
    )

    for make_strategy in strategy_factories:
        sync = Optimizer(_problem()).set_strategy(make_strategy())
        async_optimizer = Optimizer(_problem()).set_strategy(make_strategy())
        async_optimizer.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(SerialEvaluator())
        )
        sync._resolve_defaults()
        async_optimizer._resolve_defaults()

        assert _graph_signature(sync) == _graph_signature(async_optimizer)


def test_five_strategy_graphs_compile_with_runtime_neutral_topology() -> None:
    strategy_factories = (
        lambda: DirectStrategy(n_offspring=4),
        SteadyStateStrategy,
        lambda: GenerationBasedStrategy(gen_ctrl=2),
        lambda: IndividualBasedStrategy(evaluation_ratio=0.5),
        lambda: PreSelectionStrategy(n_candidates=8, n_select=2),
    )

    for make_strategy in strategy_factories:
        sync = Optimizer(_problem()).set_strategy(make_strategy())
        asynchronous = Optimizer(_problem()).set_strategy(make_strategy())
        asynchronous.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(SerialEvaluator())
        )
        sync._resolve_defaults()
        asynchronous._resolve_defaults()
        sync_graph = cast(Any, sync.strategy).build_graph(sync)
        async_graph = cast(Any, asynchronous.strategy).build_graph(asynchronous)

        assert _graph_signature(sync) == _graph_signature(asynchronous)
        for optimizer, graph in ((sync, sync_graph), (asynchronous, async_graph)):
            plan = Compiler().compile(
                graph,
                CompileContext(
                    space=optimizer.problem.space,
                    problem=optimizer.problem,
                    offered_runtime_capabilities=(
                        frozenset({"partial_feedback"})
                        if optimizer.async_evaluation_scheduler is not None
                        else frozenset()
                    ),
                    initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
                ),
            )
            assert [
                diagnostic
                for diagnostic in plan.diagnostics
                if diagnostic.severity.name == "ERROR"
            ] == []


def test_async_runtime_uses_canonical_sync_graph_contract() -> None:
    optimizer = Optimizer(_problem()).set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(SerialEvaluator())
    )
    optimizer._resolve_defaults()
    graph = cast(Any, optimizer.strategy).build_graph(optimizer)
    node = graph.node_by_id("evaluation_plan")
    stage = node.component.stage
    assert isinstance(stage, EvaluationPlanStage)
    assert all(
        "runtime" not in type(item.component.stage).__name__.lower()
        for item in graph.nodes
        if hasattr(item.component, "stage")
    )
    stage_ids = [
        item.component_id for item in graph.nodes if hasattr(item.component, "stage")
    ]
    assert stage_ids[-1] == "evaluation_acknowledge"


def test_replaced_default_registration_is_selected() -> None:
    from saealib.execution.runtime import default_runtime_registry

    selected = object()
    registry = RuntimeRegistry(default_runtime_registry.registrations())
    registry.replace(
        "sync",
        RuntimeRegistration(
            name="sync",
            matches=lambda provider: provider == "sync",
            factory=cast(Any, lambda provider, plan: selected),
        ),
    )

    assert registry.create("sync", cast(Any, object())) is selected
    with pytest.raises(ValidationError, match="already exists"):
        registry.register(registry.registrations()[-1])


def test_initialize_reads_each_node_contract_once_and_rejects_stale_plan() -> None:
    node = _Node()
    second = _Node()
    component_node = ComponentNode(component_id="node", component=node)
    second_node = ComponentNode(component_id="second", component=second)
    plan = _plan(component_node, second_node)
    # The plan snapshot is valid for the first initialize call.
    runtime = PipelineRuntime()
    runtime.initialize(plan, _state())
    assert node.reads == 2
    assert second.reads == 2

    node.capabilities = ("partial_feedback",)
    with pytest.raises(StalePlanError, match="stale_plan") as error:
        runtime.initialize(plan, _state())
    assert error.value.code == "stale_plan"
    assert node.reads == 3
    assert second.reads == 3


def test_runtime_capabilities_are_checked_as_a_required_subset() -> None:
    node = _Node(capabilities=("partial_feedback",))
    component_node = ComponentNode(component_id="node", component=node)
    plan = _plan(component_node)

    class LimitedRuntime(PipelineRuntime):
        capabilities = frozenset()

    runtime = LimitedRuntime()

    with pytest.raises(ValidationError, match="partial_feedback"):
        runtime.initialize(plan, _state())


def test_refusal_remains_a_step_outcome_and_recompile_is_step_boundary_data() -> None:
    # L6 owns the protocol boundary only. Node dispatch and command execution
    # remain Phase 7 work, so refusal is asserted as a normal RuntimeStep
    # outcome and RECOMPILE_REQUIRED is exposed for the next step boundary.
    step = RuntimeStep(
        state=_state(),
        refused_commands=(RequestRecompile(reason="policy"),),
    )
    assert step.refused_commands[0].reason == "policy"

    boundary = RuntimeStep(
        state=_state(),
        node_results=(
            NodeResult(
                patch=StatePatch(writes={}),
                status=NodeStatus.RECOMPILE_REQUIRED,
            ),
        ),
    )
    assert boundary.recompile_required
    assert boundary.refused_commands == ()
