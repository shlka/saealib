"""Focused tests for the Phase 6 runtime extension boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest
from pymoo.algorithms.soo.nonconvex.ga import GA as PymooGA  # noqa: N811

from saealib.algorithms.pymoo_algorithm import PymooAlgorithm
from saealib.context import OptimizationState
from saealib.core.compiler.compiler import ExecutablePlan
from saealib.core.compiler.graph import ComponentGraph, ComponentNode
from saealib.core.contracts import ComponentContract, ExecutionContract
from saealib.core.runtime import (
    NodeResult,
    NodeStatus,
    RequestRecompile,
    RuntimeStep,
)
from saealib.core.state.patch import StatePatch
from saealib.exceptions import StalePlanError, ValidationError
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
from saealib.strategies.direct import DirectStrategy
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
    with pytest.raises(ValidationError, match="no registered runtime"):
        registry.create("other", cast(Any, object()))


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
    expected_plan_shapes = {"island": (11, 3), "pymoo": (11, 3)}

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


def test_added_provider_overrides_default_registrations_deterministically() -> None:
    from saealib.execution.runtime import default_runtime_registry

    selected = object()
    registry = RuntimeRegistry(default_runtime_registry.registrations())
    registry.register(
        RuntimeRegistration(
            name="custom",
            matches=lambda provider: provider == "sync",
            factory=cast(Any, lambda provider, plan: selected),
        )
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
