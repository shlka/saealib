"""Focused tests for compiler-resolved service bindings."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from _algorithm_boundary import ask as algorithm_ask
from pymoo.algorithms.soo.nonconvex.ga import GA as PymooGA  # noqa: N811

from saealib import LHSInitializer, Optimizer, Termination, max_fe
from saealib.algorithms.ga import GA
from saealib.algorithms.pso import PSO
from saealib.algorithms.pymoo_algorithm import PymooAlgorithm
from saealib.context import OptimizationState
from saealib.core.compiler.compiler import ExecutablePlan
from saealib.core.compiler.graph import ComponentGraph, ComponentNode, NodeRef
from saealib.core.contracts import ComponentContract
from saealib.exceptions import ValidationError
from saealib.execution.initializer import RandomInitializer
from saealib.execution.runtime import PipelineRuntime, resolve_plan
from saealib.operators.crossover import CrossoverSBX
from saealib.operators.mutation import MutationPolynomial
from saealib.operators.selection import TournamentSelection, TruncationSelection
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem


def _state(extra_attrs: list[PopulationAttribute] | None = None) -> OptimizationState:
    problem = Problem(
        func=lambda x: float(np.sum(np.asarray(x) ** 2)),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )
    provider = MagicMock(seed=7)
    if extra_attrs:
        provider.algorithm.population_class = lambda attrs, init_capacity: Population(
            attrs=list(attrs) + extra_attrs, init_capacity=init_capacity
        )
    else:
        provider.algorithm.population_class = Population
    provider.algorithm.archive_class = Archive
    provider.algorithm.create_pareto_archive = lambda attrs, init_capacity, problem: (
        ParetoArchive(attrs=attrs, init_capacity=init_capacity)
    )
    provider.evaluator.evaluate_batch = lambda x_arr, problem: MagicMock(
        f=np.sum(x_arr**2, axis=1, keepdims=True),
        g=np.empty((len(x_arr), 0)),
        cv=np.zeros(len(x_arr)),
    )
    return RandomInitializer(4, 4, seed=7).initialize(provider, problem)


class _Component:
    def contract(self) -> ComponentContract:
        return ComponentContract()


def _plan(*services: object) -> ExecutablePlan:
    nodes = tuple(
        ComponentNode(
            component_id=f"node{index}",
            component=_Component(),
            resolved_services={"BoundsService": service},
        )
        for index, service in enumerate(services)
    )
    return ExecutablePlan(
        graph=ComponentGraph(
            nodes=nodes,
            entry_points=(NodeRef(component_id="node0"),),
        ),
        diagnostics=(),
        required_runtime_capabilities=frozenset(),
        active_rule_namespaces=frozenset(),
        active_rule_names=(),
        contract_snapshots=tuple((node.component_id, node.contract) for node in nodes),
    )


def test_runtime_binds_plan_reference_identity_and_replace_keeps_binding() -> None:
    state = _state()
    service = state.problem.space.services.require("BoundsService")
    PipelineRuntime().initialize(_plan(service), state)

    assert state.compiled_service("BoundsService") is service
    assert state.replace(data={"x": 1}).compiled_service("BoundsService") is service


def test_actual_compiled_plan_has_no_bounds_lookup_in_next_generation() -> None:
    problem = Problem(
        func=lambda x: float(np.sum(np.asarray(x) ** 2)),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )
    optimizer = (
        Optimizer(problem)
        .set_initializer(LHSInitializer(4, 4, seed=7))
        .set_termination(Termination(max_fe(100)))
    )
    registry = problem.space.services
    real_get, real_require = registry.get, registry.require
    get = MagicMock(side_effect=real_get)
    require = MagicMock(side_effect=real_require)
    registry.get = get  # type: ignore[assignment]
    registry.require = require  # type: ignore[assignment]

    iterator = optimizer.iterate()
    state = next(iterator)
    plan = resolve_plan(optimizer)
    bounds = registry.require("BoundsService")
    assert any(
        node.resolved_services.get("BoundsService") is bounds
        for node in plan.graph.nodes
    )
    assert state.compiled_service("BoundsService") is bounds

    get.reset_mock()
    require.reset_mock()
    next(iterator)
    assert not any(call.args[0] == "BoundsService" for call in get.call_args_list)
    assert not any(call.args[0] == "BoundsService" for call in require.call_args_list)
    iterator.close()


def test_registry_is_not_used_by_ga_pso_or_pymoo_bound_paths() -> None:
    state = _state(
        [
            PopulationAttribute("velocity", float, (2,), default=0.0),
            PopulationAttribute("pbest_x", float, (2,), default=np.nan),
            PopulationAttribute("pbest_f", float, (1,), default=np.nan),
            PopulationAttribute("pbest_cv", float, (), default=np.nan),
        ]
    )
    registry = state.problem.space.services
    require = MagicMock(side_effect=registry.require)
    registry.require = require  # type: ignore[assignment]

    state.compiled_service("BoundsService")
    require.reset_mock()
    ga = GA(
        crossover=CrossoverSBX(eta=20.0, prob=0.9),
        mutation=MutationPolynomial(eta=20.0, prob=0.1),
        parent_selection=TournamentSelection(tournament_size=2),
        survivor_selection=TruncationSelection(),
    )
    algorithm_ask(
        ga,
        state,
        cast(Any, SimpleNamespace(dispatch=lambda event: None)),
    )
    assert not require.called

    require.reset_mock()
    algorithm_ask(
        PSO(),
        state,
        cast(Any, SimpleNamespace(dispatch=lambda event: None)),
    )
    assert not require.called

    state = _state()
    registry = state.problem.space.services
    require = MagicMock(side_effect=registry.require)
    registry.require = require  # type: ignore[assignment]
    algorithm = PymooAlgorithm(PymooGA(pop_size=4))
    algorithm._build_pymoo_problem(state)
    assert not require.called


def test_missing_or_conflicting_binding_fails_without_fallback() -> None:
    state = object.__new__(OptimizationState)
    with pytest.raises(ValidationError, match="not bound"):
        state.compiled_service("BoundsService")

    first, second = object(), object()
    state.bind_compiled_services({"BoundsService": first})
    with pytest.raises(ValidationError, match="conflicting"):
        state.bind_compiled_services({"BoundsService": second})

    with pytest.raises(ValidationError, match="conflicting"):
        PipelineRuntime().initialize(_plan(first, second), state)


def test_binding_is_not_a_checkpoint_dataclass_field(tmp_path) -> None:
    state = _state()
    state.bind_compiled_services(
        {"BoundsService": state.problem.space.services.require("BoundsService")}
    )
    assert "_compiled_services" not in {item.name for item in fields(state)}
    path = tmp_path / "state.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)
    assert restored.compiled_service(
        "BoundsService"
    ) is state.problem.space.services.get("BoundsService")
