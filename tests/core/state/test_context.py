"""Tests for the restricted graph-native runtime context."""

import numpy as np
import pytest

from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.core.state import USER_DATA, RuntimeContext, StateStore
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem


def _state() -> OptimizationState:
    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(1,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
        PopulationAttribute(name="g", dtype=np.float64, shape=(0,)),
        PopulationAttribute(name="cv", dtype=np.float64, shape=()),
    ]
    problem = Problem(
        func=lambda x: np.array([x[0] ** 2]),
        dim=1,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0],
        ub=[1.0],
        comparator=SingleObjectiveComparator(),
    )
    population = Population(attrs, init_capacity=1)
    archive = Archive(attrs, init_capacity=1)
    pareto_archive = ParetoArchive(attrs, init_capacity=1, direction=np.array([-1.0]))
    return OptimizationState(
        problem=problem,
        population=population,
        archive=archive,
        pareto_archive=pareto_archive,
        data={"marker": 1},
    )


def test_runtime_context_restricts_graph_native_context_capabilities() -> None:
    state = _state()
    service = object()
    state.bind_compiled_services({"example": service})
    events: list[object] = []
    context = RuntimeContext(state, dispatch=events.append)
    view = StateStore().view((), context=context)

    def graph_native_component(bound_view):
        return bound_view.context

    runtime_context = graph_native_component(view)

    assert isinstance(runtime_context, RuntimeContext)
    assert not isinstance(runtime_context, OptimizationState)
    assert runtime_context.problem is state.problem
    assert runtime_context.population is state.population
    assert runtime_context.compiled_service("example") is service
    runtime_context.dispatch("event")
    assert events == ["event"]

    forbidden = (
        "set_state",
        "replace",
        "get_state",
        "bind_compiled_services",
        "_store",
    )
    for name in forbidden:
        assert not hasattr(runtime_context, name)
    with pytest.raises(AttributeError):
        runtime_context.set_state  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        runtime_context.fe = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        runtime_context.data["new"] = 2  # type: ignore[index]


def test_runtime_context_state_capabilities_follow_declared_reads() -> None:
    state = _state()
    context = RuntimeContext(state, reads=(USER_DATA,))

    assert context.data == state.data
    with pytest.raises(AttributeError, match="population"):
        context.population
