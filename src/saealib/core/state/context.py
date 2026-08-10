"""Restricted runtime context exposed to graph-native components."""

# The facade's public surface is self-documenting through its explicit
# property names; individual forwarding properties do not need duplicate
# docstrings.
# ruff: noqa: D102

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from saealib.context import OptimizationState

__all__ = ["RuntimeContext"]


class RuntimeContext:
    """Read, service, and event capabilities for graph-native execution.

    This facade deliberately does not delegate arbitrary attributes to the
    underlying :class:`OptimizationState`.  State changes remain the graph's
    ``StatePatch`` responsibility; the runtime context only exposes the
    capabilities needed while a component executes.
    """

    __slots__ = ("_dispatch", "_state")

    def __init__(
        self,
        state: OptimizationState,
        *,
        dispatch: Callable[[object], None] | None = None,
    ) -> None:
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_dispatch", dispatch)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"RuntimeContext is read-only; cannot set {name!r}")

    @property
    def problem(self) -> Any:
        return self._state.problem

    @property
    def population(self) -> Any:
        return self._state.population

    @property
    def populations(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self._state.populations))

    @property
    def archive(self) -> Any:
        return self._state.archive

    @property
    def archives(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self._state.archives))

    @property
    def pareto_archive(self) -> Any:
        return self._state.pareto_archive

    # Runtime values used by graph-native algorithms and policies.
    @property
    def rng(self) -> Any:
        return self._state.rng

    @property
    def candidate_id_allocator(self) -> Any:
        return self._state.candidate_id_allocator

    @property
    def proposal_id_allocator(self) -> Any:
        return self._state.proposal_id_allocator

    @property
    def request_id_allocator(self) -> Any:
        return self._state.request_id_allocator

    @property
    def fe(self) -> int:
        return self._state.fe

    @property
    def gen(self) -> int:
        return self._state.gen

    @property
    def dim(self) -> int:
        return self._state.dim

    @property
    def n_obj(self) -> int:
        return self._state.n_obj

    @property
    def lb(self) -> Any:
        return self._state.lb

    @property
    def ub(self) -> Any:
        return self._state.ub

    @property
    def direction(self) -> Any:
        return self._state.direction

    @property
    def comparator(self) -> Any:
        return self._state.comparator

    @property
    def offspring(self) -> Any:
        return self._state.offspring

    @property
    def evaluated_offspring(self) -> Any:
        return self._state.evaluated_offspring

    @property
    def scores(self) -> Any:
        return self._state.scores

    @property
    def acquisition_result(self) -> Any:
        return self._state.acquisition_result

    @property
    def predictions(self) -> Any:
        return self._state.predictions

    @property
    def evaluation_request(self) -> Any:
        return self._state.evaluation_request

    @property
    def evaluation_plan(self) -> Any:
        return self._state.evaluation_plan

    @property
    def evaluation_plan_state(self) -> Any:
        return self._state.evaluation_plan_state

    @property
    def evaluation_updates(self) -> Any:
        return self._state.evaluation_updates

    @property
    def evaluation_plan_updates(self) -> Any:
        return self._state.evaluation_plan_updates

    @property
    def evaluation_update_new_ids(self) -> Any:
        return self._state.evaluation_update_new_ids

    @property
    def evaluation_new_ids(self) -> Any:
        return self._state.evaluation_new_ids

    @property
    def evaluation_handles(self) -> Any:
        return self._state.evaluation_handles

    @property
    def evaluation_owners(self) -> Any:
        return self._state.evaluation_owners

    @property
    def pending_evaluations(self) -> Any:
        return self._state.pending_evaluations

    @property
    def pending_candidate_ids(self) -> Any:
        return self._state.pending_candidate_ids

    @property
    def reserved_fe(self) -> int:
        return self._state.reserved_fe

    @property
    def reserved_cost(self) -> float:
        return self._state.reserved_cost

    @property
    def feedback_result(self) -> Any:
        return self._state.feedback_result

    @property
    def feedback_accumulator(self) -> Any:
        return self._state.feedback_accumulator

    @property
    def async_fatal(self) -> Any:
        return self._state.async_fatal

    @property
    def data(self) -> Mapping[str, Any]:
        return MappingProxyType(self._state.data)

    def compiled_service(self, name: str) -> object:
        """Return a service resolved by the compiled graph."""
        return self._state.compiled_service(name)

    def dispatch(self, event: object) -> None:
        """Dispatch an event through the owning runtime."""
        if self._dispatch is not None:
            self._dispatch(event)
