"""Restricted runtime context exposed to graph-native components."""

# ruff: noqa: D102

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from saealib.core.state.keys import (
    ACQUISITION_RESULT,
    ARCHIVES_MAIN,
    ARCHIVES_PARETO,
    EVALUATED_OFFSPRING,
    EVALUATION_HANDLES,
    EVALUATION_NEW_IDS,
    EVALUATION_REQUEST,
    EVALUATION_UPDATE_NEW_IDS,
    EVALUATION_UPDATES,
    EVALUATIONS_COUNT,
    EVALUATIONS_OWNERS,
    EVALUATIONS_PENDING,
    EVALUATIONS_PLAN,
    EVALUATIONS_PLAN_STATE,
    EVALUATIONS_PLAN_UPDATES,
    FEEDBACK_ACCUMULATOR,
    FEEDBACK_RESULT,
    POPULATIONS_MAIN,
    PROPOSALS_CURRENT,
    PROPOSALS_ID_ALLOCATOR,
    PROPOSALS_OFFSPRING,
    RUNTIME_ASYNC_FATAL,
    RUNTIME_CANDIDATE_ID_ALLOCATOR,
    RUNTIME_GENERATION,
    RUNTIME_REQUEST_ID_ALLOCATOR,
    RUNTIME_RNG,
    SCORES,
    USER_DATA,
    StateKey,
)

if TYPE_CHECKING:
    from saealib.context import OptimizationState

__all__ = ["RuntimeContext"]


_CONTEXT_STATE_KEYS: dict[str, StateKey[object] | tuple[StateKey[object], ...]] = {
    "population": POPULATIONS_MAIN,
    "populations": (POPULATIONS_MAIN,),
    "archive": ARCHIVES_MAIN,
    "archives": (ARCHIVES_MAIN, ARCHIVES_PARETO),
    "pareto_archive": ARCHIVES_PARETO,
    "rng": RUNTIME_RNG,
    "candidate_id_allocator": RUNTIME_CANDIDATE_ID_ALLOCATOR,
    "proposal_id_allocator": PROPOSALS_ID_ALLOCATOR,
    "request_id_allocator": RUNTIME_REQUEST_ID_ALLOCATOR,
    "fe": EVALUATIONS_COUNT,
    "gen": RUNTIME_GENERATION,
    "offspring": PROPOSALS_OFFSPRING,
    "evaluated_offspring": EVALUATED_OFFSPRING,
    "scores": SCORES,
    "acquisition_result": ACQUISITION_RESULT,
    "evaluation_request": EVALUATION_REQUEST,
    "evaluation_plan": EVALUATIONS_PLAN,
    "evaluation_plan_state": EVALUATIONS_PLAN_STATE,
    "evaluation_updates": EVALUATION_UPDATES,
    "evaluation_plan_updates": EVALUATIONS_PLAN_UPDATES,
    "evaluation_update_new_ids": EVALUATION_UPDATE_NEW_IDS,
    "evaluation_new_ids": EVALUATION_NEW_IDS,
    "evaluation_handles": EVALUATION_HANDLES,
    "evaluation_owners": EVALUATIONS_OWNERS,
    "pending_evaluations": EVALUATIONS_PENDING,
    "pending_candidate_ids": EVALUATIONS_PENDING,
    "feedback_result": FEEDBACK_RESULT,
    "feedback_accumulator": FEEDBACK_ACCUMULATOR,
    "async_fatal": RUNTIME_ASYNC_FATAL,
    "data": USER_DATA,
    "proposal_id": PROPOSALS_CURRENT,
}


class RuntimeContext:
    """Read, service, and event capabilities for graph-native execution.

    This facade deliberately does not delegate arbitrary attributes to the
    underlying :class:`OptimizationState`.  State changes remain the graph's
    ``StatePatch`` responsibility; the runtime context only exposes the
    capabilities needed while a component executes.

    Parameters
    ----------
    state
        State owner used to resolve declared runtime capabilities.
    dispatch
        Optional event sink owned by the execution runtime.
    reads
        State keys whose context capabilities are available to the current
        component.  Directly constructed contexts may omit this restriction;
        runtimes bind it to the component contract.
    """

    _dispatch: Callable[[Any], None] | None
    _state: OptimizationState
    _reads: frozenset[StateKey[object]] | None

    __slots__ = ("_dispatch", "_reads", "_state")

    def __init__(
        self,
        state: OptimizationState,
        *,
        dispatch: Callable[[Any], None] | None = None,
        reads: Iterable[StateKey[object]] | None = None,
    ) -> None:
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_dispatch", dispatch)
        object.__setattr__(
            self,
            "_reads",
            None if reads is None else frozenset(reads),
        )

    def __getattribute__(self, name: str) -> Any:
        key = _CONTEXT_STATE_KEYS.get(name)
        if key is not None:
            reads = object.__getattribute__(self, "_reads")
            if reads is not None:
                required = key if isinstance(key, tuple) else (key,)
                if not all(item in reads for item in required):
                    raise AttributeError(
                        f"RuntimeContext capability {name!r} was not declared"
                    )
        return object.__getattribute__(self, name)

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
        return MappingProxyType({"main": self._state.population})

    @property
    def archive(self) -> Any:
        return self._state.archive

    @property
    def archives(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {"main": self._state.archive, "pareto": self._state.pareto_archive}
        )

    @property
    def pareto_archive(self) -> Any:
        return self._state.pareto_archive

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
