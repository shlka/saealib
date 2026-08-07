"""OptimizationState: immutable-style state passed through the optimization pipeline."""

from __future__ import annotations

import dataclasses
import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote, unquote

import numpy as np

from saealib.core.state import (
    EVALUATIONS_COUNT,
    EVALUATIONS_OWNERS,
    EVALUATIONS_PLAN,
    EVALUATIONS_PLAN_STATE,
    EVALUATIONS_PLAN_UPDATES,
    PENDING_EVALUATIONS,
    RUNTIME_ASYNC_FATAL,
    RUNTIME_CANDIDATE_ID_ALLOCATOR,
    RUNTIME_GENERATION,
    RUNTIME_REQUEST_ID_ALLOCATOR,
    RUNTIME_RNG,
    USER_DATA,
    StateKey,
    StatePatch,
    StateStore,
)
from saealib.exceptions import CheckpointError, ValidationError
from saealib.identity import IDAllocator

if TYPE_CHECKING:
    from saealib.acquisition.base import AcquisitionResult
    from saealib.comparators import Comparator
    from saealib.execution.evaluator import (
        EvaluationHandle,
        EvaluationRequest,
        EvaluationUpdate,
        PendingEvaluation,
    )
    from saealib.policies.evaluation import EvaluationPlan
    from saealib.policies.feedback import FeedbackResult
    from saealib.population import Archive, ParetoArchive, Population
    from saealib.problem import Problem
    from saealib.surrogate.prediction import SurrogatePrediction


CURRENT_CHECKPOINT_SCHEMA_VERSION = 2
SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS = frozenset({1, 2})
_SAFE_EMPTY_PENDING = frozenset(
    pickle.dumps({}, protocol=protocol)
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
)
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_MISSING = object()

_STORE_FIELDS = {
    "rng": RUNTIME_RNG,
    "candidate_id_allocator": RUNTIME_CANDIDATE_ID_ALLOCATOR,
    "request_id_allocator": RUNTIME_REQUEST_ID_ALLOCATOR,
    "fe": EVALUATIONS_COUNT,
    "gen": RUNTIME_GENERATION,
    "evaluation_plan": EVALUATIONS_PLAN,
    "evaluation_plan_state": EVALUATIONS_PLAN_STATE,
    "evaluation_plan_updates": EVALUATIONS_PLAN_UPDATES,
    "evaluation_owners": EVALUATIONS_OWNERS,
    "pending_evaluations": PENDING_EVALUATIONS,
    "async_fatal": RUNTIME_ASYNC_FATAL,
    "data": USER_DATA,
}


def _collection_key(kind: str, name: str) -> StateKey[object]:
    return StateKey(namespace=kind, name=name, schema_version=1)


class _NamedCollection(dict[str, Any]):
    def __init__(self, kind: str, values: dict[str, Any]) -> None:
        self._kind = kind
        self._on_change: Any = None
        super().__init__()
        for name, value in values.items():
            self[name] = value

    def _bind(self, on_change: Any) -> None:
        self._on_change = on_change

    def _commit(self, values: dict[str, Any]) -> None:
        if self._on_change is not None:
            self._on_change(values)

    def _replace_local(self, values: dict[str, Any]) -> None:
        dict.clear(self)
        for name, value in values.items():
            dict.__setitem__(self, name, value)

    def __setitem__(self, name: str, value: Any) -> None:
        self._validate_entry(name, value)
        values = dict(self)
        values[name] = value
        self._commit(values)
        super().__setitem__(name, value)

    def _validate_entry(self, name: str, value: Any) -> None:
        _validate_collection_name(name)
        from saealib.population import Archive, ParetoArchive, Population

        if self._kind == "populations":
            valid = isinstance(value, Population)
        else:
            valid = (
                isinstance(value, (Archive, ParetoArchive))
                and (name != "main" or isinstance(value, Archive))
                and (name != "pareto" or isinstance(value, ParetoArchive))
            )
        if not valid:
            raise ValidationError(f"{self._kind}[{name!r}] has an invalid type")

    def __delitem__(self, name: str) -> None:
        _validate_collection_name(name)
        if (self._kind == "populations" and name == "main") or (
            self._kind == "archives" and name in {"main", "pareto"}
        ):
            raise ValidationError(f"cannot remove required {self._kind} entry {name!r}")
        values = dict(self)
        del values[name]
        self._commit(values)
        super().__delitem__(name)

    def clear(self) -> None:
        required = {"main"} if self._kind == "populations" else {"main", "pareto"}
        values = {name: value for name, value in self.items() if name in required}
        self._commit(values)
        self._replace_local(values)

    def pop(self, name: object, default: Any = _MISSING) -> Any:
        if not isinstance(name, str):
            raise ValidationError(f"invalid collection name: {name!r}")
        _validate_collection_name(name)
        required = {"main"} if self._kind == "populations" else {"main", "pareto"}
        if name in required:
            raise ValidationError(f"cannot remove required {self._kind} entry {name!r}")
        if name not in self:
            if default is _MISSING:
                raise KeyError(name)
            return default
        value = self[name]
        values = dict(self)
        del values[name]
        self._commit(values)
        super().__delitem__(name)
        return value

    def popitem(self) -> tuple[str, Any]:
        required = {"main"} if self._kind == "populations" else {"main", "pareto"}
        for name in reversed(list(self)):
            if name not in required:
                value = self[name]
                values = dict(self)
                del values[name]
                self._commit(values)
                super().__delitem__(name)
                return name, value
        raise ValidationError(f"cannot remove required {self._kind} entries")

    def setdefault(self, name: str, default: Any = None) -> Any:
        _validate_collection_name(name)
        if name in self:
            self._validate_entry(name, self[name])
            return self[name]
        self[name] = default
        return default

    def update(self, other: Any = (), /, **kwargs: Any) -> None:
        values = dict(other, **kwargs)
        for name, value in values.items():
            self._validate_entry(name, value)
        merged = dict(self)
        merged.update(values)
        self._commit(merged)
        for name, value in values.items():
            super().__setitem__(name, value)

    def __ior__(self, other: Any):
        self.update(other)
        return self

    def __reduce__(self):
        return (type(self), (self._kind, dict(self)))


def _validate_collection_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or name.startswith("_")
        or "//" in name
        or not _NAME_RE.fullmatch(name)
    ):
        raise ValidationError(f"invalid collection name: {name!r}")


def _encoded_name(name: str) -> str:
    return quote(name, safe="")


def _decoded_name(value: str) -> str:
    name = unquote(value)
    _validate_collection_name(name)
    return name


@dataclass(frozen=True)
class EvaluationPlanState:
    """Checkpointable lifecycle state for a multi-request evaluation plan."""

    submitted: tuple[int, ...] = ()
    completed: tuple[int, ...] = ()
    acknowledged: tuple[int, ...] = ()
    deferred: tuple[int, ...] = ()
    continuation: Any = None
    feedback: Any = None

    def __post_init__(self) -> None:
        """Validate request ID collections."""
        for name in ("submitted", "completed", "acknowledged", "deferred"):
            values = tuple(int(value) for value in getattr(self, name))
            if len(values) != len(set(values)):
                raise ValidationError(f"evaluation plan state {name} has duplicates")
            object.__setattr__(self, name, values)
        submitted = set(self.submitted)
        completed = set(self.completed)
        acknowledged = set(self.acknowledged)
        deferred = set(self.deferred)
        if not completed <= submitted:
            raise ValidationError("completed requests must be submitted")
        if not acknowledged <= completed:
            raise ValidationError("acknowledged requests must be completed")
        if deferred & (submitted | completed | acknowledged):
            raise ValidationError("deferred requests cannot be terminal or submitted")


def _validate_plan_state(
    plan: EvaluationPlan | None, plan_state: EvaluationPlanState | None
) -> None:
    if plan is None:
        return
    if plan_state is None:
        raise ValidationError(
            "evaluation plan state is required while a plan is active"
        )
    plan_ids = {int(request.request_id) for request in plan.requests}
    state_ids = (
        set(plan_state.submitted)
        | set(plan_state.completed)
        | set(plan_state.acknowledged)
        | set(plan_state.deferred)
    )
    if not state_ids <= plan_ids:
        raise ValidationError("evaluation plan state references an unknown request")


@dataclass(init=False)
class OptimizationState:
    """
    Optimization State class.

    Holds the state of the optimization process.  Passed through the pipeline
    as a value object; use :meth:`replace` to produce an updated copy rather
    than mutating fields directly.

    Controlled mutable exceptions (documented):

    - ``archive`` is append-only; copying on every evaluation would incur
      O(FE²) cost, so in-place appends are permitted.
    - ``rng`` advances its internal state as a controlled side effect.
    - ``candidate_id_allocator`` / ``request_id_allocator`` advance their
      internal counters as a controlled side effect, identically to ``rng``.

    Attributes
    ----------
    problem : Problem
        Problem instance.
    population : Population
        Population instance.
    archive : Archive
        Archive instance.  Append-only in-place mutation is permitted.
    pareto_archive : ParetoArchive
        Pareto archive instance.
    rng : np.random.Generator
        Random number generator.  Advances its state as a side effect.
    candidate_id_allocator : IDAllocator
        Allocates stable, unique int64 candidate IDs.  Advances its state as
        a side effect.
    request_id_allocator : IDAllocator
        Allocates stable, unique int64 evaluation request IDs.  Advances its
        state as a side effect.
    fe : int
        Number of function evaluations.
    gen : int
        Number of generations.
    offspring : Population or None
        Candidate population produced by the current generation's ask step.
        Set by :class:`~saealib.stages.AskStage`; consumed and updated by
        downstream stages.
    evaluated_offspring : Population or None
        Sub-population that has received true objective values.
        Set by :class:`~saealib.stages.TrueEvaluationStage`; consumed by
        :class:`~saealib.stages.ArchiveUpdateStage`.
    scores : np.ndarray or None
        Acquisition scores for ``offspring``, shape ``(n_candidates,)``.
        Set by :class:`~saealib.stages.AcquisitionStage`.
    acquisition_result : AcquisitionResult or None
        Complete acquisition output, including artifacts needed by planners.
    predictions : SurrogatePrediction or None
        Batched surrogate prediction covering every row of ``offspring``.
        Set by :class:`~saealib.stages.SurrogatePredictStage`.
    pending_candidate_ids : np.ndarray
        Unique candidate IDs derived from ``pending_evaluations``.
    reserved_fe : int
        Number of candidates represented by ``pending_evaluations``.
    reserved_cost : float
        Total estimated cost represented by ``pending_evaluations``.
    async_fatal : dict[str, Any] or None
        Cross-node asynchronous evaluation failure signal.
    data : dict[str, Any]
        User-extensible key-value store.  Custom stages and callbacks may
        store arbitrary values here.  Use ``state.replace(data={**state.data,
        "key": value})`` — never mutate this dict in place.
    """

    problem: Problem

    populations: dict[str, Population]
    archives: dict[str, Archive | ParetoArchive]
    rng: np.random.Generator
    candidate_id_allocator: IDAllocator = field(default_factory=IDAllocator)
    request_id_allocator: IDAllocator = field(default_factory=IDAllocator)

    fe: int = 0
    gen: int = 0

    # Pipeline stage data (typed)
    offspring: Population | None = None
    evaluated_offspring: Population | None = None
    scores: np.ndarray | None = None
    acquisition_result: AcquisitionResult | None = None
    predictions: SurrogatePrediction | None = None
    evaluation_request: EvaluationRequest | None = None
    evaluation_plan: EvaluationPlan | None = None
    evaluation_plan_state: EvaluationPlanState | None = None
    evaluation_updates: list[EvaluationUpdate] = field(default_factory=list)
    evaluation_plan_updates: dict[int, list[EvaluationUpdate]] = field(
        default_factory=dict
    )
    evaluation_update_new_ids: list[np.ndarray] = field(default_factory=list)
    evaluation_new_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    evaluation_handles: dict[int, EvaluationHandle] = field(default_factory=dict)
    evaluation_owners: dict[int, Population] = field(default_factory=dict)
    pending_evaluations: dict[int, PendingEvaluation] = field(default_factory=dict)
    feedback_result: FeedbackResult | None = None
    async_fatal: dict[str, Any] | None = None

    # User-extensible data
    data: dict[str, Any] = field(default_factory=dict)

    _custom_state: dict[StateKey, object] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __getattr__(self, name: str) -> Any:
        """Read store-backed fields only when ordinary lookup misses."""
        if name == "populations":
            try:
                return object.__getattribute__(self, "_population_collection")
            except AttributeError as exc:
                raise AttributeError(name) from exc
        if name == "archives":
            try:
                return object.__getattribute__(self, "_archive_collection")
            except AttributeError as exc:
                raise AttributeError(name) from exc

        key = _STORE_FIELDS.get(name)
        if key is None:
            raise AttributeError(name)

        try:
            store = object.__getattribute__(self, "_store")
        except AttributeError:
            try:
                return object.__getattribute__(self, "_pending_" + name)
            except AttributeError as exc:
                raise AttributeError(name) from exc
        return store.get(key)

    def __setattr__(self, name: str, value: Any) -> None:
        """Patch store-backed fields instead of exposing store moves."""
        if name in {"populations", "archives"}:
            try:
                object.__getattribute__(self, "_store")
            except AttributeError:
                object.__setattr__(self, "_pending_" + name, value)
                return
            self._replace_collection(name, value)
            return
        key = _STORE_FIELDS.get(name)
        if key is not None:
            try:
                object.__getattribute__(self, "_store")
            except AttributeError:
                object.__setattr__(self, "_pending_" + name, value)
                return
            self._write_store(key, value)
            return
        object.__setattr__(self, name, value)

    def _write_store(self, key: StateKey, value: object) -> None:
        self._store = self._store.apply_patch(StatePatch(writes={key: value}))

    def _replace_collection(self, kind: str, values: dict[str, Any]) -> None:
        collection = _NamedCollection(kind, dict(values))
        required = {"main"} if kind == "populations" else {"main", "pareto"}
        if not required <= set(collection):
            raise ValidationError(f"{kind} must contain {sorted(required)!r}")
        writes = {
            _collection_key(kind, name): value for name, value in collection.items()
        }
        deletes = frozenset(
            key for key in self._store_keys(kind) if key.name not in collection
        )
        self._store = self._store.apply_patch(
            StatePatch(writes=writes, deletes=deletes)
        )
        collection._bind(lambda new: self._commit_collection(kind, new))
        object.__setattr__(
            self,
            "_"
            + ("population" if kind == "populations" else "archive")
            + "_collection",
            collection,
        )

    def _store_keys(self, namespace: str) -> tuple[StateKey, ...]:
        return tuple(key for key in self._store._values if key.namespace == namespace)

    def _commit_collection(self, kind: str, values: dict[str, Any]) -> None:
        collection = _NamedCollection(kind, values)
        required = {"main"} if kind == "populations" else {"main", "pareto"}
        if not required <= set(collection):
            raise ValidationError(f"{kind} must contain {sorted(required)!r}")
        current = {key.name: key for key in self._store_keys(kind)}
        writes = {
            current.get(name, _collection_key(kind, name)): value
            for name, value in collection.items()
        }
        deletes = frozenset(
            key for name, key in current.items() if name not in collection
        )
        self._store = self._store.apply_patch(
            StatePatch(writes=writes, deletes=deletes)
        )

    def get_state(self, key: StateKey[Any]) -> Any:
        """Return a custom or built-in value held by this state's store.

        ``key`` is validated by :class:`StateStore`; user components may use
        ``namespace="user"`` keys without adding a core constant.
        """
        return self._store.get(key)

    def set_state(self, key: StateKey[Any], value: Any) -> None:
        """Replace one custom or built-in value through the state store."""
        self._write_store(key, value)
        if key not in _STORE_FIELDS.values() and key.namespace not in {
            "populations",
            "archives",
        }:
            object.__setattr__(
                self, "_custom_state", {**self._custom_state, key: value}
            )

    def __init__(
        self,
        problem: Problem,
        population: Population | None = None,
        archive: Archive | None = None,
        pareto_archive: ParetoArchive | None = None,
        rng: np.random.Generator | None = None,
        *,
        populations: dict[str, Population] | None = None,
        archives: dict[str, Archive | ParetoArchive] | None = None,
        candidate_id_allocator: IDAllocator | None = None,
        request_id_allocator: IDAllocator | None = None,
        fe: int = 0,
        gen: int = 0,
        offspring: Population | None = None,
        evaluated_offspring: Population | None = None,
        scores: np.ndarray | None = None,
        acquisition_result: AcquisitionResult | None = None,
        predictions: SurrogatePrediction | None = None,
        evaluation_request: EvaluationRequest | None = None,
        evaluation_plan: EvaluationPlan | None = None,
        evaluation_plan_state: EvaluationPlanState | None = None,
        evaluation_updates: list[EvaluationUpdate] | None = None,
        evaluation_plan_updates: dict[int, list[EvaluationUpdate]] | None = None,
        evaluation_update_new_ids: list[np.ndarray] | None = None,
        evaluation_new_ids: np.ndarray | None = None,
        evaluation_handles: dict[int, EvaluationHandle] | None = None,
        evaluation_owners: dict[int, Population] | None = None,
        pending_evaluations: dict[int, PendingEvaluation] | None = None,
        feedback_result: FeedbackResult | None = None,
        async_fatal: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        _custom_state: dict[StateKey, object] | None = None,
    ) -> None:
        if populations is None:
            if population is None:
                raise ValidationError("population or populations is required")
            populations = {"main": population}
        elif population is not None and populations.get("main") is not population:
            raise ValidationError("population must alias populations['main']")
        if archives is None:
            if archive is None or pareto_archive is None:
                raise ValidationError("archive and pareto_archive are required")
            archives = {"main": archive, "pareto": pareto_archive}
        else:
            if archive is not None and archives.get("main") is not archive:
                raise ValidationError("archive must alias archives['main']")
            if (
                pareto_archive is not None
                and archives.get("pareto") is not pareto_archive
            ):
                raise ValidationError("pareto_archive must alias archives['pareto']")
        object.__setattr__(
            self, "_population_collection", _NamedCollection("populations", populations)
        )
        object.__setattr__(
            self, "_archive_collection", _NamedCollection("archives", archives)
        )
        if "main" not in self.populations:
            raise ValidationError("populations must contain 'main'")
        if "main" not in self.archives or "pareto" not in self.archives:
            raise ValidationError("archives must contain 'main' and 'pareto'")
        self.problem = problem
        self.rng = rng if rng is not None else np.random.default_rng()
        self.candidate_id_allocator = candidate_id_allocator or IDAllocator()
        self.request_id_allocator = request_id_allocator or IDAllocator()
        self.fe = fe
        self.gen = gen
        self.offspring = offspring
        self.evaluated_offspring = evaluated_offspring
        self.scores = scores
        self.acquisition_result = acquisition_result
        self.predictions = predictions
        self.evaluation_request = evaluation_request
        self.evaluation_plan = evaluation_plan
        self.evaluation_plan_state = evaluation_plan_state
        _validate_plan_state(evaluation_plan, evaluation_plan_state)
        if evaluation_plan is not None and evaluation_plan_updates is not None:
            plan_ids = {int(request.request_id) for request in evaluation_plan.requests}
            if not set(map(int, evaluation_plan_updates)) <= plan_ids:
                raise ValidationError(
                    "evaluation plan updates reference an unknown request"
                )
        self.evaluation_updates = (
            [] if evaluation_updates is None else evaluation_updates
        )
        self.evaluation_plan_updates = (
            {} if evaluation_plan_updates is None else evaluation_plan_updates
        )
        self.evaluation_update_new_ids = (
            [] if evaluation_update_new_ids is None else evaluation_update_new_ids
        )
        self.evaluation_new_ids = (
            np.empty(0, dtype=np.int64)
            if evaluation_new_ids is None
            else evaluation_new_ids
        )
        self.evaluation_handles = (
            {} if evaluation_handles is None else evaluation_handles
        )
        self.evaluation_owners = {} if evaluation_owners is None else evaluation_owners
        self.pending_evaluations = (
            {} if pending_evaluations is None else pending_evaluations
        )
        self.feedback_result = feedback_result
        self.async_fatal = async_fatal
        self.data = {} if data is None else data
        initial_store: dict[StateKey, object] = {
            _collection_key("populations", name): value
            for name, value in populations.items()
        }
        initial_store.update(
            {
                _collection_key("archives", name): value
                for name, value in archives.items()
            }
        )
        for name, key in _STORE_FIELDS.items():
            initial_store[key] = object.__getattribute__(self, "_pending_" + name)
        custom_state = {} if _custom_state is None else dict(_custom_state)
        initial_store.update(custom_state)
        object.__setattr__(self, "_store", StateStore(initial_store))
        object.__setattr__(self, "_custom_state", custom_state)
        self._population_collection._bind(
            lambda values: self._commit_collection("populations", values)
        )
        self._archive_collection._bind(
            lambda values: self._commit_collection("archives", values)
        )
        for name in (*_STORE_FIELDS, "populations", "archives"):
            self.__dict__.pop("_pending_" + name, None)

    @property
    def population(self) -> Population:
        """Return the main population."""
        return self.populations["main"]

    @population.setter
    def population(self, value: Population) -> None:
        self.populations["main"] = value

    @property
    def archive(self) -> Archive:
        """Return the main archive."""
        return cast(Any, self.archives["main"])

    @archive.setter
    def archive(self, value: Archive) -> None:
        self.archives["main"] = value

    @property
    def pareto_archive(self) -> ParetoArchive:
        """Return the Pareto archive."""
        return cast(Any, self.archives["pareto"])

    @pareto_archive.setter
    def pareto_archive(self, value: ParetoArchive) -> None:
        self.archives["pareto"] = value

    @property
    def pending_candidate_ids(self) -> np.ndarray:
        """Return unique candidate IDs reserved by pending evaluations."""
        values = [
            pending.request.candidate_ids
            for pending in self.pending_evaluations.values()
        ]
        return (
            np.unique(np.concatenate(values)).astype(np.int64, copy=False)
            if values
            else np.empty(0, dtype=np.int64)
        )

    @property
    def reserved_fe(self) -> int:
        """Return the number of candidates reserved by pending evaluations."""
        return sum(
            len(pending.request.candidate_ids)
            for pending in self.pending_evaluations.values()
        )

    @property
    def reserved_cost(self) -> float:
        """Return the total estimated cost reserved by pending evaluations."""
        from math import fsum

        return fsum(
            pending.reserved_cost for pending in self.pending_evaluations.values()
        )

    def __getstate__(self) -> dict[str, Any]:
        """Exclude runtime evaluation handles from serialized state."""
        state = {
            item.name: getattr(self, item.name)
            for item in dataclasses.fields(self)
            if item.init
        }
        state["evaluation_handles"] = {}
        state["evaluation_request"] = None
        state["evaluation_updates"] = []
        state["evaluation_update_new_ids"] = []
        return state

    def replace(self, **kwargs: Any) -> OptimizationState:
        """Return a new state with the given fields replaced.

        Parameters
        ----------
        **kwargs
            Field names and new values.

        Returns
        -------
        OptimizationState
        """
        legacy = {"population", "archive", "pareto_archive"}
        if any(key in kwargs for key in legacy):
            if "populations" in kwargs and "population" in kwargs:
                raise ValidationError(
                    "replace() cannot combine population and populations"
                )
            if "archives" in kwargs and legacy.intersection(kwargs) - {"population"}:
                raise ValidationError(
                    "replace() cannot combine archive aliases and archives"
                )
            populations = kwargs.pop("populations", self.populations)
            archives = kwargs.pop("archives", self.archives)
            if "population" in kwargs:
                populations = dict(self.populations)
                populations["main"] = kwargs.pop("population")
            if "archive" in kwargs or "pareto_archive" in kwargs:
                archives = dict(self.archives)
                if "archive" in kwargs:
                    archives["main"] = kwargs.pop("archive")
                if "pareto_archive" in kwargs:
                    archives["pareto"] = kwargs.pop("pareto_archive")
            kwargs["populations"] = populations
            kwargs["archives"] = archives
        return dataclasses.replace(self, **kwargs)

    def add_population(self, name: str, population: Population) -> None:
        """Add a named population."""
        self.populations[name] = population

    def add_archive(self, name: str, archive: Archive) -> None:
        """Add a named archive."""
        self.archives[name] = archive

    def get_population(self, name: str = "main") -> Population:
        """Return a named population."""
        return self.populations[name]

    def get_archive(self, name: str = "main") -> Archive:
        """Return a named archive."""
        return cast(Any, self.archives[name])

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore canonical collections from current or legacy pickle state."""
        if "populations" not in state:
            state["populations"] = {"main": state.pop("population")}
        if "archives" not in state:
            state["archives"] = {
                "main": state.pop("archive"),
                "pareto": state.pop("pareto_archive"),
            }
        self.__init__(**state)

    # ------------------------------------------------------------------
    # Convenience properties (delegate to problem)
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Return the dimension of the problem."""
        return self.problem.dim

    @property
    def n_obj(self) -> int:
        """Return the number of objectives."""
        return self.problem.n_obj

    @property
    def lb(self) -> np.ndarray:
        """Return the lower bounds of the problem."""
        return self.problem.lb

    @property
    def ub(self) -> np.ndarray:
        """Return the upper bounds of the problem."""
        return self.problem.ub

    @property
    def direction(self) -> np.ndarray:
        """Return the optimization direction of the problem."""
        return self.problem.direction

    @property
    def comparator(self) -> Comparator:
        """Return the comparator of the problem."""
        return self.problem.comparator

    def count_fe(self, count: int = 1) -> None:
        """
        Count function evaluations.

        Parameters
        ----------
        count : int
            Number of function evaluations to add.
        """
        self.fe += count

    def count_generation(self) -> None:
        """Count generations."""
        self.gen += 1

    # ------------------------------------------------------------------
    # Checkpoint: npz (best-effort reproducibility)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save optimization state to an npz file.

        Saves archive, population, Pareto archive arrays and the RNG state.
        Reproducibility is best-effort: bit-exact resume is expected within
        the same NumPy version and environment, but not guaranteed across
        versions.

        Only ``self.rng`` is saved. Components that own a private RNG spawned
        from ``self.rng`` (e.g. :class:`~saealib.comparators.NSGA3Comparator`'s
        niche tie-breaking generator) are not serialized; on resume, such a
        component gets a fresh spawn from the restored ``rng`` rather than a
        continuation of its own pre-checkpoint draw sequence.

        Parameters
        ----------
        path : str or Path
            Destination file path.  The ``.npz`` extension is added if absent.
        """
        if any(
            not pending.checkpointable and pending.fatal_error is None
            for pending in self.pending_evaluations.values()
        ):
            raise ValidationError(
                "cannot checkpoint while synchronous evaluations are pending"
            )
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".npz")

        save_dict: dict[str, np.ndarray] = {}
        manifest: dict[str, Any] = {
            "schema_version": CURRENT_CHECKPOINT_SCHEMA_VERSION,
            "populations": [],
            "archives": [],
            "offspring": None,
            "evaluation_owners": [],
        }
        for kind, collections in (
            ("population", self.populations),
            ("archive", self.archives),
        ):
            for name, collection in collections.items():
                descriptor = _collection_descriptor(kind, name, collection)
                manifest["populations" if kind == "population" else "archives"].append(
                    descriptor
                )
                encoded = _encoded_name(name)
                for attr_name, array in collection._data.items():
                    if array.dtype == object:
                        raise CheckpointError(
                            f"object dtype is not checkpointable: {kind} {name!r}"
                        )
                    save_dict[f"{kind}__{encoded}__{_encoded_name(attr_name)}"] = (
                        np.array(array[: len(collection)], copy=True)
                    )
        if self.offspring is not None:
            if any(
                self.offspring is population for population in self.populations.values()
            ):
                alias = next(
                    name
                    for name, population in self.populations.items()
                    if population is self.offspring
                )
                manifest["offspring"] = {"alias": alias}
            else:
                descriptor = _collection_descriptor(
                    "population", "offspring", self.offspring
                )
                manifest["offspring"] = descriptor
                for attr_name, array in self.offspring._data.items():
                    if array.dtype == object:
                        raise CheckpointError("object dtype is not checkpointable")
                    save_dict[f"offspring__{_encoded_name(attr_name)}"] = np.array(
                        array[: len(self.offspring)], copy=True
                    )

        for request_id, owner in self.evaluation_owners.items():
            alias = next(
                (
                    name
                    for name, population in self.populations.items()
                    if population is owner
                ),
                None,
            )
            if alias is not None:
                manifest["evaluation_owners"].append(
                    {"request_id": int(request_id), "alias": alias}
                )
                continue
            if owner is self.offspring:
                manifest["evaluation_owners"].append(
                    {"request_id": int(request_id), "offspring": True}
                )
                continue
            name = f"owner-{int(request_id)}"
            descriptor = _collection_descriptor("population", name, owner)
            manifest["evaluation_owners"].append(
                {"request_id": int(request_id), "descriptor": descriptor}
            )
            for attr_name, array in owner._data.items():
                if array.dtype == object:
                    raise CheckpointError("object dtype is not checkpointable")
                save_dict[
                    f"evaluation_owner__{int(request_id)}__{_encoded_name(attr_name)}"
                ] = np.array(array[: len(owner)], copy=True)

        save_dict["_manifest"] = _json_array(manifest)
        save_dict["_rng_state"] = _json_array(self.rng.bit_generator.state)
        save_dict["_fe"] = np.array(self.fe, dtype=np.int64)
        save_dict["_gen"] = np.array(self.gen, dtype=np.int64)
        save_dict["_checkpoint_schema_version"] = np.array(
            CURRENT_CHECKPOINT_SCHEMA_VERSION, dtype=np.int64
        )
        save_dict["_next_candidate_id"] = np.array(
            self.candidate_id_allocator.next_value, dtype=np.int64
        )
        save_dict["_next_request_id"] = np.array(
            self.request_id_allocator.next_value, dtype=np.int64
        )
        save_dict["_pending_evaluations"] = _json_array(
            [_pending_to_json(pending) for pending in self.pending_evaluations.values()]
        )
        save_dict["_feedback_result"] = _json_array(
            None
            if self.feedback_result is None
            else _feedback_to_json(self.feedback_result)
        )
        save_dict["_predictions"] = _json_array(
            None if self.predictions is None else _prediction_to_json(self.predictions)
        )
        save_dict["_evaluation_plan"] = _json_array(
            None
            if self.evaluation_plan is None
            else _plan_to_json(self.evaluation_plan)
        )
        save_dict["_evaluation_plan_state"] = _json_array(
            None
            if self.evaluation_plan_state is None
            else _plan_state_to_json(self.evaluation_plan_state)
        )
        save_dict["_evaluation_plan_updates"] = _json_array(
            {
                str(request_id): [_update_to_json(update) for update in updates]
                for request_id, updates in self.evaluation_plan_updates.items()
            }
        )
        save_dict["_async_fatal"] = _json_array(_json_safe(self.async_fatal))
        save_dict["_data"] = _json_array(_json_safe(self.data))
        np.savez(p, **cast(Any, save_dict))

    @classmethod
    def load(cls, path: str | Path, problem: Problem) -> OptimizationState:
        """
        Restore an OptimizationState from an npz checkpoint file.

        The returned state has ``data["resumed"] = True``.

        Parameters
        ----------
        path : str or Path
            Path to the npz file.  The ``.npz`` extension is added if absent.
        problem : Problem
            The problem instance to attach (must match the one used when saving).

        Returns
        -------
        OptimizationState
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".npz")

        try:
            data = np.load(p, allow_pickle=False)
            if "_checkpoint_schema_version" not in data.files:
                raise CheckpointError(
                    "Checkpoint is missing '_checkpoint_schema_version'"
                )
            schema_version = _scalar_int(data["_checkpoint_schema_version"])
            if schema_version not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS:
                raise CheckpointError(
                    f"Unsupported checkpoint schema version {schema_version}; "
                    "supported versions are "
                    f"{sorted(SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS)}"
                )
            if schema_version == 1:
                return _load_v1(cls, data, problem)
            return _load_v2(cls, data, problem)
        except CheckpointError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as exc:
            raise CheckpointError(f"Invalid checkpoint: {exc}") from exc


# Keep dataclass field metadata and generated replacement/pickle behavior while
# forcing store-backed fields through __getattr__. A class attribute left behind
# by a field default would satisfy ordinary lookup and silently shadow the store.
for _store_field_name in _STORE_FIELDS:
    if _store_field_name in vars(OptimizationState):
        delattr(OptimizationState, _store_field_name)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            raise CheckpointError("object dtype is not checkpointable")
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise CheckpointError("checkpoint metadata keys must be strings")
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise CheckpointError(f"non-serializable checkpoint value: {type(value).__name__}")


def _json_array(value: Any) -> np.ndarray:
    try:
        encoded = json.dumps(_json_safe(value), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"value is not JSON serializable: {exc}") from exc
    return np.frombuffer(encoded, dtype=np.uint8)


def _read_json(data: Any, key: str) -> Any:
    if key not in data.files:
        raise CheckpointError(f"Checkpoint is missing {key!r}")
    try:
        return json.loads(bytes(data[key]).decode())
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"Checkpoint key {key!r} is malformed") from exc


def _result_to_json(result: Any) -> dict[str, Any]:
    return {
        "f": _json_safe(result.f),
        "g": _json_safe(result.g),
        "cv": _json_safe(result.cv),
        "candidate_ids": _json_safe(result.candidate_ids),
        "cost": _json_safe(result.cost),
        "noise": _json_safe(result.noise),
        "outputs": _json_safe(result.outputs),
    }


def _result_from_json(value: Any) -> Any:
    from saealib.execution.evaluator import EvaluationResult

    if not isinstance(value, dict):
        raise CheckpointError("pending result is malformed")
    return EvaluationResult(
        np.asarray(value["f"], dtype=np.float64),
        np.asarray(value["g"], dtype=np.float64),
        np.asarray(value["cv"], dtype=np.float64),
        None
        if value.get("candidate_ids") is None
        else np.asarray(value["candidate_ids"], dtype=np.int64),
        None
        if value.get("cost") is None
        else np.asarray(value["cost"], dtype=np.float64),
        None
        if value.get("noise") is None
        else np.asarray(value["noise"], dtype=np.float64),
        {
            str(key): np.asarray(array, dtype=np.float64)
            for key, array in value.get("outputs", {}).items()
        },
    )


def _feedback_to_json(result: Any) -> dict[str, Any]:
    return {
        "candidate_ids": _json_safe(result.candidate_ids),
        "f": _json_safe(result.f),
        "g": _json_safe(result.g),
        "cv": _json_safe(result.cv),
        "evaluated_mask": _json_safe(result.evaluated_mask),
        "source": _json_safe(result.source),
        "artifacts": _json_safe(result.artifacts),
    }


def _feedback_from_json(value: Any) -> Any:
    from saealib.policies.feedback import FeedbackResult

    if not isinstance(value, dict):
        raise CheckpointError("feedback result is malformed")
    return FeedbackResult(
        np.asarray(value["candidate_ids"], dtype=np.int64),
        np.asarray(value["f"], dtype=np.float64),
        None if value.get("g") is None else np.asarray(value["g"], dtype=np.float64),
        None if value.get("cv") is None else np.asarray(value["cv"], dtype=np.float64),
        np.asarray(value["evaluated_mask"], dtype=bool),
        np.asarray(value["source"], dtype=np.uint8),
        {
            str(name): np.asarray(array)
            for name, array in value.get("artifacts", {}).items()
        },
    )


def _request_to_json(request: Any) -> dict[str, Any]:
    return {
        "request_id": int(request.request_id),
        "candidate_ids": _json_safe(request.candidate_ids),
        "x": _json_safe(request.x),
        "outputs": list(request.outputs),
        "metadata": _json_safe(dict(request.metadata)),
    }


def _request_from_json(value: Any) -> Any:
    from saealib.execution.evaluator import EvaluationRequest

    if not isinstance(value, dict):
        raise CheckpointError("evaluation request is malformed")
    try:
        return EvaluationRequest(
            np.int64(value["request_id"]),
            np.asarray(value["candidate_ids"], dtype=np.int64),
            np.asarray(value["x"], dtype=np.float64),
            tuple(value.get("outputs", ("f", "g", "cv"))),
            value.get("metadata", {}),
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise CheckpointError("evaluation request is malformed") from exc


def _plan_to_json(plan: Any) -> dict[str, Any]:
    return {
        "requests": [_request_to_json(request) for request in plan.requests],
        "completion_rule": _json_safe(plan.completion_rule),
        "continuation": _json_safe(plan.continuation),
        "artifacts": _json_safe(dict(plan.artifacts)),
    }


def _plan_from_json(value: Any) -> Any:
    from saealib.policies.evaluation import EvaluationPlan

    if not isinstance(value, dict) or not isinstance(value.get("requests"), list):
        raise CheckpointError("evaluation plan is malformed")
    try:
        return EvaluationPlan(
            tuple(_request_from_json(item) for item in value["requests"]),
            value.get("completion_rule"),
            value.get("continuation"),
            value.get("artifacts", {}),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise CheckpointError("evaluation plan is malformed") from exc


def _plan_state_to_json(value: EvaluationPlanState) -> dict[str, Any]:
    return {
        "submitted": list(value.submitted),
        "completed": list(value.completed),
        "acknowledged": list(value.acknowledged),
        "deferred": list(value.deferred),
        "continuation": _json_safe(value.continuation),
        "feedback": _json_safe(value.feedback),
    }


def _plan_state_from_json(value: Any) -> EvaluationPlanState:
    if not isinstance(value, dict):
        raise CheckpointError("evaluation plan state is malformed")
    try:
        return EvaluationPlanState(
            tuple(value.get("submitted", ())),
            tuple(value.get("completed", ())),
            tuple(value.get("acknowledged", ())),
            tuple(value.get("deferred", ())),
            value.get("continuation"),
            value.get("feedback"),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise CheckpointError("evaluation plan state is malformed") from exc


def _update_to_json(update: Any) -> dict[str, Any]:
    return {
        "request_id": int(update.request_id),
        "status": update.status.name,
        "candidate_ids": _json_safe(update.candidate_ids),
        "result": None if update.result is None else _result_to_json(update.result),
        "error": None
        if update.error is None
        else {
            "error_type": update.error.error_type,
            "message": update.error.message,
            "details": _json_safe(dict(update.error.details)),
        },
        "sequence": int(update.sequence),
    }


def _update_from_json(value: Any) -> Any:
    from saealib.execution.evaluator import (
        EvaluationErrorInfo,
        EvaluationStatus,
        EvaluationUpdate,
    )

    if not isinstance(value, dict):
        raise CheckpointError("evaluation plan update is malformed")
    error = value.get("error")
    try:
        return EvaluationUpdate(
            np.int64(value["request_id"]),
            EvaluationStatus[value["status"]],
            np.asarray(value["candidate_ids"], dtype=np.int64),
            None if value.get("result") is None else _result_from_json(value["result"]),
            None
            if error is None
            else EvaluationErrorInfo(
                error["error_type"], error["message"], error.get("details", {})
            ),
            int(value.get("sequence", 0)),
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise CheckpointError("evaluation plan update is malformed") from exc


def _prediction_to_json(prediction: Any) -> dict[str, Any]:
    return {
        "x": _json_safe(prediction.x),
        "label": _json_safe(prediction.label),
        "metadata": _json_safe(prediction.metadata),
        "channels": {
            name: {
                "value": _json_safe(channel.value),
                "std": _json_safe(channel.std),
                "covariance": _json_safe(channel.covariance),
                "samples": _json_safe(channel.samples),
                "metadata": _json_safe(channel.metadata),
            }
            for name, channel in prediction.channels.items()
        },
    }


def _prediction_from_json(value: Any) -> Any:
    from saealib.surrogate.prediction import PredictionChannel, SurrogatePrediction

    if not isinstance(value, dict) or not isinstance(value.get("channels"), dict):
        raise CheckpointError("prediction is malformed")
    channels = {}
    for name, channel in value["channels"].items():
        if not isinstance(channel, dict):
            raise CheckpointError("prediction channel is malformed")
        channels[name] = PredictionChannel(
            np.asarray(channel["value"], dtype=np.float64),
            None
            if channel.get("std") is None
            else np.asarray(channel["std"], dtype=np.float64),
            None
            if channel.get("covariance") is None
            else np.asarray(channel["covariance"], dtype=np.float64),
            None
            if channel.get("samples") is None
            else np.asarray(channel["samples"], dtype=np.float64),
            channel.get("metadata", {}),
        )
    return SurrogatePrediction(
        channels,
        None if value.get("x") is None else np.asarray(value["x"]),
        None if value.get("label") is None else np.asarray(value["label"]),
        value.get("metadata", {}),
    )


def _pending_to_json(pending: Any) -> dict[str, Any]:
    request = pending.request
    return {
        "request_id": int(request.request_id),
        "candidate_ids": _json_safe(request.candidate_ids),
        "x": _json_safe(request.x),
        "outputs": list(request.outputs),
        "metadata": _json_safe(dict(request.metadata)),
        "status": pending.status.name,
        "applied_candidate_ids": _json_safe(pending.applied_candidate_ids),
        "last_delivered_sequence": pending.last_delivered_sequence,
        "last_acknowledged_sequence": pending.last_acknowledged_sequence,
        "processing": {str(key): value for key, value in pending.processing.items()},
        "buffered_updates": [
            {
                "request_id": int(update.request_id),
                "status": update.status.name,
                "candidate_ids": _json_safe(update.candidate_ids),
                "result": None
                if update.result is None
                else _result_to_json(update.result),
                "error": None
                if update.error is None
                else {
                    "error_type": update.error.error_type,
                    "message": update.error.message,
                    "details": _json_safe(dict(update.error.details)),
                },
                "sequence": update.sequence,
            }
            for update in pending.buffered_updates
        ],
        "reserved_cost": pending.reserved_cost,
        "retry_count": pending.retry_count,
        "checkpointable": pending.checkpointable,
        "original_candidate_ids": _json_safe(pending.original_candidate_ids),
        "feedback_result": None
        if pending.feedback_result is None
        else _feedback_to_json(pending.feedback_result),
        "fatal_error": None
        if pending.fatal_error is None
        else {
            "error_type": pending.fatal_error.error_type,
            "message": pending.fatal_error.message,
            "details": _json_safe(dict(pending.fatal_error.details)),
        },
        "prediction": None
        if pending.prediction is None
        else _prediction_to_json(pending.prediction),
    }


def _pending_from_json(value: Any) -> Any:
    from saealib.execution.evaluator import (
        EvaluationErrorInfo,
        EvaluationRequest,
        EvaluationStatus,
        EvaluationUpdate,
        PendingEvaluation,
    )

    if not isinstance(value, dict):
        raise CheckpointError("pending evaluation is malformed")
    try:
        request = EvaluationRequest(
            np.int64(value["request_id"]),
            np.asarray(value["candidate_ids"], dtype=np.int64),
            np.asarray(value["x"], dtype=np.float64),
            tuple(value.get("outputs", ("f", "g", "cv"))),
            value.get("metadata", {}),
        )
        updates = []
        for item in value.get("buffered_updates", []):
            error = item.get("error")
            updates.append(
                EvaluationUpdate(
                    np.int64(item["request_id"]),
                    EvaluationStatus[item["status"]],
                    np.asarray(item["candidate_ids"], dtype=np.int64),
                    None
                    if item.get("result") is None
                    else _result_from_json(item["result"]),
                    None
                    if error is None
                    else EvaluationErrorInfo(
                        error["error_type"], error["message"], error.get("details", {})
                    ),
                    int(item["sequence"]),
                )
            )
        return PendingEvaluation(
            request,
            EvaluationStatus[value["status"]],
            np.asarray(value["applied_candidate_ids"], dtype=np.int64),
            int(value["last_delivered_sequence"]),
            int(value["last_acknowledged_sequence"]),
            {
                int(key): str(status)
                for key, status in value.get("processing", {}).items()
            },
            tuple(updates),
            float(value.get("reserved_cost", 0.0)),
            int(value.get("retry_count", 0)),
            bool(value.get("checkpointable", False)),
            None
            if value.get("original_candidate_ids") is None
            else np.asarray(value["original_candidate_ids"], dtype=np.int64),
            None
            if value.get("feedback_result") is None
            else _feedback_from_json(value["feedback_result"]),
            None
            if value.get("fatal_error") is None
            else EvaluationErrorInfo(
                value["fatal_error"]["error_type"],
                value["fatal_error"]["message"],
                value["fatal_error"].get("details", {}),
            ),
            None
            if value.get("prediction") is None
            else _prediction_from_json(value["prediction"]),
        )
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        ValidationError,
    ) as exc:
        raise CheckpointError("pending evaluation is malformed") from exc


def _scalar_int(value: Any) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu":
        raise CheckpointError("checkpoint scalar must be an integer")
    return int(array)


def _allocator_scalar(value: Any, name: str) -> int:
    result = _scalar_int(value)
    if result < 0 or result > np.iinfo(np.int64).max:
        raise CheckpointError(f"{name} is outside the int64 allocator range")
    return result


def _default_json(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return "__nan__"
    return _json_safe(value)


def _default_value(value: Any) -> Any:
    if isinstance(value, str) and value == "__nan__":
        return np.nan
    return value


def _attribute_descriptors(collection: Any) -> list[dict[str, Any]]:
    result = []
    for name, attr in collection.schema.items():
        dtype = np.dtype(attr.dtype)
        if dtype == np.dtype(object):
            raise CheckpointError(f"object dtype is not checkpointable: {name!r}")
        result.append(
            {
                "name": name,
                "dtype": dtype.str,
                "shape": list(attr.shape),
                "default": _default_json(attr.default),
            }
        )
    return result


def _collection_descriptor(kind: str, name: str, collection: Any) -> dict[str, Any]:
    _validate_collection_name(name)
    descriptor: dict[str, Any] = {
        "name": name,
        "subtype": "Population" if kind == "population" else "Archive",
        "schema": _attribute_descriptors(collection),
        "size": len(collection),
        "capacity": collection._capacity,
    }
    if kind == "archive":
        from saealib.population import ParetoArchive

        if isinstance(collection, ParetoArchive):
            descriptor["subtype"] = "ParetoArchive"
            descriptor["direction"] = _json_safe(collection.direction)
            descriptor["eps_cv"] = _json_safe(collection.eps_cv)
        descriptor.update(
            {
                "duplicate_policy": getattr(
                    collection, "duplicate_policy", "keep_first"
                ),
                "key_attr": getattr(collection, "key_attr", "x"),
                "atol": getattr(collection, "atol", 0.0),
                "rtol": getattr(collection, "rtol", 0.0),
            }
        )
    return descriptor


def _attrs_from_descriptor(descriptor: dict[str, Any]) -> list[Any]:
    from saealib.population import PopulationAttribute

    schema = descriptor.get("schema")
    if not isinstance(schema, list):
        raise CheckpointError("collection schema must be a list")
    attrs = []
    seen: set[str] = set()
    for item in schema:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise CheckpointError("malformed collection attribute descriptor")
        name = item["name"]
        if name in seen:
            raise CheckpointError(f"duplicate attribute name: {name!r}")
        seen.add(name)
        try:
            dtype = np.dtype(item["dtype"])
            shape = tuple(item["shape"])
            if dtype == np.dtype(object) or any(
                not isinstance(axis, int) or axis < 0 for axis in shape
            ):
                raise ValueError
            attrs.append(
                PopulationAttribute(
                    name=name,
                    dtype=dtype,
                    shape=shape,
                    default=_default_value(item.get("default")),
                )
            )
        except (TypeError, ValueError) as exc:
            raise CheckpointError(f"malformed attribute {name!r}") from exc
    return attrs


def _restore_collection(
    kind: str, descriptor: dict[str, Any], data: Any, storage_prefix: str | None = None
) -> Any:
    from saealib.population import Archive, ParetoArchive, Population
    from saealib.population.archive import _validate_observation_schema

    name = descriptor.get("name")
    if not isinstance(name, str):
        raise CheckpointError("collection name is missing")
    _validate_collection_name(name)
    size = descriptor.get("size")
    capacity = descriptor.get("capacity")
    if (
        not isinstance(size, int)
        or not isinstance(capacity, int)
        or size < 0
        or capacity < size
    ):
        raise CheckpointError(f"invalid size/capacity for collection {name!r}")
    attrs = _attrs_from_descriptor(descriptor)
    schema = {attr.name: attr for attr in attrs}
    for id_name in ("id", "request_id"):
        if id_name in schema and np.dtype(schema[id_name].dtype) != np.dtype(np.int64):
            raise CheckpointError(f"{id_name} column must use int64")
    subtype = descriptor.get("subtype")
    if kind == "population":
        if subtype != "Population":
            raise CheckpointError(f"unknown population subtype: {subtype!r}")
        collection = Population(attrs, capacity)
    else:
        if subtype not in {"Archive", "ParetoArchive"}:
            raise CheckpointError(f"unknown archive subtype: {subtype!r}")
        params = {
            "key_attr": descriptor.get("key_attr", "x"),
            "atol": descriptor.get("atol", 0.0),
            "rtol": descriptor.get("rtol", 0.0),
            "duplicate_policy": descriptor.get("duplicate_policy", "keep_first"),
        }
        if subtype == "ParetoArchive" and params["duplicate_policy"] == "append":
            raise CheckpointError("ParetoArchive does not support append policy")
        if subtype == "Archive":
            _validate_observation_schema(attrs, params["duplicate_policy"])
        if subtype == "ParetoArchive":
            direction_value = descriptor.get("direction")
            collection = ParetoArchive(
                attrs,
                capacity,
                direction=(
                    None
                    if direction_value is None
                    else np.asarray(direction_value, dtype=np.float64)
                ),
                eps_cv=float(descriptor.get("eps_cv", 0.0)),
                **params,
            )
        else:
            collection = Archive(attrs, capacity, **params)
    encoded = _encoded_name(name)
    values: dict[str, np.ndarray] = {}
    for attr in attrs:
        if storage_prefix is None:
            key = f"{kind}__{encoded}__{_encoded_name(attr.name)}"
        else:
            key = f"{storage_prefix}__{_encoded_name(attr.name)}"
        if key not in data.files:
            raise CheckpointError(f"Checkpoint is missing array {key!r}")
        array = np.asarray(data[key])
        expected = (size, *attr.shape)
        if array.dtype != np.dtype(attr.dtype) or array.shape != expected:
            raise CheckpointError(f"array {key!r} has an invalid dtype or shape")
        values[attr.name] = np.array(array, copy=True)
    if getattr(collection, "duplicate_policy", None) == "append":
        if not {"id", "request_id"}.issubset(schema) or any(
            np.dtype(schema[name].dtype) != np.dtype(np.int64)
            for name in ("id", "request_id")
        ):
            raise CheckpointError(
                "append archives require int64 id and request_id columns"
            )
        if size and "id" in values:
            pairs = np.column_stack((values["request_id"], values["id"]))
            if len(np.unique(pairs, axis=0)) != len(pairs):
                raise CheckpointError("duplicate (request_id, candidate_id) pair")
    if size:
        collection._extend_internal(
            values,
            preserve_ids=True,
            allow_duplicate_ids=(
                getattr(collection, "duplicate_policy", None) == "append"
            ),
        )
    return collection


def _load_v2(
    cls: type[OptimizationState], data: Any, problem: Problem
) -> OptimizationState:
    manifest = _read_json(data, "_manifest")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise CheckpointError("manifest schema version does not match checkpoint")
    populations = manifest.get("populations")
    archives = manifest.get("archives")
    if not isinstance(populations, list) or not isinstance(archives, list):
        raise CheckpointError("checkpoint manifest collections are malformed")
    if any(not isinstance(item, dict) for item in populations + archives):
        raise CheckpointError("checkpoint collection descriptor is malformed")
    if any(not isinstance(item.get("name"), str) for item in populations + archives):
        raise CheckpointError("checkpoint collection name is malformed")
    if len({item.get("name") for item in populations}) != len(populations):
        raise CheckpointError("checkpoint contains duplicate population names")
    if len({item.get("name") for item in archives}) != len(archives):
        raise CheckpointError("checkpoint contains duplicate archive names")
    restored_populations = {
        item["name"]: _restore_collection("population", item, data)
        for item in populations
    }
    restored_archives = {
        item["name"]: _restore_collection("archive", item, data) for item in archives
    }
    if "main" not in restored_populations or not {"main", "pareto"} <= set(
        restored_archives
    ):
        raise CheckpointError("checkpoint is missing required main/pareto collections")
    offspring_payload = manifest.get("offspring")
    if offspring_payload is None:
        offspring = None
    elif (
        isinstance(offspring_payload, dict)
        and set(offspring_payload) == {"alias"}
        and offspring_payload["alias"] in restored_populations
    ):
        offspring = restored_populations[offspring_payload["alias"]]
    elif isinstance(offspring_payload, dict):
        offspring = _restore_collection(
            "population", offspring_payload, data, storage_prefix="offspring"
        )
    else:
        raise CheckpointError("checkpoint offspring descriptor is malformed")
    owner_payload = manifest.get("evaluation_owners", [])
    if not isinstance(owner_payload, list):
        raise CheckpointError("checkpoint evaluation owners are malformed")
    owners = {}
    for item in owner_payload:
        if not isinstance(item, dict) or not isinstance(item.get("request_id"), int):
            raise CheckpointError("checkpoint evaluation owner is malformed")
        request_id = item["request_id"]
        if "alias" in item and item["alias"] in restored_populations:
            owner = restored_populations[item["alias"]]
        elif item.get("offspring") is True and offspring is not None:
            owner = offspring
        elif isinstance(item.get("descriptor"), dict):
            owner = _restore_collection(
                "population",
                item["descriptor"],
                data,
                storage_prefix=f"evaluation_owner__{request_id}",
            )
        else:
            raise CheckpointError("checkpoint evaluation owner is missing")
        if request_id in owners:
            raise CheckpointError("duplicate evaluation owner")
        owners[request_id] = owner
    pending_payload = _read_json(data, "_pending_evaluations")
    if not isinstance(pending_payload, list):
        raise CheckpointError("checkpoint pending evaluations are malformed")
    pending = {}
    for item in pending_payload:
        restored = _pending_from_json(item)
        if not restored.checkpointable and restored.fatal_error is None:
            raise CheckpointError("non-checkpointable pending evaluation")
        request_id = int(restored.request.request_id)
        if request_id in pending:
            raise CheckpointError("duplicate pending request id")
        pending[request_id] = restored
    state_data = _read_json(data, "_data") if "_data" in data.files else {}
    if not isinstance(state_data, dict):
        raise CheckpointError("checkpoint data metadata must be a mapping")
    if "_async_fatal" in data.files:
        async_fatal = _read_json(data, "_async_fatal")
        if async_fatal is not None and not isinstance(async_fatal, dict):
            raise CheckpointError("checkpoint async fatal state is malformed")
    else:
        for key in ("pending_candidate_ids", "reserved_fe", "reserved_cost"):
            state_data.pop(key, None)
        async_fatal = state_data.pop("async_fatal", None)
        if async_fatal is not None and not isinstance(async_fatal, dict):
            raise CheckpointError("checkpoint legacy async fatal state is malformed")
    state_data.pop("evaluation_plan", None)
    state_data.pop("evaluation_updates", None)
    feedback_result = (
        None
        if "_feedback_result" not in data.files
        else (
            None
            if (value := _read_json(data, "_feedback_result")) is None
            else _feedback_from_json(value)
        )
    )
    predictions = (
        None
        if "_predictions" not in data.files
        else (
            None
            if (value := _read_json(data, "_predictions")) is None
            else _prediction_from_json(value)
        )
    )
    evaluation_plan = (
        None
        if "_evaluation_plan" not in data.files
        else (
            None
            if (value := _read_json(data, "_evaluation_plan")) is None
            else _plan_from_json(value)
        )
    )
    evaluation_plan_state = (
        None
        if "_evaluation_plan_state" not in data.files
        else (
            None
            if (value := _read_json(data, "_evaluation_plan_state")) is None
            else _plan_state_from_json(value)
        )
    )
    plan_updates_payload = (
        {}
        if "_evaluation_plan_updates" not in data.files
        else _read_json(data, "_evaluation_plan_updates")
    )
    if not isinstance(plan_updates_payload, dict):
        raise CheckpointError("checkpoint evaluation plan updates are malformed")
    evaluation_plan_updates = {}
    try:
        for request_id, updates in plan_updates_payload.items():
            if not isinstance(updates, list):
                raise CheckpointError(
                    "checkpoint evaluation plan updates are malformed"
                )
            evaluation_plan_updates[int(request_id)] = [
                _update_from_json(update) for update in updates
            ]
    except (TypeError, ValueError) as exc:
        raise CheckpointError(
            "checkpoint evaluation plan updates are malformed"
        ) from exc
    if evaluation_plan is not None and evaluation_plan_state is not None:
        plan_ids = {int(request.request_id) for request in evaluation_plan.requests}
        state_ids = (
            set(evaluation_plan_state.submitted)
            | set(evaluation_plan_state.completed)
            | set(evaluation_plan_state.acknowledged)
            | set(evaluation_plan_state.deferred)
        )
        if not state_ids <= plan_ids:
            raise CheckpointError("evaluation plan state references an unknown request")
        if not set(evaluation_plan_updates) <= plan_ids:
            raise CheckpointError(
                "evaluation plan updates reference an unknown request"
            )
    rng = np.random.default_rng()
    rng.bit_generator.state = _read_json(data, "_rng_state")
    return cls(
        problem=problem,
        populations=restored_populations,
        archives=restored_archives,
        rng=rng,
        candidate_id_allocator=IDAllocator(
            _allocator_scalar(data["_next_candidate_id"], "_next_candidate_id")
        ),
        request_id_allocator=IDAllocator(
            _allocator_scalar(data["_next_request_id"], "_next_request_id")
        ),
        fe=_scalar_int(data["_fe"]),
        gen=_scalar_int(data["_gen"]),
        offspring=offspring,
        pending_evaluations=pending,
        evaluation_owners=owners,
        feedback_result=feedback_result,
        predictions=predictions,
        evaluation_plan=evaluation_plan,
        evaluation_plan_state=evaluation_plan_state,
        evaluation_plan_updates=evaluation_plan_updates,
        async_fatal=async_fatal,
        data={**state_data, "resumed": True},
    )


def _load_v1(
    cls: type[OptimizationState], data: Any, problem: Problem
) -> OptimizationState:
    from saealib.population import (
        Archive,
        ParetoArchive,
        Population,
    )

    schema_list = _read_json(data, "_schema")
    descriptor = {
        "schema": schema_list,
        "size": _scalar_int(data["_archive_size"]),
        "capacity": max(_scalar_int(data["_archive_size"]), 1),
    }
    attrs = _attrs_from_descriptor(descriptor)

    def old_collection(prefix: str, size: int, factory: Any) -> Any:
        collection = factory(attrs, max(size, 1))
        values = {}
        for attr in attrs:
            key = f"{prefix}__{attr.name}"
            if key not in data.files:
                raise CheckpointError(f"legacy checkpoint is missing {key!r}")
            array = np.asarray(data[key])
            expected = (size, *attr.shape)
            if array.dtype != np.dtype(attr.dtype) or array.shape != expected:
                raise CheckpointError(f"legacy array {key!r} has an invalid shape")
            values[attr.name] = array
        if size:
            collection._extend_internal(values, preserve_ids=True)
        return collection

    archive = old_collection("archive", descriptor["size"], Archive)
    population = old_collection("pop", _scalar_int(data["_pop_size"]), Population)
    pareto = old_collection(
        "pareto",
        _scalar_int(data["_pareto_size"]),
        lambda a, c: ParetoArchive(a, c, direction=problem.direction),
    )
    if "_pending_evaluations" in data.files:
        pending_bytes = bytes(data["_pending_evaluations"])
        if pending_bytes not in _SAFE_EMPTY_PENDING and pending_bytes not in (
            b"",
            b"[]",
            b"{}",
            b"null",
        ):
            raise CheckpointError("legacy pending state is not a safe empty value")
    rng = np.random.default_rng()
    rng.bit_generator.state = _read_json(data, "_rng_state")
    return cls(
        problem=problem,
        population=population,
        archive=archive,
        pareto_archive=pareto,
        rng=rng,
        candidate_id_allocator=IDAllocator(
            _allocator_scalar(data["_next_candidate_id"], "_next_candidate_id")
        ),
        request_id_allocator=IDAllocator(
            _allocator_scalar(data["_next_request_id"], "_next_request_id")
        ),
        fe=_scalar_int(data["_fe"]),
        gen=_scalar_int(data["_gen"]),
        data={"resumed": True},
    )


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

OptimizationContext = OptimizationState
