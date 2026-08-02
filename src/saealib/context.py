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

from saealib.exceptions import CheckpointError, ValidationError
from saealib.identity import IDAllocator

if TYPE_CHECKING:
    from saealib.comparators import Comparator
    from saealib.execution.evaluator import (
        EvaluationHandle,
        EvaluationRequest,
        EvaluationUpdate,
        PendingEvaluation,
    )
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


class _NamedCollection(dict[str, Any]):
    def __init__(self, kind: str, values: dict[str, Any]) -> None:
        self._kind = kind
        super().__init__()
        for name, value in values.items():
            self[name] = value

    def __setitem__(self, name: str, value: Any) -> None:
        self._validate_entry(name, value)
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
        super().__delitem__(name)

    def clear(self) -> None:
        required = {"main"} if self._kind == "populations" else {"main", "pareto"}
        for name in list(self):
            if name not in required:
                super().__delitem__(name)

    def pop(self, name: object, default: Any = _MISSING) -> Any:
        if not isinstance(name, str):
            raise ValidationError(f"invalid collection name: {name!r}")
        _validate_collection_name(name)
        required = {"main"} if self._kind == "populations" else {"main", "pareto"}
        if name in required:
            raise ValidationError(f"cannot remove required {self._kind} entry {name!r}")
        if default is _MISSING:
            return super().pop(name)
        return super().pop(name, default)

    def popitem(self) -> tuple[str, Any]:
        required = {"main"} if self._kind == "populations" else {"main", "pareto"}
        for name in reversed(list(self)):
            if name not in required:
                return name, super().pop(name)
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
    predictions : SurrogatePrediction or None
        Batched surrogate prediction covering every row of ``offspring``.
        Set by :class:`~saealib.stages.SurrogatePredictStage`.
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
    predictions: SurrogatePrediction | None = None
    evaluation_request: EvaluationRequest | None = None
    evaluation_updates: list[EvaluationUpdate] = field(default_factory=list)
    evaluation_update_new_ids: list[np.ndarray] = field(default_factory=list)
    evaluation_new_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    evaluation_handles: dict[int, EvaluationHandle] = field(default_factory=dict)
    pending_evaluations: dict[int, PendingEvaluation] = field(default_factory=dict)
    feedback_result: FeedbackResult | None = None

    # User-extensible data
    data: dict[str, Any] = field(default_factory=dict)

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
        predictions: SurrogatePrediction | None = None,
        evaluation_request: EvaluationRequest | None = None,
        evaluation_updates: list[EvaluationUpdate] | None = None,
        evaluation_update_new_ids: list[np.ndarray] | None = None,
        evaluation_new_ids: np.ndarray | None = None,
        evaluation_handles: dict[int, EvaluationHandle] | None = None,
        pending_evaluations: dict[int, PendingEvaluation] | None = None,
        feedback_result: FeedbackResult | None = None,
        data: dict[str, Any] | None = None,
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
        self.populations = _NamedCollection("populations", populations)
        self.archives = _NamedCollection("archives", archives)
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
        self.predictions = predictions
        self.evaluation_request = evaluation_request
        self.evaluation_updates = (
            [] if evaluation_updates is None else evaluation_updates
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
        self.pending_evaluations = (
            {} if pending_evaluations is None else pending_evaluations
        )
        self.feedback_result = feedback_result
        self.data = {} if data is None else data

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

    def __getstate__(self) -> dict[str, Any]:
        """Exclude runtime evaluation handles from serialized state."""
        state = self.__dict__.copy()
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
        if self.pending_evaluations:
            raise ValidationError("cannot checkpoint while evaluations are pending")
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".npz")

        save_dict: dict[str, np.ndarray] = {}
        manifest: dict[str, Any] = {
            "schema_version": CURRENT_CHECKPOINT_SCHEMA_VERSION,
            "populations": [],
            "archives": [],
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
        if self.pending_evaluations:
            raise ValidationError("cannot checkpoint while evaluations are pending")
        save_dict["_pending_evaluations"] = _json_array([])
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


def _restore_collection(kind: str, descriptor: dict[str, Any], data: Any) -> Any:
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
        key = f"{kind}__{encoded}__{_encoded_name(attr.name)}"
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
    pending = _read_json(data, "_pending_evaluations")
    if pending != []:
        raise CheckpointError("non-empty pending evaluations cannot be restored")
    state_data = _read_json(data, "_data") if "_data" in data.files else {}
    if not isinstance(state_data, dict):
        raise CheckpointError("checkpoint data metadata must be a mapping")
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
