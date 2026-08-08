"""Columnar observation records and their dense batch views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
from typing_extensions import Self

from saealib.core.contracts.observation import (
    CONSTRAINT,
    CV,
    OBJECTIVE,
    OBSERVATION_QUANTITY_KINDS,
    OBSERVATION_SOURCES,
    OBSERVATION_STATUSES,
    OBSERVATION_SUBJECT_KINDS,
    OK,
    TRUE,
    ObservationSource,
    ObservationStatus,
    ObservationSubjectPayload,
    ObservationValue,
    PortableValue,
    QuantityKind,
)
from saealib.exceptions import ValidationError
from saealib.population.population import CandidateIds

__all__ = [
    "ObservationBatch",
    "ObservationRecord",
    "ObservationRecords",
    "ObservationSchema",
    "ObservationSubject",
    "QuantityRef",
]


_QuantitySpace = int | Sequence[int | str]


@dataclass(frozen=True, kw_only=True)
class QuantityRef:
    """A quantity kind and its index or name."""

    kind: QuantityKind
    index: int | str

    def __post_init__(self) -> None:
        """Validate the stable parts of a quantity reference."""
        if not isinstance(self.kind, str) or not self.kind:
            raise ValidationError("quantity kind must be a non-empty string")
        if isinstance(self.index, bool) or not isinstance(self.index, (int, str)):
            raise ValidationError("quantity index must be an int or name")
        if isinstance(self.index, int) and self.index < 0:
            raise ValidationError("quantity index must be non-negative")
        if isinstance(self.index, str) and not self.index:
            raise ValidationError("quantity name must not be empty")

    @classmethod
    def from_value(cls, value: Any) -> QuantityRef:
        """Normalize a quantity reference from common record-facing forms."""
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            if "kind" not in value:
                raise TypeError("quantity mapping must contain 'kind'")
            index = value.get("index", value.get("name"))
            if index is None:
                raise TypeError("quantity mapping must contain 'index' or 'name'")
            return cls(kind=str(value["kind"]), index=index)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return cls(kind=str(value[0]), index=value[1])
        kind = getattr(value, "kind", None)
        index = getattr(value, "index", getattr(value, "name", None))
        if kind is not None and index is not None:
            return cls(kind=str(kind), index=index)
        raise TypeError(
            "quantity must be a QuantityRef, mapping, or (kind, index) pair"
        )


QuantityInput: TypeAlias = QuantityRef | tuple[str, int | str] | Mapping[str, Any]


@dataclass(frozen=True, kw_only=True)
class ObservationSubject:
    """A registered subject kind and its payload."""

    kind: str
    payload: ObservationSubjectPayload

    def __post_init__(self) -> None:
        """Validate the subject kind and its registered payload descriptor."""
        if not isinstance(self.kind, str) or not self.kind:
            raise ValidationError("observation subject kind must be a non-empty string")
        descriptor = OBSERVATION_SUBJECT_KINDS.get(self.kind)
        if descriptor is None:
            raise ValidationError(f"unknown observation subject kind: {self.kind!r}")

    @classmethod
    def from_value(cls, value: Any) -> ObservationSubject:
        """Normalize a subject from common record-facing forms."""
        if isinstance(value, cls):
            return value
        if isinstance(value, np.ndarray):
            return cls(kind="candidate", payload=value)
        if isinstance(value, Mapping):
            if "kind" not in value:
                raise TypeError("subject mapping must contain 'kind'")
            return cls(
                kind=str(value["kind"]),
                payload=value.get("payload", value.get("value")),
            )
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return cls(kind=str(value[0]), payload=value[1])
        kind = getattr(value, "kind", None)
        if kind is not None and hasattr(value, "payload"):
            return cls(kind=str(kind), payload=value.payload)
        raise TypeError(
            "subject must be an ObservationSubject, mapping, or (kind, payload) pair"
        )


SubjectInput: TypeAlias = (
    ObservationSubject | tuple[str, ObservationSubjectPayload] | Mapping[str, Any]
)


@dataclass(frozen=True, kw_only=True)
class ObservationRecord:
    """One observation in the record-facing API."""

    subject: SubjectInput
    quantity: QuantityInput
    value: ObservationValue
    status: ObservationStatus
    source: ObservationSource
    uncertainty: ObservationValue | None = None
    fidelity: Any = None
    cost: float | None = None
    timestamp: float | None = None
    provenance: Mapping[str, PortableValue] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        """Normalize nested references and freeze provenance metadata."""
        object.__setattr__(self, "subject", ObservationSubject.from_value(self.subject))
        object.__setattr__(self, "quantity", QuantityRef.from_value(self.quantity))
        if not OBSERVATION_STATUSES.contains(self.status):
            raise ValidationError(f"unknown observation status: {self.status!r}")
        if not OBSERVATION_SOURCES.contains(self.source):
            raise ValidationError(f"unknown observation source: {self.source!r}")
        if not isinstance(self.provenance, Mapping):
            raise ValidationError("observation provenance must be a mapping")
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(dict(self.provenance)),
        )


@dataclass(frozen=True, kw_only=True)
class ObservationSchema:
    """The fixed quantity/index space for one observation run."""

    objective_count: int = 0
    constraint_count: int = 0
    quantities: Mapping[str, _QuantitySpace] = field(default_factory=dict)
    extra_quantities: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        """Normalize and validate the registered quantity spaces."""
        for name, count in (
            ("objective_count", self.objective_count),
            ("constraint_count", self.constraint_count),
        ):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValidationError(f"{name} must be a non-negative integer")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version < 1
        ):
            raise ValidationError("schema_version must be a positive integer")
        if not isinstance(self.quantities, Mapping):
            raise ValidationError("quantities must be a mapping")
        if not isinstance(self.extra_quantities, tuple):
            raise ValidationError("extra_quantities must be a tuple of names")

        spaces: dict[str, tuple[int | str, ...]] = {
            OBJECTIVE: tuple(range(self.objective_count)),
            CONSTRAINT: tuple(range(self.constraint_count)),
        }
        for name in self.extra_quantities:
            if not isinstance(name, str) or not name:
                raise ValidationError("extra quantity names must be non-empty strings")
            spaces.setdefault(name, (0,))
        for kind, space in self.quantities.items():
            if not isinstance(kind, str) or not kind:
                raise ValidationError("quantity names must be non-empty strings")
            spaces[kind] = self._normalize_space(kind, space)
        spaces[OBJECTIVE] = tuple(range(self.objective_count))
        spaces[CONSTRAINT] = tuple(range(self.constraint_count))
        for kind in spaces:
            if not OBSERVATION_QUANTITY_KINDS.contains(kind):
                raise ValidationError(f"quantity kind is not registered: {kind!r}")
        object.__setattr__(self, "quantities", MappingProxyType(spaces))
        object.__setattr__(
            self,
            "extra_quantities",
            tuple(kind for kind in spaces if kind not in {OBJECTIVE, CONSTRAINT}),
        )

    @staticmethod
    def _normalize_space(kind: str, space: _QuantitySpace) -> tuple[int | str, ...]:
        if isinstance(space, bool):
            raise ValidationError(f"quantity space for {kind!r} is invalid")
        if isinstance(space, int):
            if space < 0:
                raise ValidationError(f"quantity space for {kind!r} is negative")
            return tuple(range(space))
        if isinstance(space, (str, bytes)):
            raise ValidationError(f"quantity space for {kind!r} must be a sequence")
        try:
            normalized = tuple(space)
        except TypeError as exc:
            raise ValidationError(
                f"quantity space for {kind!r} must be an int or sequence"
            ) from exc
        for index in normalized:
            if isinstance(index, bool) or not isinstance(index, (int, str)):
                raise ValidationError(
                    f"quantity space for {kind!r} has an invalid index"
                )
            if isinstance(index, int) and index < 0:
                raise ValidationError(
                    f"quantity space for {kind!r} has a negative index"
                )
            if isinstance(index, str) and not index:
                raise ValidationError(f"quantity space for {kind!r} has an empty name")
        if len(set(normalized)) != len(normalized):
            raise ValidationError(f"quantity space for {kind!r} has duplicate indices")
        return normalized

    @property
    def quantity_kinds(self) -> tuple[str, ...]:
        """Return the declared quantity kinds in schema order."""
        return tuple(self.quantities)

    def indices(self, kind: str) -> tuple[int | str, ...]:
        """Return the declared index/name space for one quantity kind."""
        if not OBSERVATION_QUANTITY_KINDS.contains(kind):
            raise ValidationError(f"quantity kind is not registered: {kind!r}")
        space = self.quantities.get(kind, ())
        if isinstance(space, int):
            return tuple(range(space))
        return tuple(space)


_OPTIONAL_COLUMNS = ("uncertainty", "fidelity", "cost", "timestamp", "provenance")
_COLUMNS = (
    "subject_kind",
    "subject_payload",
    "quantity_kind",
    "quantity_index",
    "value",
    "status",
    "source",
    *_OPTIONAL_COLUMNS,
)
_PUBLIC_ALIASES = {"subject", "quantity"}


def _vocabulary_code(vocabulary: Any, value: str) -> np.int8:
    """Encode a registered vocabulary name as a compact column value."""
    try:
        return np.int8(vocabulary.names().index(value))
    except ValueError as exc:
        raise ValidationError(
            f"unknown observation vocabulary value: {value!r}"
        ) from exc


def _vocabulary_names(vocabulary: Any, values: np.ndarray) -> np.ndarray:
    names = np.asarray(vocabulary.names(), dtype=object)
    return names[np.asarray(values, dtype=np.intp)]


def _subject_payload_column(
    subjects: Sequence[ObservationSubject],
) -> np.ndarray:
    """Pack fixed-shape subject payloads using their descriptor columns."""
    if not subjects:
        return np.empty(0, dtype=object)
    descriptors = [OBSERVATION_SUBJECT_KINDS.get(subject.kind) for subject in subjects]
    if any(descriptor is None for descriptor in descriptors):
        raise ValidationError("unknown observation subject kind")
    descriptor = descriptors[0]
    if descriptor is None:
        raise ValidationError("unknown observation subject kind")
    if (
        any(item is not descriptor for item in descriptors)
        or len(descriptor.columns) != 1
    ):
        return _object_column(subject.payload for subject in subjects)
    spec = descriptor.columns[0]
    shape = tuple(spec.shape)
    try:
        array = np.asarray([subject.payload for subject in subjects], dtype=spec.dtype)
        if array.shape == (len(subjects), *shape):
            return array
        # A scalar PopulationAttribute describes one scalar column in the
        # payload.  Subject payloads are still represented as one-element
        # arrays (the candidate descriptor is the canonical example).
        if shape == () and array.shape == (len(subjects), 1):
            return array
    except (TypeError, ValueError):
        pass
    return _object_column(subject.payload for subject in subjects)


def _object_column(values: Any) -> np.ndarray:
    """Make a one-dimensional object column without unpacking nested arrays."""
    if isinstance(values, np.ndarray) and values.ndim == 1:
        return np.array(values, dtype=object, order="C", copy=True)
    if isinstance(values, np.ndarray) and values.ndim == 0:
        values = [values.item()]
    elif not isinstance(values, np.ndarray):
        values = tuple(values)
    result = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        result[index] = value
    return result


def _typed_column(values: Any) -> np.ndarray:
    """Use a native homogeneous dtype, falling back only for heterogeneous values."""
    if not isinstance(values, np.ndarray):
        values = tuple(values)
    array = np.asarray(values)
    if array.dtype.kind in "biufc" and array.ndim == 1:
        return np.array(array, copy=True, order="C")
    return _object_column(values)


def _quantity_parts(value: Any) -> tuple[str, int | str]:
    """Extract quantity fields without allocating a reference object."""
    if isinstance(value, QuantityRef):
        return value.kind, value.index
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return str(value[0]), value[1]
    reference = QuantityRef.from_value(value)
    return reference.kind, reference.index


class ObservationRecords:
    """Record-facing wrapper around parallel NumPy columns."""

    def __init__(
        self,
        records: Sequence[ObservationRecord]
        | Mapping[str, Any]
        | ObservationRecords = (),
    ) -> None:
        if isinstance(records, ObservationRecords):
            self._columns = dict(records._columns)
            self._compat_missing_optional = getattr(
                records, "_compat_missing_optional", False
            )
            self._validate_columns()
            return
        if isinstance(records, Mapping):
            self._columns = self._columns_from_mapping(records)
            self._compat_missing_optional = any(
                name not in records for name in _OPTIONAL_COLUMNS
            )
        else:
            normalized = tuple(
                record
                if isinstance(record, ObservationRecord)
                else ObservationRecord(**record)
                for record in records
            )
            subjects = tuple(
                ObservationSubject.from_value(r.subject) for r in normalized
            )
            self._columns = {
                "subject_kind": np.asarray(
                    [
                        _vocabulary_code(OBSERVATION_SUBJECT_KINDS, s.kind)
                        for s in subjects
                    ],
                    dtype=np.int8,
                ),
                "subject_payload": _subject_payload_column(subjects),
                "quantity_kind": np.asarray(
                    [
                        _vocabulary_code(
                            OBSERVATION_QUANTITY_KINDS, _quantity_parts(r.quantity)[0]
                        )
                        for r in normalized
                    ],
                    dtype=np.int8,
                ),
                "quantity_index": _typed_column(
                    _quantity_parts(r.quantity)[1] for r in normalized
                ),
                "value": _typed_column(r.value for r in normalized),
                "status": np.asarray(
                    [
                        _vocabulary_code(OBSERVATION_STATUSES, r.status)
                        for r in normalized
                    ],
                    dtype=np.int8,
                ),
                "source": np.asarray(
                    [
                        _vocabulary_code(OBSERVATION_SOURCES, r.source)
                        for r in normalized
                    ],
                    dtype=np.int8,
                ),
            }
            for name in _OPTIONAL_COLUMNS:
                if any(getattr(r, name) is not None for r in normalized):
                    self._columns[name] = _object_column(
                        getattr(r, name) for r in normalized
                    )
            self._compat_missing_optional = False
        self._validate_columns()

    @classmethod
    def from_dense(
        cls,
        candidate_ids: np.ndarray,
        f: np.ndarray,
        g: np.ndarray,
        cv: np.ndarray | None = None,
        *,
        source: ObservationSource = TRUE,
        status: ObservationStatus = OK,
        subject_kind: str = "candidate",
    ) -> Self:
        """Construct complete dense records with array operations only."""
        ids = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
        objective = np.asarray(f, dtype=np.float64)
        constraint = np.asarray(g, dtype=np.float64)
        if objective.ndim != 2 or constraint.ndim != 2:
            raise ValidationError("dense f and g must be two-dimensional")
        if objective.shape[0] != len(ids) or constraint.shape[0] != len(ids):
            raise ValidationError("dense arrays must have one row per candidate")
        cv_array = None if cv is None else np.asarray(cv, dtype=np.float64).reshape(-1)
        if cv_array is not None and len(cv_array) != len(ids):
            raise ValidationError("dense cv must have one value per candidate")
        quantity_count = (
            objective.shape[1] + constraint.shape[1] + (cv_array is not None)
        )
        payload = np.repeat(ids.reshape(-1, 1), quantity_count, axis=0)
        base_kinds = np.concatenate(
            (
                np.full(
                    objective.shape[1],
                    _vocabulary_code(OBSERVATION_QUANTITY_KINDS, OBJECTIVE),
                    dtype=np.int8,
                ),
                np.full(
                    constraint.shape[1],
                    _vocabulary_code(OBSERVATION_QUANTITY_KINDS, CONSTRAINT),
                    dtype=np.int8,
                ),
                np.full(
                    1, _vocabulary_code(OBSERVATION_QUANTITY_KINDS, CV), dtype=np.int8
                )
                if cv_array is not None
                else np.empty(0, dtype=np.int8),
            )
        )
        kinds = np.tile(base_kinds, len(ids))
        base_indices = np.concatenate(
            (
                np.arange(objective.shape[1], dtype=np.int64),
                np.arange(constraint.shape[1], dtype=np.int64),
                np.zeros(1, dtype=np.int64)
                if cv_array is not None
                else np.empty(0, dtype=np.int64),
            )
        )
        indices = np.tile(base_indices, len(ids))
        value_rows = np.concatenate(
            (
                objective,
                constraint,
                cv_array.reshape(-1, 1)
                if cv_array is not None
                else np.empty((len(ids), 0)),
            ),
            axis=1,
        )
        values = value_rows.reshape(-1)
        rows = len(values)
        columns = {
            "subject_kind": np.full(
                rows,
                _vocabulary_code(OBSERVATION_SUBJECT_KINDS, subject_kind),
                dtype=np.int8,
            ),
            "subject_payload": payload,
            "quantity_kind": kinds,
            "quantity_index": indices,
            "value": values,
            "status": np.full(
                rows, _vocabulary_code(OBSERVATION_STATUSES, status), dtype=np.int8
            ),
            "source": np.full(
                rows, _vocabulary_code(OBSERVATION_SOURCES, source), dtype=np.int8
            ),
        }
        return cls._from_columns(columns)

    @classmethod
    def _from_columns(cls, columns: Mapping[str, np.ndarray]) -> Self:
        result = object.__new__(cls)
        result._columns = dict(columns)
        result._compat_missing_optional = False
        result._validate_columns()
        return result

    @classmethod
    def from_columns(cls, columns: Mapping[str, Any]) -> Self:
        """Build a record store from required and optional parallel columns."""
        return cls(columns)

    @classmethod
    def from_records(cls, records: Sequence[ObservationRecord]) -> Self:
        """Build a columnar store from record-facing inputs."""
        return cls(records)

    @classmethod
    def _columns_from_mapping(cls, values: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Normalize a column mapping while retaining columnar storage."""
        allowed = set(_COLUMNS) | _PUBLIC_ALIASES
        unknown = set(values) - allowed
        if unknown:
            raise ValidationError(f"unknown observation columns: {sorted(unknown)!r}")
        columns: dict[str, np.ndarray] = {}
        if "subject" in values:
            subjects = [
                ObservationSubject.from_value(value) for value in values["subject"]
            ]
            columns["subject_kind"] = np.asarray(
                [_vocabulary_code(OBSERVATION_SUBJECT_KINDS, s.kind) for s in subjects],
                dtype=np.int8,
            )
            columns["subject_payload"] = _subject_payload_column(subjects)
        elif "subject_kind" in values and "subject_payload" in values:
            subjects = tuple(
                ObservationSubject(kind=str(kind), payload=payload)
                for kind, payload in zip(
                    values["subject_kind"], values["subject_payload"]
                )
            )
            columns["subject_kind"] = np.asarray(
                [
                    _vocabulary_code(OBSERVATION_SUBJECT_KINDS, value)
                    for value in values["subject_kind"]
                ],
                dtype=np.int8,
            )
            columns["subject_payload"] = _subject_payload_column(subjects)
        else:
            raise ValidationError(
                "observation columns require subject or "
                "subject_kind and subject_payload"
            )
        if "quantity" in values:
            quantity_values = tuple(
                _quantity_parts(value) for value in values["quantity"]
            )
            columns["quantity_kind"] = np.asarray(
                [
                    _vocabulary_code(OBSERVATION_QUANTITY_KINDS, q[0])
                    for q in quantity_values
                ],
                dtype=np.int8,
            )
            columns["quantity_index"] = _typed_column(q[1] for q in quantity_values)
        elif "quantity_kind" in values and "quantity_index" in values:
            columns["quantity_kind"] = np.asarray(
                [
                    _vocabulary_code(OBSERVATION_QUANTITY_KINDS, q)
                    for q in values["quantity_kind"]
                ],
                dtype=np.int8,
            )
            columns["quantity_index"] = _typed_column(values["quantity_index"])
        else:
            raise ValidationError(
                "observation columns require quantity or "
                "quantity_kind and quantity_index"
            )
        for name in ("value", "status", "source"):
            if name not in values:
                raise ValidationError(f"observation columns require {name}")
            if name == "value":
                columns[name] = _typed_column(values[name])
            elif name == "status":
                columns[name] = np.asarray(
                    [
                        _vocabulary_code(OBSERVATION_STATUSES, value)
                        for value in values[name]
                    ],
                    dtype=np.int8,
                )
            else:
                columns[name] = np.asarray(
                    [
                        _vocabulary_code(OBSERVATION_SOURCES, value)
                        for value in values[name]
                    ],
                    dtype=np.int8,
                )
        for name in _OPTIONAL_COLUMNS:
            if name in values:
                columns[name] = _object_column(values[name])
        return columns

    def _validate_columns(self) -> None:
        required = set(_COLUMNS) - set(_OPTIONAL_COLUMNS)
        if not required.issubset(self._columns):
            raise ValidationError(
                f"Observation columns must include {sorted(required)!r}"
            )
        lengths: set[int] = set()
        for name, column in self._columns.items():
            if not isinstance(column, np.ndarray) or column.ndim < 1:
                raise ValidationError(f"Observation column {name!r} must be 1-D")
            lengths.add(len(column))
            column.flags.writeable = False
        if len(lengths) != 1:
            raise ValidationError("Observation columns must have equal lengths")

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return the names of the stored columns."""
        return tuple(self._columns)

    @property
    def columns(self) -> Mapping[str, np.ndarray]:
        """Return the read-only internal column mapping."""
        return MappingProxyType(self._columns)

    def __len__(self) -> int:
        """Return the number of stored observation rows."""
        return len(self._columns["value"])

    def __getitem__(self, index: int) -> ObservationRecord:
        """Materialize one record from the parallel columns."""
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("ObservationRecords indices must be integers")
        i = int(index)
        subject_kind = _vocabulary_names(
            OBSERVATION_SUBJECT_KINDS, self._columns["subject_kind"]
        )[i]
        quantity_kind = _vocabulary_names(
            OBSERVATION_QUANTITY_KINDS, self._columns["quantity_kind"]
        )[i]
        status = _vocabulary_names(OBSERVATION_STATUSES, self._columns["status"])[i]
        source = _vocabulary_names(OBSERVATION_SOURCES, self._columns["source"])[i]
        optional = {
            name: self._columns[name][i]
            for name in _OPTIONAL_COLUMNS
            if name in self._columns
        }
        return ObservationRecord(
            subject=ObservationSubject(
                kind=subject_kind,
                payload=self._columns["subject_payload"][i],
            ),
            quantity=QuantityRef(
                kind=quantity_kind,
                index=int(self._columns["quantity_index"][i])
                if isinstance(self._columns["quantity_index"][i], np.integer)
                else self._columns["quantity_index"][i],
            ),
            value=self._columns["value"][i],
            status=status,
            source=source,
            **optional,
        )

    def column(self, name: str) -> np.ndarray:
        """Return a read-only column, materializing only a compatibility alias."""
        if name == "subject":
            result = np.empty(len(self), dtype=object)
            for index, (kind, payload) in enumerate(
                zip(
                    _vocabulary_names(
                        OBSERVATION_SUBJECT_KINDS, self._columns["subject_kind"]
                    ),
                    self._columns["subject_payload"],
                )
            ):
                result[index] = (kind, payload)
            result.flags.writeable = False
            return result
        if name == "quantity":
            result = np.empty(len(self), dtype=object)
            for index, (kind, quantity_index) in enumerate(
                zip(
                    _vocabulary_names(
                        OBSERVATION_QUANTITY_KINDS, self._columns["quantity_kind"]
                    ),
                    self._columns["quantity_index"],
                )
            ):
                result[index] = QuantityRef(kind=kind, index=quantity_index)
            result.flags.writeable = False
            return result
        if name not in self._columns:
            if name in _OPTIONAL_COLUMNS and getattr(
                self, "_compat_missing_optional", False
            ):
                result = np.empty(len(self), dtype=object)
                result[:] = None
                result.flags.writeable = False
                return result
            raise KeyError(name)
        if name == "subject_kind":
            return _vocabulary_names(OBSERVATION_SUBJECT_KINDS, self._columns[name])
        if name == "quantity_kind":
            return _vocabulary_names(OBSERVATION_QUANTITY_KINDS, self._columns[name])
        if name == "status":
            return _vocabulary_names(OBSERVATION_STATUSES, self._columns[name])
        if name == "source":
            return _vocabulary_names(OBSERVATION_SOURCES, self._columns[name])
        return self._columns[name]

    @staticmethod
    def _matches(value: Any, expected: Any) -> bool:
        if callable(expected):
            return bool(expected(value))
        if isinstance(value, np.ndarray) or isinstance(expected, np.ndarray):
            return bool(np.array_equal(value, expected))
        result = value == expected
        return (
            bool(result) if not isinstance(result, np.ndarray) else bool(np.all(result))
        )

    def select(self, **predicate: Any) -> Self:
        """Select rows by record fields, returning a view when it is possible."""
        if not predicate:
            return self
        mask = np.ones(len(self), dtype=bool)
        for name, expected in predicate.items():
            if name == "subject":
                subject = ObservationSubject.from_value(expected)
                mask &= self._columns["subject_kind"] == _vocabulary_code(
                    OBSERVATION_SUBJECT_KINDS, subject.kind
                )
                mask &= np.fromiter(
                    (
                        self._matches(payload, subject.payload)
                        for payload in self._columns["subject_payload"]
                    ),
                    dtype=bool,
                    count=len(self),
                )
                continue
            if name == "quantity":
                quantity = QuantityRef.from_value(expected)
                mask &= np.fromiter(
                    (
                        self._matches(
                            QuantityRef(
                                kind=_vocabulary_names(
                                    OBSERVATION_QUANTITY_KINDS, np.asarray([kind])
                                )[0],
                                index=int(quantity_index)
                                if isinstance(quantity_index, np.integer)
                                else quantity_index,
                            ),
                            quantity,
                        )
                        for kind, quantity_index in zip(
                            self._columns["quantity_kind"],
                            self._columns["quantity_index"],
                        )
                    ),
                    dtype=bool,
                    count=len(self),
                )
                continue
            if name not in self._columns:
                raise KeyError(name)
            if name in {"subject_kind", "quantity_kind", "status", "source"}:
                vocabulary = {
                    "subject_kind": OBSERVATION_SUBJECT_KINDS,
                    "quantity_kind": OBSERVATION_QUANTITY_KINDS,
                    "status": OBSERVATION_STATUSES,
                    "source": OBSERVATION_SOURCES,
                }[name]
                expected = (
                    _vocabulary_code(vocabulary, expected)
                    if isinstance(expected, str)
                    else expected
                )
            column = self._columns[name]
            if (
                not callable(expected)
                and column.dtype != object
                and np.asarray(expected).ndim == 0
            ):
                mask &= column == expected
                continue
            mask &= np.fromiter(
                (self._matches(value, expected) for value in column),
                dtype=bool,
                count=len(self),
            )
        return self._select_mask(mask)

    def _select_mask(self, mask: np.ndarray) -> Self:
        if mask.shape != (len(self),):
            raise ValidationError("selection mask must match observation row count")
        if np.all(mask):
            return self
        selected = np.flatnonzero(mask)
        if len(selected) == 0:
            selector: slice | np.ndarray = slice(0, 0)
        elif len(selected) == 1 or np.all(np.diff(selected) == 1):
            selector = slice(int(selected[0]), int(selected[-1]) + 1)
        else:
            selector = mask
        return self._from_columns(
            {name: column[selector] for name, column in self._columns.items()}
        )

    def take(self, indices: Any) -> Self:
        """Take rows by integer indices, retaining one array per field."""
        if isinstance(indices, slice):
            selector: slice | np.ndarray = indices
        else:
            index_array = np.asarray(indices, dtype=np.intp)
            if index_array.ndim == 0:
                index_array = index_array.reshape(1)
            selector = index_array
        return self._from_columns(
            {name: column[selector] for name, column in self._columns.items()}
        )

    @classmethod
    def concat(cls, parts: Sequence[Self]) -> Self:
        """Concatenate columnar record batches in their given order."""
        if not parts:
            return cls(
                {
                    "subject_kind": np.empty(0, dtype=np.int8),
                    "subject_payload": np.empty(0, dtype=object),
                    "quantity_kind": np.empty(0, dtype=np.int8),
                    "quantity_index": np.empty(0, dtype=np.int64),
                    "value": np.empty(0, dtype=np.float64),
                    "status": np.empty(0, dtype=np.int8),
                    "source": np.empty(0, dtype=np.int8),
                }
            )
        if len(parts) == 1:
            return parts[0]
        names = tuple(dict.fromkeys(name for part in parts for name in part._columns))
        return cls._from_columns(
            {
                name: np.concatenate(
                    [
                        part._columns.get(name, np.full(len(part), None, dtype=object))
                        for part in parts
                    ]
                )
                for name in names
            }
        )


@dataclass(frozen=True, kw_only=True)
class ObservationBatch:
    """A fixed-schema batch whose candidate IDs are derived from subjects."""

    schema: ObservationSchema
    records: ObservationRecords

    @classmethod
    def from_dense(
        cls,
        schema: ObservationSchema,
        candidate_ids: np.ndarray,
        f: np.ndarray,
        g: np.ndarray,
        cv: np.ndarray | None = None,
        *,
        source: ObservationSource = TRUE,
        status: ObservationStatus = OK,
        subject_kind: str = "candidate",
    ) -> Self:
        """Build a dense batch from complete objective/constraint arrays."""
        records = ObservationRecords.from_dense(
            candidate_ids,
            f,
            g,
            cv,
            source=source,
            status=status,
            subject_kind=subject_kind,
        )
        batch = cls(schema=schema, records=records)
        ids = np.array(
            np.asarray(candidate_ids, dtype=np.int64).reshape(-1),
            dtype=np.int64,
            order="C",
            copy=True,
        )
        objective = np.array(
            np.asarray(f, dtype=np.float64), dtype=np.float64, order="C", copy=True
        )
        constraint = np.array(
            np.asarray(g, dtype=np.float64), dtype=np.float64, order="C", copy=True
        )
        explicit_cv = (
            None
            if cv is None
            else np.array(
                np.asarray(cv, dtype=np.float64).reshape(-1),
                dtype=np.float64,
                order="C",
                copy=True,
            )
        )
        for array in (ids, objective, constraint, explicit_cv):
            if array is not None:
                array.flags.writeable = False
        object.__setattr__(
            batch,
            "_dense_inputs",
            (ids, objective, constraint, explicit_cv, subject_kind),
        )
        return batch

    def _candidate_data(self) -> tuple[CandidateIds, tuple[np.ndarray, ...]]:
        """Extract candidate IDs through the registered subject callbacks once."""
        cached = getattr(self, "_candidate_cache", None)
        if cached is not None:
            return cached
        result: list[int] = []
        seen: set[int] = set()
        subject_ids: list[np.ndarray] = []
        descriptors: dict[str, Any] = {}
        for kind, payload in zip(
            self.records.column("subject_kind"),
            self.records.column("subject_payload"),
        ):
            kind_name = str(kind)
            if kind_name not in descriptors:
                descriptor = OBSERVATION_SUBJECT_KINDS.get(kind_name)
                if descriptor is None:
                    raise ValidationError(f"unknown observation subject kind: {kind!r}")
                descriptors[kind_name] = descriptor
            descriptor = descriptors[kind_name]
            if descriptor is None:
                raise ValidationError(f"unknown observation subject kind: {kind!r}")
            ids = descriptor.candidate_ids(payload)
            if not isinstance(ids, np.ndarray):
                ids = np.asarray(ids)
            if ids.ndim != 1:
                ids = ids.reshape(-1)
            subject_ids.append(ids)
            for candidate_id in ids:
                value = int(candidate_id)
                if value not in seen:
                    seen.add(value)
                    result.append(value)
        output = np.asarray(result, dtype=np.int64)
        output.flags.writeable = False
        data = (output, tuple(subject_ids))
        object.__setattr__(self, "_candidate_cache", data)
        return data

    @property
    def candidate_ids(self) -> CandidateIds:
        """Return every candidate in first-subject-appearance order."""
        return self._candidate_data()[0]

    def _single_candidate_ids(self) -> tuple[np.ndarray, np.ndarray]:
        """Return candidate order and one candidate ID for each record."""
        candidate_ids, subject_ids = self._candidate_data()
        positions = {int(value): index for index, value in enumerate(candidate_ids)}
        record_positions = np.empty(len(self.records), dtype=np.intp)
        for row, ids in enumerate(subject_ids):
            if ids.size != 1:
                raise ValidationError(
                    "dense observation requires every subject to be single-candidate"
                )
            try:
                record_positions[row] = positions[int(ids[0])]
            except KeyError as exc:
                raise ValidationError(
                    "dense observation requires every subject to be single-candidate"
                ) from exc
        return candidate_ids, record_positions

    def _dense_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate complete dense form and materialize objective/constraint arrays."""
        cached = getattr(self, "_dense_cache", None)
        if cached is not None:
            return cached
        fast = self._dense_input_arrays()
        if fast is not None:
            object.__setattr__(self, "_dense_cache", fast)
            return fast
        candidate_ids, record_positions = self._single_candidate_ids()
        statuses = self.records.column("status")
        if not np.all(statuses == OK):
            raise ValidationError("dense observation requires all statuses to be ok")
        sources = self.records.column("source")
        if len(sources) and not np.all(sources == sources[0]):
            raise ValidationError("dense observation requires one common source")

        quantity_kinds = self.records.column("quantity_kind")
        quantity_indices = self.records.column("quantity_index")
        values = self.records.column("value")
        objective = self._dense_kind(
            OBJECTIVE,
            candidate_ids,
            record_positions,
            quantity_kinds,
            quantity_indices,
            values,
        )
        constraint = self._dense_kind(
            CONSTRAINT,
            candidate_ids,
            record_positions,
            quantity_kinds,
            quantity_indices,
            values,
        )
        result = (candidate_ids, objective, constraint)
        object.__setattr__(self, "_dense_cache", result)
        return result

    def _dense_input_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Use the vectorized dense input after checking its column layout."""
        inputs = getattr(self, "_dense_inputs", None)
        if inputs is None:
            return None
        candidate_ids, objective, constraint, explicit_cv, subject_kind = inputs
        objective_count = len(self.schema.indices(OBJECTIVE))
        constraint_count = len(self.schema.indices(CONSTRAINT))
        if (
            objective.shape[1] != objective_count
            or constraint.shape[1] != constraint_count
        ):
            return None

        columns = self.records.columns
        ok_code = _vocabulary_code(OBSERVATION_STATUSES, OK)
        statuses = columns["status"]
        if not np.all(statuses == ok_code):
            raise ValidationError("dense observation requires all statuses to be ok")
        sources = columns["source"]
        if len(sources) and not np.all(sources == sources[0]):
            raise ValidationError("dense observation requires one common source")

        quantity_count = objective_count + constraint_count + (explicit_cv is not None)
        row_count = len(candidate_ids) * quantity_count
        if len(self.records) != row_count:
            return None
        if len(np.unique(candidate_ids)) != len(candidate_ids):
            return None

        descriptor = OBSERVATION_SUBJECT_KINDS.get(subject_kind)
        if descriptor is None or descriptor.arity != "ONE":
            return None
        if len(descriptor.columns) != 1 or tuple(descriptor.columns[0].shape) != ():
            return None
        payload = columns["subject_payload"]
        if payload.dtype != np.dtype(descriptor.columns[0].dtype):
            return None
        expected_payload = np.repeat(
            candidate_ids.reshape(-1, 1), quantity_count, axis=0
        )
        if payload.shape != expected_payload.shape or not np.array_equal(
            payload, expected_payload
        ):
            return None
        subject_code = _vocabulary_code(OBSERVATION_SUBJECT_KINDS, subject_kind)
        if not np.all(columns["subject_kind"] == subject_code):
            return None

        base_kinds = np.concatenate(
            (
                np.full(
                    objective_count,
                    _vocabulary_code(OBSERVATION_QUANTITY_KINDS, OBJECTIVE),
                    dtype=np.int8,
                ),
                np.full(
                    constraint_count,
                    _vocabulary_code(OBSERVATION_QUANTITY_KINDS, CONSTRAINT),
                    dtype=np.int8,
                ),
                np.full(
                    1, _vocabulary_code(OBSERVATION_QUANTITY_KINDS, CV), dtype=np.int8
                )
                if explicit_cv is not None
                else np.empty(0, dtype=np.int8),
            )
        )
        base_indices = np.concatenate(
            (
                np.arange(objective_count, dtype=np.int64),
                np.arange(constraint_count, dtype=np.int64),
                np.zeros(1, dtype=np.int64)
                if explicit_cv is not None
                else np.empty(0, dtype=np.int64),
            )
        )
        if not np.array_equal(
            columns["quantity_kind"], np.tile(base_kinds, len(candidate_ids))
        ) or not np.array_equal(
            columns["quantity_index"], np.tile(base_indices, len(candidate_ids))
        ):
            return None

        expected_values = np.concatenate(
            (
                objective,
                constraint,
                explicit_cv.reshape(-1, 1)
                if explicit_cv is not None
                else np.empty((len(candidate_ids), 0), dtype=np.float64),
            ),
            axis=1,
        ).reshape(-1)
        if not np.array_equal(columns["value"], expected_values, equal_nan=True):
            return None

        candidate_ids = np.array(candidate_ids, dtype=np.int64, copy=True)
        candidate_ids.flags.writeable = False
        return candidate_ids, objective, constraint

    def _dense_kind(
        self,
        kind: str,
        candidate_ids: np.ndarray,
        record_positions: np.ndarray,
        quantity_kinds: np.ndarray,
        quantity_indices: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        """Materialize one declared dense quantity from a validated column set."""
        index_space = self.schema.indices(kind)
        index_positions = {index: column for column, index in enumerate(index_space)}
        rows = np.flatnonzero(quantity_kinds == kind)
        column_positions = np.empty(len(rows), dtype=np.intp)
        valid = np.ones(len(rows), dtype=bool)
        for offset, row in enumerate(rows):
            try:
                column_positions[offset] = index_positions[quantity_indices[row]]
            except (KeyError, TypeError):
                valid[offset] = False
        counts = np.zeros((len(candidate_ids), len(index_space)), dtype=np.intp)
        valid_rows = rows[valid]
        if len(valid_rows):
            np.add.at(
                counts,
                (record_positions[valid_rows], column_positions[valid]),
                1,
            )
        if not np.all(counts == 1):
            raise ValidationError(
                f"dense observation requires exactly one {kind} record per "
                "candidate and declared index"
            )
        result = np.empty((len(candidate_ids), len(index_space)), dtype=np.float64)
        try:
            result[record_positions[valid_rows], column_positions[valid]] = np.asarray(
                values[valid_rows], dtype=np.float64
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"dense observation requires scalar {kind} values"
            ) from exc
        result.flags.writeable = False
        return result

    @property
    def f(self) -> np.ndarray:
        """Return objectives only when the full batch is dense and complete."""
        return self._dense_arrays()[1]

    @property
    def g(self) -> np.ndarray:
        """Return constraints only when the full batch is dense and complete."""
        return self._dense_arrays()[2]

    @property
    def cv(self) -> np.ndarray:
        """Return explicit or derived scalar constraint violations."""
        dense_cache = self._dense_arrays()
        dense_inputs = getattr(self, "_dense_inputs", None)
        if dense_inputs is not None:
            explicit_cv = dense_inputs[3]
            if explicit_cv is not None:
                return explicit_cv
            result = np.maximum(dense_cache[2], 0.0).sum(axis=1)
            result.flags.writeable = False
            return result

        candidate_ids, _, constraint = dense_cache
        _, record_positions = self._single_candidate_ids()
        quantity_kinds = self.records.column("quantity_kind")
        quantity_indices = self.records.column("quantity_index")
        value_column = self.records.column("value")
        cv_mask = quantity_kinds == CV
        cv_rows = np.flatnonzero(cv_mask)
        if len(cv_rows) == 0:
            result = np.maximum(constraint, 0.0).sum(axis=1)
            result.flags.writeable = False
            return result
        if np.any(quantity_indices[cv_rows] != 0) or len(cv_rows) != len(candidate_ids):
            raise ValidationError(
                "dense observation requires exactly one cv record per candidate"
            )
        try:
            result = np.asarray(value_column[cv_rows], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "dense observation requires scalar cv values"
            ) from exc
        if len(np.unique(record_positions[cv_rows])) != len(candidate_ids):
            raise ValidationError(
                "dense observation requires exactly one cv record per candidate"
            )
        result.flags.writeable = False
        return result
