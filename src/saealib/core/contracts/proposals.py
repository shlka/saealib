"""Proposal batches and the requirements attached to them."""

from __future__ import annotations

from collections.abc import ItemsView, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, cast, runtime_checkable

import numpy as np

from saealib.core.contracts.feedback import (
    COMPLETE_BATCH,
    COMPLETION_MODES,
    CompletionMode,
)
from saealib.core.contracts.observation import (
    OBSERVATION_SOURCES,
    TRUE,
    ObservationSource,
    PortableValue,
)
from saealib.core.contracts.observations import QuantityRef
from saealib.core.contracts.relations import MANY, RELATION_KINDS, RelationKind
from saealib.exceptions import ValidationError
from saealib.identity import IDAllocator

__all__ = [
    "FeedbackRequirement",
    "FidelityRef",
    "ProposalBatch",
    "ProposalId",
    "ProposalRelations",
    "QuantityRequirement",
]

ProposalId: TypeAlias = int
FidelityRef: TypeAlias = int
_RelationStore: TypeAlias = np.ndarray | Mapping[str, np.ndarray]


@runtime_checkable
class _CandidatePopulation(Protocol):
    def __len__(self) -> int: ...

    def extract(self, indices: Any) -> Any:
        """Return the selected candidate rows."""
        ...


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{name} must be a non-negative integer")
    value = int(value)
    if value < 0 or value > np.iinfo(np.int64).max:
        raise ValidationError(f"{name} must be a non-negative int64 integer")
    return value


@dataclass(frozen=True, kw_only=True)
class QuantityRequirement:
    """The accepted source and minimum fidelity for one quantity."""

    quantity: QuantityRef
    sources: frozenset[ObservationSource] = frozenset({TRUE})
    fidelity: FidelityRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantity", QuantityRef.from_value(self.quantity))
        try:
            sources = frozenset(self.sources)
        except TypeError as exc:
            raise ValidationError("quantity sources must be an iterable") from exc
        if not sources or any(
            not OBSERVATION_SOURCES.contains(source) for source in sources
        ):
            raise ValidationError(
                "quantity sources must be registered observation sources"
            )
        object.__setattr__(self, "sources", sources)
        if self.fidelity is not None:
            object.__setattr__(
                self,
                "fidelity",
                _non_negative_int(self.fidelity, "fidelity"),
            )


@dataclass(frozen=True, kw_only=True)
class FeedbackRequirement:
    """Feedback needed before a proposer can act on a proposal."""

    quantities: tuple[QuantityRequirement, ...]
    completion: CompletionMode = COMPLETE_BATCH

    def __post_init__(self) -> None:
        try:
            quantities = tuple(
                item
                if isinstance(item, QuantityRequirement)
                else QuantityRequirement(**item)
                if isinstance(item, Mapping)
                else item
                for item in self.quantities
            )
        except TypeError as exc:
            raise ValidationError("quantities must be an iterable") from exc
        if any(not isinstance(item, QuantityRequirement) for item in quantities):
            raise ValidationError("quantities must contain QuantityRequirement values")
        if not COMPLETION_MODES.contains(self.completion):
            raise ValidationError(f"unknown completion mode: {self.completion!r}")
        object.__setattr__(self, "quantities", quantities)


class ProposalRelations(Mapping[str, _RelationStore]):
    """Immutable, column-oriented values for registered proposal relations."""

    def __init__(
        self,
        relations: Mapping[str, Any] | None = None,
        *,
        row_count: int | None = None,
    ) -> None:
        if relations is None:
            relations = {}
        if not isinstance(relations, Mapping):
            raise ValidationError("relations must be a mapping")
        if row_count is not None:
            row_count = _non_negative_int(row_count, "relation row_count")
        stores: dict[str, _RelationStore] = {}
        inferred_row_count: int | None = None
        for name, value in relations.items():
            kind = RELATION_KINDS.get(name)
            if kind is None:
                raise ValidationError(f"unknown relation kind: {name!r}")
            payload, rows = self._normalize(kind, value)
            if inferred_row_count is None:
                inferred_row_count = rows
            elif rows != inferred_row_count:
                raise ValidationError("relation columns must have the same row count")
            stores[name] = payload
        if (
            row_count is not None
            and inferred_row_count is not None
            and row_count != inferred_row_count
        ):
            raise ValidationError("relation columns must have the same row count")
        self._stores = MappingProxyType(stores)
        self._row_count = (
            row_count if row_count is not None else inferred_row_count or 0
        )

    @staticmethod
    def _normalize(kind: RelationKind, value: Any) -> tuple[_RelationStore, int]:
        if not kind.columns:
            raise ValidationError(f"relation {kind.name!r} must declare columns")
        if len(kind.columns) == 1 and not isinstance(value, Mapping):
            values: Mapping[str, Any] = {kind.columns[0].name: value}
            single = True
        elif isinstance(value, Mapping):
            values = value
            single = False
        else:
            raise ValidationError(f"relation {kind.name!r} must provide its columns")
        arrays: dict[str, np.ndarray] = {}
        rows: int | None = None
        for column in kind.columns:
            if column.name not in values:
                raise ValidationError(
                    f"relation {kind.name!r} is missing column {column.name!r}"
                )
            try:
                raw = np.asarray(values[column.name])
            except ValueError:
                # NumPy refuses ragged nested input without an explicit object
                # dtype; variable-length MANY is the one permitted fallback.
                raw = np.asarray(values[column.name], dtype=object)
            expected_tail = tuple(column.shape)
            if raw.ndim == 0:
                raise ValidationError(
                    f"relation column {column.name!r} must contain rows"
                )
            if (
                raw.dtype == np.dtype(object)
                and kind.arity == MANY
                and not expected_tail
            ):
                try:
                    normalized = np.asarray(raw, dtype=column.dtype)
                except (TypeError, ValueError):
                    normalized = raw
            elif raw.shape[1:] == expected_tail:
                try:
                    normalized = np.asarray(raw, dtype=column.dtype)
                except (TypeError, ValueError) as exc:
                    raise ValidationError(
                        f"relation column {column.name!r} has the wrong dtype"
                    ) from exc
            elif kind.arity == MANY and not expected_tail and raw.ndim >= 2:
                # Fixed-width MANY remains a typed numeric array, not object.
                normalized = np.asarray(raw, dtype=column.dtype)
            else:
                raise ValidationError(
                    f"relation column {column.name!r} has the wrong shape"
                )
            if normalized.dtype != np.dtype(column.dtype) and not (
                kind.arity == MANY and normalized.dtype == np.dtype(object)
            ):
                raise ValidationError(
                    f"relation column {column.name!r} has the wrong dtype"
                )
            if rows is None:
                rows = len(normalized)
            elif len(normalized) != rows:
                raise ValidationError("relation columns must have the same row count")
            if normalized.dtype == object and kind.arity != MANY:
                raise ValidationError(
                    "object relation columns are only allowed for MANY"
                )
            normalized = np.array(
                normalized, dtype=normalized.dtype, copy=True, order="C"
            )
            # Object dtype is reserved for variable-length MANY payloads.
            # The outer row store remains vectorized; fixed-width columns take
            # the native typed path above without a per-row Python loop.
            normalized.flags.writeable = False
            arrays[column.name] = normalized
        assert rows is not None
        if single:
            return arrays[kind.columns[0].name], rows
        return MappingProxyType(arrays), rows

    def __getitem__(self, name: str) -> _RelationStore:
        return self._stores[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._stores)

    def __len__(self) -> int:
        return len(self._stores)

    @property
    def row_count(self) -> int:
        """Return the number of candidate rows represented by the relations."""
        return self._row_count

    def items(self) -> ItemsView[str, _RelationStore]:
        """Return the immutable relation mapping items."""
        return self._stores.items()

    @property
    def columns(self) -> Mapping[str, _RelationStore]:
        """Return the read-only relation column mapping."""
        return self._stores

    def take(self, indices: Any) -> ProposalRelations:
        """Select rows and let every registered kind reindex its payload."""
        selected = _normalize_indices(indices, self._row_count)
        if np.any(selected < 0) or np.any(selected >= self._row_count):
            raise ValidationError("relation index is outside the row range")
        result: dict[str, Any] = {}
        for name, payload in self._stores.items():
            kind = cast(RelationKind, RELATION_KINDS.get(name))
            if isinstance(payload, Mapping):
                selected_payload: Any = {
                    key: value[selected] for key, value in payload.items()
                }
            else:
                selected_payload = payload[selected]
            # This call is intentional even for identity relations.
            result[name] = kind.reindex(selected_payload, selected)
        return ProposalRelations(result, row_count=len(selected))


@dataclass(frozen=True, kw_only=True)
class ProposalBatch:
    """A candidate population plus its relations and feedback requirement."""

    proposal_id: ProposalId
    candidates: Any
    relations: ProposalRelations
    requirements: FeedbackRequirement
    metadata: Mapping[str, PortableValue] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        proposal_id = _non_negative_int(self.proposal_id, "proposal_id")
        if not isinstance(self.candidates, _CandidatePopulation):
            raise ValidationError("candidates must be a Population")
        if not isinstance(self.relations, ProposalRelations):
            raise ValidationError("relations must be ProposalRelations")
        if not isinstance(self.requirements, FeedbackRequirement):
            raise ValidationError("requirements must be FeedbackRequirement")
        relations = self.relations
        if len(relations) == 0 and relations.row_count == 0:
            relations = ProposalRelations({}, row_count=len(self.candidates))
            object.__setattr__(self, "relations", relations)
        if len(self.candidates) != relations.row_count:
            raise ValidationError("candidate and relation row counts must match")
        if not isinstance(self.metadata, Mapping):
            raise ValidationError("metadata must be a mapping")
        object.__setattr__(self, "proposal_id", proposal_id)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_allocator(cls, allocator: IDAllocator, **kwargs: Any) -> ProposalBatch:
        """Allocate the proposal id explicitly from an existing allocator."""
        if not isinstance(allocator, IDAllocator):
            raise ValidationError("allocator must be an IDAllocator")
        allocated = allocator.allocate(1)
        return cls(proposal_id=int(allocated[0]), **kwargs)

    def take(self, indices: Any) -> ProposalBatch:
        """Take the same candidate rows and relation rows together."""
        selected = _normalize_indices(indices, len(self.candidates))
        return ProposalBatch(
            proposal_id=self.proposal_id,
            candidates=self.candidates.extract(selected),
            relations=self.relations.take(selected),
            requirements=self.requirements,
            metadata=self.metadata,
        )


def _normalize_indices(indices: Any, row_count: int) -> np.ndarray:
    if isinstance(indices, slice):
        return np.arange(row_count, dtype=np.intp)[indices]
    raw = np.asarray(indices)
    if raw.ndim == 0:
        raw = raw.reshape(1)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValidationError(
            "relation indices must be a one-dimensional integer array"
        )
    return raw.astype(np.intp, copy=False)
