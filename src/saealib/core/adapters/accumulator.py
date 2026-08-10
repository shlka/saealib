"""Accumulate partial feedback for a complete-batch consumer.

The accumulator is deliberately independent of the compiler and scheduler.  A
caller registers a :class:`~saealib.core.contracts.proposals.ProposalBatch`,
adds its :class:`~saealib.core.contracts.feedbacks.FeedbackBatch` deliveries,
and takes completed batches from a ready queue::

    accumulator.register(proposal)
    accumulator.add(delivery)
    completed = accumulator.pop_ready()

The implementation makes the following semantic choices, which are the
runtime form of the feedback delivery contract:

* exact duplicate ``(proposal_id, sequence)`` deliveries are idempotent while
  a proposal is buffered; a conflicting duplicate raises ``ValidationError``;
* ``in_order`` requires strictly increasing sequence values, while
  ``out_of_order_allowed`` accepts any order and passes the envelope sequence
  to ``SelectionPolicy``;
* duplicate observations are resolved only by the supplied ``SelectionPolicy``;
* ``ok``, ``failed``, ``cancelled``, and ``timeout`` are all terminal coverage
  for a candidate/quantity slot.  A failure is delivered as data so a failed
  evaluation cannot deadlock a complete-batch consumer;
* completion is the Cartesian product of proposal candidates and required
  quantities, with each record satisfying its quantity's source and fidelity
  requirement;
* an incomplete ``final=True`` delivery raises a diagnostic
  ``ValidationError`` and releases the buffer;
* deliveries from multiple channels are rejected because one
  ``FeedbackBatch.channel`` cannot faithfully represent a mixed envelope;
* observations whose source is outside ``FeedbackContract.accepted_sources``
  are dropped before buffering.  If that makes final coverage incomplete, the
  final-delivery diagnostic explains the missing slots.

Only ``complete_batch`` contracts are accepted.  Buffering in front of a
``partial_allowed`` consumer would change a property that consumer explicitly
declared, so it is not a lossless adapter for that contract.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from math import isnan
from typing import Any, cast

import numpy as np

from saealib.core.contracts.feedback import (
    COMPLETE_BATCH,
    FEEDBACK_CHANNELS,
    IN_ORDER,
    OUT_OF_ORDER_ALLOWED,
)
from saealib.core.contracts.feedbacks import FeedbackBatch, FeedbackContract
from saealib.core.contracts.observation import (
    OBSERVATION_SOURCES,
    OBSERVATION_SUBJECT_KINDS,
)
from saealib.core.contracts.observations import (
    ObservationBatch,
    ObservationRecord,
    ObservationRecords,
    ObservationSchema,
    ObservationSubject,
    QuantityRef,
)
from saealib.core.contracts.proposals import (
    FeedbackRequirement,
    ProposalBatch,
    QuantityRequirement,
)
from saealib.exceptions import ValidationError
from saealib.identity import CandidateIds
from saealib.policies.feedback import DEFAULT_SELECTION_POLICY, SelectionPolicy

__all__ = ["FeedbackAccumulator"]


_ACCUMULATOR_CODEC = "feedback_accumulator"
_ACCUMULATOR_CODEC_VERSION = 1


def _state_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"feedback accumulator {name} must be an integer")
    result = int(value)
    if result < 0 or result > np.iinfo(np.int64).max:
        raise ValidationError(
            f"feedback accumulator {name} must be a non-negative int64 integer"
        )
    return result


def _state_sequence_key(value: Any) -> int:
    if not isinstance(value, str) or not value.isdecimal():
        raise ValidationError("feedback accumulator seen sequence is malformed")
    return _state_non_negative_int(int(value), "seen sequence")


@dataclass(frozen=True)
class _Delivery:
    sequence: int
    records: ObservationRecords


@dataclass
class _ProposalBuffer:
    proposal_id: int
    candidate_ids: CandidateIds
    candidate_id_set: frozenset[int]
    requirements: tuple[QuantityRequirement, ...]
    deliveries: list[_Delivery] = field(default_factory=list)
    seen: dict[int, FeedbackBatch] = field(default_factory=dict)
    channel: str | None = None
    schema: ObservationSchema | None = None
    last_sequence: int | None = None


class FeedbackAccumulator:
    """Buffer partial feedback until one complete final batch is possible.

    Parameters
    ----------
    contract:
        The feedback contract of the downstream consumer.  It must declare
        ``complete_batch``; the accumulator is the adapter for that consumer
        only.
    selection_policy:
        The existing named policy used to resolve records competing for one
        candidate/quantity.  No source, status, fidelity, or sequence ranking
        is implemented in this class.

    Notes
    -----
    ``register`` captures candidate IDs and the proposal requirement.  This
    makes later ``add`` calls need only carry the existing delivery envelope,
    and it prevents a mutable caller-side proposal from changing the
    completion boundary halfway through accumulation.  ``add`` never returns
    a batch; ``pop_ready`` is the explicit retrieval operation.  Completion
    removes all per-proposal delivery state immediately, before the ready
    batch is returned.
    """

    def __init__(
        self,
        contract: FeedbackContract,
        *,
        selection_policy: SelectionPolicy = DEFAULT_SELECTION_POLICY,
    ) -> None:
        if not isinstance(contract, FeedbackContract):
            raise ValidationError("contract must be a FeedbackContract")
        if contract.completion != COMPLETE_BATCH:
            raise ValidationError(
                "FeedbackAccumulator requires a complete_batch consumer"
            )
        if not isinstance(selection_policy, SelectionPolicy):
            raise ValidationError("selection_policy must be a SelectionPolicy")
        self.contract = contract
        self.selection_policy = selection_policy
        self._buffers: dict[int, _ProposalBuffer] = {}
        self._ready: deque[FeedbackBatch] = deque()

    @property
    def buffered_proposal_count(self) -> int:
        """Return the number of proposals that still retain delivery state."""
        return len(self._buffers)

    @property
    def ready_count(self) -> int:
        """Return the number of completed batches waiting to be taken."""
        return len(self._ready)

    def last_sequence(self, proposal_id: int) -> int:
        """Return the greatest accepted sequence for a buffered proposal."""
        buffer = self._buffers.get(int(proposal_id))
        if buffer is None or not buffer.seen:
            return -1
        return max(buffer.seen)

    def register(self, proposal: ProposalBatch) -> None:
        """Register one proposal before adding its feedback deliveries.

        Registration is idempotent for the same captured candidate and
        requirement values.  A different proposal with the same ID is
        rejected because merging their completion boundaries would be silent
        data corruption.
        """
        if not isinstance(proposal, ProposalBatch):
            raise ValidationError("proposal must be a ProposalBatch")
        if not self.contract.contains_requirement(proposal.requirements):
            raise ValidationError(
                "proposal feedback requirement is wider than the consumer "
                "FeedbackContract"
            )
        candidate_ids = _proposal_candidate_ids(proposal)
        if len(np.unique(candidate_ids)) != len(candidate_ids):
            raise ValidationError("proposal candidate IDs must be unique")
        if np.any(candidate_ids < 0):
            raise ValidationError("proposal candidate IDs must be non-negative")
        requirements = tuple(proposal.requirements.quantities)
        existing = self._buffers.get(proposal.proposal_id)
        if existing is not None:
            if np.array_equal(existing.candidate_ids, candidate_ids) and (
                existing.requirements == requirements
            ):
                return
            raise ValidationError(
                f"proposal {proposal.proposal_id} is already registered"
            )
        candidate_ids.flags.writeable = False
        self._buffers[proposal.proposal_id] = _ProposalBuffer(
            proposal_id=proposal.proposal_id,
            candidate_ids=candidate_ids,
            candidate_id_set=frozenset(int(value) for value in candidate_ids),
            requirements=requirements,
        )

    def add(self, batch: FeedbackBatch) -> None:
        """Add one delivery and queue a final batch when completion is met.

        The proposal must have been registered first.  Exact duplicate
        deliveries are ignored before any ordering check; this makes a retry
        harmless.  Once a final delivery completes or invalidates a proposal,
        its buffer is removed and later deliveries are protocol errors.
        """
        if not isinstance(batch, FeedbackBatch):
            raise ValidationError("batch must be a FeedbackBatch")
        state = self._buffers.get(batch.proposal_id)
        if state is None:
            raise ValidationError(
                f"proposal {batch.proposal_id} is not registered or is finalized"
            )

        duplicate = state.seen.get(batch.sequence)
        if duplicate is not None:
            if _feedback_batches_equal(duplicate, batch):
                return
            raise ValidationError(
                "conflicting duplicate feedback delivery for "
                f"proposal {batch.proposal_id}, sequence {batch.sequence}"
            )

        if batch.channel not in self.contract.accepted_channels:
            raise ValidationError(
                f"feedback channel {batch.channel!r} is not accepted by the consumer"
            )
        if state.channel is not None and batch.channel != state.channel:
            raise ValidationError(
                "cannot merge feedback deliveries from different channels "
                f"({state.channel!r} and {batch.channel!r})"
            )
        if (
            self.contract.ordering == IN_ORDER
            and state.last_sequence is not None
            and batch.sequence <= state.last_sequence
        ):
            raise ValidationError(
                "feedback sequence arrived out of order for an in_order consumer"
            )
        if state.schema is not None and not _schemas_equal(
            state.schema, batch.observations.schema
        ):
            raise ValidationError(
                f"proposal {batch.proposal_id} feedback schemas do not match"
            )

        records = _accepted_records(batch.observations.records, self.contract)
        record_ids = _record_candidate_ids(records)
        if len(record_ids) and not np.all(
            np.isin(record_ids, tuple(state.candidate_id_set))
        ):
            raise ValidationError(
                f"feedback for proposal {batch.proposal_id} contains an unknown "
                "candidate ID"
            )

        state.seen[batch.sequence] = batch
        state.deliveries.append(_Delivery(batch.sequence, records))
        state.channel = batch.channel
        state.schema = batch.observations.schema
        if self.contract.ordering == IN_ORDER:
            state.last_sequence = batch.sequence

        if not batch.final:
            return

        try:
            completed = self._finalize(state)
        except ValidationError:
            # ``final`` promises that no recovery delivery is valid.  Release
            # the records even on the diagnostic path so an impossible
            # proposal cannot pin memory or deadlock the caller.
            self._buffers.pop(batch.proposal_id, None)
            raise
        self._buffers.pop(batch.proposal_id, None)
        self._ready.append(completed)

    def finalize(self, proposal_id: int) -> None:
        """Complete a proposal after its terminal scheduler update arrives."""
        state = self._buffers.get(proposal_id)
        if state is None:
            raise ValidationError(
                f"proposal {proposal_id} is not registered or is finalized"
            )
        try:
            completed = self._finalize(state)
        except ValidationError:
            self._buffers.pop(proposal_id, None)
            raise
        self._buffers.pop(proposal_id, None)
        self._ready.append(completed)

    def discard(self, proposal_id: int) -> None:
        """Drop an incomplete proposal after a terminal non-retryable failure."""
        self._buffers.pop(proposal_id, None)

    def pop_ready(self) -> FeedbackBatch | None:
        """Remove and return the oldest completed batch, if one is ready."""
        if not self._ready:
            return None
        return self._ready.popleft()

    def drain_ready(self) -> tuple[FeedbackBatch, ...]:
        """Remove and return all currently completed batches in queue order."""
        result = tuple(self._ready)
        self._ready.clear()
        return result

    def has_delivery(self, proposal_id: int, sequence: int) -> bool:
        """Return whether a delivery is already buffered or ready for replay.

        Scheduler reattachment uses this query to distinguish a redelivered
        transport update from a new delivery without reaching into the
        accumulator's mutable buffer implementation.
        """
        buffered = self._buffers.get(proposal_id)
        if buffered is not None and sequence in buffered.seen:
            return True
        return any(
            batch.proposal_id == proposal_id and batch.sequence == sequence
            for batch in self._ready
        )

    def to_state(self) -> dict[str, Any]:
        """Return a portable, versioned snapshot of live accumulator state.

        This is intentionally a value codec rather than a pickle representation:
        every field that affects completion, duplicate handling, or selection is
        present in the payload and is validated again by :meth:`from_state`.
        """
        return {
            "codec": _ACCUMULATOR_CODEC,
            "version": _ACCUMULATOR_CODEC_VERSION,
            "buffers": [
                {
                    "proposal_id": state.proposal_id,
                    "candidate_ids": [int(value) for value in state.candidate_ids],
                    "requirements": [
                        _requirement_to_state(item) for item in state.requirements
                    ],
                    "channel": state.channel,
                    "schema": None
                    if state.schema is None
                    else _schema_to_state(state.schema),
                    "last_sequence": state.last_sequence,
                    "deliveries": [
                        {
                            "sequence": item.sequence,
                            "records": _records_to_state(item.records),
                        }
                        for item in state.deliveries
                    ],
                    "seen": {
                        str(sequence): _feedback_to_state(item)
                        for sequence, item in state.seen.items()
                    },
                }
                for state in self._buffers.values()
            ],
            "ready": [_feedback_to_state(item) for item in self._ready],
        }

    @classmethod
    def from_state(
        cls,
        contract: FeedbackContract,
        value: Any,
        *,
        selection_policy: SelectionPolicy = DEFAULT_SELECTION_POLICY,
    ) -> FeedbackAccumulator:
        """Restore an accumulator from its portable versioned snapshot."""
        accumulator = cls(contract, selection_policy=selection_policy)
        if not isinstance(value, Mapping) or value.get("codec") != _ACCUMULATOR_CODEC:
            raise ValidationError("feedback accumulator state codec is malformed")
        if value.get("version") != _ACCUMULATOR_CODEC_VERSION:
            raise ValidationError("unsupported feedback accumulator codec version")
        buffers = value.get("buffers")
        ready = value.get("ready")
        if not isinstance(buffers, list) or not isinstance(ready, list):
            raise ValidationError("feedback accumulator state is malformed")
        for item in buffers:
            state = _buffer_from_state(item, contract)
            if state.proposal_id in accumulator._buffers:
                raise ValidationError("duplicate feedback accumulator proposal")
            accumulator._buffers[state.proposal_id] = state
        ready_batches = []
        for item in ready:
            batch = _feedback_from_state(item)
            if not batch.final:
                raise ValidationError(
                    "ready feedback accumulator batches must be final"
                )
            if batch.channel not in contract.accepted_channels:
                raise ValidationError(
                    "ready feedback accumulator channel is not accepted"
                )
            if any(
                source not in contract.accepted_sources
                for source in batch.observations.records.column("source")
            ):
                raise ValidationError(
                    "ready feedback accumulator batch contains an unaccepted source"
                )
            if batch.proposal_id in accumulator._buffers or any(
                existing.proposal_id == batch.proposal_id for existing in ready_batches
            ):
                raise ValidationError("duplicate ready feedback accumulator proposal")
            ready_batches.append(batch)
        accumulator._ready.extend(ready_batches)
        return accumulator

    def restore_state(self, value: Any) -> None:
        """Replace live buffers and ready queue from a portable snapshot."""
        restored = type(self).from_state(
            self.contract, value, selection_policy=self.selection_policy
        )
        self._buffers = restored._buffers
        self._ready = restored._ready

    def _finalize(self, state: _ProposalBuffer) -> FeedbackBatch:
        if state.schema is None or state.channel is None or not state.deliveries:
            raise ValidationError(
                f"proposal {state.proposal_id} has no feedback schema or delivery"
            )
        records = ObservationRecords.concat(
            [delivery.records for delivery in state.deliveries]
        )
        record_ids = _record_candidate_ids(records)
        quantity_kinds = records.column("quantity_kind")
        quantity_indices = records.column("quantity_index")
        sources = records.column("source")
        statuses = records.column("status")
        fidelity = _fidelity_values(records)
        sequences = np.concatenate(
            [
                np.full(len(delivery.records), delivery.sequence, dtype=np.int64)
                for delivery in state.deliveries
            ]
        )
        batch_indices = np.arange(len(records), dtype=np.int64)

        unique_keys, record_group_ids, index_types, index_tokens = _dense_record_groups(
            record_ids, quantity_kinds, quantity_indices
        )
        candidate_ids = np.sort(state.candidate_ids)
        record_candidate_positions = np.searchsorted(candidate_ids, record_ids)
        requirement_count = len(state.requirements)
        required_slot_count = len(candidate_ids) * requirement_count

        required_group_mask = np.zeros(len(unique_keys), dtype=bool)
        for requirement in state.requirements:
            required_group_mask |= (
                (unique_keys["kind"] == requirement.quantity.kind)
                & (
                    unique_keys["index_type"]
                    == _quantity_index_type(requirement.quantity.index)
                )
                & (unique_keys["index"] == str(requirement.quantity.index))
            )

        # Each required quantity gets one dense group per candidate.  A row is
        # copied into every requirement group it can satisfy; this preserves
        # the pre-existing duplicate-requirement semantics while keeping one
        # SelectionPolicy kernel invocation for the whole proposal.
        selection_row_parts: list[np.ndarray] = []
        selection_group_parts: list[np.ndarray] = []
        for requirement_index, requirement in enumerate(state.requirements):
            eligible = (
                (quantity_kinds == requirement.quantity.kind)
                & (index_types == _quantity_index_type(requirement.quantity.index))
                & (index_tokens == str(requirement.quantity.index))
                & np.isin(sources, tuple(requirement.sources))
            )
            if requirement.fidelity is not None:
                eligible &= fidelity >= requirement.fidelity
            rows = np.flatnonzero(eligible)
            selection_row_parts.append(rows)
            selection_group_parts.append(
                record_candidate_positions[rows] * requirement_count + requirement_index
            )

        # Preserve accepted, non-required quantities too, but resolve their
        # duplicates in the same batch.  Remapping these groups after the
        # required slots makes all group IDs dense for SelectionPolicy.
        nonrequired_group_ids = np.flatnonzero(~required_group_mask)
        nonrequired_rank = np.full(len(unique_keys), -1, dtype=np.intp)
        nonrequired_rank[nonrequired_group_ids] = np.arange(
            len(nonrequired_group_ids), dtype=np.intp
        )
        nonrequired_rows = np.flatnonzero(~required_group_mask[record_group_ids])
        selection_row_parts.append(nonrequired_rows)
        selection_group_parts.append(
            required_slot_count + nonrequired_rank[record_group_ids[nonrequired_rows]]
        )

        if selection_row_parts:
            selection_rows = np.concatenate(selection_row_parts)
            selection_groups = np.concatenate(selection_group_parts)
        else:
            selection_rows = np.empty(0, dtype=np.intp)
            selection_groups = np.empty(0, dtype=np.intp)

        if len(selection_rows):
            selected_entry_positions = self.selection_policy.select(
                selection_groups,
                sources[selection_rows],
                fidelity[selection_rows],
                sequences[selection_rows],
                statuses[selection_rows],
                batch_indices[selection_rows],
            )
        else:
            selected_entry_positions = np.empty(0, dtype=np.intp)

        selected_groups = selection_groups[selected_entry_positions]
        if required_slot_count:
            covered = np.zeros(required_slot_count, dtype=bool)
            selected_required_groups = selected_groups[
                selected_groups < required_slot_count
            ]
            covered[selected_required_groups] = True
            missing_slots = np.flatnonzero(~covered)
            if len(missing_slots):
                missing = [
                    _missing_slot_description(
                        int(candidate_ids[slot // requirement_count]),
                        state.requirements[slot % requirement_count],
                    )
                    for slot in missing_slots
                ]
                detail = "; ".join(missing[:8])
                if len(missing) > 8:
                    detail += f"; ... ({len(missing)} missing slots)"
                raise ValidationError(
                    f"final feedback for proposal {state.proposal_id} is incomplete: "
                    f"{detail}"
                )

        selected = np.unique(selection_rows[selected_entry_positions])
        output_records = records.take(selected)
        output_sequence = max(delivery.sequence for delivery in state.deliveries)
        return FeedbackBatch(
            proposal_id=state.proposal_id,
            observations=ObservationBatch(schema=state.schema, records=output_records),
            channel=state.channel,
            final=True,
            sequence=output_sequence,
        )


def _portable_to_state(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": [_portable_to_state(item) for item in value.tolist()],
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, np.generic):
        return {
            "__scalar__": _portable_to_state(value.item()),
            "dtype": str(value.dtype),
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValidationError("accumulator metadata keys must be strings")
        return {key: _portable_to_state(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return {"__tuple__": [_portable_to_state(item) for item in value]}
    if isinstance(value, list):
        return [_portable_to_state(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValidationError(
        f"feedback accumulator value is not portable: {type(value).__name__}"
    )


def _portable_from_state(value: Any) -> Any:
    if isinstance(value, list):
        return [_portable_from_state(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    if "__ndarray__" in value:
        try:
            raw = _portable_from_state(value["__ndarray__"])
            return np.asarray(raw, dtype=value["dtype"]).reshape(tuple(value["shape"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("feedback accumulator array is malformed") from exc
    if "__scalar__" in value:
        try:
            return np.asarray(
                _portable_from_state(value["__scalar__"]), dtype=value["dtype"]
            ).item()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("feedback accumulator scalar is malformed") from exc
    if "__tuple__" in value:
        return tuple(_portable_from_state(item) for item in value["__tuple__"])
    return {str(key): _portable_from_state(item) for key, item in value.items()}


def _schema_to_state(schema: ObservationSchema) -> dict[str, Any]:
    return {
        "objective_count": schema.objective_count,
        "constraint_count": schema.constraint_count,
        "quantities": {
            str(kind): [_portable_to_state(index) for index in schema.indices(kind)]
            for kind in schema.quantity_kinds
        },
        "extra_quantities": list(schema.extra_quantities),
        "schema_version": schema.schema_version,
    }


def _schema_from_state(value: Any) -> ObservationSchema:
    if not isinstance(value, Mapping) or not isinstance(
        value.get("quantities"), Mapping
    ):
        raise ValidationError("feedback accumulator schema is malformed")
    try:
        return ObservationSchema(
            objective_count=int(value["objective_count"]),
            constraint_count=int(value["constraint_count"]),
            quantities={
                str(kind): tuple(_portable_from_state(index) for index in indices)
                for kind, indices in value["quantities"].items()
            },
            extra_quantities=tuple(value.get("extra_quantities", ())),
            schema_version=int(value["schema_version"]),
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise ValidationError("feedback accumulator schema is malformed") from exc


def _record_to_state(record: ObservationRecord) -> dict[str, Any]:
    subject = ObservationSubject.from_value(record.subject)
    quantity = QuantityRef.from_value(record.quantity)
    provenance = record.provenance if record.provenance is not None else {}
    return {
        "subject": {
            "kind": subject.kind,
            "payload": _portable_to_state(subject.payload),
        },
        "quantity": {
            "kind": quantity.kind,
            "index": _portable_to_state(quantity.index),
        },
        "value": _portable_to_state(record.value),
        "status": record.status,
        "source": record.source,
        "uncertainty": _portable_to_state(record.uncertainty),
        "fidelity": _portable_to_state(record.fidelity),
        "cost": _portable_to_state(record.cost),
        "timestamp": _portable_to_state(record.timestamp),
        "provenance": _portable_to_state(dict(provenance)),
    }


def _record_from_state(value: Any) -> ObservationRecord:
    if not isinstance(value, Mapping):
        raise ValidationError("feedback accumulator record is malformed")
    try:
        subject = value["subject"]
        quantity = value["quantity"]
        if not isinstance(subject, Mapping) or not isinstance(quantity, Mapping):
            raise ValidationError(
                "feedback accumulator record references are malformed"
            )
        provenance = _portable_from_state(value.get("provenance", {}))
        if provenance is None:
            provenance = {}
        return ObservationRecord(
            subject=ObservationSubject(
                kind=subject["kind"],
                payload=_portable_from_state(subject["payload"]),
            ),
            quantity=QuantityRef(
                kind=quantity["kind"],
                index=_portable_from_state(quantity["index"]),
            ),
            value=_portable_from_state(value["value"]),
            status=value["status"],
            source=value["source"],
            uncertainty=_portable_from_state(value.get("uncertainty")),
            fidelity=_portable_from_state(value.get("fidelity")),
            cost=_portable_from_state(value.get("cost")),
            timestamp=_portable_from_state(value.get("timestamp")),
            provenance=provenance,
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise ValidationError("feedback accumulator record is malformed") from exc


def _records_to_state(records: ObservationRecords) -> list[dict[str, Any]]:
    subject_kinds = records.column("subject_kind")
    subject_payloads = records.column("subject_payload")
    quantity_kinds = records.column("quantity_kind")
    quantity_indices = records.column("quantity_index")
    values = records.column("value")
    statuses = records.column("status")
    sources = records.column("source")
    optional_columns: dict[str, np.ndarray] = {}
    for name in ("uncertainty", "fidelity", "cost", "timestamp", "provenance"):
        try:
            optional_columns[name] = records.column(name)
        except KeyError:
            continue
    encoded = []
    for index in range(len(records)):
        optional = {
            name: column[index]
            for name, column in optional_columns.items()
            if not (name == "provenance" and column[index] is None)
        }
        encoded.append(
            _record_to_state(
                ObservationRecord(
                    subject=ObservationSubject(
                        kind=subject_kinds[index],
                        payload=subject_payloads[index],
                    ),
                    quantity=QuantityRef(
                        kind=quantity_kinds[index],
                        index=(
                            int(quantity_indices[index])
                            if isinstance(quantity_indices[index], np.integer)
                            else quantity_indices[index]
                        ),
                    ),
                    value=values[index],
                    status=statuses[index],
                    source=sources[index],
                    **optional,
                )
            )
        )
    return encoded


def _records_from_state(value: Any) -> ObservationRecords:
    if not isinstance(value, list):
        raise ValidationError("feedback accumulator records are malformed")
    return ObservationRecords.from_records([_record_from_state(item) for item in value])


def _requirement_to_state(requirement: QuantityRequirement) -> dict[str, Any]:
    return {
        "quantity": {
            "kind": requirement.quantity.kind,
            "index": _portable_to_state(requirement.quantity.index),
        },
        "sources": sorted(requirement.sources),
        "fidelity": requirement.fidelity,
    }


def _requirement_from_state(value: Any) -> QuantityRequirement:
    if not isinstance(value, Mapping):
        raise ValidationError("feedback accumulator requirement is malformed")
    try:
        quantity = value["quantity"]
        sources = value["sources"]
        if not isinstance(quantity, Mapping) or not isinstance(
            sources, (list, tuple, set, frozenset)
        ):
            raise ValidationError(
                "feedback accumulator requirement fields are malformed"
            )
        return QuantityRequirement(
            quantity=QuantityRef(
                kind=quantity["kind"],
                index=_portable_from_state(quantity["index"]),
            ),
            sources=frozenset(sources),
            fidelity=value.get("fidelity"),
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise ValidationError("feedback accumulator requirement is malformed") from exc


def _feedback_to_state(batch: FeedbackBatch) -> dict[str, Any]:
    return {
        "proposal_id": batch.proposal_id,
        "channel": batch.channel,
        "final": batch.final,
        "sequence": batch.sequence,
        "schema": _schema_to_state(batch.observations.schema),
        "records": _records_to_state(batch.observations.records),
    }


def _feedback_from_state(value: Any) -> FeedbackBatch:
    if not isinstance(value, Mapping):
        raise ValidationError("feedback accumulator delivery is malformed")
    try:
        proposal_id = _state_non_negative_int(value["proposal_id"], "proposal_id")
        channel = value["channel"]
        final = value["final"]
        if not isinstance(channel, str) or not isinstance(final, bool):
            raise ValidationError("feedback accumulator delivery envelope is malformed")
        return FeedbackBatch(
            proposal_id=proposal_id,
            observations=ObservationBatch(
                schema=_schema_from_state(value["schema"]),
                records=_records_from_state(value["records"]),
            ),
            channel=channel,
            final=final,
            sequence=_state_non_negative_int(value["sequence"], "sequence"),
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise ValidationError("feedback accumulator delivery is malformed") from exc


def _buffer_from_state(
    value: Any,
    contract: FeedbackContract,
    *,
    check_ordering: bool = True,
) -> _ProposalBuffer:
    if not isinstance(value, Mapping):
        raise ValidationError("feedback accumulator buffer is malformed")
    required = {
        "proposal_id",
        "candidate_ids",
        "requirements",
        "channel",
        "schema",
        "last_sequence",
        "deliveries",
        "seen",
    }
    if not required <= set(value):
        raise ValidationError("feedback accumulator buffer is missing required state")
    try:
        proposal_id = _state_non_negative_int(value["proposal_id"], "proposal_id")
        raw_candidate_ids = np.asarray(value["candidate_ids"])
        if raw_candidate_ids.ndim != 1:
            raise ValidationError("feedback accumulator candidate IDs are malformed")
        candidate_ids = np.asarray(raw_candidate_ids, dtype=np.int64)
        if len(candidate_ids) != len(np.unique(candidate_ids)) or np.any(
            candidate_ids < 0
        ):
            raise ValidationError("feedback accumulator candidate IDs are invalid")
        candidate_ids = np.array(candidate_ids, dtype=np.int64, copy=True)
        candidate_ids.flags.writeable = False

        raw_requirements = value["requirements"]
        if not isinstance(raw_requirements, (list, tuple)):
            raise ValidationError("feedback accumulator requirements are malformed")
        requirements = tuple(_requirement_from_state(item) for item in raw_requirements)
        if any(
            not requirement.sources <= contract.accepted_sources
            for requirement in requirements
        ):
            raise ValidationError(
                "feedback accumulator requirement exceeds the consumer contract"
            )
        if not contract.contains_requirement(
            FeedbackRequirement(quantities=requirements)
        ):
            raise ValidationError(
                "feedback accumulator requirement is incompatible with the consumer"
            )

        channel = value["channel"]
        if channel is not None and (
            not isinstance(channel, str) or channel not in contract.accepted_channels
        ):
            raise ValidationError("feedback accumulator channel is not accepted")
        schema = (
            None if value["schema"] is None else _schema_from_state(value["schema"])
        )
        raw_deliveries = value["deliveries"]
        raw_seen = value["seen"]
        if not isinstance(raw_deliveries, (list, tuple)) or not isinstance(
            raw_seen, Mapping
        ):
            raise ValidationError("feedback accumulator delivery state is malformed")
        deliveries = []
        for item in raw_deliveries:
            if not isinstance(item, Mapping):
                raise ValidationError("feedback accumulator delivery is malformed")
            sequence = _state_non_negative_int(item["sequence"], "sequence")
            records = _records_from_state(item["records"])
            record_ids = _record_candidate_ids(records)
            if len(record_ids) and not np.all(
                np.isin(record_ids, tuple(int(item) for item in candidate_ids))
            ):
                raise ValidationError(
                    "feedback accumulator delivery contains an unknown candidate"
                )
            deliveries.append(_Delivery(sequence, records))
        seen = {
            _state_sequence_key(key): _feedback_from_state(item)
            for key, item in raw_seen.items()
        }
        state = _ProposalBuffer(
            proposal_id=proposal_id,
            candidate_ids=candidate_ids,
            candidate_id_set=frozenset(int(item) for item in candidate_ids),
            requirements=requirements,
            deliveries=deliveries,
            seen=seen,
            channel=channel,
            schema=schema,
            last_sequence=(
                None
                if value["last_sequence"] is None
                else _state_non_negative_int(value["last_sequence"], "last_sequence")
            ),
        )
        if not deliveries or not seen:
            if deliveries or seen or channel is not None or schema is not None:
                raise ValidationError(
                    "empty feedback accumulator buffers cannot carry delivery state"
                )
            if state.last_sequence is not None:
                raise ValidationError(
                    "empty feedback accumulator buffers cannot carry a sequence"
                )
            return state
        if channel is None or schema is None:
            raise ValidationError(
                "feedback accumulator deliveries require channel and schema"
            )
        if len(seen) != len(deliveries) or set(seen) != {
            item.sequence for item in deliveries
        }:
            raise ValidationError(
                "feedback accumulator delivery/duplicate state is inconsistent"
            )
        for delivery in deliveries:
            original = seen[delivery.sequence]
            if (
                original.proposal_id != proposal_id
                or original.sequence != delivery.sequence
                or original.channel != state.channel
                or original.final
            ):
                raise ValidationError(
                    "feedback accumulator delivery envelope is inconsistent"
                )
            if not _schemas_equal(schema, original.observations.schema):
                raise ValidationError(
                    "feedback accumulator schema state is inconsistent"
                )
            if not _records_equal(
                delivery.records,
                _accepted_records(original.observations.records, contract),
            ):
                raise ValidationError(
                    "feedback accumulator accepted records are inconsistent"
                )
        sequences = [item.sequence for item in deliveries]
        if check_ordering and contract.ordering == IN_ORDER:
            if (
                sequences != sorted(sequences)
                or any(right <= left for left, right in pairwise(sequences))
                or state.last_sequence != sequences[-1]
            ):
                raise ValidationError(
                    "feedback accumulator sequence state is inconsistent"
                )
        elif check_ordering and state.last_sequence is not None:
            raise ValidationError(
                "out-of-order feedback accumulator buffers cannot carry last_sequence"
            )
        return state
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError("feedback accumulator buffer is malformed") from exc


def _validate_state(value: Any) -> None:
    if not isinstance(value, Mapping) or value.get("codec") != _ACCUMULATOR_CODEC:
        raise ValidationError("feedback accumulator state codec is malformed")
    if value.get("version") != _ACCUMULATOR_CODEC_VERSION:
        raise ValidationError("unsupported feedback accumulator codec version")
    buffers = value.get("buffers")
    ready = value.get("ready")
    if not isinstance(buffers, list) or not isinstance(ready, list):
        raise ValidationError("feedback accumulator state is malformed")
    # The generic checkpoint codec validates before the runtime consumer is
    # available.  Use all registered values here; ``from_state`` repeats the
    # checks against the actual consumer contract on reattachment.
    contract = FeedbackContract(
        accepted_channels=frozenset(FEEDBACK_CHANNELS.names()),
        accepted_sources=frozenset(OBSERVATION_SOURCES.names()),
        completion=COMPLETE_BATCH,
        ordering=OUT_OF_ORDER_ALLOWED,
    )
    seen_ready: set[int] = set()
    for item in buffers:
        _buffer_from_state(item, contract, check_ordering=False)
    for item in ready:
        batch = _feedback_from_state(item)
        if not batch.final:
            raise ValidationError("ready feedback accumulator batches must be final")
        if batch.proposal_id in seen_ready:
            raise ValidationError("duplicate ready feedback accumulator proposal")
        seen_ready.add(batch.proposal_id)
        if any(
            source not in contract.accepted_sources
            for source in batch.observations.records.column("source")
        ):
            raise ValidationError(
                "ready feedback accumulator batch contains an unregistered source"
            )


def _proposal_candidate_ids(proposal: ProposalBatch) -> CandidateIds:
    if "id" in proposal.candidates.schema:
        values = proposal.candidates.get_array("id")
    else:
        values = np.arange(len(proposal.candidates), dtype=np.int64)
    return np.array(values, dtype=np.int64, order="C", copy=True).reshape(-1)


def _accepted_records(
    records: ObservationRecords, contract: FeedbackContract
) -> ObservationRecords:
    if len(records) == 0:
        return records
    sources = records.column("source")
    mask = np.isin(sources, tuple(contract.accepted_sources))
    return records.take(np.flatnonzero(mask))


def _record_candidate_ids(records: ObservationRecords) -> np.ndarray:
    if len(records) == 0:
        return np.empty(0, dtype=np.int64)
    result = np.empty(len(records), dtype=np.int64)
    for row, (kind, payload) in enumerate(
        zip(records.column("subject_kind"), records.column("subject_payload"))
    ):
        descriptor = OBSERVATION_SUBJECT_KINDS.get(str(kind))
        if descriptor is None:
            raise ValidationError(f"unknown observation subject kind: {kind!r}")
        candidate_ids = np.asarray(descriptor.candidate_ids(payload)).reshape(-1)
        if len(candidate_ids) != 1:
            raise ValidationError(
                "FeedbackAccumulator requires one candidate per observation subject"
            )
        try:
            result[row] = int(candidate_ids[0])
        except (TypeError, ValueError) as exc:
            raise ValidationError("observation candidate ID is not an integer") from exc
    return result


def _quantity_index_type(index: Any) -> str:
    if isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool_)):
        return "int"
    return "str"


def _quantity_index_parts(
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(indices)
    if values.dtype.kind in "iu":
        return (
            np.full(len(values), "int", dtype="U3"),
            values.astype(str),
        )
    if values.dtype.kind in "US":
        return (
            np.full(len(values), "str", dtype="U3"),
            values.astype(str),
        )
    normalized = tuple(values)
    tokens = tuple(str(value) for value in normalized)
    width = max((len(token) for token in tokens), default=1)
    return (
        np.fromiter(
            (_quantity_index_type(value) for value in normalized),
            dtype="U3",
            count=len(normalized),
        ),
        np.asarray(tokens, dtype=f"U{width}"),
    )


def _dense_record_groups(
    record_ids: np.ndarray,
    quantity_kinds: np.ndarray,
    quantity_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    index_types, index_tokens = _quantity_index_parts(quantity_indices)
    kinds = np.asarray(quantity_kinds, dtype=str)
    kind_width = max((len(kind) for kind in kinds), default=1)
    token_width = max((len(token) for token in index_tokens), default=1)
    key_dtype = np.dtype(
        [
            ("candidate", np.int64),
            ("kind", f"U{kind_width}"),
            ("index_type", "U3"),
            ("index", f"U{token_width}"),
        ]
    )
    keys = cast(np.ndarray, np.empty(len(record_ids), dtype=key_dtype))
    keys["candidate"] = np.asarray(record_ids, dtype=np.int64)
    keys["kind"] = kinds
    keys["index_type"] = index_types
    keys["index"] = index_tokens
    unique_keys, group_ids = np.unique(keys, return_inverse=True)
    return unique_keys, group_ids, index_types, index_tokens


def _missing_slot_description(
    candidate_id: int, requirement: QuantityRequirement
) -> str:
    return (
        f"candidate={candidate_id}, quantity={requirement.quantity.kind}["
        f"{requirement.quantity.index}]"
    )


def _fidelity_values(records: ObservationRecords) -> np.ndarray:
    try:
        values = records.column("fidelity")
    except KeyError:
        return np.zeros(len(records), dtype=np.float64)
    result = np.zeros(len(records), dtype=np.float64)
    for row, value in enumerate(values):
        if value is None:
            continue
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            continue
        if not isnan(scalar):
            result[row] = scalar
    return result


def _schemas_equal(left: ObservationSchema, right: ObservationSchema) -> bool:
    return (
        left.objective_count == right.objective_count
        and left.constraint_count == right.constraint_count
        and dict(left.quantities) == dict(right.quantities)
        and left.extra_quantities == right.extra_quantities
        and left.schema_version == right.schema_version
    )


def _feedback_batches_equal(left: FeedbackBatch, right: FeedbackBatch) -> bool:
    return (
        left.proposal_id == right.proposal_id
        and left.channel == right.channel
        and left.final == right.final
        and left.sequence == right.sequence
        and _schemas_equal(left.observations.schema, right.observations.schema)
        and _records_equal(left.observations.records, right.observations.records)
    )


def _records_equal(left: ObservationRecords, right: ObservationRecords) -> bool:
    if len(left) != len(right):
        return False
    names = set(left.column_names) | set(right.column_names)
    for name in names:
        if not _arrays_equal(
            _optional_column(left, name), _optional_column(right, name)
        ):
            return False
    return True


def _optional_column(records: ObservationRecords, name: str) -> np.ndarray:
    try:
        return records.column(name)
    except KeyError:
        result = np.empty(len(records), dtype=object)
        result[:] = None
        return result


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape:
        return False
    try:
        if bool(np.array_equal(left, right, equal_nan=True)):
            return True
    except (TypeError, ValueError):
        pass
    return all(_values_equal(a, b) for a, b in zip(left.flat, right.flat))


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return False
        return _arrays_equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return left.keys() == right.keys() and all(
            _values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        if not isinstance(right, Sequence) or isinstance(right, (str, bytes)):
            return False
        return len(left) == len(right) and all(
            _values_equal(a, b) for a, b in zip(left, right)
        )
    if (
        isinstance(left, (float, np.floating))
        and isinstance(right, (float, np.floating))
        and isnan(float(left))
        and isnan(float(right))
    ):
        return True
    try:
        result = left == right
    except (TypeError, ValueError):
        return left is right
    if isinstance(result, np.ndarray):
        return bool(np.all(result))
    return bool(result)
