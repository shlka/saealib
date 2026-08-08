"""Accumulate partial feedback for a complete-batch consumer.

The accumulator is deliberately independent of the compiler and scheduler.  A
caller registers a :class:`~saealib.core.contracts.proposals.ProposalBatch`,
adds its :class:`~saealib.core.contracts.feedbacks.FeedbackBatch` deliveries,
and takes completed batches from a ready queue::

    accumulator.register(proposal)
    accumulator.add(delivery)
    completed = accumulator.pop_ready()

The implementation makes the following semantic choices, which are the
runtime form of ADR-0007 section 1:

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
declared, so it is not the lossless adapter described by ADR-0007.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isnan
from typing import Any, cast

import numpy as np

from saealib.core.contracts.feedback import COMPLETE_BATCH, IN_ORDER
from saealib.core.contracts.feedbacks import FeedbackBatch, FeedbackContract
from saealib.core.contracts.observation import OBSERVATION_SUBJECT_KINDS
from saealib.core.contracts.observations import (
    ObservationBatch,
    ObservationRecords,
    ObservationSchema,
)
from saealib.core.contracts.proposals import (
    ProposalBatch,
    QuantityRequirement,
)
from saealib.exceptions import ValidationError
from saealib.policies.feedback import DEFAULT_SELECTION_POLICY, SelectionPolicy
from saealib.population.population import CandidateIds

__all__ = ["FeedbackAccumulator"]


@dataclass(frozen=True)
class _Delivery:
    """The columnar records retained for one accepted delivery."""

    sequence: int
    records: ObservationRecords


@dataclass
class _ProposalBuffer:
    """Mutable state for one registered proposal."""

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
        """Validate the consumer contract and initialize empty queues."""
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

    def _finalize(self, state: _ProposalBuffer) -> FeedbackBatch:
        """Resolve a complete proposal into exactly one final delivery."""
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


def _proposal_candidate_ids(proposal: ProposalBatch) -> CandidateIds:
    """Capture candidate IDs using the existing population ID convention."""
    if "id" in proposal.candidates.schema:
        values = proposal.candidates.get_array("id")
    else:
        values = np.arange(len(proposal.candidates), dtype=np.int64)
    return np.array(values, dtype=np.int64, order="C", copy=True).reshape(-1)


def _accepted_records(
    records: ObservationRecords, contract: FeedbackContract
) -> ObservationRecords:
    """Drop observations whose provenance the consumer did not accept."""
    if len(records) == 0:
        return records
    sources = records.column("source")
    mask = np.isin(sources, tuple(contract.accepted_sources))
    return records.take(np.flatnonzero(mask))


def _record_candidate_ids(records: ObservationRecords) -> np.ndarray:
    """Return one candidate ID per record through subject descriptors."""
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
    """Return a type tag so integer index 1 differs from name ``"1"``."""
    if isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool_)):
        return "int"
    return "str"


def _quantity_index_parts(
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize heterogeneous quantity indices into vectorized key columns."""
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
    """Build dense record group IDs with one structured NumPy unique pass."""
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
    """Describe one missing candidate/quantity slot for the final diagnostic."""
    return (
        f"candidate={candidate_id}, quantity={requirement.quantity.kind}["
        f"{requirement.quantity.index}]"
    )


def _fidelity_values(records: ObservationRecords) -> np.ndarray:
    """Materialize optional fidelity metadata for the policy and predicates."""
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
    """Compare schema values without relying on MappingProxyType equality."""
    return (
        left.objective_count == right.objective_count
        and left.constraint_count == right.constraint_count
        and dict(left.quantities) == dict(right.quantities)
        and left.extra_quantities == right.extra_quantities
        and left.schema_version == right.schema_version
    )


def _feedback_batches_equal(left: FeedbackBatch, right: FeedbackBatch) -> bool:
    """Compare duplicate deliveries by payload, not object identity."""
    return (
        left.proposal_id == right.proposal_id
        and left.channel == right.channel
        and left.final == right.final
        and left.sequence == right.sequence
        and _schemas_equal(left.observations.schema, right.observations.schema)
        and _records_equal(left.observations.records, right.observations.records)
    )


def _records_equal(left: ObservationRecords, right: ObservationRecords) -> bool:
    """Compare columnar records, including optional object-valued columns."""
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
    """Read a column or represent an absent optional column as ``None``."""
    try:
        return records.column(name)
    except KeyError:
        result = np.empty(len(records), dtype=object)
        result[:] = None
        return result


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare arrays without assuming object columns contain scalars."""
    if left.shape != right.shape:
        return False
    try:
        if bool(np.array_equal(left, right, equal_nan=True)):
            return True
    except (TypeError, ValueError):
        pass
    return all(_values_equal(a, b) for a, b in zip(left.flat, right.flat))


def _values_equal(left: Any, right: Any) -> bool:
    """Recursively compare one possibly nested column value."""
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
