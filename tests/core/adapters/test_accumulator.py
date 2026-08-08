"""K6a tests for complete-batch feedback accumulation semantics."""

from __future__ import annotations

import numpy as np
import pytest

from saealib.core.adapters import FeedbackAccumulator
from saealib.core.contracts.feedback import (
    COMPLETE_BATCH,
    IN_ORDER,
    OUT_OF_ORDER_ALLOWED,
    PARTIAL_ALLOWED,
)
from saealib.core.contracts.feedbacks import FeedbackBatch, FeedbackContract
from saealib.core.contracts.observation import (
    CANCELLED,
    FAILED,
    OBJECTIVE,
    OK,
    SURROGATE,
    TIMEOUT,
    TRUE,
)
from saealib.core.contracts.observations import (
    ObservationBatch,
    ObservationRecord,
    ObservationRecords,
    ObservationSchema,
    QuantityRef,
)
from saealib.core.contracts.proposals import (
    FeedbackRequirement,
    ProposalBatch,
    ProposalRelations,
    QuantityRequirement,
)
from saealib.exceptions import ValidationError
from saealib.policies.feedback import SelectionPolicy
from saealib.population import Population, PopulationAttribute


def _population(candidate_ids: tuple[int, ...]) -> Population:
    population = Population(
        [PopulationAttribute("id", np.int64, default=-1)],
        init_capacity=max(1, len(candidate_ids)),
    )
    if candidate_ids:
        population._extend_internal(
            {"id": np.asarray(candidate_ids, dtype=np.int64)},
            preserve_ids=True,
        )
    return population


def _proposal(
    proposal_id: int,
    candidate_ids: tuple[int, ...],
    *,
    sources: frozenset[str] = frozenset({TRUE}),
    fidelity: int | None = None,
) -> ProposalBatch:
    return ProposalBatch(
        proposal_id=proposal_id,
        candidates=_population(candidate_ids),
        relations=ProposalRelations({}, row_count=len(candidate_ids)),
        requirements=FeedbackRequirement(
            quantities=(
                QuantityRequirement(
                    quantity=QuantityRef(kind=OBJECTIVE, index=0),
                    sources=sources,
                    fidelity=fidelity,
                ),
            )
        ),
    )


def _contract(
    *,
    accepted_sources: frozenset[str] = frozenset({TRUE}),
    accepted_channels: frozenset[str] = frozenset({TRUE}),
    ordering: str = IN_ORDER,
) -> FeedbackContract:
    return FeedbackContract(
        accepted_channels=accepted_channels,
        accepted_sources=accepted_sources,
        completion=COMPLETE_BATCH,
        ordering=ordering,
    )


def _record(
    candidate_id: int,
    value: float,
    *,
    source: str = TRUE,
    status: str = OK,
    fidelity: int | None = None,
) -> ObservationRecord:
    return ObservationRecord(
        subject=("candidate", np.array([candidate_id], dtype=np.int64)),
        quantity=(OBJECTIVE, 0),
        value=value,
        status=status,
        source=source,
        fidelity=fidelity,
    )


def _batch(
    proposal_id: int,
    sequence: int,
    records: tuple[ObservationRecord, ...],
    *,
    channel: str = TRUE,
    final: bool = False,
) -> FeedbackBatch:
    return FeedbackBatch(
        proposal_id=proposal_id,
        observations=ObservationBatch(
            schema=ObservationSchema(objective_count=1),
            records=ObservationRecords.from_records(records),
        ),
        channel=channel,
        final=final,
        sequence=sequence,
    )


def test_two_partial_deliveries_produce_exactly_one_final_batch() -> None:
    """Mutation: removing cross-delivery accumulation loses the first row."""
    accumulator = FeedbackAccumulator(_contract())
    proposal = _proposal(1, (10, 11))
    accumulator.register(proposal)
    accumulator.add(_batch(1, 0, (_record(10, 1.0),)))
    assert accumulator.pop_ready() is None
    accumulator.add(_batch(1, 1, (_record(11, 2.0),), final=True))

    completed = accumulator.pop_ready()
    assert completed is not None
    assert completed.final is True
    np.testing.assert_array_equal(completed.observations.candidate_ids, [10, 11])
    assert accumulator.pop_ready() is None


def test_unfinished_accumulation_has_no_ready_batch() -> None:
    """Mutation: emitting before ``final`` makes this incomplete-state test fail."""
    accumulator = FeedbackAccumulator(_contract())
    accumulator.register(_proposal(2, (20, 21)))
    accumulator.add(_batch(2, 0, (_record(20, 1.0),)))

    assert accumulator.pop_ready() is None
    assert accumulator.ready_count == 0
    assert accumulator.buffered_proposal_count == 1


def test_out_of_order_delivery_is_allowed_only_for_declared_ordering() -> None:
    """Mutation: ignoring ``FeedbackContract.ordering`` breaks both branches."""
    accumulator = FeedbackAccumulator(_contract(ordering=OUT_OF_ORDER_ALLOWED))
    accumulator.register(_proposal(3, (30, 31)))
    accumulator.add(_batch(3, 10, (_record(31, 2.0),)))
    accumulator.add(_batch(3, 2, (_record(30, 1.0),), final=True))
    assert accumulator.pop_ready() is not None

    ordered = FeedbackAccumulator(_contract())
    ordered.register(_proposal(4, (40, 41)))
    ordered.add(_batch(4, 10, (_record(40, 1.0),)))
    with pytest.raises(ValidationError, match="out of order"):
        ordered.add(_batch(4, 2, (_record(41, 2.0),), final=True))


def test_exact_duplicate_delivery_is_idempotent_and_conflict_is_rejected() -> None:
    """Mutation: appending a duplicate delivery yields duplicate output rows."""
    accumulator = FeedbackAccumulator(_contract())
    accumulator.register(_proposal(5, (50, 51)))
    first = _batch(5, 0, (_record(50, 1.0),))
    accumulator.add(first)
    accumulator.add(_batch(5, 0, (_record(50, 1.0),)))
    with pytest.raises(ValidationError, match="conflicting duplicate"):
        accumulator.add(_batch(5, 0, (_record(50, 9.0),)))
    accumulator.add(_batch(5, 1, (_record(51, 2.0),), final=True))

    completed = accumulator.pop_ready()
    assert completed is not None
    assert len(completed.observations.records) == 2


def test_failed_cancelled_and_timeout_records_are_terminal_coverage() -> None:
    """Mutation: counting only ``ok`` records deadlocks terminal failures."""
    accumulator = FeedbackAccumulator(_contract())
    accumulator.register(_proposal(6, (60, 61, 62)))
    accumulator.add(_batch(6, 0, (_record(60, 0.0, status=FAILED),)))
    accumulator.add(_batch(6, 1, (_record(61, 0.0, status=CANCELLED),)))
    accumulator.add(_batch(6, 2, (_record(62, 0.0, status=TIMEOUT),), final=True))

    completed = accumulator.pop_ready()
    assert completed is not None
    assert set(completed.observations.records.column("status")) == {
        FAILED,
        CANCELLED,
        TIMEOUT,
    }


def test_duplicate_observations_are_resolved_by_selection_policy() -> None:
    """Mutation: choosing the first row instead of the policy changes the value."""
    policy = SelectionPolicy(source_priority=(SURROGATE, TRUE))
    accumulator = FeedbackAccumulator(
        _contract(accepted_sources=frozenset({TRUE, SURROGATE})),
        selection_policy=policy,
    )
    accumulator.register(_proposal(7, (70,), sources=frozenset({TRUE, SURROGATE})))
    accumulator.add(_batch(7, 0, (_record(70, 1.0, source=TRUE),)))
    accumulator.add(_batch(7, 1, (_record(70, 2.0, source=SURROGATE),), final=True))

    completed = accumulator.pop_ready()
    assert completed is not None
    assert completed.observations.records[0].value == 2.0
    assert completed.observations.records[0].source == SURROGATE


def test_completion_requires_the_declared_quantity_fidelity_floor() -> None:
    """Mutation: ignoring ``QuantityRequirement.fidelity`` accepts low fidelity."""
    accumulator = FeedbackAccumulator(_contract())
    accumulator.register(_proposal(71, (710,), fidelity=2))
    accumulator.add(_batch(71, 0, (_record(710, 1.0, fidelity=1),)))
    accumulator.add(_batch(71, 1, (_record(710, 2.0, fidelity=2),), final=True))

    completed = accumulator.pop_ready()
    assert completed is not None
    assert completed.observations.records[0].value == 2.0
    assert completed.observations.records[0].fidelity == 2


def test_final_incomplete_delivery_reports_and_releases_buffer() -> None:
    """Mutation: suppressing the final coverage diagnostic leaves a deadlock."""
    accumulator = FeedbackAccumulator(_contract())
    accumulator.register(_proposal(8, (80, 81)))
    with pytest.raises(ValidationError, match="incomplete"):
        accumulator.add(_batch(8, 0, (_record(80, 1.0),), final=True))
    assert accumulator.pop_ready() is None
    assert accumulator.buffered_proposal_count == 0


def test_mixed_channels_cannot_be_represented_by_one_final_envelope() -> None:
    """ADR-0004 channel provenance is preserved by rejecting mixed delivery."""
    accumulator = FeedbackAccumulator(
        _contract(accepted_channels=frozenset({TRUE, SURROGATE}))
    )
    accumulator.register(_proposal(9, (90, 91)))
    accumulator.add(_batch(9, 0, (_record(90, 1.0),)))
    with pytest.raises(ValidationError, match="different channels"):
        accumulator.add(
            _batch(
                9,
                1,
                (_record(91, 2.0),),
                channel=SURROGATE,
                final=True,
            )
        )


def test_unaccepted_source_is_dropped_before_completion_selection() -> None:
    """Mutation: forwarding an unaccepted source violates the consumer contract."""
    accumulator = FeedbackAccumulator(_contract())
    accumulator.register(_proposal(10, (100,)))
    accumulator.add(_batch(10, 0, (_record(100, 99.0, source=SURROGATE),)))
    accumulator.add(_batch(10, 1, (_record(100, 1.0),), final=True))

    completed = accumulator.pop_ready()
    assert completed is not None
    assert len(completed.observations.records) == 1
    assert completed.observations.records[0].value == 1.0
    assert completed.observations.records[0].source == TRUE


def test_completion_releases_each_proposal_buffer() -> None:
    """Mutation: retaining per-proposal state makes buffered count grow linearly."""
    accumulator = FeedbackAccumulator(_contract())
    for proposal_id in range(64):
        candidate_id = 1000 + proposal_id
        accumulator.register(_proposal(proposal_id, (candidate_id,)))
        accumulator.add(
            _batch(
                proposal_id,
                0,
                (_record(candidate_id, float(proposal_id)),),
                final=True,
            )
        )
        assert accumulator.pop_ready() is not None

    assert accumulator.buffered_proposal_count == 0
    assert accumulator.ready_count == 0


def test_accumulator_rejects_partial_consumer_because_buffering_changes_timing() -> (
    None
):
    """ADR-0007 makes this adapter applicable only to complete-batch consumers."""
    contract = FeedbackContract(
        accepted_channels=frozenset({TRUE}),
        completion=PARTIAL_ALLOWED,
    )
    with pytest.raises(ValidationError, match="complete_batch"):
        FeedbackAccumulator(contract)


def test_selection_policy_call_count_is_constant_across_candidate_counts() -> None:
    """Mutation: slot-by-slot selection makes calls grow with candidates."""
    call_counts: list[int] = []

    class CountingSelectionPolicy(SelectionPolicy):
        def select(
            self,
            candidate_positions: np.ndarray,
            source: np.ndarray,
            fidelity: np.ndarray,
            sequence: np.ndarray,
            status: np.ndarray,
            batch_index: np.ndarray,
        ) -> np.ndarray:
            call_counts.append(1)
            return super().select(
                candidate_positions,
                source,
                fidelity,
                sequence,
                status,
                batch_index,
            )

    policy = CountingSelectionPolicy()
    for proposal_id, count in ((72, 3), (73, 30)):
        candidate_ids = tuple(range(7200 + proposal_id, 7200 + proposal_id + count))
        accumulator = FeedbackAccumulator(_contract(), selection_policy=policy)
        accumulator.register(_proposal(proposal_id, candidate_ids))
        accumulator.add(
            _batch(
                proposal_id,
                0,
                tuple(
                    _record(candidate_id, float(candidate_id))
                    for candidate_id in candidate_ids
                ),
                final=True,
            )
        )
        assert accumulator.pop_ready() is not None

    assert call_counts == [1, 1]
