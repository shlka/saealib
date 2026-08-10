"""Feedback batch and contract tests."""

from __future__ import annotations

from typing import Any, cast

import pytest

from saealib.algorithms import GA, PSO, Algorithm, PymooAlgorithm
from saealib.core.contracts import (
    BY_PROPOSAL,
    COMPLETE_BATCH,
    IN_ORDER,
    PARTIAL_ALLOWED,
    SINGLE,
    TRUE,
    FeedbackBatch,
    FeedbackContract,
    FeedbackRequirement,
    LifecycleContract,
    ObservationBatch,
    ObservationRecords,
    ObservationSchema,
    QuantityRef,
    QuantityRequirement,
)
from saealib.core.contracts.execution import ExecutionContract
from saealib.core.contracts.observation import SURROGATE
from saealib.exceptions import ValidationError


def _requirement(
    *sources: str, completion: str = COMPLETE_BATCH
) -> FeedbackRequirement:
    return FeedbackRequirement(
        quantities=(
            QuantityRequirement(
                quantity=QuantityRef(kind="objective", index=0),
                sources=frozenset(sources),
            ),
        ),
        completion=completion,
    )


def _contract(
    *,
    accepted_sources: frozenset[str] = frozenset({TRUE}),
    completion: str = COMPLETE_BATCH,
) -> FeedbackContract:
    return FeedbackContract(
        accepted_channels=frozenset({"true"}),
        accepted_sources=accepted_sources,
        completion=completion,
    )


def test_feedback_batch_reuses_j2_batch_and_validates_delivery_envelope() -> None:
    """Mutation: replacing ObservationBatch or dropping envelope validation fails."""
    observations = ObservationBatch(
        schema=ObservationSchema(),
        records=ObservationRecords(),
    )
    batch = FeedbackBatch(
        proposal_id=3,
        observations=observations,
        channel="true",
        final=True,
        sequence=0,
    )
    assert batch.observations is observations
    with pytest.raises(ValidationError):
        FeedbackBatch(
            proposal_id=3,
            observations=observations,
            channel="unknown",
            final=True,
            sequence=0,
        )


def test_feedback_contract_requires_channels_and_uses_restrictive_defaults() -> None:
    """Mutation: adding a channel default or loosening any default fails."""
    with pytest.raises(TypeError):
        cast(Any, FeedbackContract)()
    contract = FeedbackContract(accepted_channels=frozenset({"true"}))
    assert contract.accepted_sources == frozenset({TRUE})
    assert contract.completion == COMPLETE_BATCH
    assert contract.ordering == IN_ORDER
    assert contract.multiplicity == SINGLE
    assert contract.grouping == BY_PROPOSAL


def test_lifecycle_feedback_is_optional_for_non_consumers() -> None:
    """Lifecycle keeps non-feedback components at the empty default."""
    assert LifecycleContract().feedback is None
    with pytest.raises(ValidationError):
        LifecycleContract(feedback=cast(Any, object()))


def test_sources_and_channels_are_independent_axes() -> None:
    """Mutation: merging source and channel sets fails this axis check."""
    contract = FeedbackContract(
        accepted_channels=frozenset({"true"}),
        accepted_sources=frozenset({TRUE, SURROGATE}),
    )
    assert contract.accepted_sources == frozenset({TRUE, SURROGATE})
    assert contract.accepted_channels == frozenset({"true"})
    assert SURROGATE not in contract.accepted_channels


def test_feedback_contract_contains_only_narrower_requirements() -> None:
    """Mutation: reversing subset direction or completion ordering fails."""
    contract = _contract(accepted_sources=frozenset({TRUE, SURROGATE}))
    assert contract.contains_requirement(_requirement(TRUE))
    assert contract.contains_requirement(_requirement(TRUE, SURROGATE))
    assert not _contract().contains_requirement(_requirement(TRUE, SURROGATE))
    assert not contract.contains_requirement(
        _requirement(TRUE, completion=PARTIAL_ALLOWED)
    )
    assert _contract(completion=PARTIAL_ALLOWED).contains_requirement(
        _requirement(TRUE, completion=PARTIAL_ALLOWED)
    )


def test_algorithm_family_declares_complete_feedback_and_subclasses_inherit_it() -> (
    None
):
    """Mutation: removing the family declaration fails the inheritance check."""
    # The family method only reads its contract definition, so a structural
    # placeholder is sufficient and avoids constructing GA's collaborators.
    base_contract = Algorithm.contract(cast(Algorithm, object()))
    assert base_contract.lifecycle.feedback is not None
    assert base_contract.lifecycle.feedback.completion == COMPLETE_BATCH
    assert GA.contract is not Algorithm.contract
    assert PSO.contract is not Algorithm.contract
    assert PSO().contract().lifecycle.feedback == base_contract.lifecycle.feedback


def test_pymoo_completion_matches_partial_tell_and_keeps_runtime_capability() -> None:
    """Mutation: removing either completion or capability declaration fails."""
    baseline = PymooAlgorithm(cast(Any, object())).contract()
    partial = PymooAlgorithm(cast(Any, object()), allow_partial_tell=True).contract()
    assert baseline.lifecycle.feedback is not None
    assert partial.lifecycle.feedback is not None
    assert baseline.lifecycle.feedback.completion == COMPLETE_BATCH
    assert partial.lifecycle.feedback.completion == PARTIAL_ALLOWED
    assert baseline.execution == ExecutionContract()
    assert partial.execution == ExecutionContract(
        required_runtime_capabilities=("partial_feedback",)
    )
