"""Feedback batches and consumer-side feedback contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from saealib.core.contracts.feedback import (
    BY_PROPOSAL,
    COMPLETE_BATCH,
    COMPLETION_MODES,
    FEEDBACK_CHANNELS,
    FEEDBACK_GROUPINGS,
    IN_ORDER,
    MULTIPLICITY_MODES,
    ORDERING_MODES,
    SINGLE,
    CompletionMode,
    FeedbackChannel,
    FeedbackGrouping,
    MultiplicityMode,
    OrderingMode,
)
from saealib.core.contracts.observation import (
    OBSERVATION_SOURCES,
    TRUE,
    ObservationSource,
)
from saealib.core.contracts.observations import ObservationBatch
from saealib.core.contracts.proposals import FeedbackRequirement, ProposalId
from saealib.exceptions import ValidationError

__all__ = ["FeedbackBatch", "FeedbackContract"]


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{name} must be a non-negative integer")
    value = int(value)
    if value < 0 or value > np.iinfo(np.int64).max:
        raise ValidationError(f"{name} must be a non-negative int64 integer")
    return value


@dataclass(frozen=True, kw_only=True)
class FeedbackBatch:
    """Observations delivered for one proposal."""

    proposal_id: ProposalId
    observations: ObservationBatch
    channel: FeedbackChannel
    final: bool
    sequence: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "proposal_id", _non_negative_int(self.proposal_id, "proposal_id")
        )
        if not isinstance(self.observations, ObservationBatch):
            raise ValidationError("observations must be an ObservationBatch")
        if not FEEDBACK_CHANNELS.contains(self.channel):
            raise ValidationError(f"unknown feedback channel: {self.channel!r}")
        if not isinstance(self.final, bool):
            raise ValidationError("final must be a boolean")
        object.__setattr__(
            self, "sequence", _non_negative_int(self.sequence, "sequence")
        )


@dataclass(frozen=True, kw_only=True)
class FeedbackContract:
    """Declare the feedback deliveries a component can consume."""

    accepted_channels: frozenset[FeedbackChannel]
    accepted_sources: frozenset[ObservationSource] = frozenset({TRUE})
    completion: CompletionMode = COMPLETE_BATCH
    ordering: OrderingMode = IN_ORDER
    multiplicity: MultiplicityMode = SINGLE
    grouping: FeedbackGrouping = BY_PROPOSAL

    def __post_init__(self) -> None:
        try:
            channels = frozenset(self.accepted_channels)
        except TypeError as exc:
            raise ValidationError(
                "accepted_channels must be an iterable of registered channels"
            ) from exc
        if not channels or any(
            not FEEDBACK_CHANNELS.contains(value) for value in channels
        ):
            raise ValidationError("accepted_channels must contain registered channels")
        try:
            sources = frozenset(self.accepted_sources)
        except TypeError as exc:
            raise ValidationError(
                "accepted_sources must be an iterable of registered sources"
            ) from exc
        if not sources or any(
            not OBSERVATION_SOURCES.contains(value) for value in sources
        ):
            raise ValidationError("accepted_sources must contain registered sources")
        if not COMPLETION_MODES.contains(self.completion):
            raise ValidationError(f"unknown completion mode: {self.completion!r}")
        if not ORDERING_MODES.contains(self.ordering):
            raise ValidationError(f"unknown ordering mode: {self.ordering!r}")
        if not MULTIPLICITY_MODES.contains(self.multiplicity):
            raise ValidationError(f"unknown multiplicity mode: {self.multiplicity!r}")
        if not FEEDBACK_GROUPINGS.contains(self.grouping):
            raise ValidationError(f"unknown feedback grouping: {self.grouping!r}")
        object.__setattr__(self, "accepted_channels", channels)
        object.__setattr__(self, "accepted_sources", sources)

    def contains_requirement(self, requirement: FeedbackRequirement) -> bool:
        """Return whether a requirement is no wider than this contract."""
        if not isinstance(requirement, FeedbackRequirement):
            raise TypeError("requirement must be a FeedbackRequirement")
        if requirement.completion == COMPLETE_BATCH:
            completion_contained = True
        else:
            completion_contained = self.completion != COMPLETE_BATCH
        return completion_contained and all(
            quantity.sources <= self.accepted_sources
            for quantity in requirement.quantities
        )

    def contains(self, requirement: FeedbackRequirement) -> bool:
        """Alias for :meth:`contains_requirement`."""
        return self.contains_requirement(requirement)
