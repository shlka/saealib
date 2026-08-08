"""Feedback delivery and policy vocabularies."""

from __future__ import annotations

from typing import TypeAlias

from saealib.core.contracts.vocabulary import Vocabulary, VocabularyDescriptor

__all__ = [
    "BY_CANDIDATE",
    "BY_PROPOSAL",
    "COMPLETE_BATCH",
    "COMPLETION_MODES",
    "DEFAULT_COMPLETION_MODE",
    "DEFAULT_FEEDBACK_GROUPING",
    "DEFAULT_MULTIPLICITY_MODE",
    "DEFAULT_ORDERING_MODE",
    "FEEDBACK_CHANNELS",
    "FEEDBACK_GROUPINGS",
    "IN_ORDER",
    "MULTIPLICITY_MODES",
    "ORDERING_MODES",
    "OUT_OF_ORDER_ALLOWED",
    "PARTIAL_ALLOWED",
    "REPEATED_ALLOWED",
    "SINGLE",
    "CompletionMode",
    "FeedbackChannel",
    "FeedbackGrouping",
    "MultiplicityMode",
    "OrderingMode",
]

FeedbackChannel: TypeAlias = str
CompletionMode: TypeAlias = str
OrderingMode: TypeAlias = str
MultiplicityMode: TypeAlias = str
FeedbackGrouping: TypeAlias = str

COMPLETE_BATCH: CompletionMode = "complete_batch"
PARTIAL_ALLOWED: CompletionMode = "partial_allowed"
IN_ORDER: OrderingMode = "in_order"
OUT_OF_ORDER_ALLOWED: OrderingMode = "out_of_order_allowed"
SINGLE: MultiplicityMode = "single"
REPEATED_ALLOWED: MultiplicityMode = "repeated_allowed"
BY_PROPOSAL: FeedbackGrouping = "by_proposal"
BY_CANDIDATE: FeedbackGrouping = "by_candidate"

# J4's FeedbackContract will consume these defaults.  Keeping them as named
# constants makes the strongest-default rule explicit without creating that
# contract in J1.
DEFAULT_COMPLETION_MODE: CompletionMode = COMPLETE_BATCH
DEFAULT_ORDERING_MODE: OrderingMode = IN_ORDER
DEFAULT_MULTIPLICITY_MODE: MultiplicityMode = SINGLE
DEFAULT_FEEDBACK_GROUPING: FeedbackGrouping = BY_PROPOSAL


def _vocabulary(
    entries: tuple[tuple[str, str], ...],
) -> Vocabulary[VocabularyDescriptor]:
    registry: Vocabulary[VocabularyDescriptor] = Vocabulary()
    for name, description in entries:
        registry.register(
            name, VocabularyDescriptor(name=name, description=description)
        )
    return registry


FEEDBACK_CHANNELS = _vocabulary(
    tuple(
        (name, f"Feedback delivered through the {name} channel.")
        for name in ("true", "surrogate", "human", "simulator")
    )
)
COMPLETION_MODES = _vocabulary(
    (
        (COMPLETE_BATCH, "Wait for a complete batch."),
        (PARTIAL_ALLOWED, "Allow partial feedback."),
    )
)
ORDERING_MODES = _vocabulary(
    (
        (IN_ORDER, "Feedback arrives in order."),
        (OUT_OF_ORDER_ALLOWED, "Allow out-of-order feedback."),
    )
)
MULTIPLICITY_MODES = _vocabulary(
    ((SINGLE, "Deliver feedback once."), (REPEATED_ALLOWED, "Allow repeated feedback."))
)
FEEDBACK_GROUPINGS = _vocabulary(
    (
        (BY_PROPOSAL, "Group feedback by proposal."),
        (BY_CANDIDATE, "Group feedback by candidate."),
    )
)
