from __future__ import annotations

from dataclasses import dataclass

from saealib.core.contracts.feedbacks import FeedbackContract
from saealib.core.contracts.vocabulary import (
    Vocabulary,
    VocabularyDescriptor,
    validate_name,
)
from saealib.exceptions import ValidationError

__all__ = ["EVENT_VOCABULARY", "EventSubscription", "LifecycleContract"]


EVENT_VOCABULARY: Vocabulary[VocabularyDescriptor] = Vocabulary()


@dataclass(frozen=True, kw_only=True)
class EventSubscription:
    """Identify one event consumed by a component."""

    event: str

    def __post_init__(self) -> None:
        if not isinstance(self.event, str):
            raise ValidationError("Event identities must be strings")
        validate_name(self.event)


@dataclass(frozen=True, kw_only=True)
class LifecycleContract:
    """Declare events consumed by a component."""

    events: tuple[EventSubscription, ...] = ()
    feedback: FeedbackContract | None = None

    def __post_init__(self) -> None:
        events = tuple(self.events)
        if any(not isinstance(event, EventSubscription) for event in events):
            raise ValidationError(
                "Lifecycle events must contain EventSubscription values"
            )
        object.__setattr__(self, "events", events)
        if self.feedback is not None and not isinstance(
            self.feedback, FeedbackContract
        ):
            raise ValidationError(
                "Lifecycle feedback must be a FeedbackContract or None"
            )
