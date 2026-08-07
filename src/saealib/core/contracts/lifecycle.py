from __future__ import annotations

from dataclasses import dataclass

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
        """Validate the event identity."""
        if not isinstance(self.event, str):
            raise ValidationError("Event identities must be strings")
        validate_name(self.event)


@dataclass(frozen=True, kw_only=True)
class LifecycleContract:
    """Declare events consumed by a component."""

    events: tuple[EventSubscription, ...] = ()

    def __post_init__(self) -> None:
        """Validate and normalize event subscriptions."""
        events = tuple(self.events)
        if any(not isinstance(event, EventSubscription) for event in events):
            raise ValidationError(
                "Lifecycle events must contain EventSubscription values"
            )
        object.__setattr__(self, "events", events)
