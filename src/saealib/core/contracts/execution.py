from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from saealib.core.contracts.vocabulary import (
    Vocabulary,
    VocabularyDescriptor,
    validate_name,
)
from saealib.exceptions import ValidationError

__all__ = ["RUNTIME_CAPABILITIES", "ExecutionContract", "RuntimeCapability"]


RuntimeCapability: TypeAlias = str
RUNTIME_CAPABILITIES: Vocabulary[VocabularyDescriptor] = Vocabulary()
RUNTIME_CAPABILITIES.register(
    "partial_feedback",
    VocabularyDescriptor(
        name="partial_feedback",
        description="accept feedback for only part of a previously proposed batch",
    ),
)


@dataclass(frozen=True, kw_only=True)
class ExecutionContract:
    """Declare runtime capabilities required or offered by a component."""

    required_runtime_capabilities: tuple[RuntimeCapability, ...] = ()
    offered_runtime_capabilities: tuple[RuntimeCapability, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "required_runtime_capabilities",
            "offered_runtime_capabilities",
        ):
            capabilities = tuple(getattr(self, field_name))
            if any(not isinstance(capability, str) for capability in capabilities):
                raise ValidationError("Runtime capabilities must be strings")
            for capability in capabilities:
                validate_name(capability)
            object.__setattr__(self, field_name, capabilities)
