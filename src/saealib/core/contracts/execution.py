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


@dataclass(frozen=True, kw_only=True)
class ExecutionContract:
    """Declare runtime capabilities required by a component."""

    required_runtime_capabilities: tuple[RuntimeCapability, ...] = ()

    def __post_init__(self) -> None:
        """Validate and normalize required capability names."""
        capabilities = tuple(self.required_runtime_capabilities)
        if any(not isinstance(capability, str) for capability in capabilities):
            raise ValidationError("Runtime capabilities must be strings")
        for capability in capabilities:
            validate_name(capability)
        object.__setattr__(self, "required_runtime_capabilities", capabilities)
