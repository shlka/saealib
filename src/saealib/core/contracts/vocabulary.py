from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

from saealib.exceptions import ConfigurationError, ValidationError

__all__ = [
    "Vocabulary",
    "VocabularyDescriptor",
    "is_valid_name",
    "validate_name",
]


_NAME_PART = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
DescriptorT = TypeVar("DescriptorT")


def validate_name(name: str) -> str:
    """Validate and return a vocabulary name."""
    if not isinstance(name, str):
        raise ValidationError("Vocabulary names must be strings")
    parts = name.split(":")
    if len(parts) > 2 or any(_NAME_PART.fullmatch(part) is None for part in parts):
        raise ValidationError(
            "Vocabulary names must be an identifier or two identifiers separated by ':'"
        )
    return name


@dataclass(frozen=True, kw_only=True)
class VocabularyDescriptor:
    """Metadata for a vocabulary entry."""

    name: str
    description: str

    def __post_init__(self) -> None:
        """Validate descriptor metadata."""
        validate_name(self.name)
        if not isinstance(self.description, str):
            raise ValidationError("Vocabulary descriptions must be strings")


class Vocabulary(Generic[DescriptorT]):
    """Registry of explicitly named vocabulary descriptors."""

    def __init__(self) -> None:
        self._entries: dict[str, DescriptorT] = {}
        self._deprecation_reasons: dict[str, str] = {}

    def register(self, name: str, descriptor: DescriptorT) -> None:
        """Register a descriptor under a stable name."""
        validate_name(name)
        if name in self._entries:
            raise ConfigurationError(f"Vocabulary name is already registered: {name!r}")
        descriptor_name = getattr(descriptor, "name", None)
        if descriptor_name is not None and descriptor_name != name:
            raise ValidationError(
                f"Descriptor name {descriptor_name!r} does not match {name!r}"
            )
        self._entries[name] = descriptor

    def get(self, name: str) -> DescriptorT | None:
        """Return the descriptor for a name, or ``None`` when it is unknown."""
        if not is_valid_name(name):
            return None
        return self._entries.get(name)

    def contains(self, name: str) -> bool:
        """Return whether a name is registered."""
        return self.get(name) is not None

    def names(self) -> tuple[str, ...]:
        """Return registered names in registration order."""
        return tuple(self._entries)

    def deprecate(self, name: str, reason: str) -> None:
        """Mark a registered name deprecated without removing it."""
        validate_name(name)
        if name not in self._entries:
            raise ValidationError(f"Cannot deprecate unknown vocabulary name: {name!r}")
        if not isinstance(reason, str) or not reason:
            raise ValidationError("A deprecation reason must not be empty")
        self._deprecation_reasons[name] = reason

    def is_deprecated(self, name: str) -> bool:
        """Return whether a registered name is deprecated."""
        if not is_valid_name(name):
            return False
        return name in self._deprecation_reasons

    def deprecation_reason(self, name: str) -> str | None:
        """Return a deprecation reason, or ``None`` when none is recorded."""
        if not is_valid_name(name):
            return None
        return self._deprecation_reasons.get(name)

    def __contains__(self, name: object) -> bool:
        """Return whether an object names a registered value."""
        return isinstance(name, str) and self.contains(name)

    def __iter__(self) -> Iterator[str]:
        """Iterate over registered names."""
        return iter(self._entries)

    def __len__(self) -> int:
        """Return the number of registered names."""
        return len(self._entries)


def is_valid_name(name: object) -> bool:
    """Return whether a value has a valid vocabulary name shape."""
    if not isinstance(name, str):
        return False
    try:
        validate_name(name)
    except ValidationError:
        return False
    return True
