"""Default resolution models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from saealib.defaults.keys import DefaultKey
from saealib.exceptions import ValidationError


class DefaultStrength(IntEnum):
    """Strength of a default hint.

    Higher values take precedence over lower values when resolving conflicts.
    """

    FALLBACK = 1
    """Generic fallback hint (e.g., 4*dim for population size)."""

    RECOMMENDED = 2
    """Recommended hint based on composition (e.g., NSGA-III reference points)."""

    REQUIRED = 3
    """Required hint that must be used (rarely used)."""


@dataclass(frozen=True)
class DefaultHint:
    """A hint provided by a component for a default value."""

    key: DefaultKey
    value: Any
    strength: DefaultStrength
    source: str
    reason: str = ""

    def __post_init__(self) -> None:
        """Validate that the hinted value matches its semantic key."""
        if not isinstance(self.key, DefaultKey):
            raise ValidationError("DefaultHint.key must be a DefaultKey")
        if not isinstance(self.strength, DefaultStrength):
            raise ValidationError("DefaultHint.strength must be a DefaultStrength")

        value_type = self.key.value_type
        if not isinstance(value_type, type):
            raise ValidationError(
                f"DefaultKey {self.key.name!r} must declare a type as value_type"
            )
        # ``bool`` is an ``int`` subclass, but accepting it for integer
        # defaults would turn a malformed hint into a valid configuration.
        if isinstance(self.value, bool) and value_type is int:
            raise ValidationError(
                f"DefaultHint value for {self.key.name!r} must be an int, not bool"
            )
        if not isinstance(self.value, value_type):
            article = "an" if value_type.__name__[0] in "aeiou" else "a"
            raise ValidationError(
                f"DefaultHint value for {self.key.name!r} must be {article} "
                f"{value_type.__name__}, got {type(self.value).__name__}"
            )

    def __repr__(self) -> str:
        return (
            f"DefaultHint(key={self.key.name!r}, value={self.value!r}, "
            f"strength={self.strength.name}, source={self.source!r})"
        )


@dataclass(frozen=True)
class ResolvedDefault:
    """A resolved default value with metadata."""

    key: DefaultKey
    value: Any
    selected_hint: DefaultHint
    alternatives: tuple[DefaultHint, ...] = ()

    def __repr__(self) -> str:
        return (
            f"ResolvedDefault(key={self.key.name!r}, value={self.value!r}, "
            f"source={self.selected_hint.source!r})"
        )


@dataclass(frozen=True)
class DefaultResolution:
    """Result of default resolution."""

    values: dict[DefaultKey, Any] = field(default_factory=dict)
    resolved: dict[DefaultKey, ResolvedDefault] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()

    def get(self, key: DefaultKey, default: Any = None) -> Any:
        """Get a resolved value by key."""
        return self.values.get(key, default)

    def __repr__(self) -> str:
        return f"DefaultResolution(values={self.values!r})"
