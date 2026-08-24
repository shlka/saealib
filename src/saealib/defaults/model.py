"""Default resolution models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from saealib.defaults.keys import DefaultKey


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

    values: dict[str, Any] = field(default_factory=dict)
    resolved: dict[str, ResolvedDefault] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()

    def get(self, key: DefaultKey, default: Any = None) -> Any:
        """Get a resolved value by key."""
        return self.values.get(key.name, default)

    def __repr__(self) -> str:
        return f"DefaultResolution(values={self.values!r})"
