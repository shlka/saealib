"""DefaultHintProvider protocol and DefaultResolver."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from saealib.defaults.context import DefaultContext
from saealib.defaults.keys import DefaultKey
from saealib.defaults.model import (
    DefaultHint,
    DefaultResolution,
    DefaultStrength,
    ResolvedDefault,
)

if TYPE_CHECKING:
    pass


@runtime_checkable
class DefaultHintProvider(Protocol):
    """Protocol for components that can provide default hints."""

    def default_hints(self, ctx: DefaultContext) -> tuple[DefaultHint, ...]:
        """Provide default hints based on the current context.

        Parameters
        ----------
        ctx : DefaultContext
            The default resolution context.

        Returns
        -------
        tuple[DefaultHint, ...]
            Hints provided by this component.
        """
        ...


class DefaultResolver:
    """Resolver for default values.

    Collects hints from all components and resolves them based on precedence:
    explicit > REQUIRED > RECOMMENDED > FALLBACK.

    For same-strength hints with different values, the first hint wins
    (deterministic based on component traversal order).
    """

    def resolve(
        self,
        ctx: DefaultContext,
        providers: list,
        explicit: dict[DefaultKey, Any] | None = None,
    ) -> DefaultResolution:
        """Resolve default values from hints and explicit overrides.

        Parameters
        ----------
        ctx : DefaultContext
            The default resolution context.
        providers : list[DefaultHintProvider]
            Components that provide default hints.
        explicit : dict[DefaultKey, Any] | None
            Explicitly set values that override all hints.

        Returns
        -------
        DefaultResolution
            The resolved default values.
        """
        explicit = explicit or {}
        all_hints: dict[str, list[DefaultHint]] = {}
        diagnostics: list[str] = []

        # Collect hints from all providers
        for provider in providers:
            try:
                hints = provider.default_hints(ctx)
                for hint in hints:
                    key_name = hint.key.name
                    if key_name not in all_hints:
                        all_hints[key_name] = []
                    all_hints[key_name].append(hint)
            except Exception as e:
                diagnostics.append(
                    f"Warning: {type(provider).__name__}.default_hints() "
                    f"raised {type(e).__name__}: {e}"
                )

        # Resolve each key
        values: dict[str, Any] = {}
        resolved: dict[str, ResolvedDefault] = {}

        for key_name, hints in all_hints.items():
            if not hints:
                continue

            # Get the key from the first hint
            key = hints[0].key

            # Check for explicit override
            if key in explicit:
                values[key_name] = explicit[key]
                resolved[key_name] = ResolvedDefault(
                    key=key,
                    value=explicit[key],
                    selected_hint=DefaultHint(
                        key=key,
                        value=explicit[key],
                        strength=DefaultStrength.REQUIRED,
                        source="explicit",
                        reason="Explicitly set by user",
                    ),
                    alternatives=tuple(hints),
                )
                continue

            # Sort by strength (highest first), then by order of appearance
            sorted_hints = sorted(hints, key=lambda h: h.strength, reverse=True)

            # Select the highest strength hint
            selected = sorted_hints[0]

            # Check for conflicts at the same strength level
            same_strength = [h for h in sorted_hints if h.strength == selected.strength]
            if len(same_strength) > 1:
                # Check if they agree on the value
                unique_values = set(h.value for h in same_strength)
                if len(unique_values) > 1:
                    diagnostics.append(
                        f"Conflict for {key_name}: multiple hints at "
                        f"strength {selected.strength.name} with different "
                        f"values. Using first hint from {selected.source}."
                    )

            values[key_name] = selected.value
            resolved[key_name] = ResolvedDefault(
                key=key,
                value=selected.value,
                selected_hint=selected,
                alternatives=tuple(hints),
            )

        return DefaultResolution(
            values=values,
            resolved=resolved,
            diagnostics=tuple(diagnostics),
        )


# Global resolver instance
DEFAULT_RESOLVER = DefaultResolver()
