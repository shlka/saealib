"""DefaultHintProvider protocol and DefaultResolver."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from saealib.defaults.context import DefaultContext
from saealib.defaults.keys import DefaultKey
from saealib.defaults.model import (
    DefaultHint,
    DefaultResolution,
    DefaultStrength,
    ResolvedDefault,
)
from saealib.exceptions import ConfigurationError


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

    Same-strength hints must agree on their value.  A disagreement is a
    configuration error rather than an order-dependent choice.
    """

    def resolve(
        self,
        ctx: DefaultContext,
        providers: Sequence[DefaultHintProvider],
        explicit: Mapping[DefaultKey, Any] | None = None,
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
        all_hints: dict[DefaultKey, list[DefaultHint]] = {}
        diagnostics: list[str] = []

        for provider in providers:
            try:
                hints = tuple(provider.default_hints(ctx))
            except ConfigurationError:
                raise
            except Exception as error:
                raise ConfigurationError(
                    f"{type(provider).__name__}.default_hints() failed: "
                    f"{type(error).__name__}: {error}"
                ) from error

            for index, hint in enumerate(hints):
                if not isinstance(hint, DefaultHint):
                    raise ConfigurationError(
                        f"{type(provider).__name__}.default_hints() returned "
                        f"item {index} of type {type(hint).__name__}; expected "
                        "DefaultHint"
                    )
                all_hints.setdefault(hint.key, []).append(hint)

        values: dict[DefaultKey, Any] = {}
        resolved: dict[DefaultKey, ResolvedDefault] = {}

        for key, hints in all_hints.items():
            if not hints:
                continue

            if key in explicit:
                selected = self._explicit_hint(key, explicit[key])
                values[key] = selected.value
                resolved[key] = ResolvedDefault(
                    key=key,
                    value=selected.value,
                    selected_hint=selected,
                    alternatives=tuple(hints),
                )
                continue

            sorted_hints = sorted(hints, key=lambda h: h.strength, reverse=True)

            # Every precedence level is required to be internally consistent.
            # Checking lower-strength groups too catches contradictory provider
            # implementations even when a stronger hint happens to mask them.
            for strength in DefaultStrength:
                same_strength = [
                    hint for hint in sorted_hints if hint.strength == strength
                ]
                if len(same_strength) > 1 and not all(
                    _values_equal(same_strength[0].value, hint.value)
                    for hint in same_strength[1:]
                ):
                    details = ", ".join(
                        f"{hint.source}={hint.value!r}" for hint in same_strength
                    )
                    raise ConfigurationError(
                        f"Conflict for {key.name}: multiple hints at strength "
                        f"{strength.name} have different values ({details})"
                    )

            selected = sorted_hints[0]
            values[key] = selected.value
            resolved[key] = ResolvedDefault(
                key=key,
                value=selected.value,
                selected_hint=selected,
                alternatives=tuple(hints),
            )

        # Explicit values are useful even when no provider happened to emit a
        # hint for the key.  Constructing the synthetic hint also validates the
        # explicit value against the key's declared type.
        for key, value in explicit.items():
            if key in values:
                continue
            selected = self._explicit_hint(key, value)
            values[key] = selected.value
            resolved[key] = ResolvedDefault(
                key=key,
                value=selected.value,
                selected_hint=selected,
            )

        return DefaultResolution(
            values=values,
            resolved=resolved,
            diagnostics=tuple(diagnostics),
        )

    @staticmethod
    def _explicit_hint(key: DefaultKey, value: Any) -> DefaultHint:
        """Build a validated hint for an explicit user-provided value."""
        try:
            return DefaultHint(
                key=key,
                value=value,
                strength=DefaultStrength.REQUIRED,
                source="explicit",
                reason="Explicitly set by user",
            )
        except Exception as error:
            key_name = getattr(key, "name", repr(key))
            raise ConfigurationError(
                f"Explicit default for {key_name!r} is invalid: "
                f"{type(error).__name__}: {error}"
            ) from error


def _values_equal(left: Any, right: Any) -> bool:
    """Compare hint values without assuming they are hashable scalars."""
    try:
        equal = left == right
        if isinstance(equal, bool):
            return equal
        return bool(equal.all())
    except (AttributeError, TypeError, ValueError):
        return False


DEFAULT_RESOLVER = DefaultResolver()
