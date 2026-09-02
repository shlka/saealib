"""Builtin default hint provider with generic fallback rules."""

from __future__ import annotations

from saealib.defaults.context import DefaultContext
from saealib.defaults.keys import INITIAL_ARCHIVE_SIZE, MAX_EVALUATIONS, POPULATION_SIZE
from saealib.defaults.model import DefaultHint, DefaultStrength


class BuiltinDefaultProvider:
    """Generic fallback default provider.

    Provides dimension-scaled defaults:
    - population.size = 4 * dim
    - archive.initial_size = 5 * dim
    - termination.max_evaluations = 200 * dim
    """

    def default_hints(self, ctx: DefaultContext) -> tuple[DefaultHint, ...]:
        """Provide generic fallback hints based on problem dimension."""
        dim = ctx.dim
        return (
            DefaultHint(
                key=POPULATION_SIZE,
                value=4 * dim,
                strength=DefaultStrength.FALLBACK,
                source="builtin",
                reason=f"Generic fallback: 4 * dim = {4 * dim}",
            ),
            DefaultHint(
                key=INITIAL_ARCHIVE_SIZE,
                value=5 * dim,
                strength=DefaultStrength.FALLBACK,
                source="builtin",
                reason=f"Generic fallback: 5 * dim = {5 * dim}",
            ),
            DefaultHint(
                key=MAX_EVALUATIONS,
                value=200 * dim,
                strength=DefaultStrength.FALLBACK,
                source="builtin",
                reason=f"Generic fallback: 200 * dim = {200 * dim}",
            ),
        )


BUILTIN_DEFAULT_PROVIDER = BuiltinDefaultProvider()
