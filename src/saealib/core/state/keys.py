from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from saealib.core.contracts.vocabulary import (
    Vocabulary,
    VocabularyDescriptor,
    validate_name,
)
from saealib.exceptions import ValidationError

__all__ = [
    "PENDING_EVALUATIONS",
    "POPULATIONS_MAIN",
    "RUNTIME_RNG",
    "STATE_NAMESPACES",
    "SURROGATES_DEFAULT",
    "StateKey",
]


ValueT = TypeVar("ValueT")


STATE_NAMESPACES: Vocabulary[VocabularyDescriptor] = Vocabulary()
for _name, _description in (
    ("populations", "Population state."),
    ("archives", "Archive state."),
    ("proposals", "Proposal state."),
    ("feedback", "Feedback state."),
    ("evaluations", "Evaluation state."),
    ("surrogates", "Surrogate state."),
    ("algorithms", "Algorithm state."),
    ("runtime", "Runtime state."),
    ("user", "User-owned state."),
):
    STATE_NAMESPACES.register(
        _name,
        VocabularyDescriptor(name=_name, description=_description),
    )


@dataclass(frozen=True, kw_only=True)
class StateKey(Generic[ValueT]):
    """Identify one versioned state value."""

    namespace: str
    name: str
    schema_version: int

    def __post_init__(self) -> None:
        """Validate the state-key fields."""
        validate_name(self.namespace)
        validate_name(self.name)
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version < 1
        ):
            raise ValidationError("State schema_version must be a positive integer")


POPULATIONS_MAIN = StateKey[object](
    namespace="populations", name="main", schema_version=1
)
PENDING_EVALUATIONS = StateKey[object](
    namespace="evaluations", name="pending", schema_version=1
)
RUNTIME_RNG = StateKey[object](namespace="runtime", name="rng", schema_version=1)
SURROGATES_DEFAULT = StateKey[object](
    namespace="surrogates", name="default", schema_version=1
)
