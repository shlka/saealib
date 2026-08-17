from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from saealib.core.contracts.vocabulary import Vocabulary, VocabularyDescriptor
from saealib.exceptions import ValidationError

__all__ = [
    "ASSUMPTION_KEYS",
    "AssumptionDescriptor",
    "AssumptionSet",
    "register_assumption",
    "validate_assumption_name",
]


_ASSUMPTION_PART = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def validate_assumption_name(name: str) -> str:
    """Validate a dotted assumption key name."""
    if not isinstance(name, str):
        raise ValidationError("Assumption names must be strings")
    parts = name.split(".")
    if len(parts) < 2 or any(
        _ASSUMPTION_PART.fullmatch(part) is None for part in parts
    ):
        raise ValidationError(
            "Assumption names must contain identifier parts separated by dots"
        )
    return name


@dataclass(frozen=True, kw_only=True)
class AssumptionDescriptor(VocabularyDescriptor):
    """Describe an assumption and its unaware-contract default."""

    unaware_default: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.unaware_default, bool):
            raise ValidationError("Assumption defaults must be booleans")


ASSUMPTION_KEYS: Vocabulary[AssumptionDescriptor] = Vocabulary(
    name_validator=validate_assumption_name
)


def register_assumption(
    name: str,
    *,
    description: str,
    unaware_default: bool,
    registry: Vocabulary[AssumptionDescriptor] | None = None,
) -> AssumptionDescriptor:
    """Register an assumption key with its unaware default."""
    target = ASSUMPTION_KEYS if registry is None else registry
    descriptor = AssumptionDescriptor(
        name=name,
        description=description,
        unaware_default=unaware_default,
    )
    target.register(name, descriptor)
    return descriptor


register_assumption(
    "observation_schema.fixed",
    description="The observation schema remains fixed during a run.",
    unaware_default=True,
)
register_assumption(
    "evaluation.deterministic",
    description="Re-evaluating a genome yields the same observation.",
    unaware_default=True,
)
register_assumption(
    "population.fixed_size",
    description="The main population size remains fixed across generations.",
    unaware_default=True,
)


class AssumptionSet(Mapping[str, bool]):
    """Map declared assumptions while supplying unaware defaults."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, bool] | None = None) -> None:
        if values is None:
            values = {}
        if not isinstance(values, Mapping):
            raise ValidationError("AssumptionSet values must be a mapping")
        copied: dict[str, bool] = {}
        for name, value in values.items():
            validate_assumption_name(name)
            if not isinstance(value, bool):
                raise ValidationError("Assumption values must be booleans")
            copied[name] = value
        self._values = copied

    @classmethod
    def empty(cls) -> AssumptionSet:
        """Return an empty assumption set."""
        return cls()

    def __getitem__(self, name: str) -> bool:
        if name in self._values:
            return self._values[name]
        descriptor = ASSUMPTION_KEYS.get(name)
        if descriptor is None:
            raise KeyError(name)
        return descriptor.unaware_default

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AssumptionSet):
            return NotImplemented
        return self._values == other._values
