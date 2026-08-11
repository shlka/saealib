"""Immutable structured-control vocabulary for compiler front ends."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, TypeAlias

from saealib.core.contracts.state import StateContract
from saealib.core.contracts.vocabulary import validate_name
from saealib.core.state.keys import StateKey
from saealib.exceptions import ValidationError

if TYPE_CHECKING:
    from saealib.core.compiler.structured import StructuredGraph
    from saealib.core.state.store import StateView

__all__ = [
    "BranchRegion",
    "Condition",
    "LoopRegion",
    "RegionEffect",
    "RegionNode",
    "RepeatRegion",
    "SequenceRegion",
    "StructuredRegion",
    "compose_effects",
]

RegionEffect: TypeAlias = StateContract


class Condition(Protocol):
    """A side-effect-free predicate used by a loop or branch region."""

    def contract(self) -> StateContract:
        """Return the condition's declared state contract."""

    def evaluate(self, view: StateView) -> bool:
        """Evaluate the predicate against its declared state view."""


def _region_id(value: str, label: str = "region_id") -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{label} must be a non-empty string")
    return validate_name(value)


def _merge_keys(
    effects: Iterable[StateContract],
    accessor: Callable[[StateContract], tuple[StateKey[object], ...]],
) -> tuple[StateKey[object], ...]:
    result: list[StateKey[object]] = []
    for effect in effects:
        for key in accessor(effect):
            if key not in result:
                result.append(key)
    return tuple(result)


def compose_effects(effects: Iterable[StateContract]) -> StateContract:
    """Compose state effects, retaining first-seen key order and precision."""
    values = tuple(effects)
    if any(not isinstance(effect, StateContract) for effect in values):
        raise ValidationError("Region effects must be StateContract values")

    return StateContract(
        reads=_merge_keys(values, lambda effect: effect.reads),
        writes=_merge_keys(values, lambda effect: effect.writes),
        exports=_merge_keys(values, lambda effect: effect.exports),
        reads_enumerable=all(effect.reads_enumerable for effect in values),
    )


@dataclass(frozen=True, kw_only=True)
class StructuredRegion:
    """Base immutable description of a structured control region."""

    region_id: str
    body: StructuredGraph | tuple[object, ...]
    namespace: str = ""
    effect: RegionEffect = field(default_factory=StateContract)

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_id", _region_id(self.region_id))
        if not isinstance(self.namespace, str):
            raise ValidationError("Region namespace must be a string")
        object.__setattr__(self, "namespace", self.namespace)
        if isinstance(self.body, list):
            object.__setattr__(self, "body", tuple(self.body))
        if not isinstance(self.effect, StateContract):
            raise ValidationError("Region effect must be a StateContract")

    @property
    def qualified_id(self) -> str:
        """Return the region id qualified by its enclosing namespace."""
        return (
            f"{self.namespace}.{self.region_id}" if self.namespace else self.region_id
        )

    def with_body(
        self, body: StructuredGraph, *, effect: RegionEffect
    ) -> StructuredRegion:
        """Return an immutable copy retaining the lowered body and effect."""
        return replace(self, body=body, effect=effect)


@dataclass(frozen=True, kw_only=True)
class SequenceRegion(StructuredRegion):
    """A sequential nested pipeline body."""


@dataclass(frozen=True, kw_only=True)
class RepeatRegion(StructuredRegion):
    """A body repeated by a fixed count."""

    count: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.count, bool):
            raise ValidationError("Repeat count must be a non-negative integer")
        if not isinstance(self.count, int) or self.count < 0:
            raise ValidationError("Repeat count must be a non-negative integer")


def _validate_condition(condition: Condition) -> None:
    if not callable(getattr(condition, "contract", None)):
        raise ValidationError("Region condition must provide contract()")
    if not callable(getattr(condition, "evaluate", None)):
        raise ValidationError("Region condition must provide evaluate(context)")
    contract = condition.contract()
    if not isinstance(contract, StateContract):
        raise ValidationError("Region condition contract() must return StateContract")


@dataclass(frozen=True, kw_only=True)
class LoopRegion(StructuredRegion):
    """A body controlled by a condition; it is not represented by graph cycles."""

    condition: Condition

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_condition(self.condition)


@dataclass(frozen=True, kw_only=True)
class BranchRegion(StructuredRegion):
    """A condition-controlled body, optionally with an alternate body."""

    condition: Condition
    otherwise: StructuredGraph | tuple[object, ...] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_condition(self.condition)
        if isinstance(self.otherwise, list):
            object.__setattr__(self, "otherwise", tuple(self.otherwise))


@dataclass(frozen=True, kw_only=True)
class RegionNode:
    """Retain a region and its compilation-local namespace metadata."""

    region: StructuredRegion
    metadata: MappingProxyType | dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.region, StructuredRegion):
            raise ValidationError("RegionNode region must be a StructuredRegion")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
