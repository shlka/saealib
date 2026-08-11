"""Immutable structured-control vocabulary for compiler front ends."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypeGuard

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

    def evaluate(self, view: StateView, /) -> bool:
        """Evaluate the predicate against its declared state view."""


class _NamedStructuralValue(Protocol):
    name: str


class _PipelineLike(Protocol):
    stages: Sequence[object]


class _ComponentLike(Protocol):
    def contract(self) -> object: ...


class _RepeatLike(Protocol):
    body: object
    count: int


class _LoopLike(Protocol):
    body: object
    condition: Condition


class _BranchLike(Protocol):
    then: object
    condition: Condition
    else_: object | None


_repeat_type: type[object] | None = None
_loop_type: type[object] | None = None
_branch_type: type[object] | None = None
_stage_type: type[object] | None = None


def _register_structural_types(
    repeat_type: type[object],
    loop_type: type[object],
    branch_type: type[object],
    stage_type: type[object],
) -> None:
    global _repeat_type, _loop_type, _branch_type, _stage_type
    _repeat_type = repeat_type
    _loop_type = loop_type
    _branch_type = branch_type
    _stage_type = stage_type


def _is_pipeline_like(value: object) -> TypeGuard[_PipelineLike]:
    """Return whether *value* exposes a structural pipeline sequence."""
    stages = getattr(value, "stages", None)
    return (
        not callable(getattr(value, "contract", None))
        and isinstance(stages, Sequence)
        and not isinstance(stages, (str, bytes))
    )


def _is_component(value: object) -> TypeGuard[_ComponentLike]:
    """Return whether *value* exposes the minimal component contract."""
    return callable(getattr(value, "contract", None))


def _is_condition(value: object) -> TypeGuard[Condition]:
    """Return whether *value* exposes the condition protocol at runtime."""
    return callable(getattr(value, "contract", None)) and callable(
        getattr(value, "evaluate", None)
    )


def _is_named_structural_value(value: object) -> TypeGuard[_NamedStructuralValue]:
    """Return whether *value* has a string structural name."""
    return isinstance(getattr(value, "name", None), str)


def _structural_name(value: object, default: str = "") -> str:
    """Read a structural name without requiring one from every entry."""
    return value.name if _is_named_structural_value(value) else default


def _is_repeat(value: object) -> TypeGuard[_RepeatLike]:
    return _repeat_type is not None and isinstance(value, _repeat_type)


def _is_loop(value: object) -> TypeGuard[_LoopLike]:
    return _loop_type is not None and isinstance(value, _loop_type)


def _is_branch(value: object) -> TypeGuard[_BranchLike]:
    return _branch_type is not None and isinstance(value, _branch_type)


def _is_stage(value: object) -> bool:
    return _stage_type is not None and isinstance(value, _stage_type)


def _structural_stages(value: object) -> tuple[object, ...] | None:
    if _is_repeat(value) or _is_loop(value):
        return (value.body,)
    if _is_branch(value):
        children = (value.then,)
        return children if value.else_ is None else (*children, value.else_)
    if _is_pipeline_like(value):
        return tuple(value.stages)
    return None


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
