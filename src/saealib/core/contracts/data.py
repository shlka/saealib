from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias

from saealib.core.contracts.vocabulary import (
    Vocabulary,
    VocabularyDescriptor,
    validate_name,
)
from saealib.exceptions import ValidationError

__all__ = [
    "DATA_SPEC_KINDS",
    "Contained",
    "DataSpec",
    "DataSpecKind",
    "Fixed",
    "Product",
    "SchemaBinding",
    "Var",
    "data_spec_kind",
    "is_data_spec_compatible",
    "is_set_like",
    "register_data_spec",
]


@dataclass(frozen=True, kw_only=True)
class DataSpecKind(VocabularyDescriptor):
    """Metadata for a registered nominal data kind."""

    variables: tuple[str, ...] = ()
    supertypes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "variables", tuple(self.variables))
        supertypes = tuple(self.supertypes)
        for supertype in supertypes:
            validate_name(supertype)
        object.__setattr__(self, "supertypes", supertypes)


DATA_SPEC_KINDS: Vocabulary[DataSpecKind] = Vocabulary()


def register_data_spec(
    name: str,
    *,
    variables: tuple[str, ...] = (),
    description: str = "",
    supertypes: tuple[str, ...] = (),
    registry: Vocabulary[DataSpecKind] | None = None,
) -> DataSpecKind:
    """Register a nominal data kind and its explicit supertypes."""
    target = DATA_SPEC_KINDS if registry is None else registry
    descriptor = DataSpecKind(
        name=name,
        description=description,
        variables=variables,
        supertypes=supertypes,
    )
    target.register(name, descriptor)
    return descriptor


def data_spec_kind(
    name: str,
    *,
    registry: Vocabulary[DataSpecKind] | None = None,
) -> DataSpecKind | None:
    """Return a registered data kind, or ``None`` when it is unknown."""
    target = DATA_SPEC_KINDS if registry is None else registry
    return target.get(name)


@dataclass(frozen=True, kw_only=True)
class Var:
    """An unbound schema-variable reference."""

    name: str

    def __post_init__(self) -> None:
        validate_name(self.name)


@dataclass(frozen=True, kw_only=True)
class Fixed:
    """A fixed, hashable schema binding value."""

    value: Hashable

    def __post_init__(self) -> None:
        try:
            hash(self.value)
        except TypeError as error:
            raise ValidationError("Fixed schema values must be hashable") from error


def is_set_like(value: object) -> bool:
    """Return whether a value supports collection-style containment."""
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes))


@dataclass(frozen=True, kw_only=True)
class Contained:
    """A consumer-side requirement for a set of schema values."""

    values: frozenset[Hashable]

    def __post_init__(self) -> None:
        if isinstance(self.values, (str, bytes)) or not isinstance(
            self.values, Iterable
        ):
            raise ValidationError("Contained values must be a non-string iterable")
        try:
            values = frozenset(self.values)
        except TypeError as error:
            raise ValidationError("Contained values must be hashable") from error
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, kw_only=True)
class Product:
    """A positional product of schema bindings."""

    elements: tuple[SchemaBinding, ...]

    def __post_init__(self) -> None:
        try:
            elements = tuple(self.elements)
        except TypeError as error:
            raise ValidationError("Product elements must be iterable") from error
        if not elements:
            raise ValidationError("Product elements must not be empty")
        if any(not _is_schema_binding(element) for element in elements):
            raise ValidationError("Product elements must contain schema bindings")
        object.__setattr__(self, "elements", elements)


def _is_schema_binding(value: object) -> bool:
    return isinstance(value, (Var, Fixed, Contained, Product))


SchemaBinding: TypeAlias = Var | Fixed | Contained | Product


@dataclass(frozen=True, kw_only=True)
class DataSpec:
    """A nominal data kind with bound schema-variable values."""

    kind: str
    bindings: Mapping[str, SchemaBinding] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str):
            raise ValidationError("DataSpec kind must be a string")
        if not isinstance(self.bindings, Mapping):
            raise ValidationError("DataSpec bindings must be a mapping")
        bindings = dict(self.bindings)
        if any(
            not isinstance(name, str) or not _is_schema_binding(value)
            for name, value in bindings.items()
        ):
            raise ValidationError("DataSpec bindings must contain schema bindings")
        for name in bindings:
            validate_name(name)
        object.__setattr__(self, "bindings", MappingProxyType(bindings))

    def __hash__(self) -> int:
        return hash((self.kind, frozenset(self.bindings.items())))


def _kind_satisfies(
    provided: str,
    required: str,
    registry: Vocabulary[DataSpecKind],
    visited: set[str],
) -> bool:
    if provided in visited:
        return False
    visited.add(provided)
    provided_descriptor = registry.get(provided)
    required_descriptor = registry.get(required)
    if provided_descriptor is None or required_descriptor is None:
        return False
    if provided == required:
        return True
    return any(
        _kind_satisfies(supertype, required, registry, visited)
        for supertype in provided_descriptor.supertypes
    )


def is_data_spec_compatible(
    provided: DataSpec,
    required: DataSpec,
    *,
    registry: Vocabulary[DataSpecKind] | None = None,
) -> bool:
    """Check registered kind identity or explicit subtype compatibility.

    Schema-variable unification and service resolution are separate compiler
    operations and are intentionally not inferred by this function.
    """
    target = DATA_SPEC_KINDS if registry is None else registry
    return _kind_satisfies(provided.kind, required.kind, target, set())
