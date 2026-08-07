from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from saealib.core.contracts.data import (
    DATA_SPEC_KINDS,
    DataSpec,
    DataSpecKind,
    is_data_spec_compatible,
)
from saealib.core.contracts.schema import (
    Substitution,
    UnificationResult,
    unify_data_specs,
)
from saealib.core.contracts.vocabulary import (
    Vocabulary,
    VocabularyDescriptor,
    validate_name,
)
from saealib.exceptions import ValidationError

__all__ = [
    "CARDINALITIES",
    "MANY",
    "ONE",
    "OPTIONAL",
    "SERVICE_VOCABULARY",
    "Cardinality",
    "PortCompatibility",
    "PortContract",
    "PortDirection",
    "PortSpec",
    "ServiceRequirement",
    "cardinality_satisfies",
    "check_port_compatibility",
    "validate_port_contract_directions",
]

Cardinality: TypeAlias = str
ONE: Cardinality = "ONE"
MANY: Cardinality = "MANY"
OPTIONAL: Cardinality = "OPTIONAL"


class PortDirection(str, Enum):
    """The direction in which a port carries data."""

    INPUT = "input"
    OUTPUT = "output"


CARDINALITIES: Vocabulary[VocabularyDescriptor] = Vocabulary()
for _name, _description in (
    (ONE, "Exactly one item per activation."),
    (MANY, "A finite batch known in full at activation."),
    (OPTIONAL, "Zero or one item."),
):
    CARDINALITIES.register(
        _name,
        VocabularyDescriptor(name=_name, description=_description),
    )


SERVICE_VOCABULARY: Vocabulary[VocabularyDescriptor] = Vocabulary()
for _name, _description in (
    (
        "SamplingService",
        "draw genomes (initialization, restart, random immigrants)",
    ),
    ("ValidationService", "is this genome well-formed for this space"),
    ("CloneService", "produce an independent copy"),
    (
        "FingerprintService",
        "an exact, canonical, hashable identity for a genome",
    ),
    ("EquivalenceService", "configurable approximate duplicate matching"),
    ("GenomeCodec", "genome ↔ persistable primitives"),
    (
        "DenseNumericView",
        "zero-copy dense numeric access, where it exists",
    ),
    ("BoundsService", "per-variable bounds, where they exist"),
    ("DistanceService", "distance between genomes"),
    ("ComparisonService", "order candidates by their objective values"),
):
    SERVICE_VOCABULARY.register(
        _name,
        VocabularyDescriptor(name=_name, description=_description),
    )


def cardinality_satisfies(provided: Cardinality, required: Cardinality) -> bool:
    """Return whether a produced cardinality satisfies a consumer."""
    if provided not in CARDINALITIES or required not in CARDINALITIES:
        return False
    if provided == required:
        return True
    return provided == ONE and required in {MANY, OPTIONAL}


@dataclass(frozen=True, kw_only=True)
class ServiceRequirement:
    """A named service required by one port."""

    name: str

    def __post_init__(self) -> None:
        """Validate the service-name shape."""
        validate_name(self.name)


@dataclass(frozen=True, kw_only=True)
class PortSpec:
    """A named, directional data contract."""

    name: str
    direction: PortDirection
    data: DataSpec
    cardinality: Cardinality
    required_services: tuple[ServiceRequirement, ...] = ()
    optional: bool = False

    def __post_init__(self) -> None:
        """Validate and normalize port metadata."""
        validate_name(self.name)
        if not isinstance(self.direction, PortDirection):
            raise ValidationError("Port direction must be a PortDirection")
        if not isinstance(self.data, DataSpec):
            raise ValidationError("Port data must be a DataSpec")
        if not isinstance(self.cardinality, str) or not self.cardinality:
            raise ValidationError("Port cardinality must be a non-empty name")
        validate_name(self.cardinality)
        required_services = tuple(self.required_services)
        if any(
            not isinstance(service, ServiceRequirement) for service in required_services
        ):
            raise ValidationError(
                "Port required_services must contain ServiceRequirement values"
            )
        object.__setattr__(self, "required_services", required_services)


@dataclass(frozen=True, kw_only=True)
class PortContract:
    """Input and output ports for one component role."""

    inputs: tuple[PortSpec, ...] = ()
    outputs: tuple[PortSpec, ...] = ()

    def __post_init__(self) -> None:
        """Normalize ports and enforce per-direction name uniqueness."""
        inputs = tuple(self.inputs)
        outputs = tuple(self.outputs)
        for ports in (inputs, outputs):
            if any(not isinstance(port, PortSpec) for port in ports):
                raise ValidationError("Port tuples must contain PortSpec values")
            names = [port.name for port in ports]
            if len(names) != len(set(names)):
                raise ValidationError("Port names must be unique per direction")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)


def validate_port_contract_directions(contract: PortContract) -> None:
    """Validate both direction-specific tuples in a port contract."""
    for ports, expected in (
        (contract.inputs, PortDirection.INPUT),
        (contract.outputs, PortDirection.OUTPUT),
    ):
        for port in ports:
            if port.direction is not expected:
                raise ValidationError(
                    f"Port {port.name!r} is not an {expected.value} port"
                )


@dataclass(frozen=True, kw_only=True)
class PortCompatibility:
    """Result of the port checks implemented in this contract layer."""

    kind_ok: bool
    cardinality_ok: bool
    direction_ok: bool
    schema: UnificationResult
    unknown_kinds: tuple[str, ...] = ()
    unknown_cardinalities: tuple[str, ...] = ()
    unchecked: tuple[str, ...] = ()

    @property
    def schema_ok(self) -> bool:
        """Return whether schema bindings unified without unknown variables."""
        return self.schema.unified

    @property
    def compatible(self) -> bool:
        """Return whether every performed compatibility check passed."""
        return (
            self.kind_ok
            and self.cardinality_ok
            and self.direction_ok
            and self.schema_ok
        )


def check_port_compatibility(
    producer: PortSpec,
    consumer: PortSpec,
    *,
    data_registry: Vocabulary[DataSpecKind] | None = None,
    substitution: Substitution | None = None,
) -> PortCompatibility:
    """Check port kinds, schema bindings, and cardinalities."""
    registry = DATA_SPEC_KINDS if data_registry is None else data_registry
    kind_names = (producer.data.kind, consumer.data.kind)
    unknown_kinds = tuple(
        dict.fromkeys(kind for kind in kind_names if registry.get(kind) is None)
    )
    cardinality_names = (producer.cardinality, consumer.cardinality)
    unknown_cardinalities = tuple(
        dict.fromkeys(
            cardinality
            for cardinality in cardinality_names
            if cardinality not in CARDINALITIES
        )
    )
    schema = unify_data_specs(
        producer.data,
        consumer.data,
        substitution=substitution,
    )
    return PortCompatibility(
        kind_ok=not unknown_kinds
        and is_data_spec_compatible(
            producer.data,
            consumer.data,
            registry=registry,
        ),
        cardinality_ok=not unknown_cardinalities
        and cardinality_satisfies(producer.cardinality, consumer.cardinality),
        direction_ok=(
            producer.direction is PortDirection.OUTPUT
            and consumer.direction is PortDirection.INPUT
        ),
        schema=schema,
        unknown_kinds=unknown_kinds,
        unknown_cardinalities=unknown_cardinalities,
        unchecked=("services",),
    )
