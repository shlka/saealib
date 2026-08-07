import pytest

from saealib.core.contracts.data import DataSpec, DataSpecKind, register_data_spec
from saealib.core.contracts.ports import (
    CARDINALITIES,
    MANY,
    ONE,
    OPTIONAL,
    SERVICE_VOCABULARY,
    PortCompatibility,
    PortContract,
    PortDirection,
    PortSpec,
    ServiceRequirement,
    cardinality_satisfies,
    check_port_compatibility,
    validate_port_contract_directions,
)
from saealib.core.contracts.vocabulary import Vocabulary
from saealib.exceptions import ValidationError


def _port(
    name: str,
    direction: PortDirection,
    cardinality: str = ONE,
    kind: str = "test:genomes",
) -> PortSpec:
    return PortSpec(
        name=name,
        direction=direction,
        data=DataSpec(kind=kind),
        cardinality=cardinality,
        required_services=(ServiceRequirement(name="BoundsService"),),
    )


def test_cardinality_is_open_and_directional() -> None:
    assert CARDINALITIES.names() == (ONE, MANY, OPTIONAL)
    assert "STREAM" not in CARDINALITIES
    assert cardinality_satisfies(ONE, ONE)
    assert cardinality_satisfies(ONE, MANY)
    assert not cardinality_satisfies(MANY, ONE)
    assert cardinality_satisfies(ONE, OPTIONAL)
    assert not cardinality_satisfies(MANY, OPTIONAL)
    assert not cardinality_satisfies(OPTIONAL, ONE)
    assert not cardinality_satisfies(OPTIONAL, MANY)
    assert cardinality_satisfies(OPTIONAL, OPTIONAL)


def test_service_names_are_registered() -> None:
    assert SERVICE_VOCABULARY.names() == (
        "SamplingService",
        "ValidationService",
        "CloneService",
        "FingerprintService",
        "EquivalenceService",
        "GenomeCodec",
        "DenseNumericView",
        "BoundsService",
        "DistanceService",
        "ComparisonService",
    )
    descriptor = SERVICE_VOCABULARY.get("DistanceService")
    assert descriptor is not None
    assert descriptor.description


def test_port_contract_enforces_unique_names_per_direction() -> None:
    contract = PortContract(inputs=(_port("genomes", PortDirection.INPUT),))
    assert contract.inputs[0].required_services[0].name == "BoundsService"

    with pytest.raises(ValidationError):
        PortContract(
            inputs=(
                _port("x", PortDirection.INPUT),
                _port("x", PortDirection.INPUT),
            )
        )


def test_port_direction_compliance_is_an_explicit_check() -> None:
    malformed = PortContract(inputs=(_port("x", PortDirection.OUTPUT),))

    with pytest.raises(ValidationError):
        validate_port_contract_directions(malformed)


def test_kind_and_cardinality_connection_rules_are_partial() -> None:
    registry: Vocabulary[DataSpecKind] = Vocabulary()
    register_data_spec("test:genomes", description="Genomes", registry=registry)
    producer = _port("offspring", PortDirection.OUTPUT)
    consumer = _port("genomes", PortDirection.INPUT, MANY)

    result = check_port_compatibility(
        producer,
        consumer,
        data_registry=registry,
    )
    assert isinstance(result, PortCompatibility)
    assert result.kind_ok
    assert result.cardinality_ok
    assert result.direction_ok
    assert result.schema_ok
    assert result.compatible
    assert result.unchecked == ("services",)


def test_port_compatibility_surfaces_unknown_values() -> None:
    result = check_port_compatibility(
        _port("future_output", PortDirection.OUTPUT, "FUTURE", "future:kind"),
        _port("future_input", PortDirection.INPUT, "FUTURE", "future:kind"),
    )

    assert not result.compatible
    assert result.unknown_kinds == ("future:kind",)
    assert result.unknown_cardinalities == ("FUTURE",)


def test_port_compatibility_rejects_two_input_ports() -> None:
    registry: Vocabulary[DataSpecKind] = Vocabulary()
    register_data_spec("test:genomes", description="Genomes", registry=registry)

    result = check_port_compatibility(
        _port("first", PortDirection.INPUT),
        _port("second", PortDirection.INPUT),
        data_registry=registry,
    )

    assert not result.direction_ok
    assert not result.compatible
