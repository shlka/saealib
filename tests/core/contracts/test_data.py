import pytest

from saealib.core.contracts.data import (
    DATA_SPEC_KINDS,
    DataSpec,
    DataSpecKind,
    Fixed,
    Var,
    is_data_spec_compatible,
    register_data_spec,
)
from saealib.core.contracts.vocabulary import Vocabulary
from saealib.exceptions import ValidationError


def test_data_spec_equality_is_nominal_and_binding_based() -> None:
    bindings = {"representation": Var(name="R")}

    assert DataSpec(kind="genomes", bindings=bindings) != DataSpec(
        kind="objectives", bindings=bindings
    )
    assert DataSpec(kind="genomes", bindings=bindings) == DataSpec(
        kind="genomes", bindings=bindings
    )
    assert hash(DataSpec(kind="genomes", bindings=bindings))


def test_unregistered_data_spec_kind_is_constructible() -> None:
    spec = DataSpec(
        kind="future_kind",
        bindings={"representation": Fixed(value="vector")},
    )

    assert DATA_SPEC_KINDS.names() == ()
    assert spec.kind == "future_kind"
    assert spec.bindings == {"representation": Fixed(value="vector")}


def test_fixed_rejects_a_nested_unhashable_value() -> None:
    with pytest.raises(ValidationError):
        Fixed(value=([1, 2],))


def test_explicit_subtype_relation_is_directional() -> None:
    registry: Vocabulary[DataSpecKind] = Vocabulary()
    register_data_spec(
        "test:base",
        description="Base kind",
        registry=registry,
    )
    register_data_spec(
        "test:special",
        description="Specialized kind",
        supertypes=("test:base",),
        registry=registry,
    )

    producer = DataSpec(kind="test:special")
    consumer = DataSpec(kind="test:base")

    assert is_data_spec_compatible(producer, consumer, registry=registry)
    assert not is_data_spec_compatible(consumer, producer, registry=registry)
