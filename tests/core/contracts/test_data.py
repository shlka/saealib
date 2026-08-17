from collections.abc import Hashable
from typing import cast

import pytest

from saealib.core.contracts.data import (
    DATA_SPEC_KINDS,
    Contained,
    DataSpec,
    DataSpecKind,
    Fixed,
    Product,
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

    assert DATA_SPEC_KINDS.get("future_kind") is None
    assert spec.kind == "future_kind"
    assert spec.bindings == {"representation": Fixed(value="vector")}


def test_fixed_rejects_a_nested_unhashable_value() -> None:
    with pytest.raises(ValidationError):
        Fixed(value=([1, 2],))


def test_contained_normalizes_iterables_and_rejects_scalar_text() -> None:
    assert Contained(
        values=cast(frozenset[Hashable], ["real", "integer"])
    ).values == frozenset({"real", "integer"})

    with pytest.raises(ValidationError):
        Contained(values=cast(frozenset[Hashable], "real"))

    with pytest.raises(ValidationError):
        Contained(values=cast(frozenset[Hashable], [["real"]]))


def test_product_rejects_empty_or_non_binding_elements() -> None:
    with pytest.raises(ValidationError):
        Product(elements=())

    with pytest.raises(ValidationError):
        Product(elements=cast(tuple[Var, ...], ("representation",)))


def test_data_spec_hash_supports_contained_and_product_bindings() -> None:
    spec = DataSpec(
        kind="genomes",
        bindings={
            "representation": Contained(
                values=cast(frozenset[Hashable], {"real", "integer"})
            ),
            "proposal_group": Product(
                elements=(Fixed(value="algorithm"), Var(name="generation"))
            ),
        },
    )

    first_hash = hash(spec)
    second_hash = hash(spec)

    assert isinstance(first_hash, int)
    assert first_hash == second_hash


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
