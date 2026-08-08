from collections.abc import Hashable
from typing import cast

import pytest

from saealib.core.contracts.data import Contained, DataSpec, Fixed, Product, Var
from saealib.core.contracts.ports import (
    PortDirection,
    PortSpec,
    check_port_compatibility,
)
from saealib.core.contracts.schema import (
    SCHEMA_CONFLICT_REASONS,
    SCHEMA_VARIABLES,
    SchemaConstraint,
    Substitution,
    unify_bindings,
    unify_data_specs,
)
from saealib.exceptions import ValidationError


def _contained(values: object) -> Contained:
    return Contained(values=cast(frozenset[Hashable], values))


def test_schema_vocabularies_register_the_initial_names() -> None:
    assert SCHEMA_VARIABLES.names() == (
        "representation",
        "candidate_count",
        "parent_count",
        "objective_schema",
        "constraint_schema",
        "feature_schema",
        "fidelity",
        "species",
        "proposal_group",
    )
    assert SCHEMA_CONFLICT_REASONS.names() == (
        "fixed_mismatch",
        "carrier_mismatch",
        "producer_binding_missing",
        "containment_on_producer",
        "containment_requires_collection",
        "containment_unsatisfied",
        "containment_undecided",
        "product_arity_mismatch",
        "representation_kind_mismatch",
    )


def test_schema_constraint_validates_reason_and_exposes_deferred_state() -> None:
    finding = SchemaConstraint(
        variable="representation",
        reason="containment_undecided",
        detail="producer is a variable",
    )

    assert finding.deferred
    assert "representation" in str(finding)
    assert "containment_undecided" in str(finding)

    with pytest.raises(ValidationError):
        SchemaConstraint(variable="representation", reason="not_registered")


def test_substitution_is_immutable_and_resolves_variable_chains() -> None:
    original = Substitution()
    first = original.bind("A", Var(name="B"))
    second = first.bind("B", Fixed(value="integer"))

    assert original.assignments == {}
    assert first.assignments == {"A": Var(name="B")}
    assert second.resolve(Var(name="A")) == Fixed(value="integer")
    assert second.resolve(
        Product(elements=(Var(name="A"), Fixed(value="real")))
    ) == Product(elements=(Fixed(value="integer"), Fixed(value="real")))


def test_substitution_cycle_is_guarded() -> None:
    substitution = Substitution(assignments={"A": Var(name="B"), "B": Var(name="A")})

    with pytest.raises(ValidationError):
        substitution.resolve(Var(name="A"))


def test_three_node_chain_reuses_substitution_and_names_the_conflict() -> None:
    producer = PortSpec(
        name="out",
        direction=PortDirection.OUTPUT,
        data=DataSpec(
            kind="GenomeBatch",
            bindings={"representation": Fixed(value="real")},
        ),
        cardinality="ONE",
    )
    middle = DataSpec(
        kind="GenomeBatch",
        bindings={"representation": Var(name="R")},
    )
    middle_input = PortSpec(
        name="in",
        direction=PortDirection.INPUT,
        data=middle,
        cardinality="ONE",
    )
    middle_output = PortSpec(
        name="out",
        direction=PortDirection.OUTPUT,
        data=middle,
        cardinality="ONE",
    )
    consumer = PortSpec(
        name="in",
        direction=PortDirection.INPUT,
        data=DataSpec(
            kind="GenomeBatch",
            bindings={"representation": Fixed(value="integer")},
        ),
        cardinality="ONE",
    )

    first = check_port_compatibility(producer, middle_input)
    second = check_port_compatibility(
        middle_output,
        consumer,
        substitution=first.schema.substitution,
    )

    assert first.compatible
    assert not second.compatible
    assert second.schema.conflicts == (
        SchemaConstraint(
            variable="representation",
            reason="fixed_mismatch",
            detail="provided='real', required='integer'",
        ),
    )


def test_containment_against_scalar_fixed_is_a_finding_not_an_exception() -> None:
    result = unify_data_specs(
        DataSpec(
            kind="ObservationBatch",
            bindings={"representation": Fixed(value="real")},
        ),
        DataSpec(
            kind="ObservationBatch",
            bindings={"representation": _contained({"real"})},
        ),
    )

    assert not result.unified
    assert result.conflicts[0].reason == "containment_requires_collection"


def test_unbound_producer_containment_is_deferred() -> None:
    result = unify_bindings(
        {"representation": Var(name="R")},
        {"representation": _contained({"real"})},
    )

    assert len(result.findings) == 1
    assert result.findings[0].reason == "containment_undecided"
    assert result.conflicts == ()
    assert result.deferred == result.findings
    assert not result.unified
    assert result.substitution.assignments == {}
    assert result.substitution.resolve(Var(name="R")) == Var(name="R")


def test_missing_binding_and_unbound_variable_requirement_are_distinct() -> None:
    missing = unify_bindings(
        {},
        {"representation": Fixed(value="real")},
    )
    unbound = unify_bindings(
        {},
        {"representation": Var(name="R")},
    )

    assert missing.conflicts[0].reason == "producer_binding_missing"
    assert unbound.unified


def test_producer_containment_is_rejected() -> None:
    result = unify_bindings(
        {"representation": _contained({"real"})},
        {"representation": Fixed(value="real")},
    )

    assert result.conflicts[0].reason == "containment_on_producer"


def test_product_unification_recurses_and_reports_arity_and_carrier_mismatches() -> (
    None
):
    unified = unify_bindings(
        {"proposal_group": Product(elements=(Fixed(value="algorithm"), Var(name="G")))},
        {
            "proposal_group": Product(
                elements=(Fixed(value="algorithm"), Fixed(value="generation"))
            )
        },
    )
    arity = unify_bindings(
        {"proposal_group": Product(elements=(Fixed(value="algorithm"),))},
        {
            "proposal_group": Product(
                elements=(Fixed(value="algorithm"), Fixed(value="generation"))
            )
        },
    )
    carrier = unify_bindings(
        {"proposal_group": Fixed(value="algorithm")},
        {
            "proposal_group": Product(
                elements=(Fixed(value="algorithm"), Fixed(value="generation"))
            )
        },
    )

    assert unified.unified
    assert unified.substitution.resolve(Var(name="G")) == Fixed(value="generation")
    assert arity.conflicts[0].reason == "product_arity_mismatch"
    assert carrier.conflicts[0].reason == "carrier_mismatch"


def test_containment_checks_set_values_and_reports_missing_values() -> None:
    satisfied = unify_bindings(
        {"objective_schema": Fixed(value=frozenset({"a", "b"}))},
        {"objective_schema": _contained({"a"})},
    )
    unsatisfied = unify_bindings(
        {"objective_schema": Fixed(value=frozenset({"a"}))},
        {"objective_schema": _contained({"a", "b"})},
    )

    assert satisfied.unified
    assert unsatisfied.conflicts[0].reason == "containment_unsatisfied"
    assert "b" in unsatisfied.conflicts[0].detail


def test_unknown_binding_keys_are_reported_in_first_seen_order() -> None:
    result = unify_bindings(
        {
            "future_a": Fixed(value="a"),
            "representation": Var(name="R"),
        },
        {
            "future_b": Fixed(value="b"),
            "future_a": Fixed(value="a"),
        },
    )

    assert result.unknown_variables == ("future_a", "future_b")
    assert not result.unified
