"""Tests for representation specifications and unification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from saealib.core.contracts.data import Contained, Fixed
from saealib.core.contracts.representation import (
    REPRESENTATION_KINDS,
    ParameterSpec,
    RepresentationKind,
    RepresentationSpec,
    unify_representation_specs,
)
from saealib.exceptions import ConfigurationError, ValidationError

if TYPE_CHECKING:
    from saealib.core.contracts.vocabulary import Vocabulary

# ---------------------------------------------------------------------------
# 1. Different kind names do NOT unify
# ---------------------------------------------------------------------------


def test_different_kinds_do_not_unify() -> None:
    vec = RepresentationSpec(
        kind="vector",
        parameters=(ParameterSpec(name="dim", value=Fixed(value=10)),),
    )
    perm = RepresentationSpec(
        kind="permutation",
        parameters=(ParameterSpec(name="length", value=Fixed(value=10)),),
    )
    result = unify_representation_specs(vec, perm)
    assert not result.unified
    # The finding must name the right reason and mention both kind names.
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.reason == "representation_kind_mismatch"
    assert "vector" in finding.detail
    assert "permutation" in finding.detail


# ---------------------------------------------------------------------------
# 2. Same kind, equal parameters → unify
# ---------------------------------------------------------------------------


def test_same_kind_equal_parameters_unify() -> None:
    spec_a = RepresentationSpec(
        kind="vector",
        parameters=(ParameterSpec(name="dim", value=Fixed(value=10)),),
    )
    spec_b = RepresentationSpec(
        kind="vector",
        parameters=(ParameterSpec(name="dim", value=Fixed(value=10)),),
    )
    result = unify_representation_specs(spec_a, spec_b)
    assert result.unified


# ---------------------------------------------------------------------------
# 3. Same kind, unequal parameters → do NOT unify (default equality)
# ---------------------------------------------------------------------------


def test_same_kind_unequal_parameters_do_not_unify() -> None:
    provided = RepresentationSpec(
        kind="vector",
        parameters=(ParameterSpec(name="dim", value=Fixed(value=10)),),
    )
    required = RepresentationSpec(
        kind="vector",
        parameters=(ParameterSpec(name="dim", value=Fixed(value=5)),),
    )
    result = unify_representation_specs(provided, required)
    assert not result.unified
    assert any(f.reason == "fixed_mismatch" for f in result.findings)


# ---------------------------------------------------------------------------
# 4a. Custom unify (sequence, containment-ok) → unify
# ---------------------------------------------------------------------------


def test_sequence_containment_ok_unifies() -> None:
    provided = RepresentationSpec(
        kind="sequence",
        parameters=(
            ParameterSpec(
                name="alphabet", value=Fixed(value=frozenset({"a", "b", "c", "d"}))
            ),
            ParameterSpec(name="min_length", value=Fixed(value=1)),
            ParameterSpec(name="max_length", value=Fixed(value=100)),
        ),
    )
    required = RepresentationSpec(
        kind="sequence",
        parameters=(
            ParameterSpec(
                name="alphabet", value=Contained(values=frozenset({"a", "b"}))
            ),
            ParameterSpec(name="min_length", value=Fixed(value=1)),
            ParameterSpec(name="max_length", value=Fixed(value=100)),
        ),
    )
    result = unify_representation_specs(provided, required)
    assert result.unified


# ---------------------------------------------------------------------------
# 4b. Custom unify (sequence, containment-fail) → do NOT unify
# ---------------------------------------------------------------------------


def test_sequence_containment_fail_does_not_unify() -> None:
    provided = RepresentationSpec(
        kind="sequence",
        parameters=(
            ParameterSpec(name="alphabet", value=Fixed(value=frozenset({"a", "b"}))),
            ParameterSpec(name="min_length", value=Fixed(value=1)),
            ParameterSpec(name="max_length", value=Fixed(value=100)),
        ),
    )
    required = RepresentationSpec(
        kind="sequence",
        parameters=(
            ParameterSpec(
                name="alphabet", value=Contained(values=frozenset({"a", "z"}))
            ),
            ParameterSpec(name="min_length", value=Fixed(value=1)),
            ParameterSpec(name="max_length", value=Fixed(value=100)),
        ),
    )
    result = unify_representation_specs(provided, required)
    assert not result.unified
    assert any(f.reason == "containment_unsatisfied" for f in result.findings)


# ---------------------------------------------------------------------------
# 5. Unregistered kind name is rejected
# ---------------------------------------------------------------------------


def test_unregistered_kind_name_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="Unknown representation kind"):
        RepresentationSpec(kind="graph", parameters=())


# ---------------------------------------------------------------------------
# 6. Double-registration is rejected
# ---------------------------------------------------------------------------


def test_double_registration_is_rejected() -> None:
    temp: Vocabulary[RepresentationKind] = type(REPRESENTATION_KINDS)(
        name_validator=__import__(
            "saealib.core.contracts.vocabulary", fromlist=["validate_identifier"]
        ).validate_identifier
    )
    kind = RepresentationKind(
        name="custom_test_kind",
        description="A temporary kind for testing double-registration.",
    )
    temp.register("custom_test_kind", kind)
    with pytest.raises(ConfigurationError):
        temp.register("custom_test_kind", kind)


# ---------------------------------------------------------------------------
# Supplementary: ParameterSpec validates name and value carrier
# ---------------------------------------------------------------------------


def test_parameter_spec_validates_name() -> None:
    """ParameterSpec name must be a plain identifier."""
    with pytest.raises(ValidationError):
        ParameterSpec(name="not:valid", value=Fixed(value=10))


def test_parameter_spec_validates_value_carrier() -> None:
    """ParameterSpec value must be a SchemaBinding carrier."""
    with pytest.raises(ValidationError):
        ParameterSpec(name="dim", value=cast(Any, 42))


# ---------------------------------------------------------------------------
# Supplementary: RepresentationKind validates parameters
# ---------------------------------------------------------------------------


def test_representation_kind_rejects_duplicate_parameter_names() -> None:
    """RepresentationKind with duplicate parameter names must raise."""
    with pytest.raises(ValidationError, match="Duplicate parameter name"):
        RepresentationKind(
            name="bad_kind",
            description="duplicate param names",
            parameters=(
                ParameterSpec(name="x", value=Fixed(value=1)),
                ParameterSpec(name="x", value=Fixed(value=2)),
            ),
        )


# ---------------------------------------------------------------------------
# Supplementary: incompatible_representation is registered in DIAGNOSTIC_CODES
# ---------------------------------------------------------------------------


def test_incompatible_representation_is_registered_in_diagnostic_codes() -> None:
    """incompatible_representation must appear in DIAGNOSTIC_CODES."""
    from saealib.core.compiler.diagnostics import DIAGNOSTIC_CODES

    assert DIAGNOSTIC_CODES.get("incompatible_representation") is not None


# ---------------------------------------------------------------------------
# Supplementary: initial REPRESENTATION_KINDS registrations
# ---------------------------------------------------------------------------


def test_initial_representation_kinds() -> None:
    """REPRESENTATION_KINDS must contain exactly vector, permutation, sequence."""
    assert REPRESENTATION_KINDS.names() == ("vector", "permutation", "sequence")


# ---------------------------------------------------------------------------
# Supplementary: _kind_param_registry prevents unknown_variables in result
# ---------------------------------------------------------------------------


def test_param_names_are_not_flagged_as_unknown_variables() -> None:
    """Parameter names local to a kind must not appear in unknown_variables."""
    vec_a = RepresentationSpec(
        kind="vector",
        parameters=(ParameterSpec(name="dim", value=Fixed(value=7)),),
    )
    vec_b = RepresentationSpec(
        kind="vector",
        parameters=(ParameterSpec(name="dim", value=Fixed(value=7)),),
    )
    result = unify_representation_specs(vec_a, vec_b)
    assert result.unknown_variables == ()
    assert result.unified
