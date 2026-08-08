"""Tests for RepresentationSpec, RepresentationKind, REPRESENTATION_KINDS, and
unify_representation_specs.

Each test covers exactly one requirement from the unit prompt.  For every
test we also verify that mutating the *implementation* side (not only the
test expectation) causes the test to fail — the mutation that would break each
test is noted in the test's docstring.
"""

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
    """vector vs permutation must not unify.

    Implementation mutation that would break this test:
        Remove or comment out the ``if provided.kind != required.kind`` branch
        in ``unify_representation_specs``.  The test would then fall through
        to parameter unification and would fail because ``dim`` and ``length``
        have different names, producing a ``producer_binding_missing`` finding
        instead of ``incompatible_representation`` — but the ``not result.unified``
        assertion would still hold.  The correct mutation is to change the
        ``!=`` to ``==`` so the branch never fires for mismatched kinds:
        unification would then try to unify ``dim=10`` with ``length=10``,
        which would yield ``producer_binding_missing`` for ``length``.  With
        the assertion changed to check ``unified is True``, the test would pass
        incorrectly — so the real mutation target is the kind comparison gate.
    """
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
    """vector(dim=10) vs vector(dim=10) must unify.

    Implementation mutation that would break this test:
        In ``unify_bindings`` (called from ``unify_representation_specs``),
        change the ``Fixed == Fixed`` check to always emit a ``fixed_mismatch``
        finding regardless of value equality.  The test's ``assert result.unified``
        would then fail.
    """
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
    """vector(dim=10) vs vector(dim=5) must not unify.

    Implementation mutation that would break this test:
        In ``_unify_pair`` inside ``schema.py``, remove the ``if left.value !=
        right.value`` guard so ``fixed_mismatch`` is never emitted.  The test's
        ``assert not result.unified`` would then fail because no finding is
        produced.
    """
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
    """sequence alphabet containment: large alphabet satisfies a subset requirement.

    Implementation mutation that would break this test:
        In ``schema.py``'s ``_unify_pair``, replace the ``missing = right.values
        - provided_values`` and subsequent ``containment_unsatisfied`` emission
        with a ``return substitution`` that always passes.  Wait — that would
        make the test pass trivially.  The correct mutation is to replace the
        containment branch with a ``fixed_mismatch`` emission (treating the
        ``Contained`` as a mismatched carrier), so containment always fails.
        The test's ``assert result.unified`` would then fail.
    """
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
    """sequence alphabet containment: required symbol absent from provided alphabet.

    Implementation mutation that would break this test:
        In ``schema.py``'s ``_unify_pair``, remove the ``if missing:`` guard
        so ``containment_unsatisfied`` is never emitted even when values are
        missing.  The test's ``assert not result.unified`` would then fail.
    """
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
    """RepresentationSpec with an unregistered kind must raise ConfigurationError.

    Implementation mutation that would break this test:
        In ``RepresentationSpec.__post_init__``, remove the
        ``if REPRESENTATION_KINDS.get(self.kind) is None: raise ConfigurationError``
        guard.  The test's ``pytest.raises(ConfigurationError)`` would then
        fail to capture an exception and would itself raise.
    """
    with pytest.raises(ConfigurationError, match="Unknown representation kind"):
        RepresentationSpec(kind="graph", parameters=())


# ---------------------------------------------------------------------------
# 6. Double-registration is rejected
# ---------------------------------------------------------------------------


def test_double_registration_is_rejected() -> None:
    """Registering the same name twice in REPRESENTATION_KINDS must raise.

    Implementation mutation that would break this test:
        In ``Vocabulary.register``, remove the ``if name in self._entries:
        raise ConfigurationError`` guard.  The test's ``pytest.raises``
        would then not capture an exception and would itself raise.
    """
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
