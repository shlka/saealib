"""Tests for SearchSpace protocol, ServiceRegistry, VectorSpace, and ObjectSpace.

Each test covers specific requirements from Unit H4 and documents the
implementation-side mutation that would cause it to fail.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from saealib.core.contracts.data import Fixed
from saealib.core.contracts.representation import ParameterSpec, RepresentationSpec
from saealib.exceptions import ValidationError
from saealib.population.genome import DenseVectorBatch, ObjectBatch
from saealib.space import (
    DenseNumericView,
    EquivalenceService,
    FingerprintService,
    ObjectSpace,
    SearchSpace,
    ServiceRegistry,
    ValidationResult,
    VectorSpace,
)

# ---------------------------------------------------------------------------
# Protocol runtime / static compatibility check
# ---------------------------------------------------------------------------


def test_spaces_satisfy_search_space_protocol() -> None:
    """VectorSpace and ObjectSpace satisfy SearchSpace protocol at runtime.

    Implementation mutation that would break this test:
        Rename `sample` or `validate` in VectorSpace or ObjectSpace.
        `isinstance(space, SearchSpace)` would fail.
    """
    vec_space = VectorSpace(dim=2, lb=[-1.0, -1.0], ub=[1.0, 1.0])
    rep = RepresentationSpec(
        kind="sequence",
        parameters=(
            ParameterSpec(name="alphabet", value=Fixed(value=frozenset({"a"}))),
            ParameterSpec(name="min_length", value=Fixed(value=1)),
            ParameterSpec(name="max_length", value=Fixed(value=10)),
        ),
    )
    obj_space = ObjectSpace(representation=rep)

    assert isinstance(vec_space, SearchSpace)
    assert isinstance(obj_space, SearchSpace)


# ---------------------------------------------------------------------------
# 1. Requiring an unoffered service from ServiceRegistry raises ValidationError
# ---------------------------------------------------------------------------


def test_service_registry_require_unoffered_service_raises() -> None:
    """Requiring missing service from ServiceRegistry raises with name.

    Implementation mutation that would break this test:
        In `ServiceRegistry.require`, remove `name` from the exception message or
        return `None` instead of raising. `pytest.raises(ValidationError)` or
        `match="GenomeCodec"` would fail.
    """
    registry = ServiceRegistry()

    with pytest.raises(ValidationError, match="GenomeCodec"):
        registry.require("GenomeCodec")


# ---------------------------------------------------------------------------
# 2. ObjectSpace without services can sample and validate
# ---------------------------------------------------------------------------


def test_object_space_without_services_can_sample_and_validate() -> None:
    """ObjectSpace with empty ServiceRegistry can still sample and validate.

    Implementation mutation that would break this test:
        In `ObjectSpace.sample` or `ObjectSpace.validate`, call
        `self.services.require("SamplingService")` or `require("ValidationService")`.
        Since ObjectSpace does not register services, `sample` or `validate` would fail.
    """
    rep = RepresentationSpec(
        kind="sequence",
        parameters=(
            ParameterSpec(name="alphabet", value=Fixed(value=frozenset({"a"}))),
            ParameterSpec(name="min_length", value=Fixed(value=1)),
            ParameterSpec(name="max_length", value=Fixed(value=10)),
        ),
    )
    space = ObjectSpace(representation=rep)

    assert len(space.services.names()) == 0

    batch = space.sample(5)
    assert isinstance(batch, ObjectBatch)
    assert len(batch) == 5

    res = space.validate(batch)
    assert isinstance(res, ValidationResult)
    assert res.valid
    assert len(res.valid_mask) == 5


# ---------------------------------------------------------------------------
# 3. VectorSpace.validate identifies out-of-bounds genomes as invalid
# ---------------------------------------------------------------------------


def test_vector_space_validate_identifies_out_of_bounds() -> None:
    """VectorSpace.validate marks genomes outside [lb, ub] as valid=False.

    Implementation mutation that would break this test:
        In `VectorSpace.validate`, replace `in_bounds_mask` with `True`.
        Out-of-bounds genomes would be marked as `valid=True`, failing assertions.
    """
    space = VectorSpace(dim=2, lb=[-1.0, -1.0], ub=[1.0, 1.0])

    in_bounds = DenseVectorBatch([[0.0, 0.5], [-0.5, 1.0]])
    res_ok = space.validate(in_bounds)
    assert res_ok.valid
    assert res_ok.valid_mask == (True, True)

    out_of_bounds = DenseVectorBatch([[0.0, 2.0], [0.0, 0.0]])
    res_bad = space.validate(out_of_bounds)
    assert not res_bad.valid
    assert res_bad.valid_mask == (False, True)
    assert len(res_bad.errors) > 0


# ---------------------------------------------------------------------------
# 4. DenseNumericView is zero-copy (shares memory)
# ---------------------------------------------------------------------------


def test_dense_numeric_view_is_zero_copy() -> None:
    """DenseNumericView.get_view returns an array sharing memory with batch.array.

    Implementation mutation that would break this test:
        In `_VectorDenseNumericView.get_view`, return `genomes.array.copy()`.
        `np.shares_memory(view, batch.array)` would fail (return False).
    """
    space = VectorSpace(dim=2, lb=[-1.0, -1.0], ub=[1.0, 1.0])
    batch = DenseVectorBatch([[0.5, -0.5], [0.1, 0.2]])

    service = space.services.require("DenseNumericView")
    assert isinstance(service, DenseNumericView)

    view = service.get_view(batch)
    assert isinstance(view, np.ndarray)
    assert np.shares_memory(view, batch.array)


# ---------------------------------------------------------------------------
# 5. FingerprintService and EquivalenceService return distinct answers for near points
# ---------------------------------------------------------------------------


def test_fingerprint_and_equivalence_services_differ() -> None:
    """FingerprintService and EquivalenceService yield different outputs.

    Implementation mutation that would break this test:
        In `_VectorFingerprintService.fingerprint`, round values
        with `np.round(val, 4)`. `fp[0] != fp[1]` would fail for
        `[1.0, 2.0]` and `[1.0 + 1e-9, 2.0]`.
    """
    # Use custom atol=1e-5 to test distinct behavior for 1e-9 difference
    space = VectorSpace(dim=2, lb=[-10.0, -10.0], ub=[10.0, 10.0], atol=1e-5)

    p1 = [1.0, 2.0]
    p2 = [1.0 + 1e-9, 2.0]  # Very close to p1, within atol=1e-5
    batch = DenseVectorBatch([p1, p2])

    fp_service = space.services.require("FingerprintService")
    assert isinstance(fp_service, FingerprintService)
    fp = fp_service.fingerprint(batch)
    assert len(fp) == 2
    # Exact fingerprints are distinct
    assert fp[0] != fp[1]

    eq_service = space.services.require("EquivalenceService")
    assert isinstance(eq_service, EquivalenceService)
    dups = eq_service.find_duplicates(batch)
    # Approximate equivalence marks p2 as duplicate of p1
    assert dups[0] is np.bool_(False)
    assert dups[1] is np.bool_(True)


# ---------------------------------------------------------------------------
# 6. Validating mismatched GenomeBatch type raises ValidationError
# ---------------------------------------------------------------------------


def test_validate_mismatched_genome_batch_raises() -> None:
    """Passing mismatched GenomeBatch to space.validate raises ValidationError.

    Implementation mutation that would break this test:
        In `VectorSpace.validate` or `ObjectSpace.validate`, remove the
        `isinstance(genomes, ExpectedBatch)` type check. Passing ObjectBatch to
        VectorSpace would raise AttributeError or return invalid result.
    """
    vec_space = VectorSpace(dim=2, lb=[-1.0, -1.0], ub=[1.0, 1.0])
    obj_batch = ObjectBatch(["item1", "item2"])

    with pytest.raises(ValidationError, match="DenseVectorBatch"):
        vec_space.validate(obj_batch)

    rep = RepresentationSpec(
        kind="sequence",
        parameters=(
            ParameterSpec(name="alphabet", value=Fixed(value=frozenset({"a"}))),
            ParameterSpec(name="min_length", value=Fixed(value=1)),
            ParameterSpec(name="max_length", value=Fixed(value=10)),
        ),
    )
    obj_space = ObjectSpace(representation=rep)
    dense_batch = DenseVectorBatch([[0.0, 0.0]])

    with pytest.raises(ValidationError, match="ObjectBatch"):
        obj_space.validate(dense_batch)


# ---------------------------------------------------------------------------
# R1 Tests: Tolerance parameterization and default values
# ---------------------------------------------------------------------------


def test_vector_space_tolerance_defaults_and_customization() -> None:
    """VectorSpace defaults atol=1e-16, rtol=0.0 and allows custom tolerances.

    Implementation mutation that would break this test:
        Change default atol in VectorSpace.__init__ to 1e-5.
        `space.atol == 1e-16` would fail.
    """
    default_space = VectorSpace(dim=2, lb=[-1.0, -1.0], ub=[1.0, 1.0])
    assert default_space.atol == 1e-16
    assert default_space.rtol == 0.0

    # With default atol=1e-16, points differing by 1e-9 are NOT equivalent
    b = DenseVectorBatch([[1.0, 2.0], [1.0 + 1e-9, 2.0]])
    eq_default = default_space.services.require("EquivalenceService")
    assert isinstance(eq_default, EquivalenceService)
    assert not eq_default.find_duplicates(b)[1]

    # With custom atol=1e-5, points differing by 1e-9 ARE equivalent
    custom_space = VectorSpace(dim=2, lb=[-1.0, -1.0], ub=[1.0, 1.0], atol=1e-5)
    assert custom_space.atol == 1e-5
    eq_custom = custom_space.services.require("EquivalenceService")
    assert isinstance(eq_custom, EquivalenceService)
    assert eq_custom.find_duplicates(b)[1]


# ---------------------------------------------------------------------------
# R2 Tests: ValidationResult valid property behavior
# ---------------------------------------------------------------------------


def test_validation_result_valid_property_logic() -> None:
    """ValidationResult.valid is derived dynamically from valid_mask and errors.

    Implementation mutation that would break this test:
        Change `ValidationResult.valid` property to ignore `self.errors`.
        `res_err.valid` would return `True` for `errors=("type error",)`.
    """
    res1 = ValidationResult(valid_mask=(True, False))
    assert not res1.valid

    res2 = ValidationResult(valid_mask=(True, True))
    assert res2.valid

    res_empty = ValidationResult(valid_mask=(), errors=())
    assert res_empty.valid

    res_err = ValidationResult(valid_mask=(), errors=("Batch-level type error",))
    assert not res_err.valid


# ---------------------------------------------------------------------------
# R3 Tests: Fingerprint canonicalization for -0.0 and NaN
# ---------------------------------------------------------------------------


def test_fingerprint_canonicalization_minus_zero_and_nan() -> None:
    """FingerprintService normalizes -0.0 == 0.0 and NaN == NaN hashability.

    Implementation mutation that would break this test:
        In `_canonical_float`, remove `math.isnan(fval)` check.
        `fp_nan1 == fp_nan2` would fail because float('nan') != float('nan').
    """
    space = VectorSpace(dim=2, lb=[-10.0, -10.0], ub=[10.0, 10.0])
    fp_service = space.services.require("FingerprintService")
    assert isinstance(fp_service, FingerprintService)

    # -0.0 vs +0.0
    b_zero = DenseVectorBatch([[0.0, -0.0], [-0.0, 0.0]])
    fp_zero = fp_service.fingerprint(b_zero)
    assert fp_zero[0] == fp_zero[1]
    assert hash(fp_zero[0]) == hash(fp_zero[1])

    # NaN canonicalization
    b_nan1 = DenseVectorBatch([[math.nan, 1.0]])
    b_nan2 = DenseVectorBatch([[math.nan, 1.0]])
    fp_nan1 = fp_service.fingerprint(b_nan1)
    fp_nan2 = fp_service.fingerprint(b_nan2)

    assert fp_nan1[0] == fp_nan2[0]
    assert hash(fp_nan1[0]) == hash(fp_nan2[0])
