"""Contract tests for Archive identity services."""

from __future__ import annotations

import importlib
from typing import cast

import numpy as np
import pytest

from saealib.core.contracts.data import Fixed
from saealib.core.contracts.representation import ParameterSpec, RepresentationSpec
from saealib.exceptions import ValidationError
from saealib.population import Archive, PopulationAttribute
from saealib.population.genome import DenseVectorBatch, GenomeBatch, ObjectBatch
from saealib.space import EquivalenceService, FingerprintService, ObjectSpace


class _ExactService:
    """Small exact service used to prove the Archive calls the service."""

    def __init__(self) -> None:
        self.calls = 0

    def fingerprint(self, genomes: GenomeBatch) -> tuple[tuple[float, ...], ...]:
        self.calls += 1
        values = cast(DenseVectorBatch, genomes).array
        return tuple(tuple(float(value) for value in row) for row in values)


class _ApproximateService:
    """Batch duplicate service with an explicit tolerance."""

    def __init__(self, atol: float) -> None:
        self._atol = atol
        self._rtol = 0.0
        self.calls = 0

    def find_duplicates(self, genomes: GenomeBatch) -> np.ndarray:
        self.calls += 1
        values = cast(DenseVectorBatch, genomes).array
        duplicate = np.zeros(len(values), dtype=bool)
        for row in range(len(values)):
            duplicate[row] = any(
                np.all(
                    np.isclose(values[row], values[previous], atol=self._atol, rtol=0.0)
                )
                for previous in range(row)
            )
        return duplicate


def _vector_attrs() -> list[PopulationAttribute]:
    return [
        PopulationAttribute("x", np.float64, (2,), default=np.nan),
        PopulationAttribute("f", np.float64, (1,), default=np.nan),
        PopulationAttribute("id", np.int64, (), default=-1),
    ]


def _append_attrs() -> list[PopulationAttribute]:
    return [
        PopulationAttribute("f", np.float64, (1,), default=np.nan),
        PopulationAttribute("id", np.int64, (), default=-1),
        PopulationAttribute("request_id", np.int64, (), default=-1),
    ]


def _object_space() -> ObjectSpace:
    return ObjectSpace(
        RepresentationSpec(
            kind="sequence",
            parameters=(
                ParameterSpec(name="alphabet", value=Fixed(value=frozenset({"a"}))),
                ParameterSpec(name="min_length", value=Fixed(value=1)),
                ParameterSpec(name="max_length", value=Fixed(value=10)),
            ),
        )
    )


def test_exact_identity_uses_fingerprint_without_constructing_kdtree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("saealib.population.archive")

    def fail_kdtree(*args, **kwargs):
        raise AssertionError("exact identity must not construct a KD-tree")

    monkeypatch.setattr(module, "cKDTree", fail_kdtree)
    service = _ExactService()
    archive = Archive(
        _vector_attrs(),
        atol=0.0,
        rtol=0.0,
        fingerprint_service=cast(FingerprintService, service),
    )

    archive.add(x=np.array([1.0, 2.0]), f=np.array([1.0]), id=np.int64(1))
    archive.add(x=np.array([1.0, 2.0]), f=np.array([2.0]), id=np.int64(2))

    assert archive._identity_mode == "fingerprint"
    assert len(archive) == 1
    assert service.calls >= 2


def test_equivalence_service_handles_tolerance_matching() -> None:
    """Non-zero tolerances select EquivalenceService and preserve first match."""
    service = _ApproximateService(atol=1e-5)
    archive = Archive(
        _vector_attrs(),
        atol=1e-5,
        rtol=0.0,
        equivalence_service=cast(EquivalenceService, service),
    )

    archive.add(x=np.array([1.0, 2.0]), f=np.array([1.0]), id=np.int64(1))
    archive.add(x=np.array([1.0 + 1e-6, 2.0]), f=np.array([2.0]), id=np.int64(2))

    assert archive._identity_mode == "equivalence"
    assert len(archive) == 1
    assert service.calls > 0


def test_default_legacy_archive_keeps_exact_pre_service_behavior() -> None:
    """The compatibility constructor still defaults to exact matching."""
    archive = Archive(_vector_attrs())
    archive.add(x=np.array([0.0, 0.0]), f=np.array([1.0]), id=np.int64(1))
    archive.add(x=np.array([1e-12, 0.0]), f=np.array([2.0]), id=np.int64(2))

    assert archive.atol == 0.0
    assert archive.rtol == 0.0
    assert len(archive) == 2


def test_object_space_append_archive_needs_no_identity_service() -> None:
    """An opaque, service-less archive can retain observations append-only."""
    archive = Archive(
        _append_attrs(),
        genomes=ObjectBatch(),
        duplicate_policy="append",
        space=_object_space(),
    )

    archive.add(
        genome=ObjectBatch([{"candidate": "a"}]),
        id=np.int64(1),
        request_id=np.int64(10),
        f=np.array([1.0]),
    )
    archive.add(
        genome=ObjectBatch([{"candidate": "a"}]),
        id=np.int64(1),
        request_id=np.int64(11),
        f=np.array([2.0]),
    )

    assert len(archive) == 2
    assert archive._identity_mode == "none"


def test_service_less_keep_first_names_missing_fingerprint_service() -> None:
    """Opaque deduplicating archives fail with the required service name."""
    with pytest.raises(ValidationError, match="FingerprintService"):
        Archive(
            _append_attrs(),
            genomes=ObjectBatch(),
            duplicate_policy="keep_first",
            space=_object_space(),
        )


@pytest.mark.parametrize("policy", ["keep_first", "replace", "append"])
def test_duplicate_policies_remain_construction_policies(policy: str) -> None:
    """All three policies keep their pre-service row behavior."""
    attrs = [*_vector_attrs(), PopulationAttribute("request_id", np.int64, (), -1)]
    archive = Archive(attrs, duplicate_policy=policy)

    archive.add(
        x=np.array([0.0, 0.0]),
        f=np.array([1.0]),
        id=np.int64(1),
        request_id=np.int64(10),
    )
    archive.add(
        x=np.array([0.0, 0.0]),
        f=np.array([2.0]),
        id=np.int64(1),
        request_id=np.int64(11),
    )

    if policy == "keep_first":
        assert len(archive) == 1
        np.testing.assert_array_equal(archive.get_array("f"), [[1.0]])
    elif policy == "replace":
        assert len(archive) == 1
        np.testing.assert_array_equal(archive.get_array("f"), [[2.0]])
    else:
        assert len(archive) == 2
