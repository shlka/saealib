"""Unit H9 contracts for indexed service query forms and hot-loop behavior."""

from __future__ import annotations

import cProfile
import importlib
import pstats
from typing import Any, cast

import numpy as np
import pytest

from saealib.core.contracts.data import Fixed
from saealib.core.contracts.representation import ParameterSpec, RepresentationSpec
from saealib.population import Archive, PopulationAttribute
from saealib.population.genome import DenseVectorBatch, ObjectBatch
from saealib.space import (
    DistanceService,
    EquivalenceService,
    FingerprintService,
    ObjectSpace,
    VectorSpace,
)


def _attrs() -> list[PopulationAttribute]:
    return [
        PopulationAttribute("x", np.float64, (2,), default=np.nan),
        PopulationAttribute("f", np.float64, (1,), default=np.nan),
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


def test_exact_index_uses_numpy_canonicalization_and_no_archive_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact identity uses vectorized bits and a hash index.

    Mutation checks: restoring the element-wise canonicalizer makes the
    profile exceed the call budget; routing exact identity through Archive's
    cKDTree makes the first duplicate add raise.
    """
    archive_module = importlib.import_module("saealib.population.archive")

    def fail_tree(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("exact identity must not construct Archive cKDTree")

    monkeypatch.setattr(archive_module, "cKDTree", fail_tree)
    space = VectorSpace(2, [-10.0, -10.0], [10.0, 10.0], atol=0.0, rtol=0.0)
    fingerprint = cast(FingerprintService, space.services.require("FingerprintService"))
    batch = DenseVectorBatch(np.random.default_rng(4).random((10_000, 10)))
    profile = cProfile.Profile()
    profile.runcall(fingerprint.fingerprint, batch)
    stats = pstats.Stats(profile)
    # One row generator and one ``tobytes`` call per row are intentional; a
    # Python canonicalization helper per element would exceed this budget.
    assert getattr(stats, "total_calls") < len(batch) * 3

    archive = Archive(_attrs(), space=space)
    archive.add(x=np.array([0.0, -0.0]), f=np.array([1.0]))
    archive.add(x=np.array([-0.0, 0.0]), f=np.array([2.0]))
    assert len(archive) == 1


def test_canonical_fingerprint_preserves_zero_and_nan_identity() -> None:
    """The H4 -0.0 and NaN canonicalization properties remain unchanged."""
    space = VectorSpace(2, [-10.0, -10.0], [10.0, 10.0])
    service = cast(FingerprintService, space.services.require("FingerprintService"))
    zeros = service.fingerprint(DenseVectorBatch([[0.0, -0.0], [-0.0, 0.0]]))
    nans = service.fingerprint(DenseVectorBatch([[np.nan, 1.0], [np.nan, 1.0]]))
    assert zeros[0] == zeros[1]
    assert hash(zeros[0]) == hash(zeros[1])
    assert nans[0] == nans[1]
    assert hash(nans[0]) == hash(nans[1])


def test_equivalence_collection_query_is_one_pass_not_three_dimensional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Approximate collection lookup avoids the old Archive row loop.

    Mutation check: broadcasting collection and query batches into an
    (n_query, n_collection, dim) tensor makes the recorded input three-
    dimensional and fails.  The service call also proves Archive has a
    collection-query shape rather than the H7 pairwise bridge.
    """
    vector_module = importlib.import_module("saealib.space.vector")
    original_isclose = vector_module.np.isclose
    shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def record_isclose(left: Any, right: Any, **kwargs: Any) -> np.ndarray:
        shapes.append((np.shape(left), np.shape(right)))
        return original_isclose(left, right, **kwargs)

    monkeypatch.setattr(vector_module.np, "isclose", record_isclose)
    space = VectorSpace(2, [-10.0, -10.0], [10.0, 10.0], atol=1e-3)
    service = cast(EquivalenceService, space.services.require("EquivalenceService"))
    collection = DenseVectorBatch([[1.0, 1.0], [3.0, 3.0]])
    queries = DenseVectorBatch([[1.0001, 1.0001], [3.1, 3.1]])
    matches = service.find_matches(collection, queries)
    assert matches.tolist() == [0, -1]
    assert shapes
    assert all(len(left) <= 2 and len(right) <= 2 for left, right in shapes)
    assert all(left == collection.array.shape for left, _right in shapes)


def test_distance_knn_uses_lazy_service_tree_without_pairwise_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """kNN queries use the DistanceService index and never call pairwise API.

    Mutation check: replacing query_knn with a full pairwise-distance path
    invokes the patched method and fails; the service tree is lazy until the
    first query.
    """
    space = VectorSpace(2, [-10.0, -10.0], [10.0, 10.0])
    service = cast(DistanceService, space.services.require("DistanceService"))
    collection = DenseVectorBatch([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    index = service.create_index(collection)
    assert getattr(index, "tree") is None

    def fail_pairwise(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("kNN must not materialize pairwise distances")

    monkeypatch.setattr(service, "pairwise_distance", fail_pairwise)
    indices, distances = service.query_knn(index, DenseVectorBatch([[0.0, 0.0]]), k=2)
    assert indices.tolist() == [0, 1]
    assert distances[0] == pytest.approx(0.0)
    assert getattr(index, "tree") is not None


def test_append_archive_requires_no_identity_or_distance_service() -> None:
    """Append-only object archives do not resolve any identity service.

    Mutation check: making Archive eagerly require FingerprintService,
    EquivalenceService, or DistanceService for append causes construction to
    fail because ObjectSpace intentionally offers none of them.
    """
    attrs = [
        PopulationAttribute("f", np.float64, (1,), default=np.nan),
        PopulationAttribute("id", np.int64, (), default=-1),
        PopulationAttribute("request_id", np.int64, (), default=-1),
    ]
    archive = Archive(
        attrs,
        genomes=ObjectBatch(),
        duplicate_policy="append",
        space=_object_space(),
    )
    assert archive._identity_mode == "none"
    assert archive._fingerprint_service is None
    assert archive._equivalence_service is None
    assert archive._distance_service is None
    archive.add(
        genome=ObjectBatch(["a"]),
        id=np.int64(1),
        request_id=np.int64(10),
        f=np.array([1.0]),
    )


def test_service_indexes_are_invalidated_after_value_structure_change() -> None:
    """Changed rows cannot be answered by an old fingerprint or kNN handle.

    Mutation check: removing ArchiveMixin's service-index invalidation from
    mod_value leaves the old nearest-neighbor result and fails this test.
    """
    space = VectorSpace(2, [-100.0, -100.0], [100.0, 100.0], atol=0.0, rtol=0.0)
    archive = Archive(_attrs(), space=space)
    archive.add(x=np.array([0.0, 0.0]), f=np.array([0.0]))
    archive.add(x=np.array([1.0, 1.0]), f=np.array([1.0]))
    archive.get_knn(np.array([0.0, 0.0]), k=1)
    assert archive._fingerprint_index is not None
    assert archive._distance_index is not None

    archive.update_rows(np.array([0]), {"x": np.array([[100.0, 100.0]])})
    assert archive._fingerprint_index is None
    assert archive._distance_index is None
    indices, _distances = archive.get_knn(np.array([0.0, 0.0]), k=1)
    assert indices[0] == 1


def test_service_and_legacy_duplicate_results_are_identical() -> None:
    """Exact and approximate service paths preserve legacy Archive results.

    Mutation check: changing either service's first-match position or its
    tolerance relation changes one of the returned insertion-index sequences.
    """
    points = [
        np.array([0.0, 0.0]),
        np.array([1e-12, 0.0]),
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([1.0005, 1.0]),
    ]

    def run(archive: Archive) -> tuple[list[int], int]:
        indices = [
            archive.add(x=point, f=np.array([float(i)]))
            for i, point in enumerate(points)
        ]
        return indices, len(archive)

    legacy_exact = run(Archive(_attrs()))
    service_exact = run(
        Archive(
            _attrs(),
            space=VectorSpace(2, [-10.0, -10.0], [10.0, 10.0], atol=0.0),
        )
    )
    assert service_exact == legacy_exact

    legacy_approx = run(Archive(_attrs(), atol=1e-3))
    service_approx = run(
        Archive(
            _attrs(),
            space=VectorSpace(2, [-10.0, -10.0], [10.0, 10.0], atol=1e-3),
        )
    )
    assert service_approx == legacy_approx
