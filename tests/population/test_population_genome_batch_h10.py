"""Regression tests for Unit H10 genome-batch cache reuse."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from saealib.exceptions import ValidationError
from saealib.population import Population, PopulationAttribute
from saealib.population.genome import DenseVectorBatch, GenomeBatch, ObjectBatch
from saealib.space.services import DenseNumericView


def _dense_population(capacity: int = 2) -> Population:
    return Population(
        [
            PopulationAttribute("x", np.float64, shape=(2,)),
            PopulationAttribute("f", np.float64),
        ],
        init_capacity=capacity,
    )


def _assert_dense_backing(pop: Population, expected: list[list[float]]) -> None:
    assert isinstance(pop._genome_batch, DenseVectorBatch)
    assert pop._genome_batch.array.base is pop._data["x"]
    expected_array = np.asarray(expected, dtype=np.float64)
    if not expected:
        expected_array = np.empty((0, pop._data["x"].shape[1]), dtype=np.float64)
    genomes = pop.genomes
    assert isinstance(genomes, DenseVectorBatch)
    np.testing.assert_array_equal(genomes.array, expected_array)


def _object_population(capacity: int = 2) -> Population:
    return Population(
        [PopulationAttribute("f", np.float64)],
        init_capacity=capacity,
        genomes=ObjectBatch(["a", "b"]),
    )


def _assert_object_state(pop: Population, expected: tuple[object, ...]) -> None:
    assert isinstance(pop._genome_batch, ObjectBatch)
    assert pop._genome_batch.items == expected
    genomes = pop.genomes
    assert isinstance(genomes, ObjectBatch)
    assert genomes.items == expected


class _CountingDenseView:
    def __init__(self) -> None:
        self.calls = 0

    def get_view(self, genomes: GenomeBatch) -> np.ndarray:
        self.calls += 1
        return cast(DenseVectorBatch, genomes).array


class _CanonicalCountingDenseView(_CountingDenseView):
    _canonical_identity_backing = True


class _CopyingDenseView(_CountingDenseView):
    def get_view(self, genomes: GenomeBatch) -> np.ndarray:
        self.calls += 1
        return cast(DenseVectorBatch, genomes).array.copy()


class _GetterCountingPopulation(Population):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.get_array_calls = 0
        super().__init__(*args, **kwargs)

    def get_array(self, key: str) -> np.ndarray:
        self.get_array_calls += 1
        return super().get_array(key)


class _CountingArray(np.ndarray):
    def __new__(cls, value: np.ndarray) -> _CountingArray:
        result = np.asarray(value).view(cls)
        result.writes = 0
        return result

    def __array_finalize__(self, parent: np.ndarray | None) -> None:
        if parent is not None:
            self.writes = getattr(parent, "writes", 0)

    def __setitem__(self, key: Any, value: Any) -> None:
        self.writes += 1
        super().__setitem__(key, value)


class _CustomBatch:
    def __init__(self, items: tuple[object, ...] = ()) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def take(self, indices: np.ndarray) -> _CustomBatch:
        return type(self)(tuple(self.items[int(index)] for index in indices))

    @classmethod
    def concat(cls, batches: tuple[_CustomBatch, ...]) -> _CustomBatch:
        return cls(tuple(item for batch in batches for item in batch.items))


def test_dense_population_extend_uses_canonical_genome_append_once() -> None:
    """Dense extension does not copy the x column before appending genomes."""
    attrs = [
        PopulationAttribute("x", np.float64, shape=(2,)),
        PopulationAttribute("f", np.float64),
        PopulationAttribute("g", np.float64, shape=(1,)),
        PopulationAttribute("id", np.int64, default=-1),
    ]
    target = Population(attrs, init_capacity=4)
    source = Population(attrs, init_capacity=2)
    target._extend_internal(
        {
            "x": np.array([[1.0, 2.0]]),
            "f": np.array([10.0]),
            "g": np.array([[0.1]]),
            "id": np.array([101]),
        },
        preserve_ids=True,
    )
    source._extend_internal(
        {
            "x": np.array([[3.0, 4.0], [5.0, 6.0]]),
            "f": np.array([20.0, 30.0]),
            "g": np.array([[0.2], [0.3]]),
            "id": np.array([102, 103]),
        },
        preserve_ids=True,
    )

    shared_view = _CanonicalCountingDenseView()
    source_view = shared_view
    target_view = shared_view
    source._dense_numeric_view = cast(DenseNumericView, source_view)
    target._dense_numeric_view = cast(DenseNumericView, target_view)
    target_x = _CountingArray(target._data["x"])
    target._data["x"] = target_x

    target._extend_internal(source, preserve_ids=True)

    assert shared_view.calls == 0
    assert target_x.writes == 1
    assert len(target) == 3
    np.testing.assert_array_equal(target.x, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_array_equal(target.f, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(target.g, [[0.1], [0.2], [0.3]])
    np.testing.assert_array_equal(target.candidate_ids, [101, 102, 103])

    object_target = _object_population(capacity=2)
    object_source = Population(
        [PopulationAttribute("f", np.float64)],
        init_capacity=1,
        genomes=ObjectBatch(["d"]),
    )
    object_target.extend(object_source)
    _assert_object_state(object_target, ("a", "b", "d"))

    dict_target = _dense_population(capacity=1)
    dict_target.extend(
        {
            "x": np.array([[7.0, 8.0]]),
            "f": np.array([40.0]),
        }
    )
    np.testing.assert_array_equal(dict_target.x, [[7.0, 8.0]])
    np.testing.assert_array_equal(dict_target.f, [40.0])


def test_dense_population_extend_falls_back_for_custom_getter() -> None:
    attrs = [
        PopulationAttribute("x", np.float64, shape=(2,)),
        PopulationAttribute("f", np.float64),
    ]
    source = _GetterCountingPopulation(attrs, init_capacity=1)
    source.extend({"x": np.array([[1.0, 2.0]]), "f": np.array([3.0])})
    target = Population(attrs, init_capacity=1)
    target._extend_internal(source, preserve_ids=True)
    assert source.get_array_calls == 2
    np.testing.assert_array_equal(target.x, [[1.0, 2.0]])
    np.testing.assert_array_equal(target.f, [3.0])


def test_dense_population_extend_uses_backing_columns_with_distinct_services() -> None:
    attrs = [
        PopulationAttribute("x", np.float64, shape=(2,)),
        PopulationAttribute("f", np.float64),
    ]
    source = Population(attrs, init_capacity=1)
    target = Population(attrs, init_capacity=1)
    source.extend({"x": np.array([[4.0, 5.0]]), "f": np.array([6.0])})
    source._dense_numeric_view = cast(DenseNumericView, _CountingDenseView())
    target._dense_numeric_view = cast(DenseNumericView, _CountingDenseView())
    target._extend_internal(source, preserve_ids=True)
    assert cast(_CountingDenseView, source._dense_numeric_view).calls == 0
    assert cast(_CountingDenseView, target._dense_numeric_view).calls == 1
    np.testing.assert_array_equal(target.x, [[4.0, 5.0]])


def test_dense_population_replacement_uses_non_identity_dense_view() -> None:
    attrs = [
        PopulationAttribute("x", np.float64, shape=(2,)),
        PopulationAttribute("f", np.float64),
        PopulationAttribute("id", np.int64, default=-1),
    ]
    source_service = _CopyingDenseView()
    target_service = _CopyingDenseView()
    source = Population(attrs, init_capacity=2)
    target = Population(attrs, init_capacity=2)
    source._dense_numeric_view = cast(DenseNumericView, source_service)
    target._dense_numeric_view = cast(DenseNumericView, target_service)
    source._extend_internal(
        {
            "x": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "f": np.array([5.0, 6.0]),
            "id": np.array([10, 11]),
        },
        preserve_ids=True,
    )
    assert target._replace_from_population(
        source, np.array([1], dtype=np.intp), preserve_ids=True
    )
    assert source_service.calls == 0
    assert target_service.calls == 1
    np.testing.assert_array_equal(target.x, [[3.0, 4.0]])
    np.testing.assert_array_equal(target.f, [6.0])
    np.testing.assert_array_equal(target.candidate_ids, [11])


def test_dense_population_extend_preserves_ids_versions_and_cache() -> None:
    attrs = [
        PopulationAttribute("x", np.float64, shape=(2,)),
        PopulationAttribute("f", np.float64),
        PopulationAttribute("id", np.int64, default=-1),
    ]
    service = _CountingDenseView()
    target = Population(attrs, init_capacity=2)
    source = Population(attrs, init_capacity=2)
    target._dense_numeric_view = cast(DenseNumericView, service)
    source._dense_numeric_view = cast(DenseNumericView, service)
    source._extend_internal(
        {"x": np.array([[1.0, 2.0]]), "f": np.array([3.0]), "id": np.array([10])},
        preserve_ids=True,
    )
    target.set_cache("cached", object())
    structure_version = target.structure_version
    value_version = target.value_version
    target._extend_internal(source, preserve_ids=True)
    assert target.structure_version == structure_version + 1
    assert target.value_version == value_version + 1
    assert target.get_cache("cached") is None
    np.testing.assert_array_equal(target.candidate_ids, [10])
    np.testing.assert_array_equal(target.x, [[1.0, 2.0]])

    duplicate = Population(attrs, init_capacity=2)
    duplicate._dense_numeric_view = cast(DenseNumericView, service)
    duplicate._extend_internal(
        {"x": np.array([[9.0, 9.0]]), "f": np.array([8.0]), "id": np.array([10])},
        preserve_ids=True,
    )
    old_versions = (target.structure_version, target.value_version)
    with pytest.raises(ValidationError, match="Duplicate candidate id already"):
        target._extend_internal(duplicate, preserve_ids=True)
    assert (target.structure_version, target.value_version) == old_versions
    np.testing.assert_array_equal(target.candidate_ids, [10])


def test_dense_population_replacement_preserves_selection_and_versions() -> None:
    attrs = [
        PopulationAttribute("x", np.float64, shape=(2,)),
        PopulationAttribute("f", np.float64),
        PopulationAttribute("id", np.int64, default=-1),
    ]
    target = Population(attrs, init_capacity=4)
    source = Population(attrs, init_capacity=4)
    target._extend_internal(
        {
            "x": np.array([[0.0, 1.0], [1.0, 2.0]]),
            "f": np.array([0.0, 1.0]),
            "id": np.array([100, 101]),
        },
        preserve_ids=True,
    )
    source._extend_internal(
        {
            "x": np.array([[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]),
            "f": np.array([2.0, 3.0, 4.0]),
            "id": np.array([200, 201, 202]),
        },
        preserve_ids=True,
    )
    target.set_cache("survivors", object())
    structure_version = target.structure_version
    value_version = target.value_version
    old_individual = target[0]

    assert target._replace_from_population(
        source, np.array([2, 0], dtype=np.intp), preserve_ids=True
    )

    assert len(target) == 2
    np.testing.assert_array_equal(target.x, [[4.0, 5.0], [2.0, 3.0]])
    np.testing.assert_array_equal(target.f, [4.0, 2.0])
    np.testing.assert_array_equal(target.candidate_ids, [202, 200])
    assert target.structure_version == structure_version + 2
    assert target.value_version == value_version + 2
    assert target.get_cache("survivors") is None
    assert getattr(target._dense_numeric_view, "_canonical_identity_backing") is True
    assert getattr(source._dense_numeric_view, "_canonical_identity_backing") is True
    with pytest.raises(RuntimeError, match="Invalid Individual reference"):
        old_individual.f


def test_dense_population_replacement_rejects_duplicate_survivor_ids() -> None:
    attrs = [
        PopulationAttribute("x", np.float64, shape=(2,)),
        PopulationAttribute("id", np.int64, default=-1),
    ]
    target = Population(attrs, init_capacity=2)
    source = Population(attrs, init_capacity=2)
    target._extend_internal(
        {
            "x": np.array([[9.0, 9.0]]),
            "id": np.array([99]),
        },
        preserve_ids=True,
    )
    source._extend_internal(
        {
            "x": np.array([[1.0, 1.0], [2.0, 2.0]]),
            "id": np.array([10, 11]),
        },
        preserve_ids=True,
    )
    structure_version = target.structure_version
    with pytest.raises(ValidationError, match="Duplicate candidate id within"):
        target._replace_from_population(
            source, np.array([0, 0], dtype=np.intp), preserve_ids=True
        )

    assert len(target) == 0
    assert target.structure_version == structure_version + 1


def test_population_replacement_falls_back_for_object_and_custom_genomes() -> None:
    object_target = _object_population(capacity=2)
    object_source = Population(
        [PopulationAttribute("f", np.float64)],
        init_capacity=2,
        genomes=ObjectBatch(["c", "d"]),
    )
    assert not object_target._replace_from_population(
        object_source, np.array([1, 0], dtype=np.intp), preserve_ids=True
    )
    _assert_object_state(object_target, ("a", "b"))

    attrs = [PopulationAttribute("f", np.float64)]
    custom_target = Population(
        attrs, init_capacity=2, genomes=cast(GenomeBatch, _CustomBatch(("a",)))
    )
    custom_source = Population(
        attrs, init_capacity=2, genomes=cast(GenomeBatch, _CustomBatch(("b",)))
    )
    assert not custom_target._replace_from_population(
        custom_source, np.array([0], dtype=np.intp), preserve_ids=True
    )
    assert cast(_CustomBatch, custom_target.genomes).items == ("a",)


def test_dense_batch_rebinds_after_resize() -> None:
    """A capacity change installs a batch for the new backing array."""
    pop = _dense_population(capacity=1)
    pop.append(x=np.array([1.0, 2.0]))
    cached = pop._genome_batch
    backing = pop._data["x"]

    pop.append(x=np.array([3.0, 4.0]))

    assert pop._genome_batch is not cached
    assert pop._data["x"] is not backing
    _assert_dense_backing(pop, [[1.0, 2.0], [3.0, 4.0]])


def test_dense_batch_reuses_backing_for_in_place_updates() -> None:
    """In-place changes retain the batch and expose the new values."""
    pop = _dense_population(capacity=2)
    pop.append(x=np.array([1.0, 2.0]))
    pop.append(x=np.array([3.0, 4.0]))
    cached = pop._genome_batch

    pop.update_rows([0], {"x": np.array([[10.0, 20.0]])})
    assert pop._genome_batch is cached
    _assert_dense_backing(pop, [[10.0, 20.0], [3.0, 4.0]])
    pop.update_array("x", np.array([[10.0, 20.0], [30.0, 40.0]]))
    assert pop._genome_batch is cached
    _assert_dense_backing(pop, [[10.0, 20.0], [30.0, 40.0]])


def test_dense_genomes_view_cache_tracks_structure_and_value_updates() -> None:
    pop = _dense_population(capacity=2)
    pop.append(x=np.array([1.0, 2.0]))
    first = pop.genomes
    assert first is pop.genomes
    pop.update_array("x", np.array([[3.0, 4.0]]))
    assert first is pop.genomes
    np.testing.assert_array_equal(cast(DenseVectorBatch, first).array, [[3.0, 4.0]])

    pop.append(x=np.array([5.0, 6.0]))
    assert first is not pop.genomes
    pop.clear()
    assert len(pop.genomes) == 0

    custom = Population(
        [PopulationAttribute("f", np.float64)],
        init_capacity=1,
        genomes=cast(GenomeBatch, _CustomBatch(("a",))),
    )
    assert custom.genomes is not custom.genomes


def test_dense_batch_operations_keep_values_and_current_backing() -> None:
    """extract, extend, delete, truncate, and clear preserve dense cache validity."""
    pop = _dense_population(capacity=4)
    for value in ([1.0, 2.0], [3.0, 4.0], [5.0, 6.0]):
        pop.append(x=np.asarray(value))

    extracted = pop.extract([2, 0])
    _assert_dense_backing(extracted, [[5.0, 6.0], [1.0, 2.0]])

    other = _dense_population(capacity=1)
    other.append(x=np.array([7.0, 8.0]))
    extracted.extend(other)
    _assert_dense_backing(extracted, [[5.0, 6.0], [1.0, 2.0], [7.0, 8.0]])

    extracted.delete(1)
    _assert_dense_backing(extracted, [[5.0, 6.0], [7.0, 8.0]])
    extracted.truncate(1)
    _assert_dense_backing(extracted, [[5.0, 6.0]])
    extracted.clear()
    _assert_dense_backing(extracted, [])


def test_object_batch_rebuilds_after_mutating_object_storage() -> None:
    """ObjectBatch copies the mutable list, so each list mutation is republished."""
    pop = _object_population(capacity=2)
    pop.append(genome=ObjectBatch(["c"]))
    _assert_object_state(pop, ("a", "b", "c"))

    pop.update_rows([1], {}, genome=ObjectBatch(["updated"]))
    _assert_object_state(pop, ("a", "updated", "c"))
    pop.update_array("f", np.array([1.0, 2.0, 3.0]))
    _assert_object_state(pop, ("a", "updated", "c"))

    extracted = pop.extract([2, 0])
    _assert_object_state(extracted, ("c", "a"))
    other = Population(
        [PopulationAttribute("f", np.float64)],
        init_capacity=1,
        genomes=ObjectBatch(["d"]),
    )
    extracted.extend(other)
    _assert_object_state(extracted, ("c", "a", "d"))

    extracted.delete(1)
    _assert_object_state(extracted, ("c", "d"))
    extracted.truncate(1)
    _assert_object_state(extracted, ("c",))
    extracted.clear()
    _assert_object_state(extracted, ())
