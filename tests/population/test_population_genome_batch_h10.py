"""Regression tests for Unit H10 genome-batch cache reuse."""

from __future__ import annotations

import numpy as np

from saealib.population import Population, PopulationAttribute
from saealib.population.genome import DenseVectorBatch, ObjectBatch


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
