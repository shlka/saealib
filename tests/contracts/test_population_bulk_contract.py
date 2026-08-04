"""
Contract tests for ``Population.update_rows()`` (ADR-0001 §2.3) and its
interaction with ``ArchiveMixin``'s KD-tree cache (ADR-0001 §2.4).

Covers the documented contract only: validation ordering,
atomicity, version-bump semantics, and copy-not-alias behavior.
"""

import numpy as np
import pytest

from saealib.exceptions import ValidationError
from saealib.population import Archive, Population, PopulationAttribute


@pytest.fixture
def attrs() -> list[PopulationAttribute]:
    return [
        PopulationAttribute(name="x", dtype=np.float64, shape=(2,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
    ]


@pytest.fixture
def pop(attrs: list[PopulationAttribute]) -> Population:
    p = Population(attrs, init_capacity=10)
    for i in range(5):
        p.append(x=np.array([i, i], dtype=np.float64), f=np.array([float(i)]))
    return p


def _snapshot(p: Population) -> dict[str, np.ndarray]:
    return {k: p.get_array(k).copy() for k in p.schema}


class TestUpdateRowsValidUpdate:
    def test_applies_values_and_bumps_version_once(self, pop: Population) -> None:
        version_before = pop._value_version
        pop.update_rows(
            np.array([1, 3]),
            {
                "x": np.array([[9.0, 9.0], [8.0, 8.0]]),
                "f": np.array([[99.0], [88.0]]),
            },
        )
        assert pop._value_version == version_before + 1
        np.testing.assert_array_equal(pop.get_array("x")[1], [9.0, 9.0])
        np.testing.assert_array_equal(pop.get_array("x")[3], [8.0, 8.0])
        np.testing.assert_array_equal(pop.get_array("f")[1], [99.0])
        np.testing.assert_array_equal(pop.get_array("f")[3], [88.0])

    def test_single_column_update_bumps_version_once(self, pop: Population) -> None:
        version_before = pop._value_version
        pop.update_rows(np.array([0]), {"f": np.array([[42.0]])})
        assert pop._value_version == version_before + 1


class TestUpdateRowsNoOps:
    def test_empty_indices_is_noop(self, pop: Population) -> None:
        version_before = pop._value_version
        snapshot = _snapshot(pop)
        pop.update_rows(np.array([], dtype=np.intp), {"f": np.array([]).reshape(0, 1)})
        assert pop._value_version == version_before
        for k, v in snapshot.items():
            np.testing.assert_array_equal(pop.get_array(k), v)

    def test_empty_values_dict_is_noop(self, pop: Population) -> None:
        version_before = pop._value_version
        snapshot = _snapshot(pop)
        pop.update_rows(np.array([0, 1]), {})
        assert pop._value_version == version_before
        for k, v in snapshot.items():
            np.testing.assert_array_equal(pop.get_array(k), v)


class TestUpdateRowsValidationFailures:
    def test_out_of_range_index_raises_and_leaves_data_unchanged(
        self, pop: Population
    ) -> None:
        snapshot = _snapshot(pop)
        with pytest.raises(ValidationError):
            pop.update_rows(np.array([0, 100]), {"f": np.array([[1.0], [2.0]])})
        for k, v in snapshot.items():
            np.testing.assert_array_equal(pop.get_array(k), v)

    def test_duplicate_indices_raise_and_leave_data_unchanged(
        self, pop: Population
    ) -> None:
        snapshot = _snapshot(pop)
        with pytest.raises(ValidationError):
            pop.update_rows(np.array([1, 1]), {"f": np.array([[1.0], [2.0]])})
        for k, v in snapshot.items():
            np.testing.assert_array_equal(pop.get_array(k), v)

    def test_boolean_indices_raise(self, pop: Population) -> None:
        snapshot = _snapshot(pop)
        mask = np.zeros(5, dtype=bool)
        mask[0] = True
        with pytest.raises(ValidationError):
            pop.update_rows(mask, {"f": np.array([[1.0]])})
        for k, v in snapshot.items():
            np.testing.assert_array_equal(pop.get_array(k), v)

    def test_unknown_column_raises_all_or_nothing(self, pop: Population) -> None:
        """A bad key in a multi-key call must not apply the good key either."""
        snapshot = _snapshot(pop)
        with pytest.raises(ValidationError):
            pop.update_rows(
                np.array([0, 1]),
                {
                    "f": np.array([[123.0], [456.0]]),
                    "bogus": np.array([[1.0], [2.0]]),
                },
            )
        for k, v in snapshot.items():
            np.testing.assert_array_equal(pop.get_array(k), v)

    def test_id_key_raises_even_when_schema_has_no_id_column(
        self, pop: Population
    ) -> None:
        assert "id" not in pop.schema
        snapshot = _snapshot(pop)
        with pytest.raises(ValidationError):
            pop.update_rows(np.array([0]), {"id": np.array([[1]])})
        for k, v in snapshot.items():
            np.testing.assert_array_equal(pop.get_array(k), v)

    def test_shape_mismatch_raises_and_leaves_data_unchanged(
        self, pop: Population
    ) -> None:
        snapshot = _snapshot(pop)
        with pytest.raises(ValidationError):
            # f column has shape (1,) per row; 2 values requested but a
            # (2, 2)-shaped array supplied instead of (2, 1).
            pop.update_rows(np.array([0, 1]), {"f": np.array([[1.0, 2.0], [3.0, 4.0]])})
        for k, v in snapshot.items():
            np.testing.assert_array_equal(pop.get_array(k), v)

    def test_dtype_mismatch_raises_and_leaves_data_unchanged(
        self, pop: Population
    ) -> None:
        snapshot = _snapshot(pop)
        with pytest.raises(ValidationError):
            pop.update_rows(np.array([0]), {"f": np.array([[1.0]], dtype=np.float32)})
        for k, v in snapshot.items():
            np.testing.assert_array_equal(pop.get_array(k), v)


class TestUpdateRowsCopySemantics:
    def test_committed_values_are_copies_not_aliases(self, pop: Population) -> None:
        caller_arr = np.array([[7.0, 7.0]])
        pop.update_rows(np.array([0]), {"x": caller_arr})
        caller_arr[:] = -1.0
        np.testing.assert_array_equal(pop.get_array("x")[0], [7.0, 7.0])


class TestUpdateRowsScalarAndObjectDtypeRejection:
    def test_scalar_indices_raise_validation_error(self, pop: Population) -> None:
        snapshot = _snapshot(pop)
        with pytest.raises(ValidationError):
            pop.update_rows(np.array(0), {"f": np.array([[1.0]])})
        for k, v in snapshot.items():
            np.testing.assert_array_equal(pop.get_array(k), v)

    def test_empty_values_is_a_noop_even_with_invalid_indices(
        self, pop: Population
    ) -> None:
        """An empty ``values`` mapping short-circuits before indices are
        even inspected, per the ADR's "empty values mapping is a no-op"."""
        version_before = pop._value_version
        pop.update_rows(np.array(0), {})
        assert pop._value_version == version_before

    def test_object_dtype_value_rejected(self) -> None:
        attrs = [PopulationAttribute(name="tag", dtype=object, shape=())]
        p = Population(attrs, init_capacity=3)
        p.append(tag="a")
        p.append(tag="b")
        with pytest.raises(ValidationError):
            p.update_rows(np.array([0]), {"tag": np.array(["z"], dtype=object)})


class TestArchiveKdtreeInvalidation:
    def test_update_rows_rebuilds_stale_kdtree(self) -> None:
        archive = Archive(
            attrs=[
                PopulationAttribute(name="x", dtype=np.float64, shape=(2,)),
                PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
            ],
            init_capacity=10,
            key_attr="x",
        )
        for i in range(3):
            archive.add(x=np.array([float(i), float(i)]), f=np.array([float(i)]))

        # Populate the KD-tree cache against the original coordinates.
        idx, _ = archive.get_knn(np.array([0.0, 0.0]), k=1)
        assert idx[0] == 0

        # Move point 0 far away via update_rows; the cache must be rebuilt,
        # not left stale, on the next get_knn() call.
        archive.update_rows(np.array([0]), {"x": np.array([[100.0, 100.0]])})

        idx, _ = archive.get_knn(np.array([0.0, 0.0]), k=1)
        assert idx[0] != 0
        np.testing.assert_array_equal(archive.get_array("x")[idx[0]], [1.0, 1.0])
