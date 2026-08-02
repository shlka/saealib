"""
Contract tests for stable int64 candidate IDs (ADR-0001 §2.1, §2.2).

Covers ``IDAllocator`` in isolation, the public/internal ``append()``/
``extend()`` split on ``Population``, ID preservation through structural
operations, the reserved-column mutation guards, ``Population._assign_ids()``,
and ``Archive``/``ParetoArchive`` ``add()`` ``preserve_ids=True`` wiring.
"""

import numpy as np
import pytest

from saealib.exceptions import ValidationError
from saealib.identity import IDAllocator
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute

# ---------------------------------------------------------------------------
# IDAllocator
# ---------------------------------------------------------------------------


class TestIDAllocator:
    def test_allocate_returns_sequential_unique_ids(self) -> None:
        alloc = IDAllocator(start=5)
        ids = alloc.allocate(3)
        np.testing.assert_array_equal(ids, [5, 6, 7])
        assert ids.dtype == np.int64

    def test_allocate_advances_state(self) -> None:
        alloc = IDAllocator(start=0)
        alloc.allocate(3)
        second = alloc.allocate(2)
        np.testing.assert_array_equal(second, [3, 4])

    def test_allocate_zero_returns_owned_empty_array_and_does_not_advance(
        self,
    ) -> None:
        alloc = IDAllocator(start=10)
        ids = alloc.allocate(0)
        assert len(ids) == 0
        assert ids.dtype == np.int64
        assert alloc.next_value == 10

    def test_allocate_negative_raises(self) -> None:
        alloc = IDAllocator()
        with pytest.raises(ValidationError):
            alloc.allocate(-1)

    def test_next_value_reflects_state(self) -> None:
        alloc = IDAllocator(start=0)
        assert alloc.next_value == 0
        alloc.allocate(4)
        assert alloc.next_value == 4

    def test_allocate_exact_int64_max_succeeds(self) -> None:
        alloc = IDAllocator(start=np.iinfo(np.int64).max)
        ids = alloc.allocate(1)
        assert ids[0] == np.iinfo(np.int64).max

    def test_allocate_past_int64_max_raises_overflow(self) -> None:
        alloc = IDAllocator(start=np.iinfo(np.int64).max)
        with pytest.raises(OverflowError):
            alloc.allocate(2)


# ---------------------------------------------------------------------------
# Population with an "id" column
# ---------------------------------------------------------------------------


@pytest.fixture
def id_attrs() -> list[PopulationAttribute]:
    return [
        PopulationAttribute(name="x", dtype=np.float64, shape=(2,)),
        PopulationAttribute(name="id", dtype=np.int64, shape=(), default=-1),
    ]


@pytest.fixture
def pop(id_attrs: list[PopulationAttribute]) -> Population:
    return Population(id_attrs, init_capacity=10)


def _snapshot(p: Population) -> dict[str, np.ndarray]:
    return {k: p.get_array(k).copy() for k in p.schema}


class TestPublicAppendExtendRejectExplicitId:
    def test_append_rejects_explicit_nonneg1_id(self, pop: Population) -> None:
        snapshot = _snapshot(pop)
        with pytest.raises(ValidationError):
            pop.append(x=np.array([1.0, 2.0]), id=5)
        assert len(pop) == 0
        for k, v in snapshot.items():
            np.testing.assert_array_equal(pop.get_array(k), v)

    def test_extend_rejects_explicit_nonneg1_id(self, pop: Population) -> None:
        snapshot = _snapshot(pop)
        with pytest.raises(ValidationError):
            pop.extend({"x": np.array([[1.0, 2.0]]), "id": np.array([5])})
        assert len(pop) == 0
        for k, v in snapshot.items():
            np.testing.assert_array_equal(pop.get_array(k), v)

    def test_append_omitting_id_defaults_to_sentinel(self, pop: Population) -> None:
        pop.append(x=np.array([1.0, 2.0]))
        assert pop.get_array("id")[0] == -1

    def test_append_explicit_sentinel_id_succeeds(self, pop: Population) -> None:
        pop.append(x=np.array([1.0, 2.0]), id=-1)
        assert pop.get_array("id")[0] == -1

    def test_extend_omitting_id_defaults_to_sentinel(self, pop: Population) -> None:
        pop.extend({"x": np.array([[1.0, 2.0], [3.0, 4.0]])})
        np.testing.assert_array_equal(pop.get_array("id"), [-1, -1])

    def test_extend_explicit_sentinel_id_succeeds(self, pop: Population) -> None:
        pop.extend({"x": np.array([[1.0, 2.0]]), "id": np.array([-1])})
        assert pop.get_array("id")[0] == -1


class TestInternalPreserveIdsPath:
    def test_append_internal_accepts_explicit_real_id(self, pop: Population) -> None:
        pop._append_internal(x=np.array([1.0, 2.0]), id=42, preserve_ids=True)
        assert pop.get_array("id")[0] == 42

    def test_extend_internal_accepts_explicit_real_ids(self, pop: Population) -> None:
        pop._extend_internal(
            {"x": np.array([[1.0, 2.0], [3.0, 4.0]]), "id": np.array([1, 2])},
            preserve_ids=True,
        )
        np.testing.assert_array_equal(pop.get_array("id"), [1, 2])

    def test_append_internal_duplicate_within_existing_raises(
        self, pop: Population
    ) -> None:
        pop._append_internal(x=np.array([1.0, 2.0]), id=7, preserve_ids=True)
        snapshot = _snapshot(pop)
        with pytest.raises(ValidationError):
            pop._append_internal(x=np.array([3.0, 4.0]), id=7, preserve_ids=True)
        assert len(pop) == 1
        for k, v in snapshot.items():
            np.testing.assert_array_equal(pop.get_array(k), v)

    def test_extend_internal_duplicate_within_single_batch_raises(
        self, pop: Population
    ) -> None:
        snapshot = _snapshot(pop)
        with pytest.raises(ValidationError):
            pop._extend_internal(
                {
                    "x": np.array([[1.0, 2.0], [3.0, 4.0]]),
                    "id": np.array([9, 9]),
                },
                preserve_ids=True,
            )
        assert len(pop) == 0
        for k, v in snapshot.items():
            np.testing.assert_array_equal(pop.get_array(k), v)

    def test_extend_internal_duplicate_against_existing_raises(
        self, pop: Population
    ) -> None:
        pop._append_internal(x=np.array([1.0, 2.0]), id=3, preserve_ids=True)
        snapshot = _snapshot(pop)
        with pytest.raises(ValidationError):
            pop._extend_internal(
                {"x": np.array([[5.0, 6.0]]), "id": np.array([3])},
                preserve_ids=True,
            )
        assert len(pop) == 1
        for k, v in snapshot.items():
            np.testing.assert_array_equal(pop.get_array(k), v)


class TestAppendInternalDefaultGateSymmetry:
    def test_omitted_id_checked_against_non_sentinel_schema_default(self) -> None:
        attrs = [
            PopulationAttribute(name="x", dtype=np.float64, shape=(2,)),
            PopulationAttribute(name="id", dtype=np.int64, shape=(), default=0),
        ]
        p = Population(attrs, init_capacity=10)
        with pytest.raises(ValidationError):
            p.append(x=np.array([1.0, 2.0]))


class TestStructuralOperationsPreserveId:
    def _make_pop(self, id_attrs: list[PopulationAttribute]) -> Population:
        p = Population(id_attrs, init_capacity=10)
        p._extend_internal(
            {
                "x": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
                "id": np.array([10, 11, 12]),
            },
            preserve_ids=True,
        )
        return p

    def test_extract_preserves_ids(self, id_attrs: list[PopulationAttribute]) -> None:
        p = self._make_pop(id_attrs)
        sub = p.extract([2, 0])
        np.testing.assert_array_equal(sub.get_array("id"), [12, 10])

    def test_reorder_preserves_ids(self, id_attrs: list[PopulationAttribute]) -> None:
        p = self._make_pop(id_attrs)
        p.reorder(np.array([2, 1, 0]))
        np.testing.assert_array_equal(p.get_array("id"), [12, 11, 10])

    def test_delete_preserves_remaining_ids(
        self, id_attrs: list[PopulationAttribute]
    ) -> None:
        p = self._make_pop(id_attrs)
        p.delete([1])
        np.testing.assert_array_equal(p.get_array("id"), [10, 12])


class TestReservedIdColumnMutationGuards:
    def test_update_array_id_raises(self, pop: Population) -> None:
        pop.append(x=np.array([1.0, 2.0]))
        with pytest.raises(ValidationError):
            pop.update_array("id", np.array([99]))

    def test_individual_id_assignment_raises(self, pop: Population) -> None:
        pop.append(x=np.array([1.0, 2.0]))
        with pytest.raises(ValidationError):
            pop[0].id = 99


class TestAssignIds:
    def test_assigns_real_ids_and_bumps_value_version(self, pop: Population) -> None:
        pop.extend({"x": np.array([[1.0, 2.0], [3.0, 4.0]])})
        version_before = pop._value_version
        pop._assign_ids(np.array([0, 1]), np.array([100, 101], dtype=np.int64))
        np.testing.assert_array_equal(pop.get_array("id"), [100, 101])
        assert pop._value_version == version_before + 1

    def test_reassigning_already_assigned_row_raises(self, pop: Population) -> None:
        pop._append_internal(x=np.array([1.0, 2.0]), id=5, preserve_ids=True)
        with pytest.raises(ValidationError):
            pop._assign_ids(np.array([0]), np.array([6], dtype=np.int64))

    def test_assignment_creating_duplicate_raises_and_leaves_unchanged(
        self, pop: Population
    ) -> None:
        pop._extend_internal(
            {
                "x": np.array([[1.0, 2.0], [3.0, 4.0]]),
                "id": np.array([1, -1]),
            },
            preserve_ids=True,
        )
        snapshot = _snapshot(pop)
        with pytest.raises(ValidationError):
            pop._assign_ids(np.array([1]), np.array([1], dtype=np.int64))
        for k, v in snapshot.items():
            np.testing.assert_array_equal(pop.get_array(k), v)


# ---------------------------------------------------------------------------
# Archive / ParetoArchive: add() preserve_ids=True wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def archive_attrs() -> list[PopulationAttribute]:
    return [
        PopulationAttribute(name="x", dtype=np.float64, shape=(2,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
        PopulationAttribute(name="cv", dtype=np.float64, shape=(), default=0.0),
        PopulationAttribute(name="id", dtype=np.int64, shape=(), default=-1),
    ]


class TestArchiveAddPreservesId:
    def test_archive_add_preserves_explicit_id(
        self, archive_attrs: list[PopulationAttribute]
    ) -> None:
        archive = Archive(archive_attrs, init_capacity=10, key_attr="x")
        archive.add(x=np.array([1.0, 2.0]), f=np.array([0.5]), id=77)
        assert archive.get_array("id")[0] == 77

    def test_pareto_archive_add_preserves_explicit_id(
        self, archive_attrs: list[PopulationAttribute]
    ) -> None:
        pareto = ParetoArchive(
            archive_attrs, init_capacity=10, direction=np.array([-1.0])
        )
        pareto.add(x=np.array([1.0, 2.0]), f=np.array([0.5]), id=88)
        assert pareto.get_array("id")[0] == 88

    def test_archive_add_dict_element_preserves_id(
        self, archive_attrs: list[PopulationAttribute]
    ) -> None:
        archive = Archive(archive_attrs, init_capacity=10, key_attr="x")
        archive.add({"x": np.array([1.0, 2.0]), "f": np.array([0.5]), "id": 55})
        assert archive.get_array("id")[0] == 55


class TestArchiveAddRejectsSentinelWhenSchemaHasId:
    def test_archive_add_without_id_raises(
        self, archive_attrs: list[PopulationAttribute]
    ) -> None:
        archive = Archive(archive_attrs, init_capacity=10, key_attr="x")
        with pytest.raises(ValidationError):
            archive.add(x=np.array([1.0, 2.0]), f=np.array([0.5]))
        assert len(archive) == 0

    def test_pareto_archive_add_without_id_raises(
        self, archive_attrs: list[PopulationAttribute]
    ) -> None:
        pareto = ParetoArchive(
            archive_attrs, init_capacity=10, direction=np.array([-1.0])
        )
        with pytest.raises(ValidationError):
            pareto.add(x=np.array([1.0, 2.0]), f=np.array([0.5]))
        assert len(pareto) == 0


class TestParetoAddAtomicityOnDuplicateId:
    def test_rejected_duplicate_id_does_not_delete_dominated_row(self) -> None:
        two_obj_attrs = [
            PopulationAttribute(name="x", dtype=np.float64, shape=(2,)),
            PopulationAttribute(name="f", dtype=np.float64, shape=(2,)),
            PopulationAttribute(name="cv", dtype=np.float64, shape=(), default=0.0),
            PopulationAttribute(name="id", dtype=np.int64, shape=(), default=-1),
        ]
        pareto = ParetoArchive(
            two_obj_attrs, init_capacity=10, direction=np.array([-1.0, -1.0])
        )
        pareto.add(x=np.array([0.0, 0.0]), f=np.array([2.0, 2.0]), id=1)  # A
        pareto.add(x=np.array([1.0, 1.0]), f=np.array([0.0, 3.0]), id=2)  # B
        assert len(pareto) == 2

        with pytest.raises(ValidationError):
            pareto.add(x=np.array([2.0, 2.0]), f=np.array([1.0, 2.0]), id=2)

        assert len(pareto) == 2
        np.testing.assert_array_equal(sorted(pareto.get_array("id")), [1, 2])


class TestPublicExtendCannotRoundTripOwnedIds:
    def test_extract_then_public_extend_of_owned_ids_raises(
        self, id_attrs: list[PopulationAttribute]
    ) -> None:
        p = Population(id_attrs, init_capacity=10)
        p._extend_internal(
            {
                "x": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
                "id": np.array([10, 11, 12]),
            },
            preserve_ids=True,
        )
        kept = p.extract([0, 1])
        p.clear()
        with pytest.raises(ValidationError):
            p.extend(kept)
