"""
Tests for the population module.

Tests cover:
- PopulationAttribute dataclass
- Population: init, append, extend, extract, truncate, delete, reorder,
  argsort, clear, empty_like, get, get_array, schema, attrs, len,
  __getattr__, __getitem__, name conflict warning, resize
- Individual: getattr, setattr, version invalidation, pop property
- bind_property / bind_property_array
- ArchiveMixin / Archive: add, duplicate detection, get_duplicated_population,
  get_knn, tolerance-based matching
- Cache: set_cache, get_cache, automatic invalidation on mutation
- ParetoMixin.add() fast-path (#224): differential equivalence between the
  vectorized ``dominates_many`` broadcast path and the original per-row loop
"""

import warnings

import numpy as np
import pytest

from saealib.comparators import Dominator, EpsilonDominator, ParetoDominator
from saealib.population import (
    Archive,
    Individual,
    ParetoArchive,
    Population,
    PopulationAttribute,
    bind_property,
    bind_property_array,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def basic_attrs() -> list[PopulationAttribute]:
    """Basic attribute list (x: 3-dim vector, f: scalar)."""
    return [
        PopulationAttribute(name="x", dtype=np.float64, shape=(3,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=()),
    ]


@pytest.fixture
def pop(basic_attrs: list[PopulationAttribute]) -> Population:
    """Basic Population instance."""
    return Population(basic_attrs, init_capacity=10)


@pytest.fixture
def populated_pop(pop: Population) -> Population:
    """Population pre-filled with 5 individuals."""
    for i in range(5):
        pop.append(x=np.array([i, i + 1, i + 2], dtype=np.float64), f=float(i * 10))
    return pop


@pytest.fixture
def archive_attrs() -> list[PopulationAttribute]:
    """Attribute list for Archive."""
    return [
        PopulationAttribute(name="x", dtype=np.float64, shape=(2,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=()),
    ]


@pytest.fixture
def archive(archive_attrs: list[PopulationAttribute]) -> Archive:
    """Basic Archive instance."""
    return Archive(archive_attrs, init_capacity=10)


@pytest.fixture
def moo_attrs() -> list[PopulationAttribute]:
    """Attribute list for ParetoArchive (x: 2-dim, f: 2-dim objective, cv: scalar)."""
    return [
        PopulationAttribute(name="x", dtype=np.float64, shape=(2,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(2,)),
        PopulationAttribute(name="cv", dtype=np.float64, shape=()),
    ]


@pytest.fixture
def pareto_archive(moo_attrs: list[PopulationAttribute]) -> ParetoArchive:
    return ParetoArchive(moo_attrs, init_capacity=20)


# ===========================================================================
# PopulationAttribute Tests
# ===========================================================================
class TestPopulationAttribute:
    """Tests for the PopulationAttribute dataclass."""

    def test_defaults(self) -> None:
        attr = PopulationAttribute(name="x", dtype=np.float64)
        assert attr.name == "x"
        assert attr.dtype == np.float64
        assert attr.shape == ()
        assert np.isnan(attr.default)

    def test_custom_values(self) -> None:
        attr = PopulationAttribute(name="flag", dtype=np.int32, shape=(2,), default=0)
        assert attr.name == "flag"
        assert attr.dtype == np.int32
        assert attr.shape == (2,)
        assert attr.default == 0

    def test_frozen(self) -> None:
        attr = PopulationAttribute(name="x", dtype=np.float64)
        with pytest.raises(AttributeError):
            attr.name = "y"  # type: ignore  # intentional: testing read-only attribute assignment raises


# ===========================================================================
# Population Initialization Tests
# ===========================================================================
class TestPopulationInit:
    """Tests for Population initialization."""

    def test_empty_population(self, pop: Population) -> None:
        assert len(pop) == 0
        assert pop._capacity == 10

    def test_schema(self, pop: Population) -> None:
        schema = pop.schema
        assert "x" in schema
        assert "f" in schema
        assert schema["x"].shape == (3,)
        assert schema["f"].shape == ()

    def test_schema_is_immutable(self, pop: Population) -> None:
        schema = pop.schema
        with pytest.raises(TypeError):
            schema["new_key"] = None  # type: ignore  # intentional: testing invalid schema value raises

    def test_attrs_property(self, pop: Population) -> None:
        attrs = pop.attrs
        assert len(attrs) == 2
        assert all(isinstance(a, PopulationAttribute) for a in attrs)

    def test_name_conflict_warning(self) -> None:
        """A warning is raised when an attribute name conflicts with a method name."""
        attrs = [
            PopulationAttribute(name="clear", dtype=np.float64),
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Population(attrs, init_capacity=5)
            assert len(w) == 1
            assert "conflicts" in str(w[0].message)

    def test_no_warning_for_bind_property_names(self) -> None:
        """No warning for names defined via bind_property_array (x, f, g, cv)."""
        attrs = [
            PopulationAttribute(name="x", dtype=np.float64, shape=(2,)),
            PopulationAttribute(name="f", dtype=np.float64),
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Population(attrs, init_capacity=5)
            conflict_warnings = [x for x in w if "conflicts" in str(x.message)]
            assert len(conflict_warnings) == 0

    def test_dot_setter_property(self, populated_pop: Population) -> None:
        arr = np.zeros_like(populated_pop.x)
        populated_pop.x = arr
        np.testing.assert_array_equal(populated_pop.x, arr)

    def test_get_readonly_array(self, pop: Population) -> None:
        arr = pop.get_readonly_array("f")
        assert not arr.flags.writeable

    def test_mod_value_and_mod_structure(self, pop: Population) -> None:
        v0 = pop._value_version
        s0 = pop._structure_version
        pop.mod_value()
        assert pop._value_version == v0 + 1
        pop.mod_structure()
        assert pop._structure_version == s0 + 1
        assert pop._value_version == v0 + 2


# ===========================================================================
# Population Append Tests
# ===========================================================================
class TestPopulationAppend:
    """Tests for Population.append."""

    def test_append_kwargs(self, pop: Population) -> None:
        pop.append(x=np.array([1.0, 2.0, 3.0]), f=0.5)
        assert len(pop) == 1
        np.testing.assert_array_equal(pop.x[0], [1.0, 2.0, 3.0])
        assert pop.f[0] == 0.5

    def test_append_dict(self, pop: Population) -> None:
        pop.append({"x": np.array([4.0, 5.0, 6.0]), "f": 1.0})
        assert len(pop) == 1
        np.testing.assert_array_equal(pop.x[0], [4.0, 5.0, 6.0])

    def test_append_individual(self, populated_pop: Population) -> None:
        ind = populated_pop[0]
        new_pop = populated_pop.empty_like()
        new_pop.append(ind)
        assert len(new_pop) == 1
        np.testing.assert_array_equal(new_pop.x[0], populated_pop.x[0])

    def test_append_individual_with_override(self, populated_pop: Population) -> None:
        ind = populated_pop[0]
        new_pop = populated_pop.empty_like()
        new_pop.append(ind, f=999.0)
        assert new_pop.f[0] == 999.0

    def test_append_default_values(self, pop: Population) -> None:
        """Unspecified attributes are filled with their default values."""
        pop.append(x=np.array([1.0, 2.0, 3.0]))
        assert np.isnan(pop.f[0])

    def test_append_triggers_resize(self) -> None:
        attrs = [PopulationAttribute(name="x", dtype=np.float64)]
        pop = Population(attrs, init_capacity=2)
        for i in range(5):
            pop.append(x=float(i))
        assert len(pop) == 5
        assert pop._capacity >= 5

    def test_structure_version_increments(self, pop: Population) -> None:
        v0 = pop._structure_version
        pop.append(x=np.array([1.0, 2.0, 3.0]), f=0.0)
        assert pop._structure_version == v0 + 1

    def test_value_version_increments(self, pop: Population) -> None:
        v0 = pop._value_version
        pop.x = np.zeros_like(pop.x)
        assert pop._value_version == v0 + 1

    def test_readonly_view(self, pop: Population) -> None:
        with pytest.raises(ValueError, match="read-only"):
            pop.f[:] = 1.0
        with pytest.raises(ValueError, match="read-only"):
            pop.f[0] = 1.0


# ===========================================================================
# Population Extend Tests
# ===========================================================================
class TestPopulationExtend:
    """Tests for Population.extend."""

    def test_extend_population(self, populated_pop: Population) -> None:
        new_pop = populated_pop.empty_like()
        new_pop.extend(populated_pop)
        assert len(new_pop) == 5
        np.testing.assert_array_equal(new_pop.x, populated_pop.x)

    def test_extend_dict(self, pop: Population) -> None:
        data = {
            "x": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "f": np.array([0.1, 0.2]),
        }
        pop.extend(data)
        assert len(pop) == 2

    def test_extend_empty_population(self, pop: Population) -> None:
        """Extending with an empty Population is a no-op."""
        empty = pop.empty_like()
        pop.append(x=np.array([1.0, 2.0, 3.0]), f=0.0)
        v_before = pop._structure_version
        pop.extend(empty)
        assert len(pop) == 1
        assert pop._structure_version == v_before  # version unchanged

    def test_extend_triggers_resize(self) -> None:
        attrs = [PopulationAttribute(name="x", dtype=np.float64)]
        pop = Population(attrs, init_capacity=2)
        data = {"x": np.arange(10, dtype=np.float64)}
        pop.extend(data)
        assert len(pop) == 10
        assert pop._capacity >= 10


# ===========================================================================
# Population Extract Tests
# ===========================================================================
class TestPopulationExtract:
    """Tests for Population.extract."""

    def test_extract_indices(self, populated_pop: Population) -> None:
        sub = populated_pop.extract([0, 2, 4])
        assert len(sub) == 3
        np.testing.assert_array_equal(sub.x[0], populated_pop.x[0])
        np.testing.assert_array_equal(sub.x[1], populated_pop.x[2])
        np.testing.assert_array_equal(sub.x[2], populated_pop.x[4])

    def test_extract_slice(self, populated_pop: Population) -> None:
        sub = populated_pop.extract(slice(1, 4))
        assert len(sub) == 3
        np.testing.assert_array_equal(sub.f, populated_pop.f[1:4])

    def test_extract_returns_new_population(self, populated_pop: Population) -> None:
        sub = populated_pop.extract([0])
        assert sub is not populated_pop
        assert isinstance(sub, Population)


# ===========================================================================
# Population Truncate Tests
# ===========================================================================
class TestPopulationTruncate:
    """Tests for Population.truncate."""

    def test_truncate(self, populated_pop: Population) -> None:
        populated_pop.truncate(3)
        assert len(populated_pop) == 3

    def test_truncate_larger_than_size(self, populated_pop: Population) -> None:
        """Truncating to a value >= current size has no effect."""
        populated_pop.truncate(100)
        assert len(populated_pop) == 5

    def test_truncate_to_zero(self, populated_pop: Population) -> None:
        populated_pop.truncate(0)
        assert len(populated_pop) == 0

    def test_truncate_negative_raises(self, populated_pop: Population) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            populated_pop.truncate(-1)


# ===========================================================================
# Population Delete Tests
# ===========================================================================
class TestPopulationDelete:
    """Tests for Population.delete."""

    def test_delete_single(self, populated_pop: Population) -> None:
        original_x1 = populated_pop.x[1].copy()
        populated_pop.delete(0)
        assert len(populated_pop) == 4
        np.testing.assert_array_equal(populated_pop.x[0], original_x1)

    def test_delete_multiple(self, populated_pop: Population) -> None:
        populated_pop.delete([0, 2, 4])
        assert len(populated_pop) == 2

    def test_delete_slice(self, populated_pop: Population) -> None:
        populated_pop.delete(slice(0, 3))
        assert len(populated_pop) == 2


# ===========================================================================
# Population Reorder Tests
# ===========================================================================
class TestPopulationReorder:
    """Tests for Population.reorder."""

    def test_reorder(self, populated_pop: Population) -> None:
        original_x = populated_pop.x.copy()
        order = np.array([4, 3, 2, 1, 0])
        populated_pop.reorder(order)
        np.testing.assert_array_equal(populated_pop.x[0], original_x[4])
        np.testing.assert_array_equal(populated_pop.x[4], original_x[0])

    def test_reorder_wrong_length_raises(self, populated_pop: Population) -> None:
        with pytest.raises(ValueError, match="must match population size"):
            populated_pop.reorder(np.array([0, 1]))


# ===========================================================================
# Population Argsort Tests
# ===========================================================================
class TestPopulationArgsort:
    """Tests for Population.argsort."""

    def test_argsort_ascending(self, populated_pop: Population) -> None:
        order = populated_pop.argsort("f")
        f_sorted = populated_pop.f[order]
        np.testing.assert_array_equal(f_sorted, np.sort(populated_pop.f))

    def test_argsort_descending(self, populated_pop: Population) -> None:
        order = populated_pop.argsort("f", reverse=True)
        f_sorted = populated_pop.f[order]
        np.testing.assert_array_equal(f_sorted, np.sort(populated_pop.f)[::-1])

    def test_argsort_invalid_key_raises(self, populated_pop: Population) -> None:
        with pytest.raises(KeyError, match="not found"):
            populated_pop.argsort("nonexistent")


# ===========================================================================
# Population Clear & empty_like Tests
# ===========================================================================
class TestPopulationClearAndEmptyLike:
    """Tests for Population.clear and empty_like."""

    def test_clear(self, populated_pop: Population) -> None:
        populated_pop.clear()
        assert len(populated_pop) == 0

    def test_empty_like_default_capacity(self, populated_pop: Population) -> None:
        new_pop = populated_pop.empty_like()
        assert len(new_pop) == 0
        assert new_pop._capacity == populated_pop._capacity
        assert set(new_pop.schema.keys()) == set(populated_pop.schema.keys())

    def test_empty_like_custom_capacity(self, populated_pop: Population) -> None:
        new_pop = populated_pop.empty_like(capacity=50)
        assert new_pop._capacity == 50


# ===========================================================================
# Population get / get_array Tests
# ===========================================================================
class TestPopulationGetAndGetArray:
    """Tests for Population.get and get_array."""

    def test_get_existing_key(self, populated_pop: Population) -> None:
        result = populated_pop.get("x")
        assert result is not None
        assert result.shape == (5, 3)

    def test_get_missing_key_returns_default(self, populated_pop: Population) -> None:
        result = populated_pop.get("nonexistent")
        assert result is None

    def test_get_missing_key_custom_default(self, populated_pop: Population) -> None:
        result = populated_pop.get("nonexistent", default=42)
        assert result == 42

    def test_get_array(self, populated_pop: Population) -> None:
        arr = populated_pop.get_array("f")
        assert arr.shape == (5,)
        assert arr[0] == 0.0

    def test_get_array_returns_view(self, populated_pop: Population) -> None:
        """get_array returns a slice view, so mutations are reflected."""
        arr = populated_pop.get_array("f")
        arr[0] = 999.0
        assert populated_pop.get_array("f")[0] == 999.0


# ===========================================================================
# Population __getattr__ / __getitem__ Tests
# ===========================================================================
class TestPopulationAccess:
    """Tests for Population dot-access and bracket-access."""

    def test_dot_access(self, populated_pop: Population) -> None:
        x = populated_pop.x
        assert x.shape == (5, 3)

    def test_dot_access_invalid_raises(self, pop: Population) -> None:
        with pytest.raises(AttributeError):
            _ = pop.nonexistent

    def test_bracket_int_returns_individual(self, populated_pop: Population) -> None:
        ind = populated_pop[0]
        assert isinstance(ind, Individual)

    def test_bracket_int_out_of_range_raises(self, populated_pop: Population) -> None:
        with pytest.raises(IndexError):
            _ = populated_pop[100]

    def test_bracket_negative_raises(self, populated_pop: Population) -> None:
        with pytest.raises(IndexError):
            _ = populated_pop[-1]

    def test_bracket_slice_returns_population(self, populated_pop: Population) -> None:
        sub = populated_pop[:3]
        assert isinstance(sub, Population)
        assert len(sub) == 3

    def test_bracket_invalid_type_raises(self, populated_pop: Population) -> None:
        with pytest.raises(TypeError):
            _ = populated_pop["invalid"]  # type: ignore  # intentional: testing invalid index type raises TypeError


# ===========================================================================
# Population bind_property_array Tests
# ===========================================================================
class TestBindPropertyArray:
    """Tests for getter/setter via bind_property_array."""

    def test_setter_via_property(self, populated_pop: Population) -> None:
        new_f = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        populated_pop.f = new_f
        np.testing.assert_array_equal(populated_pop.get_array("f"), new_f)

    def test_getter_via_property(self, populated_pop: Population) -> None:
        result = populated_pop.f
        assert isinstance(result, np.ndarray)
        assert len(result) == 5


# ===========================================================================
# Individual Tests
# ===========================================================================
class TestIndividual:
    """Tests for the Individual class."""

    def test_getattr(self, populated_pop: Population) -> None:
        ind = populated_pop[0]
        np.testing.assert_array_equal(ind.x, populated_pop.x[0])
        assert ind.f == populated_pop.f[0]

    def test_setattr(self, populated_pop: Population) -> None:
        ind = populated_pop[0]
        ind.f = 999.0
        assert populated_pop.f[0] == 999.0

    def test_setattr_array(self, populated_pop: Population) -> None:
        ind = populated_pop[0]
        new_x = np.array([10.0, 20.0, 30.0])
        ind.x = new_x
        np.testing.assert_array_equal(populated_pop.x[0], new_x)

    def test_getattr_invalid_raises(self, populated_pop: Population) -> None:
        ind = populated_pop[0]
        with pytest.raises(AttributeError):
            _ = ind.nonexistent

    def test_setattr_invalid_raises(self, populated_pop: Population) -> None:
        ind = populated_pop[0]
        with pytest.raises(AttributeError):
            ind.nonexistent = 1.0

    def test_version_invalidation(self, populated_pop: Population) -> None:
        """An Individual becomes invalid after the Population is modified."""
        ind = populated_pop[0]
        populated_pop.append(x=np.array([99.0, 99.0, 99.0]), f=99.0)
        with pytest.raises(RuntimeError, match="Invalid Individual reference"):
            _ = ind.x

    def test_pop_property(self, populated_pop: Population) -> None:
        ind = populated_pop[0]
        assert ind.pop is populated_pop

    def test_pop_property_after_invalidation(self, populated_pop: Population) -> None:
        ind = populated_pop[0]
        populated_pop.clear()
        with pytest.raises(RuntimeError, match="Invalid Individual reference"):
            _ = ind.pop

    def test_weakref_garbage_collection(
        self, basic_attrs: list[PopulationAttribute]
    ) -> None:
        """Individual becomes invalid when parent Population is GC'd."""
        pop = Population(basic_attrs, init_capacity=5)
        pop.append(x=np.array([1.0, 2.0, 3.0]), f=0.0)
        ind = pop[0]
        del pop
        with pytest.raises(RuntimeError, match="Invalid Individual reference"):
            _ = ind.x

    def test_readonly_view(self, populated_pop: Population) -> None:
        ind = populated_pop[0]
        with pytest.raises(ValueError, match="read-only"):
            ind.x[:] = np.array([1.0, 2.0, 3.0])

    def test_get_readonly_value(self, populated_pop: Population) -> None:
        ind = populated_pop[0]
        arr = ind.get_readonly_value("x")
        assert not arr.flags.writeable


# ===========================================================================
# bind_property Tests
# ===========================================================================
class TestBindProperty:
    """Tests for the bind_property helper functions."""

    def test_bind_property_creates_property(self) -> None:
        prop = bind_property("test_key", doc="docstring")
        assert isinstance(prop, property)

    def test_bind_property_array_creates_property(self) -> None:
        prop = bind_property_array("test_key", doc="docstring")
        assert isinstance(prop, property)


# ===========================================================================
# Archive Tests
# ===========================================================================
class TestArchive:
    """Tests for Archive / ArchiveMixin."""

    def test_add_unique(self, archive: Archive) -> None:
        idx = archive.add(x=np.array([1.0, 2.0]), f=0.1)
        assert idx == 0
        assert len(archive) == 1

    def test_add_duplicate_is_ignored(self, archive: Archive) -> None:
        archive.add(x=np.array([1.0, 2.0]), f=0.1)
        idx = archive.add(x=np.array([1.0, 2.0]), f=0.2)
        assert idx == 0  # returns existing index
        assert len(archive) == 1  # size unchanged

    def test_add_different_solutions(self, archive: Archive) -> None:
        archive.add(x=np.array([1.0, 2.0]), f=0.1)
        archive.add(x=np.array([3.0, 4.0]), f=0.2)
        assert len(archive) == 2

    def test_add_from_dict(self, archive: Archive) -> None:
        idx = archive.add({"x": np.array([1.0, 2.0]), "f": 0.1})
        assert idx == 0
        assert len(archive) == 1

    def test_add_missing_key_attr_raises(self, archive: Archive) -> None:
        with pytest.raises(ValueError, match="must have"):
            archive.add(f=0.1)

    def test_add_from_individual(self, archive: Archive) -> None:
        archive.add(x=np.array([1.0, 2.0]), f=0.1)
        ind = archive[0]
        new_archive = Archive(archive.attrs, init_capacity=10)
        idx = new_archive.add(ind)
        assert idx == 0
        assert len(new_archive) == 1

    def test_tolerance_based_duplicate(
        self, archive_attrs: list[PopulationAttribute]
    ) -> None:
        archive = Archive(archive_attrs, init_capacity=10, atol=0.1)
        archive.add(x=np.array([1.0, 2.0]), f=0.1)
        # within atol=0.1, so treated as duplicate
        idx = archive.add(x=np.array([1.05, 2.05]), f=0.2)
        assert idx == 0
        assert len(archive) == 1

    def test_no_duplicate_beyond_tolerance(
        self, archive_attrs: list[PopulationAttribute]
    ) -> None:
        archive = Archive(archive_attrs, init_capacity=10, atol=0.01)
        archive.add(x=np.array([1.0, 2.0]), f=0.1)
        # exceeds atol=0.01, so added as a new entry
        idx = archive.add(x=np.array([1.1, 2.1]), f=0.2)
        assert idx == 1
        assert len(archive) == 2

    def test_get_duplicated_population(self, archive: Archive) -> None:
        archive.add(x=np.array([1.0, 2.0]), f=0.1)
        archive.add(x=np.array([3.0, 4.0]), f=0.2)
        archive.add(x=np.array([1.0, 2.0]), f=0.3)  # duplicate

        dup_pop = archive.get_duplicated_population()
        assert len(dup_pop) == 3  # total number of add() calls
        # first 2 are unique (idx 0, 1), 3rd is a duplicate (refers to idx 0)
        np.testing.assert_array_equal(dup_pop.x[0], [1.0, 2.0])
        np.testing.assert_array_equal(dup_pop.x[2], [1.0, 2.0])

    def test_get_knn(self, archive: Archive) -> None:
        archive.add(x=np.array([0.0, 0.0]), f=0.0)
        archive.add(x=np.array([1.0, 0.0]), f=1.0)
        archive.add(x=np.array([0.0, 1.0]), f=2.0)
        archive.add(x=np.array([10.0, 10.0]), f=3.0)

        idx, dist = archive.get_knn(np.array([0.0, 0.0]), k=2)
        assert len(idx) == 2
        assert idx[0] == 0  # nearest is the point itself
        assert dist[0] == pytest.approx(0.0)

    def test_get_knn_empty(self, archive: Archive) -> None:
        idx, dist = archive.get_knn(np.array([0.0, 0.0]), k=3)
        assert len(idx) == 0
        assert len(dist) == 0

    def test_get_knn_k_larger_than_size(self, archive: Archive) -> None:
        archive.add(x=np.array([1.0, 2.0]), f=0.0)
        idx, _dist = archive.get_knn(np.array([1.0, 2.0]), k=100)
        assert len(idx) == 1

    def test_get_knn_kdtree_built_on_demand(self, archive: Archive) -> None:
        archive.add(x=np.array([0.0, 0.0]), f=0.0)
        assert archive._kdtree is None
        archive.get_knn(np.array([0.0, 0.0]), k=1)
        assert archive._kdtree is not None

    def test_get_knn_cache_invalidated_on_add(self, archive: Archive) -> None:
        archive.add(x=np.array([0.0, 0.0]), f=0.0)
        archive.get_knn(np.array([0.0, 0.0]), k=1)
        assert archive._kdtree is not None
        archive.add(x=np.array([1.0, 0.0]), f=1.0)
        assert archive._kdtree is None

    def test_invalid_key_attr_raises(
        self, archive_attrs: list[PopulationAttribute]
    ) -> None:
        with pytest.raises(ValueError, match="not defined"):
            Archive(archive_attrs, key_attr="nonexistent")

    def test_archive_inherits_population(self, archive: Archive) -> None:
        """Archive can use Population methods."""
        archive.add(x=np.array([1.0, 2.0]), f=0.1)
        archive.add(x=np.array([3.0, 4.0]), f=0.2)
        assert len(archive) == 2
        # extract is also available
        sub = archive.extract([0])
        assert len(sub) == 1


# ===========================================================================
# ParetoArchive Tests
# ===========================================================================
class TestParetoArchive:
    """Tests for ParetoArchive (Pareto-non-dominated archive)."""

    def test_add_to_empty(self, pareto_archive: ParetoArchive) -> None:
        idx = pareto_archive.add(x=np.array([0.5, 0.5]), f=np.array([0.0, 1.0]), cv=0.0)
        assert idx == 0
        assert len(pareto_archive) == 1

    def test_add_non_dominated(self, pareto_archive: ParetoArchive) -> None:
        pareto_archive.add(x=np.array([0.0, 0.0]), f=np.array([0.0, 1.0]), cv=0.0)
        idx = pareto_archive.add(x=np.array([1.0, 1.0]), f=np.array([1.0, 0.0]), cv=0.0)
        assert idx >= 0
        assert len(pareto_archive) == 2

    def test_add_dominated_rejected(self, pareto_archive: ParetoArchive) -> None:
        pareto_archive.add(x=np.array([0.0, 0.0]), f=np.array([1.0, 1.0]), cv=0.0)
        idx = pareto_archive.add(x=np.array([1.0, 1.0]), f=np.array([2.0, 2.0]), cv=0.0)
        assert idx == -1
        assert len(pareto_archive) == 1

    def test_add_dominates_existing(self, pareto_archive: ParetoArchive) -> None:
        pareto_archive.add(x=np.array([0.0, 0.0]), f=np.array([2.0, 2.0]), cv=0.0)
        pareto_archive.add(x=np.array([1.0, 1.0]), f=np.array([1.0, 1.0]), cv=0.0)
        assert len(pareto_archive) == 1
        np.testing.assert_array_equal(pareto_archive.get_array("f")[0], [1.0, 1.0])

    def test_add_dominates_multiple(self, pareto_archive: ParetoArchive) -> None:
        pareto_archive.add(x=np.array([0.0, 0.0]), f=np.array([3.0, 3.0]), cv=0.0)
        pareto_archive.add(x=np.array([1.0, 0.0]), f=np.array([4.0, 2.0]), cv=0.0)
        pareto_archive.add(x=np.array([0.0, 1.0]), f=np.array([2.0, 4.0]), cv=0.0)
        pareto_archive.add(x=np.array([0.5, 0.5]), f=np.array([1.0, 1.0]), cv=0.0)
        assert len(pareto_archive) == 1
        np.testing.assert_array_equal(pareto_archive.get_array("f")[0], [1.0, 1.0])

    def test_feasible_dominates_infeasible(self, pareto_archive: ParetoArchive) -> None:
        pareto_archive.add(x=np.array([0.0, 0.0]), f=np.array([5.0, 5.0]), cv=1.0)
        pareto_archive.add(x=np.array([1.0, 1.0]), f=np.array([9.0, 9.0]), cv=0.0)
        assert len(pareto_archive) == 1
        assert float(pareto_archive.get_array("cv")[0]) == pytest.approx(0.0)

    def test_infeasible_rejected_by_feasible(
        self, pareto_archive: ParetoArchive
    ) -> None:
        pareto_archive.add(x=np.array([0.0, 0.0]), f=np.array([1.0, 1.0]), cv=0.0)
        idx = pareto_archive.add(x=np.array([1.0, 1.0]), f=np.array([0.5, 0.5]), cv=1.0)
        assert idx == -1
        assert len(pareto_archive) == 1

    def test_infeasible_lower_cv_wins(self, pareto_archive: ParetoArchive) -> None:
        pareto_archive.add(x=np.array([0.0, 0.0]), f=np.array([1.0, 1.0]), cv=2.0)
        pareto_archive.add(x=np.array([1.0, 1.0]), f=np.array([1.0, 1.0]), cv=1.0)
        assert len(pareto_archive) == 1
        assert float(pareto_archive.get_array("cv")[0]) == pytest.approx(1.0)

    def test_add_from_dict(self, pareto_archive: ParetoArchive) -> None:
        idx = pareto_archive.add(
            {"x": np.array([0.0, 0.0]), "f": np.array([1.0, 2.0]), "cv": 0.0}
        )
        assert idx >= 0

    def test_add_from_kwargs(self, pareto_archive: ParetoArchive) -> None:
        idx = pareto_archive.add(x=np.array([0.0, 0.0]), f=np.array([1.0, 2.0]), cv=0.0)
        assert idx >= 0

    def test_pareto_archive_inherits_population(
        self, pareto_archive: ParetoArchive
    ) -> None:
        """ParetoArchive can use Population methods."""
        pareto_archive.add(x=np.array([0.0, 0.0]), f=np.array([0.0, 1.0]), cv=0.0)
        pareto_archive.add(x=np.array([1.0, 1.0]), f=np.array([1.0, 0.0]), cv=0.0)
        assert len(pareto_archive) == 2
        f_arr = pareto_archive.get_array("f")
        assert f_arr.shape == (2, 2)

    def test_custom_direction_maximize(
        self, moo_attrs: list[PopulationAttribute]
    ) -> None:
        """direction=[1,1] means maximize both objectives."""
        archive = ParetoArchive(
            moo_attrs, init_capacity=20, direction=np.array([1.0, 1.0])
        )
        archive.add(x=np.array([0.0, 0.0]), f=np.array([1.0, 1.0]), cv=0.0)
        archive.add(x=np.array([1.0, 1.0]), f=np.array([2.0, 2.0]), cv=0.0)
        assert len(archive) == 1
        np.testing.assert_array_equal(archive.get_array("f")[0], [2.0, 2.0])


# ===========================================================================
# Population with int dtype / default=0 Tests
# ===========================================================================
class TestPopulationIntDtype:
    """Tests for Population with integer-typed attributes."""

    def test_int_attribute(self) -> None:
        attrs = [
            PopulationAttribute(name="label", dtype=np.int32, shape=(), default=0),
        ]
        pop = Population(attrs, init_capacity=5)
        pop.append(label=42)
        assert pop.get_array("label")[0] == 42

    def test_int_default_zero(self) -> None:
        attrs = [
            PopulationAttribute(name="count", dtype=np.int64, shape=(), default=0),
        ]
        pop = Population(attrs, init_capacity=5)
        pop.append()
        assert pop.get_array("count")[0] == 0


# ===========================================================================
# Population Resize Tests
# ===========================================================================
class TestPopulationResize:
    """Tests for internal resize behavior."""

    def test_data_preserved_after_resize(self) -> None:
        attrs = [
            PopulationAttribute(name="x", dtype=np.float64, shape=(2,)),
        ]
        pop = Population(attrs, init_capacity=2)
        pop.append(x=np.array([1.0, 2.0]))
        pop.append(x=np.array([3.0, 4.0]))
        # 3rd append triggers resize
        pop.append(x=np.array([5.0, 6.0]))
        assert len(pop) == 3
        np.testing.assert_array_equal(pop.x[0], [1.0, 2.0])
        np.testing.assert_array_equal(pop.x[1], [3.0, 4.0])
        np.testing.assert_array_equal(pop.x[2], [5.0, 6.0])


# ===========================================================================
# Cache Tests
# ===========================================================================
class TestPopulationCache:
    """Tests for Population cache mechanism (set_cache / get_cache)."""

    def test_set_and_get_cache(self, populated_pop: Population) -> None:
        """Basic cache round-trip."""
        populated_pop.set_cache("rank", [0, 1, 2, 3, 4])
        assert populated_pop.get_cache("rank") == [0, 1, 2, 3, 4]

    def test_get_cache_missing_key(self, pop: Population) -> None:
        """get_cache returns None for a missing key."""
        assert pop.get_cache("nonexistent") is None

    def test_cache_cleared_on_mod_value(self, populated_pop: Population) -> None:
        """Cache is cleared when mod_value() is called."""
        populated_pop.set_cache("rank", [0, 1, 2, 3, 4])
        populated_pop.mod_value()
        assert populated_pop.get_cache("rank") is None

    def test_cache_cleared_on_mod_structure(self, populated_pop: Population) -> None:
        """Cache is cleared when mod_structure() is called."""
        populated_pop.set_cache("rank", [0, 1, 2, 3, 4])
        populated_pop.mod_structure()
        assert populated_pop.get_cache("rank") is None

    def test_cache_cleared_on_append(self, populated_pop: Population) -> None:
        """Cache is cleared when a new individual is appended."""
        populated_pop.set_cache("cd", np.ones(5))
        populated_pop.append(x=np.array([9.0, 9.0, 9.0]), f=9.0)
        assert populated_pop.get_cache("cd") is None

    def test_cache_cleared_on_extend(self, populated_pop: Population) -> None:
        """Cache is cleared when the population is extended."""
        populated_pop.set_cache("cd", np.ones(5))
        other = populated_pop.empty_like()
        other.append(x=np.array([9.0, 9.0, 9.0]), f=9.0)
        populated_pop.extend(other)
        assert populated_pop.get_cache("cd") is None

    def test_cache_cleared_on_delete(self, populated_pop: Population) -> None:
        """Cache is cleared when individuals are deleted."""
        populated_pop.set_cache("rank", [0, 1, 2, 3, 4])
        populated_pop.delete(0)
        assert populated_pop.get_cache("rank") is None

    def test_cache_cleared_on_truncate(self, populated_pop: Population) -> None:
        """Cache is cleared when the population is truncated."""
        populated_pop.set_cache("rank", [0, 1, 2, 3, 4])
        populated_pop.truncate(3)
        assert populated_pop.get_cache("rank") is None

    def test_cache_cleared_on_reorder(self, populated_pop: Population) -> None:
        """Cache is cleared when the population is reordered."""
        populated_pop.set_cache("rank", [0, 1, 2, 3, 4])
        populated_pop.reorder(np.array([4, 3, 2, 1, 0]))
        assert populated_pop.get_cache("rank") is None

    def test_cache_cleared_on_clear(self, populated_pop: Population) -> None:
        """Cache is cleared when the population is cleared."""
        populated_pop.set_cache("rank", [0, 1, 2, 3, 4])
        populated_pop.clear()
        assert populated_pop.get_cache("rank") is None

    def test_cache_cleared_on_update_array(self, populated_pop: Population) -> None:
        """Cache is cleared when update_array is called."""
        populated_pop.set_cache("rank", [0, 1, 2, 3, 4])
        populated_pop.update_array("f", np.zeros(5))
        assert populated_pop.get_cache("rank") is None

    def test_cache_cleared_on_dot_setter(self, populated_pop: Population) -> None:
        """Cache is cleared when a value is set via dot access."""
        populated_pop.set_cache("rank", [0, 1, 2, 3, 4])
        populated_pop.f = np.zeros(5)
        assert populated_pop.get_cache("rank") is None

    def test_cache_cleared_on_individual_setattr(
        self, populated_pop: Population
    ) -> None:
        """Cache is cleared when an Individual modifies a value."""
        populated_pop.set_cache("rank", [0, 1, 2, 3, 4])
        ind = populated_pop[0]
        ind.f = 999.0
        assert populated_pop.get_cache("rank") is None

    def test_cache_not_inherited_by_extract(self, populated_pop: Population) -> None:
        """Extracted population does not inherit cache from the parent."""
        populated_pop.set_cache("rank", [0, 1, 2, 3, 4])
        sub = populated_pop.extract([0, 1])
        assert sub.get_cache("rank") is None

    def test_cache_not_inherited_by_empty_like(self, populated_pop: Population) -> None:
        """empty_like population does not inherit cache."""
        populated_pop.set_cache("rank", [0, 1, 2, 3, 4])
        new_pop = populated_pop.empty_like()
        assert new_pop.get_cache("rank") is None

    def test_cache_overwrite(self, populated_pop: Population) -> None:
        """Setting the same key twice overwrites the previous value."""
        populated_pop.set_cache("rank", [0, 1, 2, 3, 4])
        populated_pop.set_cache("rank", [4, 3, 2, 1, 0])
        assert populated_pop.get_cache("rank") == [4, 3, 2, 1, 0]

    def test_multiple_cache_keys(self, populated_pop: Population) -> None:
        """Multiple independent cache keys can coexist."""
        populated_pop.set_cache("rank", [0, 1, 2])
        populated_pop.set_cache("cd", [0.5, 0.3, 0.1])
        assert populated_pop.get_cache("rank") == [0, 1, 2]
        assert populated_pop.get_cache("cd") == [0.5, 0.3, 0.1]

    def test_all_cache_keys_cleared_on_mutation(
        self, populated_pop: Population
    ) -> None:
        """All cache keys are cleared on a single mutation."""
        populated_pop.set_cache("rank", [0, 1, 2])
        populated_pop.set_cache("cd", [0.5, 0.3, 0.1])
        populated_pop.mod_value()
        assert populated_pop.get_cache("rank") is None
        assert populated_pop.get_cache("cd") is None


# ===========================================================================
# ParetoMixin.add() fast-path differential tests (#224)
# ===========================================================================
#
# The fast path (a vectorized broadcast, using Dominator.dominates_many) and
# the original per-row Python loop must produce byte-identical accept/reject
# and survivor decisions.  The functions below are standalone, independent
# transcriptions of the *pre-fast-path* add() logic (they call only
# `dominator.dominates`/`dominator.dominates_many` and plain Python/NumPy —
# never ParetoMixin.add() itself) so that comparing against them is a genuine
# differential test, not the new code checked against itself.


def _loop_new_dominates_existing(
    f_a: np.ndarray | None,
    cv_a: float,
    f_b: np.ndarray | None,
    cv_b: float,
    dominator: Dominator,
    direction: np.ndarray | None,
    eps_cv: float,
) -> bool:
    """Standalone transcription of the pre-#224 ParetoMixin._new_dominates_existing."""
    a_feasible = cv_a <= eps_cv
    b_feasible = cv_b <= eps_cv
    if a_feasible and not b_feasible:
        return True
    if not a_feasible and b_feasible:
        return False
    if a_feasible and b_feasible:
        if f_a is None:
            return False
        if f_b is None:
            return True
        return bool(dominator.dominates(f_a, f_b, direction))
    return cv_a < cv_b


def _loop_masks(
    f_new: np.ndarray | None,
    cv_new: float,
    f_arr: np.ndarray | None,
    cv_arr: np.ndarray | None,
    dominator: Dominator,
    direction: np.ndarray | None,
    eps_cv: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Standalone transcription of the pre-#224 two-loop add() comparison."""
    if f_arr is not None:
        n = f_arr.shape[0]
    elif cv_arr is not None:
        n = cv_arr.shape[0]
    else:
        n = 0
    existing_dominates_new = np.zeros(n, dtype=bool)
    new_dominates_existing = np.zeros(n, dtype=bool)
    for i in range(n):
        f_ex = f_arr[i] if f_arr is not None else None
        cv_ex = float(cv_arr[i]) if cv_arr is not None else 0.0
        existing_dominates_new[i] = _loop_new_dominates_existing(
            f_ex, cv_ex, f_new, cv_new, dominator, direction, eps_cv
        )
        new_dominates_existing[i] = _loop_new_dominates_existing(
            f_new, cv_new, f_ex, cv_ex, dominator, direction, eps_cv
        )
    return existing_dominates_new, new_dominates_existing


def _fast_masks(
    f_new: np.ndarray,
    cv_new: float,
    f_arr: np.ndarray,
    cv_arr: np.ndarray | None,
    dominator: Dominator,
    direction: np.ndarray | None,
    eps_cv: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Standalone transcription of ParetoMixin.add()'s vectorized fast-path formula."""
    n = f_arr.shape[0]
    cv_ex_arr = cv_arr.astype(float) if cv_arr is not None else np.zeros(n)
    new_feasible = np.bool_(cv_new <= eps_cv)
    ex_feasible = cv_ex_arr <= eps_cv

    existing_dominates_new = np.zeros(n, dtype=bool)
    new_dominates_existing = np.zeros(n, dtype=bool)

    existing_dominates_new |= (~new_feasible) & ex_feasible
    new_dominates_existing |= new_feasible & (~ex_feasible)

    both_infeasible = (~new_feasible) & (~ex_feasible)
    existing_dominates_new |= both_infeasible & (cv_ex_arr < cv_new)
    new_dominates_existing |= both_infeasible & (cv_new < cv_ex_arr)

    both_feasible = new_feasible & ex_feasible
    if np.any(both_feasible):
        # Caller (_check_scenario) only invokes this helper when the
        # dominator overrides dominates_many; ty can't see that invariant
        # across the function boundary, hence the ignore (same convention as
        # ParetoMixin.add() elsewhere in this repo).
        new_dom, ex_dom = dominator.dominates_many(  # ty: ignore[not-iterable]
            f_new, f_arr, direction
        )
        new_dominates_existing |= both_feasible & new_dom
        existing_dominates_new |= both_feasible & ex_dom
    return existing_dominates_new, new_dominates_existing


def _build_raw_archive(
    n: int,
    f_arr: np.ndarray | None,
    cv_arr: np.ndarray | None,
    direction: np.ndarray | None = None,
    dominator: Dominator | None = None,
    eps_cv: float = 0.0,
) -> ParetoArchive:
    """
    Build a ParetoArchive pre-filled with *n* raw rows.

    Uses the inherited ``Population.append`` (not ``ParetoMixin.add``) so the
    existing rows are placed exactly as given, bypassing the dominance-
    preserving filter — required so the differential tests can exercise
    otherwise-invariant-violating combinations (ties, mutually dominated
    rows, etc.) directly.
    """
    attrs = [PopulationAttribute(name="x", dtype=np.float64, shape=(1,))]
    if f_arr is not None:
        attrs.append(
            PopulationAttribute(name="f", dtype=np.float64, shape=(f_arr.shape[1],))
        )
    if cv_arr is not None:
        attrs.append(PopulationAttribute(name="cv", dtype=np.float64, shape=()))
    archive = ParetoArchive(
        attrs,
        init_capacity=max(n + 5, 10),
        direction=direction,
        dominator=dominator,
        eps_cv=eps_cv,
    )
    for i in range(n):
        kwargs: dict = {"x": np.array([float(i)])}
        if f_arr is not None:
            kwargs["f"] = f_arr[i]
        if cv_arr is not None:
            kwargs["cv"] = float(cv_arr[i])
        archive.append(**kwargs)
    return archive


def _check_scenario(
    n: int,
    f_arr: np.ndarray | None,
    cv_arr: np.ndarray | None,
    f_new: np.ndarray | None,
    cv_new: float,
    dominator: Dominator,
    direction: np.ndarray | None = None,
    eps_cv: float = 0.0,
) -> None:
    """
    Core differential check, shared by every scenario below.

    1. Computes the independent loop-based reference masks.
    2. If the fast-path preconditions hold, also computes the standalone
       fast-path formula and asserts it matches the reference exactly.
    3. Runs the *real* ``ParetoArchive.add()`` end-to-end and asserts its
       return index and the archive's post-add contents match what the
       reference masks predict — this is what actually proves the production
       fast path (not just its standalone transcription above) is correct.
    """
    ref_existing_dom, ref_new_dom = _loop_masks(
        f_new, cv_new, f_arr, cv_arr, dominator, direction, eps_cv
    )

    has_nan = (f_new is not None and np.any(np.isnan(f_new))) or (
        f_arr is not None and np.any(np.isnan(f_arr))
    )
    dominates_many_supported = (
        type(dominator).dominates_many is not Dominator.dominates_many
    )
    can_use_fast = (
        dominates_many_supported
        and f_arr is not None
        and f_new is not None
        and not has_nan
    )
    if can_use_fast:
        fast_existing_dom, fast_new_dom = _fast_masks(
            f_new, cv_new, f_arr, cv_arr, dominator, direction, eps_cv
        )
        np.testing.assert_array_equal(ref_existing_dom, fast_existing_dom)
        np.testing.assert_array_equal(ref_new_dom, fast_new_dom)

    archive = _build_raw_archive(n, f_arr, cv_arr, direction, dominator, eps_cv)
    kwargs_new: dict = {"x": np.array([999.0])}
    if f_new is not None:
        kwargs_new["f"] = f_new
    kwargs_new["cv"] = cv_new

    real_idx = archive.add(**kwargs_new)

    if np.any(ref_existing_dom):
        assert real_idx == -1
        assert len(archive) == n
        if f_arr is not None:
            np.testing.assert_array_equal(archive.get_array("f"), f_arr)
        if cv_arr is not None:
            np.testing.assert_array_equal(archive.get_array("cv"), cv_arr)
    else:
        survivors = ~ref_new_dom
        n_survivors = int(np.sum(survivors))
        # add() computes new_idx *after* deleting dominated existing rows, so
        # the assigned index is the survivor count, not the original size n.
        assert real_idx == n_survivors
        assert len(archive) == n_survivors + 1
        if f_arr is not None:
            np.testing.assert_array_equal(
                archive.get_array("f")[:n_survivors], f_arr[survivors]
            )
            if f_new is not None:
                np.testing.assert_array_equal(
                    archive.get_array("f")[n_survivors], f_new
                )
        if cv_arr is not None:
            np.testing.assert_array_equal(
                archive.get_array("cv")[:n_survivors], cv_arr[survivors]
            )
            assert archive.get_array("cv")[n_survivors] == pytest.approx(
                cv_new, nan_ok=True
            )


class TestParetoMixinFastPathEquivalence:
    """
    Differential tests: ``ParetoMixin.add()``'s vectorized fast path vs the
    original per-row Python loop (#224).

    Every scenario asserts exact (not approximate) agreement — dominance
    decisions are pure deterministic math, so there is no RNG-order excuse.
    """

    # -----------------------------------------------------------------------
    # 1. Randomized battery: both feasible/infeasible mixes, ties, boundary
    #    cv==eps_cv, direction combinations, ParetoDominator/EpsilonDominator.
    # -----------------------------------------------------------------------
    @pytest.mark.parametrize("n", [1, 2, 5, 20])
    @pytest.mark.parametrize("eps_cv", [0.0, 0.5])
    @pytest.mark.parametrize(
        "direction_kind", ["none", "all_minimize", "all_maximize", "mixed"]
    )
    @pytest.mark.parametrize(
        "dominator_kind",
        ["pareto", "epsilon"],
    )
    def test_random_battery(
        self, n: int, eps_cv: float, direction_kind: str, dominator_kind: str
    ) -> None:
        m = 3
        if direction_kind == "none":
            direction = None
        elif direction_kind == "all_minimize":
            direction = -np.ones(m)
        elif direction_kind == "all_maximize":
            direction = np.ones(m)
        else:
            direction = np.array([1.0, -1.0, 1.0])
        dominator: Dominator = (
            ParetoDominator() if dominator_kind == "pareto" else EpsilonDominator(0.3)
        )

        # Fixed (not hash()-based) per-case seed: Python's hash() on str/tuple
        # is salted per-process (PYTHONHASHSEED), which would make a failing
        # seed non-reproducible across runs -- exactly the wrong property for
        # a differential test.
        direction_offset = {
            "none": 0,
            "all_minimize": 11,
            "all_maximize": 23,
            "mixed": 37,
        }[direction_kind]
        dominator_offset = {"pareto": 0, "epsilon": 101}[dominator_kind]
        seed = int(n * 1000 + eps_cv * 100 + direction_offset + dominator_offset)
        rng = np.random.default_rng(seed)
        for _ in range(10):
            # cv values deliberately include the eps_cv boundary and ties.
            cv_pool = np.array([0.0, eps_cv, eps_cv + 0.1, 1.0, 1.0, 2.0])
            cv_arr = rng.choice(cv_pool, size=n)
            cv_new = float(rng.choice(cv_pool))

            f_arr = rng.integers(0, 3, size=(n, m)).astype(float)  # heavy tie rate
            f_new = rng.integers(0, 3, size=m).astype(float)
            # Occasionally force an exact tie with an existing row.
            if n > 0 and rng.random() < 0.3:
                f_new = f_arr[0].copy()

            _check_scenario(
                n, f_arr, cv_arr, f_new, cv_new, dominator, direction, eps_cv
            )

    # -----------------------------------------------------------------------
    # 2. Archive size 0 and size 1
    # -----------------------------------------------------------------------
    def test_size_zero(self) -> None:
        """Empty archive: comparison block is skipped entirely by both paths."""
        f_arr = np.zeros((0, 2))
        cv_arr = np.zeros(0)
        _check_scenario(0, f_arr, cv_arr, np.array([1.0, 1.0]), 0.0, ParetoDominator())

    def test_size_one(self) -> None:
        f_arr = np.array([[1.0, 1.0]])
        cv_arr = np.array([0.0])
        _check_scenario(1, f_arr, cv_arr, np.array([2.0, 2.0]), 0.0, ParetoDominator())
        _check_scenario(1, f_arr, cv_arr, np.array([0.5, 0.5]), 0.0, ParetoDominator())

    # -----------------------------------------------------------------------
    # 3. Schema without "f" and/or "cv" — fast path must decline
    # -----------------------------------------------------------------------
    def test_no_f_attribute_wipes_archive(self) -> None:
        """
        No "f" schema attribute at all -> f_arr is None -> loop fallback.

        Pre-existing (not-a-bug) behavior: the new solution, having an
        objective value while the existing rows structurally cannot, is
        treated as dominating every existing row (per
        ``_new_dominates_existing``'s ``f_ex is None -> return True`` rule).
        """
        n = 3
        cv_arr = np.zeros(n)
        archive = _build_raw_archive(n, None, cv_arr, dominator=ParetoDominator())
        idx = archive.add(x=np.array([999.0]), f=np.array([1.0, 1.0]), cv=0.0)
        assert idx == 0  # every existing row wiped, single survivor at index 0
        assert len(archive) == 1

    def test_no_f_and_no_cv_attribute(self) -> None:
        """Neither "f" nor "cv" in schema -> loop fallback, same wipe behavior."""
        n = 3
        archive = _build_raw_archive(n, None, None, dominator=ParetoDominator())
        idx = archive.add(x=np.array([999.0]), f=np.array([1.0, 1.0]), cv=0.0)
        assert idx == 0
        assert len(archive) == 1

    def test_cv_only_missing_still_uses_fast_path(self) -> None:
        """ "f" present but "cv" absent from schema: fast path still engages."""
        n = 5
        f_arr = np.array([[3.0, 3.0], [4.0, 2.0], [2.0, 4.0], [1.0, 1.0], [5.0, 5.0]])
        dominator = ParetoDominator()
        counting = _CountingDominator(dominator)
        archive = _build_raw_archive(n, f_arr, None, dominator=counting)
        f_new = np.array([0.5, 0.5])
        idx = archive.add(x=np.array([999.0]), f=f_new, cv=0.0)
        assert counting.calls > 0  # fast path (dominates_many) was engaged
        assert idx == 0
        assert len(archive) == 1
        np.testing.assert_array_equal(archive.get_array("f")[0], f_new)

    def test_all_infeasible_skips_dominates_many_call(self) -> None:
        """
        When every existing row and the new candidate are infeasible, the
        fast path takes the "both feasible" short-circuit (empty mask) and
        never calls dominates_many -- exercises that branch explicitly,
        not just via the masked-out-either-way argument.
        """
        n = 4
        f_arr = np.array([[3.0, 3.0], [4.0, 2.0], [2.0, 4.0], [1.0, 1.0]])
        cv_arr = np.full(n, 1.0)  # all infeasible (eps_cv=0.0)
        dominator = ParetoDominator()
        counting = _CountingDominator(dominator)
        archive = _build_raw_archive(n, f_arr, cv_arr, dominator=counting)
        f_new = np.array([0.5, 0.5])
        idx = archive.add(x=np.array([999.0]), f=f_new, cv=2.0)  # also infeasible
        assert counting.calls == 0  # both_feasible mask empty -> never called
        # Lower cv wins among infeasible solutions: new (cv=2.0) is worse
        # than every existing row (cv=1.0) -> rejected.
        assert idx == -1
        assert len(archive) == n

    # -----------------------------------------------------------------------
    # 4. f_new is None (all-NaN new candidate) -> falls back to loop
    # -----------------------------------------------------------------------
    def test_f_new_all_nan_falls_back(self) -> None:
        n = 4
        f_arr = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [0.5, 0.5]])
        cv_arr = np.zeros(n)
        dominator = ParetoDominator()
        counting = _CountingDominator(dominator)
        archive = _build_raw_archive(n, f_arr, cv_arr, dominator=counting)
        idx = archive.add(x=np.array([999.0]), f=np.array([np.nan, np.nan]), cv=0.0)
        assert counting.calls == 0  # fast path never engaged (f_new is None)
        # Both feasible, f_new is None -> _new_dominates_existing always False
        # for "new dominates existing"; existing never dominates a None-f new
        # candidate either (new_feasible and not ex_feasible never True here,
        # both feasible -> f_a(existing) not None, f_b(new) None -> dominator
        # branch: f_a is None? no; f_b is None -> "if f_b is None: return True"
        # only applies when checking "existing dominates new" with f_a=f_ex,
        # f_b=f_new=None -> existing dominates new is True for every row.
        assert idx == -1
        assert len(archive) == n

    # -----------------------------------------------------------------------
    # 5. Partial-NaN (not all-NaN) values -> has_nan catches it, loop fallback
    # -----------------------------------------------------------------------
    def test_partial_nan_in_f_new_falls_back(self) -> None:
        n = 3
        f_arr = np.array([[3.0, 3.0], [1.0, 1.0], [5.0, 5.0]])
        cv_arr = np.zeros(n)
        f_new = np.array([1.0, np.nan])  # not all-NaN -> _extract_fv keeps it
        dominator = ParetoDominator()
        counting = _CountingDominator(dominator)

        has_nan = np.any(np.isnan(f_new)) or np.any(np.isnan(f_arr))
        assert has_nan  # sanity: has_nan correctly flags a partial-NaN row

        archive = _build_raw_archive(n, f_arr, cv_arr, dominator=counting)
        idx = archive.add(x=np.array([999.0]), f=f_new, cv=0.0)
        assert counting.calls == 0  # fast path declined due to NaN

        ref_existing_dom, ref_new_dom = _loop_masks(
            f_new, 0.0, f_arr, cv_arr, dominator, None, 0.0
        )
        if np.any(ref_existing_dom):
            assert idx == -1
            assert len(archive) == n
        else:
            n_survivors = n - int(np.sum(ref_new_dom))
            assert idx == n_survivors
            assert len(archive) == n_survivors + 1

    def test_partial_nan_in_existing_row_falls_back(self) -> None:
        n = 3
        f_arr = np.array([[3.0, np.nan], [1.0, 1.0], [5.0, 5.0]])
        cv_arr = np.zeros(n)
        f_new = np.array([0.5, 0.5])
        dominator = ParetoDominator()
        counting = _CountingDominator(dominator)

        has_nan = np.any(np.isnan(f_new)) or np.any(np.isnan(f_arr))
        assert has_nan

        archive = _build_raw_archive(n, f_arr, cv_arr, dominator=counting)
        idx = archive.add(x=np.array([999.0]), f=f_new, cv=0.0)
        assert counting.calls == 0  # fast path declined due to NaN

        ref_existing_dom, ref_new_dom = _loop_masks(
            f_new, 0.0, f_arr, cv_arr, dominator, None, 0.0
        )
        if np.any(ref_existing_dom):
            assert idx == -1
        else:
            n_survivors = n - int(np.sum(ref_new_dom))
            assert idx == n_survivors
            assert len(archive) == n_survivors + 1

    # -----------------------------------------------------------------------
    # 6. Custom Dominator subclass without dominates_many -> loop fallback
    # -----------------------------------------------------------------------
    def test_custom_dominator_without_dominates_many_falls_back(self) -> None:
        """A Dominator that only implements dominance_matrix forces the loop."""

        class MinimalDominator(Dominator):
            def dominance_matrix(self, f, direction=None):
                return ParetoDominator().dominance_matrix(f, direction)

        n = 4
        f_arr = np.array([[3.0, 3.0], [4.0, 2.0], [2.0, 4.0], [1.0, 1.0]])
        cv_arr = np.zeros(n)
        f_new = np.array([0.5, 0.5])
        dominator = MinimalDominator()

        _check_scenario(n, f_arr, cv_arr, f_new, 0.0, dominator)

        # And confirm it really was accepted/dominates as ordinary Pareto would.
        archive = _build_raw_archive(n, f_arr, cv_arr, dominator=dominator)
        idx = archive.add(x=np.array([999.0]), f=f_new, cv=0.0)
        assert idx == 0
        assert len(archive) == 1

    # -----------------------------------------------------------------------
    # 7. NaN cv comparisons: False in both loop and fast path (IEEE 754)
    # -----------------------------------------------------------------------
    def test_cv_nan_comparisons_agree(self) -> None:
        """NaN cv values compare False both ways in Python floats and NumPy."""
        cv_new = float("nan")
        cv_ex = 1.0
        assert not (cv_new <= 0.0)  # new "feasible" check: False
        assert not (cv_ex < cv_new)  # infeasible-tie check: False
        assert not (cv_new < cv_ex)  # infeasible-tie check: False
        cv_ex_arr = np.array([1.0])
        assert not np.any(cv_ex_arr < cv_new)
        assert not np.any(np.array([cv_new]) < cv_ex_arr)

        # And an end-to-end differential check with cv_new = NaN.
        n = 2
        f_arr = np.array([[1.0, 1.0], [2.0, 2.0]])
        cv_arr = np.array([0.0, 1.0])
        _check_scenario(
            n, f_arr, cv_arr, np.array([1.5, 1.5]), cv_new, ParetoDominator()
        )


class _CountingDominator(Dominator):
    """Wraps a Dominator, counting dominates_many calls (for fallback proofs)."""

    def __init__(self, inner: Dominator) -> None:
        self._inner = inner
        self.calls = 0

    def dominance_matrix(self, f, direction=None):
        return self._inner.dominance_matrix(f, direction)

    def dominates(self, fa, fb, direction=None):
        return self._inner.dominates(fa, fb, direction)

    def dominates_many(self, fa, f_matrix, direction=None):
        self.calls += 1
        return self._inner.dominates_many(fa, f_matrix, direction)


# ===========================================================================
# ParetoMixin dispatch-consistency regression (Issue #224 follow-up fix)
# ===========================================================================
#
# A Dominator subclass that overrides only `dominance_matrix()` (and thereby
# `dominates()`, which is derived from it in the base class) while leaving
# `dominates_many()` inherited unchanged must NOT have its stale
# `dominates_many()` used by ParetoArchive.add() -- doing so would silently
# ignore the subclass's overridden dominance semantics. See
# batch_override_is_consistent (saealib._dispatch).


class _ReverseDominanceMatrixDominator(ParetoDominator):
    """Overrides only ``dominance_matrix`` (reversing the relation via
    transpose, mirroring test_moo.py's ``ReverseParetoDominator``);
    ``dominates_many`` is inherited unchanged from ``ParetoDominator`` and
    is therefore stale/inconsistent with the overridden dominance_matrix."""

    def dominance_matrix(self, f, direction=None):
        return super().dominance_matrix(f, direction).T


class TestParetoMixinDispatchConsistency:
    def test_add_uses_scalar_dominates_not_stale_dominates_many(self):
        dominator = _ReverseDominanceMatrixDominator()
        # Sanity: this dominator's dominates_many is indeed inherited
        # unchanged (the scenario this fix targets).
        assert type(dominator).dominates_many is ParetoDominator.dominates_many
        # ... yet dominance_matrix (and therefore dominates(), derived from
        # it) IS overridden, so the two disagree for this pair.
        f_arr = np.array([[1.0, 1.0]])
        f_new = np.array([[2.0, 2.0]])
        # Normal Pareto semantics: f_new=[2,2] is dominated by f_arr=[1,1].
        # dominates_many (stale, forward semantics) would say f_arr
        # dominates f_new.
        fa_dom, dom_fa = ParetoDominator().dominates_many(f_new[0], f_arr)
        assert not fa_dom.any() and dom_fa.any()
        # But this dominator's overridden (reversed) dominance_matrix flips
        # that: dominates() now says f_new dominates f_arr.
        assert dominator.dominates(f_new[0], f_arr[0]) is True
        assert dominator.dominates(f_arr[0], f_new[0]) is False

        archive = _build_raw_archive(1, f_arr, None, None, dominator, 0.0)
        real_idx = archive.add(x=np.array([999.0]), f=f_new[0])

        # Scalar (dominates()-based) reference: f_new dominates the sole
        # existing row, so it survives and the existing row is evicted.
        ref_existing_dom, ref_new_dom = _loop_masks(
            f_new[0], 0.0, f_arr, None, dominator, None, 0.0
        )
        assert not ref_existing_dom.any()
        assert ref_new_dom.all()
        assert real_idx == 0  # existing row evicted, f_new takes index 0
        assert len(archive) == 1
        np.testing.assert_array_equal(archive.get_array("f")[0], f_new[0])

        # Had ParetoArchive.add() dispatched to the stale dominates_many
        # instead (the pre-fix bug), it would have rejected f_new (real_idx
        # == -1) and kept the original row -- the opposite outcome.
