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
"""

import warnings

import numpy as np
import pytest

from saealib.population import (
    Archive,
    Individual,
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
            attr.name = "y"


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
            schema["new_key"] = None

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
            _ = populated_pop["invalid"]


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
