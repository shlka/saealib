"""Population and individual container classes."""

from __future__ import annotations

import warnings
import weakref
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

import numpy as np
from typing_extensions import Self

from saealib.exceptions import ValidationError
from saealib.population.genome import DenseVectorBatch, GenomeBatch, ObjectBatch
from saealib.space.services import DenseNumericView

if TYPE_CHECKING:
    pass

T_Population = TypeVar("T_Population", bound="Population")
T_Individual = TypeVar("T_Individual", bound="Individual")
_T_Default = TypeVar("_T_Default")

CandidateIds = np.ndarray
ColumnStore = Mapping[str, np.ndarray]


@dataclass(frozen=True)
class PopulationAttribute:
    """
    Population attribute definition.

    Attributes
    ----------
    name : str
        Name of the attribute.
    dtype : Type | np.dtype
        Data type of the attribute.
    shape : Tuple[int, ...]
        Shape of the attribute.
    default : Any
        Default value for the attribute.
    """

    name: str
    dtype: type | np.dtype
    shape: tuple[int, ...] = ()
    default: Any = np.nan


class _ConflictBypassProperty(property):
    """Mark bound properties as exempt from population-name conflict warnings."""

    pass


class _LegacyDenseNumericView:
    """Dense view used only by the pre-genome Population constructor path."""

    _canonical_identity_backing = True

    def get_view(self, genomes: GenomeBatch) -> np.ndarray:
        if not isinstance(genomes, DenseVectorBatch):
            raise ValidationError("DenseNumericView requires DenseVectorBatch")
        return genomes.array


def bind_property(key: str, doc: str = "") -> Any:
    """Make property for Individual attributes (helper function)."""

    def fget(self):
        return self.get_readonly_value(key)

    def fset(self, value):
        self.update_value(key, value)

    return _ConflictBypassProperty(fget, fset, doc=doc)


def bind_property_array(key: str, doc: str = "") -> Any:
    """Make property for Population attributes (helper function)."""

    def fget(self):
        return self.get_readonly_array(key)

    def fset(self, value):
        self.update_array(key, value)

    return _ConflictBypassProperty(fget, fset, doc=doc)


class Population(Generic[T_Individual]):
    """
    Container for population data.

    Attributes
    ----------
    schema : Dict[str, PopulationAttribute]
        Schema defining the attributes of the population.
    _data : Dict[str, np.ndarray]
        Dictionary to store population data arrays.
    _cache : Dict[str, Any]
        Dictionary to store cached values (ex: nds_rank).
        Cleared on every value or structure modification.
    _capacity : int
        Current capacity of the population.
    _size : int
        Current size of the population.
    _structure_version : int
        Version number to track structure modifications.
    _value_version : int
        Version number to track value modifications.
    """

    individual_class = None

    # Reserve standard expressions
    x: np.ndarray = bind_property_array("x", doc="Design variables")
    f: np.ndarray = bind_property_array("f", doc="Objective function values")
    g: np.ndarray = bind_property_array("g", doc="Constraint values")
    cv: np.ndarray = bind_property_array("cv", doc="Constraint violation")

    def __init__(
        self,
        attrs: list[PopulationAttribute],
        init_capacity: int = 100,
        *,
        genomes: GenomeBatch | None = None,
        dense_numeric_view: DenseNumericView | None = None,
        dense_view: DenseNumericView | None = None,
    ) -> None:
        """
        Initialize a Population.

        Parameters
        ----------
        attrs : List[PopulationAttribute]
            List of population attributes.
            Each attribute defines a column in the population.
        init_capacity : int, optional
            Initial capacity of the population, by default 100.
        """
        if genomes is not None:
            init_capacity = max(init_capacity, len(genomes))
        self._capacity = init_capacity
        self._size = 0
        self._structure_version = 0
        self._value_version = 0
        self._data: dict[str, np.ndarray] = {}
        self._cache: dict[Hashable, Any] = {}
        for attr in attrs:
            self._init_column(attr, self._capacity)
        self._schema = {attr.name: attr for attr in attrs}
        self._dense_genomes_view_cache: tuple[int, int, DenseVectorBatch] | None = None
        self._dense_numeric_view = (
            dense_numeric_view if dense_numeric_view is not None else dense_view
        )
        self._genome_items: list[object] | None = None
        self._legacy_scalar_x = False
        if genomes is not None:
            self._initialize_genomes(genomes)
        elif "x" in self._schema:
            # The legacy attrs=[..., PopulationAttribute("x", ...)] API is
            # still used by archives and older user factories.  Its x array
            # remains the compatibility backing store and is dense by nature.
            if self._schema["x"].shape == ():
                self._legacy_scalar_x = True
                self._genome_items = []
                self._genome_batch = ObjectBatch()
            else:
                if self._dense_numeric_view is None:
                    self._dense_numeric_view = _LegacyDenseNumericView()
                self._genome_batch = DenseVectorBatch(self._data["x"])
        else:
            self._genome_items = []
            self._genome_batch = ObjectBatch()
        self._check_name_conflicts()

    def _initialize_genomes(self, genomes: GenomeBatch) -> None:
        """Install an independent, population-owned genome backing store."""
        if len(genomes) > self._capacity:
            self._capacity = len(genomes)
        if isinstance(genomes, DenseVectorBatch):
            if self._dense_numeric_view is None:
                raise ValidationError(
                    "DenseVectorBatch genomes require a resolved DenseNumericView"
                )
            view = np.asarray(self._dense_numeric_view.get_view(genomes))
            if view.ndim != 2 or view.dtype != np.float64:
                raise ValidationError(
                    "DenseNumericView must return a 2-D float64 array"
                )
            storage = np.array(view, dtype=np.float64, order="C", copy=True)
            if "x" in self._schema:
                attr = self._schema["x"]
                if tuple(attr.shape) != (storage.shape[1],):
                    raise ValidationError("genome dimension does not match x column")
                self._data["x"][: len(storage)] = storage
                self._genome_batch = DenseVectorBatch(self._data["x"])
            else:
                self._data["x"] = np.full(
                    (self._capacity, storage.shape[1]), np.nan, dtype=np.float64
                )
                self._data["x"][: len(storage)] = storage
                self._genome_batch = DenseVectorBatch(self._data["x"])
                self._schema.setdefault(
                    "x",
                    PopulationAttribute(
                        name="x", dtype=np.float64, shape=(storage.shape[1],)
                    ),
                )
            self._size = len(storage)
        elif isinstance(genomes, ObjectBatch):
            self._genome_items = list(genomes.items)
            self._size = len(self._genome_items)
            self._genome_batch = ObjectBatch(self._genome_items)
        else:
            # Custom GenomeBatch values are retained as values.  Dense x
            # compatibility is intentionally unavailable for them.
            self._genome_batch = genomes
            self._size = len(genomes)

    def _check_name_conflicts(self):
        """
        Check conflict attributes.

        If any attributes provided during initialization have the same name
        as a method or property of the Population class, a warning message is displayed.
        """
        cls = type(self)
        for name in self.schema:
            if hasattr(cls, name):
                attr = getattr(cls, name)
                if isinstance(attr, _ConflictBypassProperty):
                    # No conflicts occur for properties added using the bind_property
                    # function or the bind_property_array function.
                    continue
                warnings.warn(
                    f"Attribute name '{name}' conflicts with a "
                    f"Population method/property. "
                    f"Access via pop.{name} will return the method. "
                    f"Use pop.get('{name}') or pop.get_array('{name}') "
                    f"to access the data.",
                    UserWarning,
                    stacklevel=3,
                )

    def _init_column(self, attr: PopulationAttribute, capacity: int) -> None:
        """
        Initialize a column in the population.

        Parameters
        ----------
        attr : PopulationAttribute
            The attribute definition for the column.
        capacity : int
            The initial capacity of the column.
        """
        shape = (capacity, *attr.shape)
        if attr.default is not None:
            arr = np.full(
                shape=shape, fill_value=attr.default, dtype=attr.dtype, order="C"
            )
        elif np.issubdtype(attr.dtype, np.floating) and np.isnan(attr.default):
            arr = np.full(shape=shape, fill_value=np.nan, dtype=attr.dtype, order="C")
        else:
            arr = np.zeros(shape=shape, dtype=attr.dtype, order="C")
        self._data[attr.name] = arr

    def _resize(self, new_capacity: int) -> None:
        """
        Resize the population to a new capacity.

        Parameters
        ----------
        new_capacity : int
            The new capacity of the population.
        """
        for k, v in self._data.items():
            attr = self._schema.get(k)
            if attr is None and k == "x":
                attr = PopulationAttribute(
                    name="x", dtype=v.dtype, shape=v.shape[1:], default=np.nan
                )
            if attr is None:
                raise RuntimeError(f"Missing schema for population storage '{k}'")
            shape = (new_capacity, *attr.shape)
            new_arr = np.full(
                shape=shape, fill_value=attr.default, dtype=attr.dtype, order="C"
            )
            if attr.default is not None:
                new_arr[:] = attr.default
            new_arr[: self._size] = v[: self._size]
            self._data[k] = new_arr
        self._capacity = new_capacity
        if isinstance(self._genome_batch, DenseVectorBatch):
            self._genome_batch = DenseVectorBatch(self._data["x"])

    def mod_value(self) -> None:
        """Public method to call when the value changes."""
        self._value_version += 1
        if "_cache" in self.__dict__:
            self._cache.clear()

    def mod_structure(self) -> None:
        """Public method to call when the structure changes."""
        self._structure_version += 1
        self.mod_value()

    def set_cache(self, key: Hashable, value: Any) -> None:
        """
        Set a cache value.

        The cache is automatically cleared when the population is modified
        (via ``mod_value`` or ``mod_structure``).

        Parameters
        ----------
        key : Hashable
            The key of the cache.
        value : Any
            The value to be cached.
        """
        self._cache[key] = value

    def get_cache(self, key: Hashable) -> Any | None:
        """
        Get a cached value.

        Returns ``None`` if the key is not found.

        Parameters
        ----------
        key : Hashable
            The key of the cache.

        Returns
        -------
        Any | None
            The cached value, or ``None`` if not found.
        """
        return self._cache.get(key)

    def append(
        self, element: T_Individual | dict[str, Any] | None = None, **kwargs
    ) -> None:
        """
        Append a new individual to the population.

        Parameters
        ----------
        element : Individual | dict | None
            Data for the additional individual
        **kwargs :
            Set attribute values individually and add them.
            Alternatively, overwrite based on the element's value and add it.

        Examples
        --------
        >>> pop.append(ind)
        >>> pop.append({"x": x_val})
        >>> pop.append(x=x_val, f=0.1)
        >>> pop.append(ind, f=0.1)
        """
        self._append_internal(element, preserve_ids=False, **kwargs)

    def _append_internal(
        self,
        element: T_Individual | dict[str, Any] | None = None,
        *,
        preserve_ids: bool,
        allow_duplicate_ids: bool = False,
        **kwargs,
    ) -> None:
        """Append a new individual; ``preserve_ids`` controls ``id`` acceptance.

        The public and internal ``id`` columns use the same acceptance rules.

        Parameters
        ----------
        element : Individual | dict | None
            Data for the additional individual
        preserve_ids : bool
            If ``False`` (the public ``append()`` path), an explicit ``id``
            other than the ``-1`` sentinel raises ``ValidationError``. If
            ``True``, an explicit real ``id`` is accepted and validated for
            uniqueness — used only by internal lifecycle code.
        **kwargs :
            Set attribute values individually and add them.
            Alternatively, overwrite based on the element's value and add it.
        """
        data: dict[str, Any] = {}
        if element is not None:
            if isinstance(element, dict):
                data.update(element)
            else:
                for key in self._schema:
                    if hasattr(element, key):
                        data[key] = getattr(element, key)
        data.update(kwargs)

        genome_value = data.pop("genome", data.pop("genomes", None))

        if "id" in self._schema:
            id_val = int(data.get("id", self._schema["id"].default))
            if not preserve_ids and id_val != -1:
                raise ValidationError(
                    "append() does not accept an explicit 'id' other than the -1 "
                    "sentinel; internal lifecycle code uses "
                    "_append_internal(preserve_ids=True)"
                )
            if (
                id_val != -1
                and not allow_duplicate_ids
                and self._size > 0
                and np.any(self._get_mutable_array("id") == id_val)
            ):
                raise ValidationError(f"Duplicate candidate id {id_val}")

        if self._size >= self._capacity:
            self._resize(self._capacity * 2)

        idx = self._size
        for key, attr in self._schema.items():
            data_self = self._data[key]
            if key in data:
                data_self[idx] = data[key]
            else:
                if attr.default is not None:
                    data_self[idx] = attr.default
                elif np.issubdtype(attr.dtype, np.floating) and np.isnan(attr.default):
                    data_self[idx] = np.nan
                else:
                    data_self[idx] = 0

        if "x" not in self._schema and self._genome_items is not None:
            if genome_value is None:
                self._genome_items.append(None)
            else:
                if not isinstance(genome_value, ObjectBatch) or len(genome_value) != 1:
                    raise ValidationError("append() expects one genome")
                self._genome_items.append(genome_value.items[0])
        elif self._legacy_scalar_x:
            if self._genome_items is None:
                raise RuntimeError("Legacy scalar genome storage is not initialized")
            self._genome_items.append(data.get("x", self._schema["x"].default))
        elif genome_value is not None:
            self._replace_genome_rows(np.array([idx], dtype=np.intp), genome_value)

        self._size += 1
        if self._genome_items is not None:
            self._genome_batch = ObjectBatch(self._genome_items)
        self.mod_structure()

    def extend(self, other: Self | dict) -> None:
        """
        Extend this population with another population.

        Parameters
        ----------
        other : Population | dict
            The other population to extend from.
        """
        self._extend_internal(other, preserve_ids=False)

    def _validate_incoming_ids(
        self,
        other_data: Mapping[str, Any],
        other_size: int,
        *,
        preserve_ids: bool,
        allow_duplicate_ids: bool,
        check_existing: bool = True,
    ) -> None:
        """Validate IDs for an incoming row batch before it is appended."""
        if "id" not in self._schema:
            return

        if "id" in other_data:
            incoming_ids = np.asarray(other_data["id"]).astype(np.int64, copy=False)
            if not preserve_ids and np.any(incoming_ids != -1):
                raise ValidationError(
                    "extend() does not accept explicit 'id' values other than the "
                    "-1 sentinel; internal lifecycle code uses "
                    "_extend_internal(preserve_ids=True)"
                )
        else:
            incoming_ids = np.full(
                other_size, self._schema["id"].default, dtype=np.int64
            )
        real_incoming = incoming_ids[incoming_ids != -1]
        if not allow_duplicate_ids and len(real_incoming) != len(
            np.unique(real_incoming)
        ):
            raise ValidationError("Duplicate candidate id within the extended batch")
        if (
            check_existing
            and self._size > 0
            and len(real_incoming) > 0
            and not allow_duplicate_ids
            and np.any(np.isin(self._get_mutable_array("id"), real_incoming))
        ):
            raise ValidationError(
                "Duplicate candidate id already present in population"
            )

    def _extend_internal(
        self,
        other: Any,
        *,
        preserve_ids: bool,
        allow_duplicate_ids: bool = False,
    ) -> None:
        """Extend this population; ``preserve_ids`` controls ``id`` acceptance.

        The public and internal ``id`` columns use the same acceptance rules.

        Parameters
        ----------
        other : Population | dict
            The other population to extend from.
        preserve_ids : bool
            If ``False`` (the public ``extend()`` path), an explicit ``id``
            other than the ``-1`` sentinel raises ``ValidationError``. If
            ``True``, explicit real ``id`` values are accepted and validated
            for uniqueness — used only by internal lifecycle code.
        """
        exact_dense_population = False
        canonical_identity_backing = False
        if isinstance(other, Population):
            other_size = len(other)
            exact_dense_population = (
                type(other) is type(self)
                and other._schema == self._schema
                and type(other).get_array is Population.get_array
                and type(self).get_array is Population.get_array
                and isinstance(other._genome_batch, DenseVectorBatch)
                and isinstance(self._genome_batch, DenseVectorBatch)
            )
            if exact_dense_population:
                canonical_identity_backing = (
                    getattr(
                        other._dense_numeric_view,
                        "_canonical_identity_backing",
                        False,
                    )
                    is True
                    and getattr(
                        self._dense_numeric_view,
                        "_canonical_identity_backing",
                        False,
                    )
                    is True
                )
                # The canonical dense path copies the x backing column below;
                # defer constructing the compatibility genome view until it is
                # actually needed by a non-canonical fallback.
                other_genomes = None
                other_data = {
                    k: other._data[k][:other_size] for k in other.schema if k != "x"
                }
                if not canonical_identity_backing:
                    other_genomes = other.genomes
            else:
                other_genomes = other.genomes
                other_data = {k: other.get_array(k) for k in other.schema}
        elif isinstance(other, dict):
            other_size = np.asarray(next(iter(other.values()))).shape[0]
            other_data = other
            other_genomes = other.get("genomes", other.get("genome"))

        if other_size == 0:
            return

        self._validate_incoming_ids(
            other_data,
            other_size,
            preserve_ids=preserve_ids,
            allow_duplicate_ids=allow_duplicate_ids,
        )

        if self._size + other_size > self._capacity:
            self._resize(max(self._capacity * 2, self._size + other_size))

        start = self._size
        for key, attr in self._schema.items():
            if key == "x" and exact_dense_population:
                continue
            val_self = self._data[key]
            if key in other_data:
                val_self[start : start + other_size] = other_data[key]
            else:
                if attr.default is not None:
                    val_self[start : start + other_size] = attr.default
                elif np.issubdtype(attr.dtype, np.floating) and np.isnan(attr.default):
                    val_self[start : start + other_size] = np.nan
                else:
                    val_self[start : start + other_size] = 0

        self._size += other_size
        if canonical_identity_backing:
            self._data["x"][start : start + other_size] = other._data["x"][:other_size]
        elif other_genomes is not None:
            self._append_genomes(other_genomes)
        elif self._genome_items is not None:
            self._genome_items.extend([None] * other_size)
            self._genome_batch = ObjectBatch(self._genome_items)
        self.mod_structure()

    def _replace_from_population(
        self,
        source: Population,
        indices: np.ndarray | list[int] | slice,
        *,
        preserve_ids: bool,
        allow_duplicate_ids: bool = False,
    ) -> bool:
        """Replace rows from a dense population without an intermediate copy.

        Return ``False`` when the populations are not an exact dense match so
        callers can use the general extract/extend path.
        """
        if (
            source is self
            or type(source) is not type(self)
            or not isinstance(source, Population)
            or not isinstance(source._genome_batch, DenseVectorBatch)
            or not isinstance(self._genome_batch, DenseVectorBatch)
            or source._schema != self._schema
            or type(source).get_array is not Population.get_array
            or type(self).get_array is not Population.get_array
        ):
            return False

        if isinstance(indices, slice):
            start, stop, step = indices.indices(source._size)
            indices_arr: Any = slice(start, stop, step)
            n_selected = len(range(start, stop, step))
        else:
            indices_arr = np.asarray(indices)
            if indices_arr.ndim != 1 or indices_arr.dtype.kind not in "iu":
                return False
            n_selected = len(indices_arr)

        # Match GA.tell's existing clear-then-extend state transition, including
        # the empty-on-validation-error behavior of the old path.
        self.clear()
        # Validate IDs before writing any destination columns.  Only the ID
        # column needs selection materialized for validation; materializing
        # every selected column here used to create a full intermediate copy
        # of the survivor payload.
        selected_ids = None
        if "id" in self._schema:
            selected_ids = {"id": source._data["id"][: source._size][indices_arr]}
        self._validate_incoming_ids(
            selected_ids or {},
            n_selected,
            preserve_ids=preserve_ids,
            allow_duplicate_ids=allow_duplicate_ids,
            check_existing=False,
        )
        if n_selected == 0:
            return True
        if n_selected > self._capacity:
            self._resize(max(self._capacity * 2, n_selected))
        source_rows = source._size
        for key in self._schema:
            if key == "x":
                continue
            source_column = source._data[key][:source_rows]
            destination_column = self._data[key][:n_selected]
            if isinstance(indices_arr, slice):
                destination_column[...] = source_column[indices_arr]
            else:
                # np.take(out=...) writes directly into the destination and
                # avoids retaining a separate selected_data dictionary.
                np.take(source_column, indices_arr, axis=0, out=destination_column)
        source_view = source._dense_numeric_view
        target_view = self._dense_numeric_view
        canonical_identity_backing = (
            getattr(source_view, "_canonical_identity_backing", False) is True
            and getattr(target_view, "_canonical_identity_backing", False) is True
        )
        if canonical_identity_backing:
            # Built-in identity views expose the population's canonical x
            # backing directly, so no batch selection or service conversion is
            # needed for this dense survivor tail.
            source_x = source._data["x"][:source_rows]
            destination_x = self._data["x"][:n_selected]
            if isinstance(indices_arr, slice):
                destination_x[...] = source_x[indices_arr]
            else:
                np.take(source_x, indices_arr, axis=0, out=destination_x)
        else:
            selected_indices = np.arange(source._size, dtype=np.intp)[indices_arr]
            self._replace_genome_rows(
                np.arange(n_selected, dtype=np.intp),
                source.genomes.take(selected_indices),
            )
        self._size = n_selected
        self.mod_structure()
        return True

    def extract(self, indices: np.ndarray | list[int] | slice) -> Self:
        """
        Extract individuals with indices, and return new Population.

        Parameters
        ----------
        indices : np.ndarray | List[int] | slice
            Indices to extract
        """
        if isinstance(indices, slice):
            start, stop, step = indices.indices(self._size)
            n_extract = len(range(start, stop, step))
            indices_arr = slice(start, stop, step)
        else:
            indices_arr = np.array(indices)
            n_extract = len(indices_arr)

        new_pop = self.empty_like(capacity=n_extract)

        for key, val in self._data.items():
            new_pop._data[key][:n_extract] = val[: self._size][indices_arr]

        # Dense genomes are backed by the x column copied above.  The helper
        # is needed only for representations whose genome storage is not in
        # _data (or for legacy scalar x, whose object mirror must be synced).
        if not isinstance(self._genome_batch, DenseVectorBatch):
            new_pop._copy_genomes_from(self, indices_arr)

        new_pop._size = n_extract
        new_pop.mod_structure()
        return new_pop

    def truncate(self, new_size: int) -> None:
        """
        Cut the population to a new size.

        Parameters
        ----------
        new_size : int
            The new size of the population.
        """
        if new_size < 0:
            raise ValueError("new_size must be non-negative")
        if new_size < self._size:
            self._size = new_size
            if self._genome_items is not None:
                del self._genome_items[new_size:]
                self._genome_batch = ObjectBatch(self._genome_items)
            self.mod_structure()

    def delete(self, index: int | slice | list[int] | np.ndarray) -> None:
        """
        Delete individuals from the population.

        Parameters
        ----------
        index : int, slice, list[int], np.ndarray
            The index or indices of individuals to delete.
        """
        bool_mask = np.ones(self._size, dtype=bool)
        bool_mask[index] = False
        new_size = np.sum(bool_mask)
        for k, v in self._data.items():
            valid_data = v[: self._size]
            v[:new_size] = valid_data[bool_mask]
        self._reorder_genomes(bool_mask)
        self._size = new_size
        self.mod_structure()

    def reorder(self, order: np.ndarray) -> None:
        """
        Reorder individuals in the population.

        Parameters
        ----------
        order : np.ndarray
            The new order of individuals.
        """
        if len(order) != self._size:
            raise ValueError(
                f"Order length {len(order)} must match population size {self._size}"
            )
        for k, v in self._data.items():
            valid_data = v[: self._size]
            v[: self._size] = valid_data[order]
        self._reorder_genomes(order)
        self.mod_structure()

    def argsort(self, name: str, reverse: bool = False) -> np.ndarray:
        """
        Get the indices that would sort the population by a specific attribute.

        Parameters
        ----------
        name : str
            The attribute name to sort by.
        reverse : bool, optional
            Whether to sort in descending order, by default False.
        """
        if name not in self._data:
            raise KeyError(f"Key '{name}' not found in population schema")
        sort_arg = np.argsort(self._data[name][: self._size])
        if reverse:
            sort_arg = sort_arg[::-1]
        return sort_arg

    def clear(self) -> None:
        """Clear the population."""
        self._size = 0
        if self._genome_items is not None:
            self._genome_items.clear()
            self._genome_batch = ObjectBatch()
        self.mod_structure()

    def empty_like(self, capacity: int | None = None):
        """
        Create an empty Population with the same schema.

        Parameters
        ----------
        capacity : int
            Initial capacity of the new Population. Defaults to ``self._capacity``.
        """
        if capacity is None:
            capacity = self._capacity
        # Dense populations can reconstruct their empty x-backed genome view
        # from the schema; creating an empty batch here is redundant.  Other
        # representations still need an empty batch to preserve their type.
        genome_template = (
            None
            if self._legacy_scalar_x or isinstance(self._genome_batch, DenseVectorBatch)
            else self.genomes.take([])
        )
        return self.__class__(
            self.attrs,
            capacity,
            genomes=genome_template,
            dense_numeric_view=self._dense_numeric_view,
        )

    @overload
    def get(self, key: str) -> np.ndarray | None: ...

    @overload
    def get(self, key: str, default: _T_Default) -> np.ndarray | _T_Default: ...

    def get(
        self, key: str, default: _T_Default | None = None
    ) -> np.ndarray | _T_Default | None:
        """
        Get the array of a specific attribute.

        Parameters
        ----------
        key : str
            The attribute name to get the array for.
        default : any, optional
            Returned when the key is absent. Defaults to ``None``.

        Returns
        -------
        np.ndarray or default
            The attribute array if ``key`` exists, otherwise ``default``.
        """
        if key in self._data:
            return self.get_array(key)
        return default

    def _get_mutable_array(self, key: str) -> np.ndarray:
        """
        Get a mutable view of a specific attribute's backing storage.

        Bypasses read-only enforcement and bumps no version — callers are
        responsible for calling ``mod_value()`` (or ``mod_structure()``)
        themselves after mutating through this view.

        Parameters
        ----------
        key : str
            The attribute name to get the array for.
        """
        return self._data[key][: self._size]

    def _validate_genomes(
        self, genomes: GenomeBatch | np.ndarray | None, count: int
    ) -> GenomeBatch | None:
        if genomes is None:
            return None
        candidate: GenomeBatch
        if isinstance(genomes, np.ndarray):
            candidate = DenseVectorBatch(genomes)
        else:
            candidate = genomes
        if len(candidate) != count:
            raise ValidationError(
                f"genome batch length must be {count}, got {len(candidate)}"
            )
        if isinstance(self._genome_batch, DenseVectorBatch):
            if self._dense_numeric_view is None:
                raise ValidationError(
                    "Dense genome updates require the DenseNumericView service"
                )
            view = np.asarray(self._dense_numeric_view.get_view(candidate))
            target = self._data["x"][: self._size]
            if view.shape != (count, target.shape[1]) or view.dtype != np.float64:
                raise ValidationError(
                    "genome batch shape/dtype does not match DenseNumericView"
                )
        elif not isinstance(candidate, ObjectBatch):
            raise ValidationError("Object populations require an ObjectBatch genome")
        return candidate

    def _require_dense_view(self) -> DenseNumericView:
        if self._dense_numeric_view is None:
            raise AttributeError(
                "Population genome access requires the DenseNumericView service"
            )
        return self._dense_numeric_view

    def _replace_genome_rows(self, indices: np.ndarray, genomes: GenomeBatch) -> None:
        if isinstance(self._genome_batch, DenseVectorBatch):
            values = np.asarray(self._require_dense_view().get_view(genomes))
            self._data["x"][indices] = values
        else:
            if not isinstance(genomes, ObjectBatch) or self._genome_items is None:
                raise ValidationError(
                    "Object populations require an ObjectBatch genome"
                )
            for index, item in zip(indices, genomes.items):
                self._genome_items[int(index)] = item
            self._genome_batch = ObjectBatch(self._genome_items)

    def _append_genomes(self, genomes: GenomeBatch) -> None:
        if isinstance(self._genome_batch, DenseVectorBatch):
            values = np.asarray(self._require_dense_view().get_view(genomes))
            self._data["x"][self._size - len(genomes) : self._size] = values
        elif isinstance(genomes, ObjectBatch) and self._genome_items is not None:
            self._genome_items.extend(genomes.items)
            self._genome_batch = ObjectBatch(self._genome_items)

    def _copy_genomes_from(self, source: Population, indices: Any) -> None:
        if isinstance(indices, slice):
            indices = np.arange(len(source), dtype=np.intp)[indices]
        selected = source.genomes.take(indices)
        if isinstance(self._genome_batch, DenseVectorBatch):
            values = np.asarray(self._require_dense_view().get_view(selected))
            self._data["x"][: len(selected)] = values
        else:
            if not isinstance(selected, ObjectBatch) or self._genome_items is None:
                raise ValidationError("incompatible genome batch representation")
            self._genome_items = list(selected.items)
            self._genome_batch = ObjectBatch(self._genome_items)

    def _reorder_genomes(self, order: Any) -> None:
        if self._genome_items is not None:
            values = self._genome_items
            self._genome_items = list(np.asarray(values, dtype=object)[order])
            self._genome_batch = ObjectBatch(self._genome_items)

    @property
    def genomes(self) -> GenomeBatch:
        """Return the current genomes as a read-only batch value."""
        if isinstance(self._genome_batch, DenseVectorBatch):
            batch_id = id(self._genome_batch)
            cached = self._dense_genomes_view_cache
            if (
                cached is not None
                and cached[0] == self._structure_version
                and cached[1] == batch_id
            ):
                return cached[2]
            view = DenseVectorBatch(self._data["x"][: self._size])
            self._dense_genomes_view_cache = (
                self._structure_version,
                batch_id,
                view,
            )
            return view
        if self._genome_items is not None:
            return ObjectBatch(self._genome_items)
        return self._genome_batch.take(np.arange(self._size, dtype=np.intp))

    @property
    def candidate_ids(self) -> CandidateIds:
        """Return candidate IDs as a read-only row-aligned array."""
        if "id" in self._data:
            return self.get_array("id")
        ids = np.full(len(self), -1, dtype=np.int64)
        ids.setflags(write=False)
        return ids

    @property
    def columns(self) -> ColumnStore:
        """Return read-only row-aligned column arrays, excluding genome and IDs."""
        return MappingProxyType(
            {key: self.get_array(key) for key in self._schema if key not in {"x", "id"}}
        )

    def get_array(self, key: str) -> np.ndarray:
        """
        Get the array of a specific attribute.

        Returns a read-only view; mutating the returned array raises
        ``ValueError``. Use ``update_array()`` or ``update_rows()`` to
        mutate, so structure/value versions and caches stay in sync.

        Parameters
        ----------
        key : str
            The attribute name to get the array for.
        """
        if key == "x":
            if self._legacy_scalar_x:
                view = self._get_mutable_array(key).view()
                view.flags.writeable = False
                return view
            if self._dense_numeric_view is None:
                raise AttributeError(
                    "Population.x requires the DenseNumericView service"
                )
            try:
                view = np.asarray(
                    self._dense_numeric_view.get_view(self.genomes)
                ).view()
            except KeyError as exc:
                raise AttributeError(
                    "Population.x requires the DenseNumericView service"
                ) from exc
            view.flags.writeable = False
            return view
        view = self._get_mutable_array(key).view()
        view.flags.writeable = False
        return view

    def get_readonly_array(self, key: str) -> np.ndarray:
        """Return a read only view of the specified key."""
        view = self.get_array(key).view()
        view.flags.writeable = False
        return view

    def update_array(self, key: str, value: Any) -> None:
        """Update array in place and bump the value version."""
        if key == "x":
            if self._legacy_scalar_x:
                self._get_mutable_array(key)[:] = value
                self.mod_value()
                return
            if self._dense_numeric_view is None:
                raise AttributeError(
                    "Population.update_array('x') requires the DenseNumericView service"
                )
            arr = self._data["x"][: self._size]
            normalized = self._copy_validated_array(
                value, arr.shape, np.dtype(arr.dtype)
            )
            arr[:] = normalized
            self.mod_value()
            return
        if key == "id":
            raise ValidationError(
                "Cannot update the reserved 'id' column via update_array()"
            )
        self._get_mutable_array(key)[:] = value
        self.mod_value()

    def _copy_validated_array(
        self, value: Any, expected_shape: tuple[int, ...], expected_dtype: np.dtype
    ) -> np.ndarray:
        """
        Validate ``value`` against an exact shape/dtype and return an owned copy.

        Parameters
        ----------
        value : Any
            The array-like value to validate.
        expected_shape : tuple[int, ...]
            The exact shape the value must have.
        expected_dtype : np.dtype
            The exact dtype the value must have (no silent casting).

        Returns
        -------
        np.ndarray
            A native-endian, C-contiguous owned copy of ``value``.
        """
        arr = np.asarray(value)
        if expected_dtype == np.dtype(object) or arr.dtype == np.dtype(object):
            raise ValidationError(
                "update_rows() rejects object-dtype columns and values"
            )
        if arr.shape != expected_shape or arr.dtype != expected_dtype:
            raise ValidationError(
                f"Expected array of shape {expected_shape} and dtype "
                f"{expected_dtype}, got shape {arr.shape} and dtype {arr.dtype}"
            )
        return np.array(arr, dtype=expected_dtype, order="C", copy=True)

    def update_rows(
        self,
        indices: np.ndarray | list[int],
        values: dict[str, np.ndarray],
        genome: GenomeBatch | np.ndarray | None = None,
    ) -> None:
        """Atomically update multiple columns for the given rows.

        Validates every index and value before mutating any backing storage,
        so a validation failure never leaves a partial update applied. Bumps
        the value version exactly once for a non-empty update; an update with
        empty ``indices`` or an empty ``values`` mapping is a no-op and does
        NOT bump the version.

        Parameters
        ----------
        indices : np.ndarray | list[int]
            Row indices to update. Must be 1-D, integer dtype, within
            ``[0, len(self))``, and free of duplicates.
        values : dict[str, np.ndarray]
            Mapping of column name to the new values for those rows. Each
            value array must match shape ``(len(indices), *schema[key].shape)``
            and dtype ``schema[key].dtype`` exactly.

        Raises
        ------
        ValidationError
            If ``indices`` is invalid (wrong dtype/ndim, out of range, or
            contains duplicates), if a key in ``values`` is unknown (or is
            ``"id"``), or if a value array's shape/dtype does not match the
            schema exactly.
        """
        genome_value = genome
        # Keep the old x mapping working for legacy callers, but the explicit
        # genome argument is the canonical route after the column split.
        if "x" in values and genome_value is not None:
            raise ValidationError(
                "'x' is not a column; pass the genome through the explicit "
                "genome argument"
            )
        if genome_value is None and "x" in values and not self._legacy_scalar_x:
            genome_value = values["x"]
            values = {key: value for key, value in values.items() if key != "x"}
        if len(values) == 0 and genome_value is None:
            return

        indices_arr = np.asarray(indices)

        if indices_arr.ndim != 1:
            raise ValidationError(
                "indices must be a 1-D integer array, not a scalar/0-D or "
                "higher-dimensional array"
            )
        if indices_arr.dtype == np.bool_:
            raise ValidationError(
                "update_rows() expects an integer row-index array, not a boolean mask"
            )
        if indices_arr.dtype.kind not in "iu":
            raise ValidationError("indices must be a 1-D integer array")

        if len(indices_arr) == 0:
            return

        normalized_indices = indices_arr.astype(np.intp, copy=False)

        if np.any((normalized_indices < 0) | (normalized_indices >= self._size)):
            raise ValidationError(f"indices must be within [0, {self._size})")
        if len(np.unique(normalized_indices)) != len(normalized_indices):
            raise ValidationError("indices must not contain duplicates")

        normalized_values: dict[str, np.ndarray] = {}
        for key, value in values.items():
            if key == "id":
                raise ValidationError("Cannot update the reserved 'id' column")
            if key not in self._schema:
                raise ValidationError(f"Cannot update unknown column '{key}'")
            attr = self._schema[key]
            expected_shape = (len(normalized_indices), *attr.shape)
            normalized_values[key] = self._copy_validated_array(
                value, expected_shape, np.dtype(attr.dtype)
            )

        normalized_genome = self._validate_genomes(
            genome_value, len(normalized_indices)
        )

        for key, arr in normalized_values.items():
            self._data[key][normalized_indices] = arr
        if normalized_genome is not None:
            self._replace_genome_rows(normalized_indices, normalized_genome)

        self.mod_value()

    def _assign_ids(self, indices: np.ndarray, ids: np.ndarray) -> None:
        """Assign real candidate IDs to rows currently holding the -1 sentinel.

        Rejects reassigning a row that already has a real (non -1) id, and
        rejects an assignment that would create a duplicate id in the live
        population. Validates fully before mutating (atomic).
        """
        idx = np.asarray(indices, dtype=np.intp)
        id_col = self._get_mutable_array("id")
        if np.any(id_col[idx] != -1):
            raise ValidationError("Cannot reassign an already-assigned candidate id")
        prospective = id_col.copy()
        prospective[idx] = ids
        live = prospective[prospective != -1]
        if len(live) != len(np.unique(live)):
            raise ValidationError("Duplicate candidate id after assignment")
        id_col[idx] = ids
        self.mod_value()

    @property
    def schema(self) -> MappingProxyType[str, PopulationAttribute]:
        """Return the schema of the population."""
        return MappingProxyType(self._schema)

    @property
    def attrs(self) -> list[PopulationAttribute]:
        """Return the list of attributes in the population."""
        return list(self._schema.values())

    @property
    def value_version(self) -> int:
        """Return the current value-version counter, bumped by ``mod_value()``."""
        return self._value_version

    @property
    def structure_version(self) -> int:
        """Return the current structure-version counter, bumped by ``mod_structure``."""
        return self._structure_version

    def __len__(self) -> int:
        """Return the size of the population."""
        return self._size

    def __getattr__(self, name: str) -> np.ndarray:
        """Support dot access (ex: pop.x)."""
        if name == "x" and "_dense_numeric_view" in self.__dict__:
            raise AttributeError("Population.x requires the DenseNumericView service")
        if "_data" in self.__dict__ and name in self.__dict__["_data"]:
            return self.get_readonly_array(name)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Support dot setter."""
        if "_data" in self.__dict__ and name in self.__dict__["_data"]:
            self.update_array(name, value)
        else:
            super().__setattr__(name, value)

    @overload
    def __getitem__(self, index: int) -> T_Individual: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    def __getitem__(self, index: int | slice) -> T_Individual | Self:
        """
        Support bracket access.

        ``pop[0]`` returns an Individual; ``pop[:10]`` returns a new Population.
        """
        if isinstance(index, int):
            if index < 0 or index >= self._size:
                raise IndexError("Index out of range")
            return self.individual_class(self, index)  # type: ignore  # individual_class is generic; ty can't verify constructor signature
        elif isinstance(index, slice):
            return self.extract(index)
        else:
            raise TypeError("Invalid argument type.")


class Individual(Generic[T_Population]):
    """
    Individual class representing a single solution in the population.

    Attributes
    ----------
    _popref : weakref.ref
        Weak reference to the parent population.
    _index : int
        Index of the individual in the population.
    _structure_version : int
        Version number to track structure modifications.
    """

    __slots__ = ("_index", "_popref", "_structure_version")

    def __init__(self, population: T_Population, index: int):
        self._popref = weakref.ref(population)
        self._index = index
        self._structure_version = population._structure_version

    def _get_pop(self) -> T_Population:
        """
        Get the referenced population, checking for validity.

        Returns
        -------
        Population
            The referenced population.
        """
        pop = self._popref()
        if pop is None or pop._structure_version != self._structure_version:
            raise RuntimeError("Invalid Individual reference")
        return pop

    def get_readonly_value(self, key: str) -> Any:
        """
        Retrieve the value of the specified key.

        If the value is a NumPy array, return a read-only view.
        """
        value = self._get_pop().get_array(key)[self._index]
        if isinstance(value, np.ndarray):
            view = value.view()
            view.flags.writeable = False
            return view
        return value

    def update_value(self, key: str, value: Any) -> None:
        """Update value in place and bump the value version."""
        if key == "id":
            raise ValidationError(
                "Cannot update the reserved 'id' column via update_value()"
            )
        self._get_pop()._get_mutable_array(key)[self._index] = value
        self._get_pop().mod_value()

    def __getattr__(self, name: str) -> Any:
        """Access attribute from the population data."""
        pop = self._get_pop()
        if name in pop._data:
            return self.get_readonly_value(name)
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute value."""
        if name in self.__slots__:
            super().__setattr__(name, value)
            return
        else:
            pop = self._get_pop()
            if name in pop._data:
                self.update_value(name, value)
                return
            else:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'"
                )

    @property
    def pop(self) -> T_Population:
        """Return the parent population."""
        pop = self._get_pop()
        return pop


Population.individual_class = Individual
