"""
Population module.

This module defines classes to handle populations and individuals.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Type, Tuple, List, Dict, Any, TypeVar, Generic
from typing_extensions import Self
from types import MappingProxyType
import weakref
import warnings

import numpy as np


T_Population = TypeVar("T_Population", bound="Population")
T_Individual = TypeVar("T_Individual", bound="Individual")


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
    dtype: Type | np.dtype
    shape: Tuple[int, ...] = ()
    default: Any = np.nan


class PropertyAvoidConfCheck(property):
    """
    A subclass of the property to avoid attribute conflict checks in Population. 
    Using for function 'bind_property', 'bind_property_array'. 
    This class and property behave identically, differing only in their class names.
    """
    pass


def bind_property(key: str, doc: str = "") -> Any:
    """
    Helper function: make property for Individual attributes.
    """
    def fget(self):
        return self.__getattr__(key)
    def fset(self, value):
        self.__setattr__(key, value)
    return PropertyAvoidConfCheck(fget, fset, doc=doc)


def bind_property_array(key: str, doc: str = "") -> Any:
    """
    Helper function: make property for Population attributes.
    Need 'get_array' method for setter.
    """
    def fget(self):
        return self.__getattr__(key)
    def fset(self, value):
        self.get_array(key)[:] = value
    return PropertyAvoidConfCheck(fget, fset, doc=doc)


class Population(Generic[T_Individual]):
    """
    Container for population data.

    Attributes
    ----------
    schema : Dict[str, PopulationAttribute]
        Schema defining the attributes of the population.
    _data : Dict[str, np.ndarray]
        Dictionary to store population data arrays.
    _capacity : int
        Current capacity of the population.
    _size : int
        Current size of the population.
    _version : int
        Version number to track modifications.
    """

    individual_class = None

    def __init__(self, attrs: List[PopulationAttribute], init_capacity: int = 100) -> None:
        """
        Initialize a Population.

        Parameters
        ----------
        attrs : List[PopulationAttribute]
            List of population attributes. Each attribute defines a column in the population.
        init_capacity : int, optional
            Initial capacity of the population, by default 100.
        """
        self._capacity = init_capacity
        self._size = 0
        self._version = 0
        self._data: Dict[str, np.ndarray] = {}
        for attr in attrs:
            self._init_column(attr, self._capacity)
        self._schema = {attr.name: attr for attr in attrs}
        self._check_name_conflicts()

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
                if isinstance(attr, PropertyAvoidConfCheck):
                    # No conflicts occur for properties added using the bind_property function or the bind_property_array function.
                    continue
                warnings.warn(
                    f"Attribute name '{name}' conflicts with a Population method/property. "
                    f"Access via pop.{name} will return the method. "
                    f"Use pop.get('{name}') or pop.get_array('{name}') to access the data.",
                    UserWarning,
                    stacklevel=3
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
        shape = (capacity, ) + attr.shape
        if attr.default is not None:
            arr = np.full(shape=shape, fill_value=attr.default, dtype=attr.dtype, order="C")
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
            attr = self._schema[k]
            shape = (new_capacity, ) + attr.shape
            new_arr = np.full(shape=shape, fill_value=attr.default, dtype=attr.dtype, order="C")
            if attr.default is not None:
                new_arr[:] = attr.default
            new_arr[:self._size] = v[:self._size]
            self._data[k] = new_arr
        self._capacity = new_capacity

    def append(self, element: T_Individual | Dict[str, Any] | None = None, **kwargs) -> None:
        """
        Append a new individual to the population.

        Parameters
        ----------
        element : Individual | dict | None
            Data for the additional individual
        **kwargs : 
            Set attribute values individually and add them. Alternatively, overwrite based on the element's value and add it.
        
        Example
        -------
        >>> pop.append(ind)             # adding ind(Individual)
        >>> pop.append({"x": x_val})    # adding dict
        >>> pop.append(x=x_val, f=0.1)  # adding with keywords
        >>> pop.append(ind, f=0.1)      # Overwrite only the 'f' based on the ind and add it.
        """
        # data: merge element and kwargs
        data: Dict[str, Any] = {}
        if element is not None:
            if isinstance(element, dict):
                data.update(element)
            else:
                for key in self._schema:
                    if hasattr(element, key):
                        data[key] = getattr(element, key)
        data.update(kwargs)

        # resizing
        if self._size >= self._capacity:
            self._resize(self._capacity * 2)

        # appending
        idx = self._size
        for key, attr in self._schema.items():
            data_self = self._data[key]
            if key in data:
                data_self[idx] = data[key]
            else:
                # fill default values
                if attr.default is not None:
                    data_self[idx] = attr.default
                elif np.issubdtype(attr.dtype, np.floating) and np.isnan(attr.default):
                    data_self[idx] = np.nan
                else:
                    data_self[idx] = 0

        self._size += 1
        self._version += 1

    def extend(self, other: Self | dict) -> None:
        """
        Extend this population with another population.

        Parameters
        ----------
        other : Population | dict
            The other population to extend from.
        """
        # Population(Self) or dict
        if isinstance(other, (Self, Population)):
            other_size = len(other)
            other_data = {k: other.get_array(k) for k in other.schema}
        elif isinstance(other, dict):
            other_size = len(next(iter(other.values())))
            other_data = other

        # if empty extended
        if other_size == 0: return

        # resizing
        if self._size + other_size > self._capacity:
            self._resize(max(self._capacity * 2, self._size + other_size))

        # extending
        start = self._size
        for key, attr in self._schema.items():
            val_self = self._data[key]
            if key in other_data:
                val_self[start:start+other_size] = other_data[key]
            else:
                # fill default values
                if attr.default is not None:
                    val_self[start:start+other_size] = attr.default
                elif np.issubdtype(attr.dtype, np.floating) and np.isnan(attr.default):
                    val_self[start:start+other_size] = np.nan
                else:
                    val_self[start:start+other_size] = 0
        
        self._size += other_size
        self._version += 1

    def extract(self, indices: np.ndarray | List[int] | slice) -> Self:
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
            new_pop._data[key][:n_extract] = val[:self._size][indices_arr]
        
        new_pop._size = n_extract
        new_pop._version += 1
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
            self._version += 1

    def delete(self, index: int | slice | List[int] | np.ndarray) -> None:
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
            valid_data = v[:self._size]
            v[:new_size] = valid_data[bool_mask]
        self._size = new_size
        self._version += 1

    def reorder(self, order: np.ndarray) -> None:
        """
        Reorder individuals in the population.

        Parameters
        ----------
        order : np.ndarray
            The new order of individuals.
        """
        if len(order) != self._size:
            raise ValueError(f"Order length {len(order)} must match population size {self._size}")
        for k, v in self._data.items():
            valid_data = v[:self._size]
            v[:self._size] = valid_data[order]
        self._version += 1

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
        sort_arg = np.argsort(self._data[name][:self._size])
        if reverse:
            sort_arg = sort_arg[::-1]
        return sort_arg

    def clear(self) -> None:
        """
        Clear the population.
        """
        self._size = 0
        self._version += 1

    def empty_like(self, capacity: int = None):
        """
        Create emplty Population object that have same schema

        Parameters
        ----------
        capacity : int
            capacity of new Population object. default is self._capacity.
        """
        if capacity is None:
            capacity = self._capacity 
        return self.__class__(self.attrs, capacity)

    def get(self, key: str, default = None) -> np.ndarray:
        """
        Get the array of a specific attribute.
        If the key does not exist in the attribute, it returns the specified value (default is None).

        Parameters
        ----------
        key : str
            The attribute name to get the array for.
        default : Any
            If the key does not exist in the attribute, return this value.
        """
        if key in self._data:
            return self.get_array(key)
        return default

    def get_array(self, key: str) -> np.ndarray:
        """
        Get the array of a specific attribute.

        Parameters
        ----------
        key : str
            The attribute name to get the array for.
        """
        return self._data[key][:self._size]

    @property
    def schema(self) -> MappingProxyType[str, PopulationAttribute]:
        return MappingProxyType(self._schema)

    @property
    def attrs(self) -> List[PopulationAttribute]:
        return list(self._schema.values())

    def __len__(self) -> int:
        return self._size

    def __getattr__(self, name: str) -> np.ndarray:
        """support dot access (ex: pop.x)"""
        if name in self._data:
            return self.get_array(name)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, index: int | slice) -> T_Individual | Self:
        """
        support bracket access

        pop[0]   -> return Individual
        pop[:10] -> return new Population
        """
        # index (return Individual)
        if isinstance(index, int):
            if index < 0 or index >= self._size:
                raise IndexError("Index out of range")
            return self.individual_class(self, index)
        # slice (return Population)
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
    _version : int
        Version number to track modifications.
    """
    __slots__ = ("_popref", "_index", "_version")

    def __init__(self, population: T_Population, index: int):
        self._popref = weakref.ref(population)
        self._index = index
        self._version = population._version

    def _get_pop(self) -> T_Population:
        """
        Get the referenced population, checking for validity.

        Returns
        -------
        Population
            The referenced population.
        """
        pop = self._popref()
        if pop is None or pop._version != self._version:
            raise RuntimeError("Invalid Individual reference")
        return pop

    def __getattr__(self, name: str) -> Any:
        pop = self._get_pop()
        if name in pop._data:
            return pop._data[name][self._index]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__slots__:
            super().__setattr__(name, value)
            return
        else:
            pop = self._get_pop()
            if name in pop._data:
                pop._data[name][self._index] = value
                return
            else:
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def pop(self) -> T_Population:
        pop = self._get_pop()
        return pop


class Archive(Population[T_Individual]):
    """
    Archive class to handle archive of evaluated solutions.
    (self.data must have at least key_attr (default is "x").)
    Duplicate removal and range queries can be performed.

    Attributes
    ----------
    data : dict[str, np.ndarray]
        Dictionary to store archive data.
    duplicate_log : list[dict]
        List to store duplicate solutions information.
    key_attr : str
        Key for duplicate checking
    atol : float
        Absolute tolerance for duplicate check.
    rtol : float
        Relative tolerance for duplicate check.
    """
    def __init__(self, attrs: List[PopulationAttribute], init_capacity: int = 100, key_attr: str = "x", atol: float = 0.0, rtol: float = 0.0):
        super().__init__(attrs, init_capacity)
        if key_attr not in self.schema:
            raise ValueError(f"key_attr '{key_attr}' is not defined in attrs")
        self._duplicate_indices: List[int] = []
        self.key_attr = key_attr
        self.atol = atol  # tolerance for duplicate check
        self.rtol = rtol  # relative tolerance for duplicate check

    def add(self, element: Individual | Dict[str, Any] | None = None, **kwargs) -> int:
        """
        Add a new solution to the archive. Duplicate solutions are ignored.

        Parameters
        ----------
        element : Individual | dict | None
            Data for the additional individual
        **kwargs : 
            Set attribute values individually and add them. Alternatively, overwrite based on the element's value and add it.
        
        Returns
        -------
        idx : int
            Destination Index

        Example
        -------
        >>> arcv.add(ind)               # adding ind(Individual)
        >>> arcv.add({"x": x_val})      # adding dict
        >>> arcv.add(x=x_val, f=0.1)    # adding with keywords
        >>> arcv.add(ind, f=0.1)        # Overwrite only the 'f' based on the ind and add it.
        """
        # get adding solution
        key_attr_val = kwargs.get(self.key_attr)
        if key_attr_val is None:
            if isinstance(element, dict):
                key_attr_val = element.get(self.key_attr)
            elif element is not None and hasattr(element, self.key_attr):
                key_attr_val = getattr(element, self.key_attr)
        if key_attr_val is None:
            # element does not have key_attr
            raise ValueError(f"Solution must have {self.key_attr} attribute")
        
        # check duplicate
        idx = self._find_idx(key_attr_val)

        if idx is not None:
            # duplicate found
            self._duplicate_indices.append(idx)
            return idx
        else:
            # no duplicate
            new_idx = self._size
            super().append(element, **kwargs)
            self._duplicate_indices.append(new_idx)
            return new_idx

    def _find_idx(self, element: np.ndarray | np.floating) -> int | None:
        """
        Search for duplicate indexes and return them if found.

        Parameters
        ----------
        element : np.ndarray | np.floating
            Search target
        
        Returns
        -------
        int | None
            Duplicate index. Return None if it does not exist.
        """
        if self._size == 0:
            return None
        # TODO: Handling cases where the element is not a np.ndarray
        key_attr_arr = self.get_array(self.key_attr)
        element = np.array(element, dtype=self._schema[self.key_attr].dtype)
        if element.ndim == 0:
            element = element.reshape(1)
        if element.shape != key_attr_arr.shape[1:]:
            element = element.reshape(key_attr_arr.shape[1:])
        matching = np.all(np.isclose(key_attr_arr, element, atol=self.atol, rtol=self.rtol), axis=1)
        indices = np.where(matching)[0]
        if indices.size > 0:
            return int(indices[0])
        return None

    def get_duplicated_population(self) -> Population:
        """
        Returns a Population object without removing duplicates.

        Returns
        -------
        Population without removing duplicates.
        """
        all_length = len(self._duplicate_indices)
        dup_pop = Population(attrs=list(self._schema.values()), init_capacity=all_length)
        indices = np.array(self._duplicate_indices)
        for k, v in self._data.items():
            dup_pop._data[k][:all_length] = v[indices]
        dup_pop._size = all_length
        dup_pop._version = self._version
        return dup_pop

    def get_knn(self, x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get k-nearest neighbors of the given solution from the archive.

        Parameters
        ----------
        x : np.ndarray
            The solution to find neighbors for.
        k : int
            The number of neighbors to retrieve.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The k-nearest neighbors' solutions and their objective values.
        """
        if self._size == 0:
            return np.array([]), np.array([])
        key_attr_arr = self.get_array(self.key_attr)
        dist = np.linalg.norm(key_attr_arr - x, axis=1)
        k = min(k, self._size)
        idx = np.argsort(dist)[:k]
        return idx, dist[idx]


Population.individual_class = Individual
