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


class Population:
    """
    Base class for population.
    (self.data must have at least "x" key.)

    Attributes
    ----------
    data : dict[str, np.ndarray]
        Dictionary to store population data.
    """
    def __init__(self):
        self.data = {}
    
    @staticmethod
    def new(key: str, value: np.ndarray) -> Population:
        pop = Population()
        pop.set(key, value)
        return pop

    @classmethod
    def merge(cls, *populations: Population) -> Population:
        """
        Merge multiple populations into new Population.

        Only keys common to all populations are merged.

        Parameters
        ----------
        *populations : Population
            Populations to merge.

        Returns
        -------
        Population
            Merged Population.
        """
        if not populations:
            return cls()
        
        all_keys = set().union(*(pop.data.keys() for pop in populations))

        new_pop = cls()
        for key in all_keys:
            merged_arr = [p.get(key) for p in populations if p.get(key) is not None]
            if len(merged_arr) == len(populations):
                new_pop.set(key, np.vstack(merged_arr))
        
        return new_pop


    def get(self, key: str) -> np.ndarray:
        """
        Get the population data for the given key.

        Parameters
        ----------
        key : str
            The key to retrieve data for.
        """
        if key not in self.data:
            return None
        return self.data.get(key)

    def set(self, key: str, value: np.ndarray) -> None:
        """
        Set the population data for the given key.

        Parameters
        ----------
        key : str
            The key to set data for.
        value : np.ndarray
            The data to set for the given key.
        """
        self.data[key] = value

    def __len__(self):
        if not self.data:
            return 0
        return len(next(iter(self.data.values())))

    def __getitem__(self, index):    
        if isinstance(index, int):
            return Individual(self, index)
        elif isinstance(index, slice):
            return [Individual(self, i) for i in range(*index.indices(len(self)))]
        else:
            raise TypeError("Invalid argument type.")

    def __iter__(self) -> iter:
        for i in range(len(self)):
            yield Individual(self, i)


class Archive(Population):
    """
    Archive class to handle archive of evaluated solutions.
    (self.data must have at least "x" and "y" keys.)

    Attributes
    ----------
    data : dict[str, np.ndarray]
        Dictionary to store archive data. keys are "x" and "y".
    duplicate_log : list[dict]
        List to store duplicate solutions information.
    atol : float
        Absolute tolerance for duplicate check.
    rtol : float
        Relative tolerance for duplicate check.
    """
    def __init__(self, atol: float = 0.0, rtol: float = 0.0):
        super().__init__()
        self.data["x"] = np.empty((0, 0))
        self.data["y"] = np.empty((0, ))
        self.duplicate_log = []
        self.atol = atol  # tolerance for duplicate check
        self.rtol = rtol  # relative tolerance for duplicate check

    @staticmethod
    def new(x: np.ndarray, y: np.ndarray, atol: float = 0.0, rtol: float = 0.0) -> Archive:
        archive = Archive(atol=atol, rtol=rtol)
        archive.set("x", x)
        archive.set("y", y)
        return archive

    def add(self, x: np.ndarray, y: float) -> None:
        """
        Add a new solution to the archive. Duplicate solutions are ignored.

        Parameters
        ----------
        x : np.ndarray
            The solution to add.
        y : float
            The objective value of the solution.
        """
        # duplicate check
        if np.any(np.all(np.isclose(self.data["x"], x.reshape(1, -1), atol=self.atol, rtol=self.rtol), axis=1)):
            # TODO: implement to match the actual evaluation count (fe) with the archive size
            # self.data["x"] = np.vstack((self.data["x"], x.reshape(1, -1)))
            # self.data["y"] = np.hstack((self.data["y"], np.inf))
            self.duplicate_log.append({
                "index": len(self.data["y"]),
                "x": x.copy(),
                "y": y
            })
            return
        self.data["x"] = np.vstack((self.data["x"], x.reshape(1, -1)))
        self.data["y"] = np.hstack((self.data["y"], y))

    def restore_duplicates(self) -> Archive:
        """
        Restore duplicate solutions and return a new Archive including them.

        This method does not modify the current Archive instance.

        Returns
        -------
        Archive
            Returns a new Archive including duplicate solutions.
        """
        x_full = self.data["x"].copy()
        y_full = self.data["y"].copy()
        for e in self.duplicate_log:
            idx = e["index"]
            x_full = np.insert(x_full, idx, e["x"], axis=0)
            y_full = np.insert(y_full, idx, e["y"], axis=0)
        arc_full = Archive.new(x_full, y_full, atol=self.atol, rtol=self.rtol)
        return arc_full

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
        dist = np.linalg.norm(self.data["x"] - x, axis=1)
        idx = np.argsort(dist)[:k]
        return self.data["x"][idx], self.data["y"][idx]