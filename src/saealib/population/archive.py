"""Archive classes built on top of Population."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial import cKDTree

from saealib.population.population import Individual, Population, PopulationAttribute

if TYPE_CHECKING:
    from saealib.comparators import Dominator


class ArchiveMixin:
    """
    A mixin class for using Population as an Archive.

    Must be subclassed via multiple inheritance as a subclass of the Population class.
    Handle archive of evaluated solutions.
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

    def __init__(
        self,
        attrs: list[PopulationAttribute],
        init_capacity: int = 100,
        key_attr: str = "x",
        atol: float = 0.0,
        rtol: float = 0.0,
        **kwargs,
    ):
        super().__init__(attrs=attrs, init_capacity=init_capacity)

        if key_attr not in self.schema:
            raise ValueError(f"key_attr '{key_attr}' is not defined in attrs")
        self._duplicate_indices: list[int] = []
        self.key_attr = key_attr
        self.atol = atol
        self.rtol = rtol
        self._kdtree: cKDTree | None = None

    def add(self, element: Individual | dict[str, Any] | None = None, **kwargs) -> int:
        """
        Add a new solution to the archive. Duplicate solutions are ignored.

        Parameters
        ----------
        element : Individual | dict | None
            Data for the additional individual
        **kwargs :
            Set attribute values individually and add them.
            Alternatively, overwrite based on the element's value and add it.

        Returns
        -------
        idx : int
            Destination Index

        Examples
        --------
        >>> arcv.add(ind)
        >>> arcv.add({"x": x_val})
        >>> arcv.add(x=x_val, f=0.1)
        >>> arcv.add(ind, f=0.1)
        """
        key_attr_val = kwargs.get(self.key_attr)
        if key_attr_val is None:
            if isinstance(element, dict):
                key_attr_val = element.get(self.key_attr)
            elif element is not None and hasattr(element, self.key_attr):
                key_attr_val = getattr(element, self.key_attr)
        if key_attr_val is None:
            raise ValueError(f"Solution must have {self.key_attr} attribute")

        idx = self._find_idx(key_attr_val)

        if idx is not None:
            self._duplicate_indices.append(idx)
            return idx
        else:
            new_idx = self._size
            super().append(element, **kwargs)
            self._duplicate_indices.append(new_idx)
            self._kdtree = None
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
        matching = np.all(
            np.isclose(key_attr_arr, element, atol=self.atol, rtol=self.rtol), axis=1
        )
        indices = np.where(matching)[0]
        if indices.size > 0:
            return int(indices[0])
        return None

    def get_duplicated_population(self) -> Population:
        """
        Return a Population object without removing duplicates.

        Returns
        -------
        Population without removing duplicates.
        """
        all_length = len(self._duplicate_indices)
        dup_pop = Population(
            attrs=list(self._schema.values()), init_capacity=all_length
        )
        indices = np.array(self._duplicate_indices)
        for k, v in self._data.items():
            dup_pop._data[k][:all_length] = v[indices]
        dup_pop._size = all_length
        dup_pop._structure_version = self._structure_version
        return dup_pop

    def delete(self, index):
        """Delete element(s) and invalidate the kNN cache."""
        super().delete(index)
        self._kdtree = None

    def _ensure_kdtree(self) -> None:
        if self._kdtree is None:
            self._kdtree = cKDTree(self.get_array(self.key_attr))

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
            Indices and distances of the k-nearest neighbors.
        """
        if self._size == 0:
            return np.array([]), np.array([])
        self._ensure_kdtree()
        k = min(k, self._size)
        dist, idx = self._kdtree.query(x, k=k)
        return np.atleast_1d(idx), np.atleast_1d(dist)


class Archive(ArchiveMixin, Population):
    """Concrete archive: ``ArchiveMixin`` mixed into ``Population``."""

    pass


class ParetoMixin:
    """
    A mixin that maintains a Pareto-non-dominated archive.

    Must be used via multiple inheritance together with ``Population``
    (or a subclass thereof).  Only non-dominated solutions are retained:
    when a new solution is added any existing solutions it dominates are
    removed, and if the new solution is itself dominated it is discarded.

    Feasibility-first dominance is applied:

    - A feasible solution (cv ≤ eps_cv) dominates every infeasible one.
    - Among two infeasible solutions the one with lower cv dominates.
    - Among two feasible solutions ``dominator.dominates`` is used.

    Parameters
    ----------
    attrs : list[PopulationAttribute]
        Forwarded to ``Population.__init__``.
    init_capacity : int, optional
        Forwarded to ``Population.__init__``.
    direction : np.ndarray or None, optional
        Per-objective direction (+1 maximize, -1 minimize).
        ``None`` defaults to all-minimize.
    dominator : Dominator or None, optional
        Dominance predicate.  ``None`` defaults to ``ParetoDominator()``.
    eps_cv : float, optional
        Feasibility threshold for constraint violation, by default 0.0.
    """

    def __init__(
        self,
        attrs: list[PopulationAttribute],
        init_capacity: int = 100,
        direction: np.ndarray | None = None,
        dominator: Dominator | None = None,
        eps_cv: float = 0.0,
        **kwargs,
    ):
        super().__init__(attrs=attrs, init_capacity=init_capacity, **kwargs)

        # Import here to avoid circular imports at module load time.
        from saealib.comparators import ParetoDominator

        self.direction = direction
        self.dominator: Dominator = (
            dominator if dominator is not None else ParetoDominator()
        )
        self.eps_cv = eps_cv

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_fv(
        self, element: Individual | dict[str, Any] | None, kwargs: dict[str, Any]
    ) -> tuple[np.ndarray | None, float]:
        """
        Extract (f, cv) from the supplied element / kwargs.

        Returns
        -------
        f : np.ndarray or None
            Objective vector, or None when the key is absent or all-NaN.
        cv : float
            Constraint violation (0.0 when absent).
        """
        # --- f ---
        f_val = kwargs.get("f")
        if f_val is None:
            if isinstance(element, dict):
                f_val = element.get("f")
            elif element is not None and hasattr(element, "f"):
                f_val = getattr(element, "f")

        if f_val is None:
            f = None
        else:
            f = np.asarray(f_val, dtype=float).ravel()
            if np.all(np.isnan(f)):
                f = None

        # --- cv ---
        cv_val = kwargs.get("cv")
        if cv_val is None:
            if isinstance(element, dict):
                cv_val = element.get("cv")
            elif element is not None and hasattr(element, "cv"):
                cv_val = getattr(element, "cv")

        cv: float = float(cv_val) if cv_val is not None else 0.0

        return f, cv

    def _new_dominates_existing(
        self,
        f_new: np.ndarray | None,
        cv_new: float,
        f_ex: np.ndarray | None,
        cv_ex: float,
    ) -> bool:
        """Return True if the new solution dominates the existing one."""
        new_feasible = cv_new <= self.eps_cv
        ex_feasible = cv_ex <= self.eps_cv

        if new_feasible and not ex_feasible:
            return True
        if not new_feasible and ex_feasible:
            return False
        if new_feasible and ex_feasible:
            # Both feasible — use objective-space dominance.
            if f_new is None:
                return False
            if f_ex is None:
                # Existing has no objective value → new dominates it.
                return True
            return bool(self.dominator.dominates(f_new, f_ex, self.direction))
        # Both infeasible — lower cv wins.
        return cv_new < cv_ex

    def _existing_dominates_new(
        self,
        f_new: np.ndarray | None,
        cv_new: float,
        f_ex: np.ndarray | None,
        cv_ex: float,
    ) -> bool:
        """Return True if an existing solution dominates the new one."""
        return self._new_dominates_existing(f_ex, cv_ex, f_new, cv_new)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, element: Individual | dict[str, Any] | None = None, **kwargs) -> int:
        """
        Add a solution to the Pareto archive.

        The solution is accepted only if it is not dominated by any existing
        member.  After insertion all existing members that are dominated by
        the new solution are removed.

        Parameters
        ----------
        element : Individual | dict | None
            Data for the new solution.
        **kwargs :
            Attribute values that override or supplement ``element``.

        Returns
        -------
        idx : int
            Index assigned to the new solution, or -1 when it was rejected.

        Examples
        --------
        >>> archive.add(ind)
        >>> archive.add({"x": x_val, "f": f_val})
        >>> archive.add(x=x_val, f=f_val)
        """
        f_new, cv_new = self._extract_fv(element, kwargs)

        # Check whether any existing solution dominates the new one.
        if self._size > 0:
            f_arr = self.get_array("f") if "f" in self._schema else None
            cv_arr = self.get_array("cv") if "cv" in self._schema else None

            for i in range(self._size):
                f_ex = f_arr[i] if f_arr is not None else None
                cv_ex = float(cv_arr[i]) if cv_arr is not None else 0.0
                if self._existing_dominates_new(f_new, cv_new, f_ex, cv_ex):
                    return -1

            # Collect indices of existing solutions dominated by the new one.
            dominated_mask = np.zeros(self._size, dtype=bool)
            for i in range(self._size):
                f_ex = f_arr[i] if f_arr is not None else None
                cv_ex = float(cv_arr[i]) if cv_arr is not None else 0.0
                if self._new_dominates_existing(f_new, cv_new, f_ex, cv_ex):
                    dominated_mask[i] = True

            # Remove dominated solutions in one pass using delete().
            if np.any(dominated_mask):
                dominated_indices = np.where(dominated_mask)[0]
                self.delete(dominated_indices)

        # Append the new solution and return its index.
        new_idx: int = self._size
        super().append(element, **kwargs)  # type: ignore[misc]
        return new_idx


class ParetoArchive(ParetoMixin, Population):
    """Concrete Pareto archive: ``ParetoMixin`` mixed into ``Population``."""

    pass
