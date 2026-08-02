"""Archive classes built on top of Population."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.spatial import cKDTree  # type: ignore  # cKDTree has no bundled type stubs

from saealib.comparators import Dominator
from saealib.exceptions import ValidationError
from saealib.population.population import Individual, Population, PopulationAttribute


def _extract_id_value(
    schema: dict[str, PopulationAttribute], element: Any, kwargs: dict[str, Any]
) -> int | None:
    """Return the effective ``id`` implied by an ``add()``/``append()`` call.

    Returns ``None`` when the schema has no ``id`` column.
    """
    if "id" not in schema:
        return None
    id_val = kwargs.get("id")
    if id_val is None:
        if isinstance(element, dict):
            id_val = element.get("id")
        elif element is not None and hasattr(element, "id"):
            id_val = getattr(element, "id")
    if id_val is None:
        id_val = schema["id"].default
    return int(id_val)


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
        duplicate_policy: str = "keep_first",
        **kwargs,
    ):
        super().__init__(attrs=attrs, init_capacity=init_capacity)  # ty: ignore[unknown-argument]

        if key_attr not in self.schema:  # ty: ignore[unresolved-attribute]
            raise ValueError(f"key_attr '{key_attr}' is not defined in attrs")
        if duplicate_policy not in {"keep_first", "replace", "append"}:
            raise ValueError(
                "duplicate_policy must be 'keep_first', 'replace', or 'append'"
            )
        self._deprecated_duplicate_indices: list[int] = []
        self.duplicate_policy = duplicate_policy
        self.key_attr = key_attr
        self.atol = atol
        self.rtol = rtol
        self._kdtree: cKDTree | None = None

    def add(
        self: Any, element: Individual | dict[str, Any] | None = None, **kwargs
    ) -> int:
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

        if idx is not None and self.duplicate_policy != "append":
            if self.duplicate_policy == "keep_first":
                self._deprecated_duplicate_indices.append(idx)
                return idx

            # A retry of the same candidate is a value-only update.  A spatial
            # duplicate from another candidate must replace the whole row so
            # that the ID remains provenance-correct.
            incoming_id = _extract_id_value(self._schema, element, kwargs)  # type: ignore[unresolved-attribute]
            existing_id = (
                int(self.get_array("id")[idx]) if "id" in self.schema else incoming_id
            )
            if incoming_id == existing_id and "id" in self.schema:
                values = {
                    key: np.asarray(value, dtype=self._schema[key].dtype).reshape(
                        (1, *self._schema[key].shape)
                    )
                    for key, value in kwargs.items()
                    if key in self._schema and key != "id"
                }
                if isinstance(element, dict):
                    values.update(
                        {
                            key: np.asarray(
                                value, dtype=self._schema[key].dtype
                            ).reshape((1, *self._schema[key].shape))
                            for key, value in element.items()
                            if key in self._schema and key != "id" and key not in values
                        }
                    )
                if values:
                    self.update_rows(np.array([idx]), values)
                self._kdtree = None
                return idx

            self.delete(idx)
            # Delete compacts rows; append the incoming observation below.
            idx = None
        else:
            pass

        id_val = _extract_id_value(self._schema, element, kwargs)
        if id_val == -1:
            raise ValidationError(
                "Archive.add() requires a real candidate id when the "
                "schema declares an 'id' column (got the -1 sentinel)"
            )
        new_idx = self._size
        super()._append_internal(
            element,
            preserve_ids=True,
            allow_duplicate_ids=self.duplicate_policy == "append",
            **kwargs,
        )
        self._deprecated_duplicate_indices.append(new_idx)
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
        if self._size == 0:  # ty: ignore[unresolved-attribute]
            return None
        # TODO: Handling cases where the element is not a np.ndarray
        key_attr_arr = self.get_array(self.key_attr)  # ty: ignore[unresolved-attribute]
        element = np.array(element, dtype=self._schema[self.key_attr].dtype)  # ty: ignore[unresolved-attribute]
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
        warnings.warn(
            "get_duplicated_population() is deprecated; use an Archive with "
            "duplicate_policy='append' for observation history.",
            DeprecationWarning,
            stacklevel=2,
        )
        all_length = len(self._deprecated_duplicate_indices)
        dup_pop = Population(
            attrs=list(self._schema.values()),  # ty: ignore[unresolved-attribute]
            init_capacity=all_length,
        )
        indices = np.array(self._deprecated_duplicate_indices)
        for k, v in self._data.items():  # ty: ignore[unresolved-attribute]
            dup_pop._data[k][:all_length] = v[indices]
        dup_pop._size = all_length
        dup_pop._structure_version = self._structure_version  # ty: ignore[unresolved-attribute]
        return dup_pop

    @property
    def _duplicate_indices(self) -> list[int]:
        """Return the old duplicate index log and emit a warning."""
        warnings.warn(
            "Archive._duplicate_indices is deprecated; use duplicate_policy='append'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._deprecated_duplicate_indices

    def delete(self, index):
        """Delete element(s) and invalidate the kNN cache."""
        super().delete(index)  # ty: ignore[unresolved-attribute]
        self._kdtree = None

    def mod_value(self) -> None:
        """Invalidate the kNN cache on every value-only mutation, too."""
        self._kdtree = None
        super().mod_value()  # ty: ignore[unresolved-attribute]

    def _ensure_kdtree(self) -> None:
        if self._kdtree is None:
            self._kdtree = cKDTree(self.get_array(self.key_attr))  # ty: ignore[unresolved-attribute]

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
        if self._size == 0:  # ty: ignore[unresolved-attribute]
            return np.array([]), np.array([])
        self._ensure_kdtree()
        k = min(k, self._size)  # ty: ignore[unresolved-attribute]
        dist, idx = self._kdtree.query(x, k=k)  # ty: ignore[unresolved-attribute]
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
        Under ``Optimizer`` execution this value is overwritten every
        generation from ``problem.handler.feasibility_threshold``; the
        default of 0.0 (strictly feasible only) is only meaningful for
        standalone (non-``Optimizer``) use of ``ParetoArchive``.
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
        super().__init__(attrs=attrs, init_capacity=init_capacity, **kwargs)  # ty: ignore[unknown-argument]

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

        id_val = _extract_id_value(self.schema, element, kwargs)  # ty: ignore[unresolved-attribute]
        if id_val == -1:
            raise ValidationError(
                "ParetoArchive.add() requires a real candidate id when the "
                "schema declares an 'id' column (got the -1 sentinel)"
            )
        if (
            id_val is not None
            and id_val != -1
            and self._size > 0  # ty: ignore[unresolved-attribute]
            and np.any(self.get_array("id") == id_val)  # ty: ignore[unresolved-attribute]
        ):
            raise ValidationError(f"Duplicate candidate id {id_val}")

        # Check whether any existing solution dominates the new one.
        if self._size > 0:  # ty: ignore[unresolved-attribute]
            f_arr = self.get_array("f") if "f" in self._schema else None  # ty: ignore[unresolved-attribute]
            cv_arr = self.get_array("cv") if "cv" in self._schema else None  # ty: ignore[unresolved-attribute]
            n = self._size  # ty: ignore[unresolved-attribute]

            has_nan = (f_new is not None and np.any(np.isnan(f_new))) or (
                f_arr is not None and np.any(np.isnan(f_arr))
            )

            # dominates_many requires NaN-free objective values.
            use_fast_path = not has_nan and f_new is not None and f_arr is not None

            existing_dominates_new = np.zeros(n, dtype=bool)
            new_dominates_existing = np.zeros(n, dtype=bool)

            if use_fast_path:
                cv_ex_arr = cv_arr.astype(float) if cv_arr is not None else np.zeros(n)
                # np.bool_ (not a plain Python bool) so that `~` below is a
                # correct logical negation rather than a bitwise int inversion.
                new_feasible = np.bool_(cv_new <= self.eps_cv)
                ex_feasible = cv_ex_arr <= self.eps_cv

                existing_dominates_new |= (~new_feasible) & ex_feasible
                new_dominates_existing |= new_feasible & (~ex_feasible)

                both_infeasible = (~new_feasible) & (~ex_feasible)
                existing_dominates_new |= both_infeasible & (cv_ex_arr < cv_new)
                new_dominates_existing |= both_infeasible & (cv_new < cv_ex_arr)

                both_feasible = new_feasible & ex_feasible
                if np.any(both_feasible):
                    # Only pass the both_feasible-masked subset to
                    # dominates_many: existing rows outside that mask may be
                    # infeasible with objective values that aren't guaranteed
                    # meaningful (e.g. non-positive under multiplicative
                    # epsilon-dominance's f > 0 requirement), so including
                    # them in the call -- even though their result would
                    # later be discarded by the mask -- can crash.
                    feasible_idx = np.where(both_feasible)[0]
                    new_dom, ex_dom = self.dominator.dominates_many(
                        f_new, f_arr[feasible_idx], self.direction
                    )
                    new_dominates_existing[feasible_idx] |= new_dom
                    existing_dominates_new[feasible_idx] |= ex_dom

            if use_fast_path:
                if existing_dominates_new.any():
                    return -1
                dominated_mask = new_dominates_existing
            else:
                for i in range(n):
                    f_ex = f_arr[i] if f_arr is not None else None
                    cv_ex = float(cv_arr[i]) if cv_arr is not None else 0.0
                    if self._existing_dominates_new(f_new, cv_new, f_ex, cv_ex):
                        return -1

                # Collect indices of existing solutions dominated by the new one.
                dominated_mask = np.zeros(n, dtype=bool)
                for i in range(n):
                    f_ex = f_arr[i] if f_arr is not None else None
                    cv_ex = float(cv_arr[i]) if cv_arr is not None else 0.0
                    if self._new_dominates_existing(f_new, cv_new, f_ex, cv_ex):
                        dominated_mask[i] = True

            # Remove dominated solutions in one pass using delete().
            if np.any(dominated_mask):
                dominated_indices = np.where(dominated_mask)[0]
                self.delete(dominated_indices)  # ty: ignore[unresolved-attribute]

        # Append the new solution and return its index.
        new_idx: int = self._size  # ty: ignore[unresolved-attribute]
        super()._append_internal(element, preserve_ids=True, **kwargs)  # type: ignore[misc]  # ty: ignore[unresolved-attribute]
        return new_idx


class ParetoArchive(ParetoMixin, Population):
    """Concrete Pareto archive: ``ParetoMixin`` mixed into ``Population``."""

    pass
