"""
Comparators module.

This module defines comparator classes and Pareto-related utility functions
for ranking and comparing solutions in single- and multi-objective optimization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol

from saealib._deprecated import deprecated_param, warn_deprecated

import numpy as np

if TYPE_CHECKING:
    from saealib.population import Population


# ---------------------------------------------------------------------------
# Dominator abstraction
# ---------------------------------------------------------------------------


class Dominator(ABC):
    """
    Abstract base for dominance predicates.

    A ``Dominator`` encapsulates the definition of dominance between
    objective vectors independently of the sorting algorithm.  Two concrete
    operations are required and MUST agree with each other:

    - ``dominance_matrix`` — batched NxN boolean matrix (primary operation).
    - ``dominates`` — scalar pairwise predicate, derived from the matrix to
      guarantee consistency.

    Parameters are the same for both methods:

    Parameters
    ----------
    f : np.ndarray
        Objective matrix. shape: (n, n_obj)  [``dominance_matrix``]
    fa, fb : np.ndarray
        Objective vectors. shape: (n_obj,)  [``dominates``]
    direction : np.ndarray or None
        Per-objective optimization direction: +1 = maximize, -1 = minimize.
        ``None`` defaults to minimization for all objectives.
    """

    @abstractmethod
    def dominance_matrix(
        self,
        f: np.ndarray,
        direction: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute the NxN boolean dominance matrix.

        ``D[i, j]`` is True if row i dominates row j.

        Parameters
        ----------
        f : np.ndarray
            Objective matrix. shape: (n, n_obj).  Must contain no NaN values
            (callers are responsible for pre-filtering).
        direction : np.ndarray or None
            Per-objective direction. ``None`` → minimize all.

        Returns
        -------
        np.ndarray
            Boolean matrix of shape (n, n).
        """

    def dominates(
        self,
        fa: np.ndarray,
        fb: np.ndarray,
        direction: np.ndarray | None = None,
    ) -> bool:
        """
        Return True if fa dominates fb.

        Derived from ``dominance_matrix`` on a 2-row stack to guarantee
        agreement with the batched path.  NaN values in fa always return
        False (consistent with ``dominance_matrix`` assuming finite input).

        Parameters
        ----------
        fa, fb : np.ndarray
            Objective vectors. shape: (n_obj,)
        direction : np.ndarray or None
            Per-objective direction. ``None`` → minimize all.

        Returns
        -------
        bool
        """
        fa = np.asarray(fa, dtype=float)
        fb = np.asarray(fb, dtype=float)
        if np.any(np.isnan(fa)):
            return False
        # Replace NaN in fb with +inf so it appears infinitely bad (dominated).
        fb_safe = np.where(np.isnan(fb), np.inf, fb)
        stacked = np.stack([fa, fb_safe])
        return bool(self.dominance_matrix(stacked, direction)[0, 1])


class ParetoDominator(Dominator):
    """
    Pareto dominance predicate.

    ``fa`` dominates ``fb`` iff ``fa`` is at most as large as ``fb`` in all
    objectives and strictly smaller in at least one (after applying the
    direction transform).

    This is the default dominance relation used throughout the library.
    """

    def dominance_matrix(
        self,
        f: np.ndarray,
        direction: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute the NxN Pareto-dominance matrix.

        Applies the ``f * (-direction)`` transform so that smaller is always
        better, then accumulates per-objective ``leq_all`` / ``less_any``
        comparisons.  Peak memory is O(N²); no (N, N, M) tensor is created.

        Parameters
        ----------
        f : np.ndarray
            Objective matrix. shape: (n, n_obj).  Must contain no NaN values.
        direction : np.ndarray or None
            Per-objective direction. ``None`` → minimize all.

        Returns
        -------
        np.ndarray
            Boolean matrix of shape (n, n).  ``D[i, j]`` = True iff row i
            Pareto-dominates row j.
        """
        g = np.asarray(f, dtype=float)
        if direction is not None:
            g = g * (-direction)
        k, m = g.shape
        leq_all = np.ones((k, k), dtype=bool)
        less_any = np.zeros((k, k), dtype=bool)
        for obj in range(m):
            col = g[:, obj]
            leq_all &= col[:, None] <= col[None, :]
            less_any |= col[:, None] < col[None, :]
        return leq_all & less_any


class EpsilonDominator(Dominator):
    """
    ε-dominance predicate (Laumanns et al., 2002).

    Quantizes each objective into ε-boxes and applies ordinary Pareto
    dominance on the resulting integer box coordinates.  Two solutions that
    fall in the same ε-box are mutually non-dominating; a solution in a
    strictly better box (lower box index in every minimization objective,
    higher in every maximization objective) dominates the other.

    Two quantization modes are supported:

    - **additive** (default): box index = ``floor(f_i / eps_i)``.
      As ``eps → 0`` the boxes shrink to individual points and the relation
      recovers ordinary Pareto dominance.

    - **multiplicative**: box index = ``floor(log f_i / log(1 + eps_i))``.
      Requires strictly positive objective values (``f > 0``).

    Parameters
    ----------
    eps : float or np.ndarray
        Box width per objective.  A scalar applies the same width to every
        objective; an array of shape ``(n_obj,)`` sets per-objective widths.
        All values must be strictly positive (> 0).
    mode : {"additive", "multiplicative"}
        Quantization mode.  Default is ``"additive"``.

    Raises
    ------
    ValueError
        If any element of ``eps`` is not strictly positive (≤ 0).
    ValueError
        If ``mode`` is not one of ``"additive"`` or ``"multiplicative"``.

    Notes
    -----
    Additive box rule::

        b_i = floor(f_i / eps_i)

    Multiplicative box rule (requires f_i > 0 for all i)::

        b_i = floor(log(f_i) / log(1 + eps_i))

    Because ``log(1 + eps_i) = log1p(eps_i)`` is used internally, the
    multiplicative rule is numerically stable for small ``eps``.

    The dominance-matrix computation delegates entirely to an internal
    :class:`ParetoDominator` instance operating on the quantized box
    coordinates.  This guarantees that :meth:`dominates` (inherited from
    :class:`Dominator`) and :meth:`dominance_matrix` always agree.

    References
    ----------
    .. [1] Laumanns, M., Thiele, L., Deb, K., & Zitzler, E. (2002).
       Combining Convergence and Diversity in Evolutionary Multiobjective
       Optimization. *Evolutionary Computation*, 10(3), 263-282.
       https://doi.org/10.1162/106365602760234108
    """

    def __init__(
        self,
        eps: float | np.ndarray,
        mode: str = "additive",
    ) -> None:
        eps_arr = np.atleast_1d(np.asarray(eps, dtype=float))
        if np.any(eps_arr <= 0):
            raise ValueError(
                f"All eps values must be strictly positive (> 0); got {eps_arr}"
            )
        if mode not in ("additive", "multiplicative"):
            raise ValueError(
                f"mode must be 'additive' or 'multiplicative'; got {mode!r}"
            )
        self._eps = eps_arr
        self._mode = mode
        # Internal delegate: Pareto dominance on box coordinates.
        self._pareto = ParetoDominator()

    @property
    def eps(self) -> np.ndarray:
        """Box widths per objective."""
        return self._eps

    @property
    def mode(self) -> str:
        """Quantization mode ('additive' or 'multiplicative')."""
        return self._mode

    def _quantize(self, f: np.ndarray) -> np.ndarray:
        """
        Map objective matrix *f* to integer ε-box coordinates.

        Parameters
        ----------
        f : np.ndarray
            Objective matrix, shape (n, n_obj).

        Returns
        -------
        np.ndarray
            Integer box-index matrix, shape (n, n_obj), dtype float64
            (floor of real-valued box coordinates; kept as float so that
            ParetoDominator can operate on them without type issues).

        Raises
        ------
        ValueError
            In multiplicative mode, if any element of *f* is ≤ 0.
        """
        if self._mode == "additive":
            return np.floor(f / self._eps)
        # multiplicative mode
        if np.any(f <= 0):
            raise ValueError(
                "Multiplicative ε-dominance requires all objective values to be "
                "strictly positive (f > 0).  Found non-positive values in f."
            )
        return np.floor(np.log(f) / np.log1p(self._eps))

    def dominance_matrix(
        self,
        f: np.ndarray,
        direction: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute the NxN ε-dominance matrix.

        Quantizes *f* into ε-box coordinates and delegates to
        :class:`ParetoDominator` on those coordinates so that direction
        handling and the accumulation logic are not duplicated.

        ``D[i, j]`` is True if row i ε-dominates row j.

        Parameters
        ----------
        f : np.ndarray
            Objective matrix. shape: (n, n_obj).  Must contain no NaN values
            (callers are responsible for pre-filtering).
        direction : np.ndarray or None
            Per-objective direction. ``None`` → minimize all.

        Returns
        -------
        np.ndarray
            Boolean matrix of shape (n, n).

        Raises
        ------
        ValueError
            In multiplicative mode, if any element of *f* is ≤ 0.
        """
        b = self._quantize(np.asarray(f, dtype=float))
        return self._pareto.dominance_matrix(b, direction)


class Comparator(ABC):
    """
    Base class for comparator.

    Attributes
    ----------
    weights : np.ndarray
        Weights for objectives. shape = (n_obj, )
    eps_cv : float
        Epsilon for constraint violation feasibility threshold.
    eps_obj : float
        Epsilon for objective value equality comparison.
    direction : np.ndarray or None
        Per-objective optimization directions (+1 = maximize, -1 = minimize).
        None means all objectives are minimized.
    """

    @abstractmethod
    def __init__(
        self,
        weights: np.ndarray,
        eps_cv: float,
        eps_obj: float,
        direction: np.ndarray | None = None,
    ):
        self.weights = weights
        self.eps_cv = eps_cv
        self.eps_obj = eps_obj
        self.direction = direction

    @property
    def eps(self) -> float:
        """Deprecated. Use eps_cv or eps_obj."""
        warn_deprecated("Comparator.eps", "eps_cv or eps_obj", "0.1.0")
        return self.eps_cv

    @abstractmethod
    def sort_population(self, population: Population) -> np.ndarray:
        """
        Sort population based on their attributes.

        Parameters
        ----------
        population : Population
            The population to sort.

        Returns
        -------
        np.ndarray
            Sorted population indices.
        """
        pass

    @abstractmethod
    def compare_population(self, population: Population, idx_a: int, idx_b: int) -> int:
        """
        Compare individual a and b in population.

        Parameters
        ----------
        population : Population
            The population to compare.
        idx_a : int
            Index of the first individual.
        idx_b : int
            Index of the second individual.

        Returns
        -------
        int
            Comparison result.
            -1 if the individual a is better than the individual b.
            1 if the individual a is worse than the individual b.
            0 if the individual a is equal to the individual b.
        """
        pass

    @abstractmethod
    def compare(self, fa: np.ndarray, cv_a: float, fb: np.ndarray, cv_b: float) -> int:
        """
        Compare two solutions directly from raw values.

        This is the low-overhead variant of ``compare_population`` that avoids
        constructing or indexing into a Population object.

        Parameters
        ----------
        fa : np.ndarray
            Objective values of solution a. shape = (n_obj,)
        cv_a : float
            Constraint violation of solution a.
        fb : np.ndarray
            Objective values of solution b. shape = (n_obj,)
        cv_b : float
            Constraint violation of solution b.

        Returns
        -------
        int
            -1 if a is better than b, 1 if b is better than a, 0 if equal.
        """
        pass


class SingleObjectiveComparator(Comparator):
    """Comparator for single-objective optimization."""

    @deprecated_param("weight", "direction", "0.1.0")
    def __init__(
        self,
        direction: float = 1.0,
        eps: float | None = None,
        *,
        eps_cv: float = 1e-6,
        eps_obj: float = 1e-6,
    ):
        if eps is not None:
            warn_deprecated("eps", "eps_cv and eps_obj", "0.1.0")
            eps_cv = eps_obj = eps
        super().__init__(
            np.array([direction]), eps_cv, eps_obj, direction=np.array([direction])
        )

    def sort_population(self, population: Population) -> np.ndarray:
        """
        Sort population based on their fitness and constraint violations.

        Parameters
        ----------
        population : Population
            The population to sort.

        Returns
        -------
        np.ndarray
            Sorted population indices.
        """
        f = population.get("f")
        cv = population.get("cv")
        return self._sort(f, cv)

    def compare_population(self, population: Population, idx_a: int, idx_b: int) -> int:
        """Compare individuals a and b; returns -1/0/1."""
        f = population.get("f")
        cv = population.get("cv")
        return self.compare(f[idx_a], cv[idx_a], f[idx_b], cv[idx_b])

    def compare(self, fa: np.ndarray, cv_a: float, fb: np.ndarray, cv_b: float) -> int:
        """Compare two solutions directly without a Population object."""
        return self._compare(fa, cv_a, fb, cv_b)

    def _compare(
        self, fitness_a: np.ndarray, cv_a: float, fitness_b: np.ndarray, cv_b: float
    ) -> int:
        """Constraint-domination comparison; -1=a better, 1=b better, 0=equal."""
        if cv_a > self.eps_cv and cv_b > self.eps_cv:
            if cv_a < cv_b:
                return -1
            elif cv_a > cv_b:
                return 1
            else:
                return 0
        elif cv_a > self.eps_cv and cv_b <= self.eps_cv:
            return 1
        elif cv_a <= self.eps_cv and cv_b > self.eps_cv:
            return -1
        else:
            if fitness_a[0] < fitness_b[0] - self.eps_obj:
                return -1
            elif fitness_a[0] > fitness_b[0] + self.eps_obj:
                return 1
            else:
                return 0

    def _sort(self, fitness: np.ndarray, cv: np.ndarray) -> np.ndarray:
        """
        Sort solutions based on their fitness and constraint violations.

        Parameters
        ----------
        fitness : np.ndarray
            Objective values of solutions. shape = (n_individuals, n_obj)
        cv : np.ndarray
            Constraint violations of solutions. shape = (n_individuals, )

        Returns
        -------
        np.ndarray
            Sorted indices of the solutions.
        """
        cv_key = np.where(cv > self.eps_cv, cv, 0)
        obj_key = fitness.flatten() * self.direction[0]
        return np.lexsort((-obj_key, cv_key))


class WeightedSumComparator(Comparator):
    """
    Comparator for multi-objective optimization via weighted scalarization.

    Aggregates multiple objectives into a single scalar using a weighted
    dot product: score = f @ weights. The sign convention for weights
    follows the same pattern as SingleObjectiveComparator: use negative
    weights for minimization (e.g., weights=np.array([-1.0, -1.0])).

    Also supports single-objective problems (n_obj == 1).

    Parameters
    ----------
    direction : np.ndarray
        Per-objective optimization directions (±1). shape = (n_obj,)
    eps : float
        Epsilon tolerance for constraint violation and fitness comparison.
    """

    @deprecated_param("weights", "direction", "0.1.0")
    def __init__(
        self,
        direction: np.ndarray | None = None,
        eps: float | None = None,
        *,
        eps_cv: float = 1e-6,
        eps_obj: float = 1e-6,
    ):
        if direction is None:
            raise TypeError(
                "WeightedSumComparator() missing required argument: 'direction'"
            )
        if eps is not None:
            warn_deprecated("eps", "eps_cv and eps_obj", "0.1.0")
            eps_cv = eps_obj = eps
        _dir = np.asarray(direction, dtype=float)
        super().__init__(_dir, eps_cv, eps_obj, direction=_dir)

    def sort_population(self, population: Population) -> np.ndarray:
        """Sort population by weighted sum of objectives, with feasibility first."""
        f = population.get("f")  # (n_ind, n_obj)
        cv = population.get("cv")  # (n_ind,)
        scalar = f @ self.direction  # (n_ind,) weighted sum per individual
        cv_key = np.where(cv > self.eps_cv, cv, 0)
        return np.lexsort((-scalar, cv_key))

    def compare_population(self, population: Population, idx_a: int, idx_b: int) -> int:
        """Compare two individuals by weighted sum; -1=a better, 1=b better, 0=equal."""
        f = population.get("f")
        cv = population.get("cv")
        return self.compare(f[idx_a], float(cv[idx_a]), f[idx_b], float(cv[idx_b]))

    def compare(self, fa: np.ndarray, cv_a: float, fb: np.ndarray, cv_b: float) -> int:
        """Compare two solutions directly without a Population object."""
        return self._compare(fa, cv_a, fb, cv_b)

    def _compare(self, fa: np.ndarray, cv_a: float, fb: np.ndarray, cv_b: float) -> int:
        if cv_a > self.eps_cv and cv_b > self.eps_cv:
            if cv_a < cv_b:
                return -1
            elif cv_a > cv_b:
                return 1
            else:
                return 0
        elif cv_a > self.eps_cv:
            return 1
        elif cv_b > self.eps_cv:
            return -1
        sa = float(np.dot(fa, self.direction))
        sb = float(np.dot(fb, self.direction))
        if sa > sb + self.eps_obj:
            return -1
        elif sa < sb - self.eps_obj:
            return 1
        return 0


# ---------------------------------------------------------------------------
# Non-dominated sorting utilities
# ---------------------------------------------------------------------------

# Module-level singleton used by legacy wrappers (_pareto_dominates,
# _dominance_matrix) so callers that import them directly stay correct.
_PARETO_DOMINATOR: ParetoDominator = ParetoDominator()


def _pareto_dominates(
    fa: np.ndarray, fb: np.ndarray, direction: np.ndarray | None = None
) -> bool:
    """
    Return True if fa Pareto-dominates fb.

    NaN values in fa are treated as non-dominating (returns False).

    .. deprecated::
        Thin wrapper kept for backward compatibility.  Use
        ``ParetoDominator().dominates(fa, fb, direction)`` instead.

    Parameters
    ----------
    fa, fb : np.ndarray
        Objective vectors to compare.
    direction : np.ndarray or None
        Per-objective optimization direction: +1 = maximize, -1 = minimize.
        None defaults to minimization for all objectives.
    """
    return _PARETO_DOMINATOR.dominates(fa, fb, direction)


def _dominance_matrix(g: np.ndarray) -> np.ndarray:
    """
    Compute the NxN boolean dominance matrix from a direction-transformed matrix.

    ``D[i, j]`` is True if row i Pareto-dominates row j.  The input ``g``
    must already have the direction transform applied (smaller = better for
    every objective) and must contain no NaN values.

    .. deprecated::
        Thin wrapper kept for backward compatibility.  Use
        ``ParetoDominator().dominance_matrix(g)`` instead.

    Parameters
    ----------
    g : np.ndarray
        Objective matrix after direction transform. shape: (k, n_obj).
        Must contain no NaN values.

    Returns
    -------
    np.ndarray
        Boolean matrix of shape (k, k).
    """
    # g is already direction-transformed; pass direction=None so no second
    # transform is applied inside ParetoDominator.
    return _PARETO_DOMINATOR.dominance_matrix(g)


def non_dominated_sort(
    f: np.ndarray,
    direction: np.ndarray | None = None,
    *,
    dominator: Dominator | None = None,
) -> tuple[np.ndarray, list[list[int]]]:
    """
    Non-dominated sorting (Deb et al., 2002) via a vectorized dominance matrix.

    Time is O(MN^2), but the pairwise dominance relations are computed with a
    NumPy dominance matrix accumulated one objective at a time (peak memory
    O(N^2), no (N, N, M) tensor), replacing the original Python double loop.

    NaN rows are treated as infinitely bad and placed in the last front.

    Parameters
    ----------
    f : np.ndarray
        Objective matrix. shape: (n, n_obj)
    direction : np.ndarray or None
        Per-objective optimization direction: +1 = maximize, -1 = minimize.
        None defaults to minimization for all objectives.
    dominator : Dominator or None
        Dominance predicate to use.  ``None`` defaults to
        ``ParetoDominator`` (standard Pareto dominance).

    Returns
    -------
    ranks : np.ndarray shape (n,)
        Pareto front index for each individual (0 = first/best front).
    fronts : list[list[int]]
        fronts[i] contains the local indices of individuals in front i.
    """
    _dom = dominator if dominator is not None else _PARETO_DOMINATOR

    n = len(f)
    nan_mask = np.any(np.isnan(f), axis=1)
    valid = np.where(~nan_mask)[0]

    ranks = np.full(n, -1, int)
    fronts: list[list[int]] = []

    if len(valid) > 0:
        # Feed only direction-untransformed valid rows to the dominator; the
        # dominator is responsible for applying the direction transform.
        g = f[valid].astype(float)

        # Build the full dominance matrix for valid individuals.
        dom = _dom.dominance_matrix(g, direction)

        # Front-peel: iteratively collect individuals with zero dominators.
        dominated_by_count = dom.sum(axis=0)  # (k,) — how many dominate each row
        remaining = np.ones(len(valid), dtype=bool)
        k = 0
        while remaining.any():
            # Local indices (within valid) of the current front.
            front_local = np.where(remaining & (dominated_by_count == 0))[0]
            # Map back to global indices.
            front_global = valid[front_local].tolist()
            ranks[valid[front_local]] = k
            fronts.append(front_global)
            # Remove this front and decrement counts for individuals they dominated.
            remaining[front_local] = False
            dominated_by_count -= dom[front_local].sum(axis=0)
            k += 1

    # Push NaN individuals to a final sentinel front (one per NaN row).
    last_rank = len(fronts)
    for i in np.where(nan_mask)[0]:
        ranks[i] = last_rank
        fronts.append([int(i)])

    return ranks, fronts


def dda_non_dominated_sort(
    f: np.ndarray,
    direction: np.ndarray | None = None,
    *,
    dominator: Dominator | None = None,
) -> tuple[np.ndarray, list[list[int]]]:
    """
    Non-dominated sorting via the Dominance-Degree Approach (DDA-ENS).

    Produces **identical** ``(ranks, fronts)`` to :func:`non_dominated_sort`
    and satisfies the :class:`NonDominatedSorter` Protocol.  It is intended
    as a scalable drop-in for large *N* and/or large *M* (M > 100).

    The dominance relation is computed once as an *NxN* boolean matrix
    (peak memory O(N^2), **independent of M**; no (N, N, M) tensor is ever
    allocated).  Front assignment then follows the DDA-ENS scheme: solutions
    are ordered by ascending dominance-in-degree (number of solutions that
    dominate them), so that a solution's dominators are always processed
    before the solution itself.  Each solution is assigned to the first
    existing front whose members do **not** dominate it, creating a new
    front when necessary.

    NaN handling is identical to :func:`non_dominated_sort`: rows that
    contain any NaN value are excluded from the dominance computation and
    appended afterwards as individual sentinel fronts at rank
    ``len(real_fronts)``.

    Parameters
    ----------
    f : np.ndarray
        Objective matrix.  shape: (n, n_obj)
    direction : np.ndarray or None
        Per-objective optimization direction: +1 = maximize, -1 = minimize.
        ``None`` defaults to minimization for all objectives.
    dominator : Dominator or None
        Dominance predicate to use.  ``None`` defaults to
        ``ParetoDominator`` (standard Pareto dominance).

    Returns
    -------
    ranks : np.ndarray
        Pareto front index for each individual (0 = first/best front).
        shape: (n,), dtype int.
    fronts : list[list[int]]
        ``fronts[i]`` contains the global indices of individuals in front i.

    References
    ----------
    .. [1] Zhou, Y., Chen, Z., & Zhang, J. (2017). Ranking Vectors by Means of
       the Dominance Degree Matrix. IEEE Transactions on Evolutionary
       Computation, 21(1), 34-51. https://ieeexplore.ieee.org/document/7469397
    .. [2] DDA-ENS: Dominance Degree Approach based Efficient Non-dominated
       Sort. IEEE Conference Publication (2020).
       https://ieeexplore.ieee.org/document/9282978
    """
    _dom = dominator if dominator is not None else _PARETO_DOMINATOR

    n = len(f)
    nan_mask = np.any(np.isnan(f), axis=1)
    valid = np.where(~nan_mask)[0]

    ranks = np.full(n, -1, int)
    fronts: list[list[int]] = []

    if len(valid) > 0:
        g = f[valid].astype(float)
        k = len(valid)

        # Build dominance matrix once: D[i, j] = True means valid[i] dominates valid[j].
        # Memory O(N²), no (N, N, M) tensor.
        dom_mat = _dom.dominance_matrix(g, direction)

        # in_degree[i] = number of solutions that dominate solution i (locally).
        in_degree = dom_mat.sum(axis=0)  # (k,)

        # Process solutions in ascending in-degree order so that dominators of
        # a solution are always assigned to a front before the solution itself.
        # This guarantees each solution can be placed in the correct front in a
        # single pass.
        order = np.argsort(in_degree, kind="stable")

        # front_members[r] = list of local indices assigned to front r.
        front_members: list[list[int]] = []

        local_ranks = np.full(k, -1, int)

        for i in order:
            assigned = False
            # Scan fronts in increasing rank order; assign to the first front
            # where no current member dominates solution i.
            for r, members in enumerate(front_members):
                # dom_mat[m, i] is True if member m dominates solution i.
                dominated = any(dom_mat[m, i] for m in members)
                if not dominated:
                    front_members[r].append(i)
                    local_ranks[i] = r
                    assigned = True
                    break
            if not assigned:
                # No existing front is safe; open a new one.
                local_ranks[i] = len(front_members)
                front_members.append([i])

        # Map local indices back to global indices and build output.
        for members in front_members:
            global_members = valid[members].tolist()
            fronts.append(global_members)
            ranks[valid[members]] = local_ranks[members]

    # Append NaN individuals as individual sentinel fronts (same contract as
    # non_dominated_sort: each NaN row gets rank = len(real_fronts)).
    last_rank = len(fronts)
    for i in np.where(nan_mask)[0]:
        ranks[i] = last_rank
        fronts.append([int(i)])

    return ranks, fronts


class NonDominatedSorter(Protocol):
    """
    Protocol for non-dominated sorting callables.

    Any callable that accepts an objective matrix ``f``, an optional
    per-objective direction array, and an optional ``dominator`` keyword
    argument and returns ``(ranks, fronts)`` satisfies this protocol.
    The free function ``non_dominated_sort`` is the default implementation.

    Parameters
    ----------
    f : np.ndarray
        Objective matrix.  shape: (n, n_obj)
    direction : np.ndarray or None
        Per-objective optimization direction: +1 = maximize, -1 = minimize.
        ``None`` defaults to minimization for all objectives.
    dominator : Dominator or None
        Dominance predicate to use.  ``None`` defaults to
        ``ParetoDominator``.

    Returns
    -------
    ranks : np.ndarray
        Pareto front index for each individual (0 = first/best front).
        shape: (n,)
    fronts : list[list[int]]
        ``fronts[i]`` contains the local indices of individuals in front i.
    """

    def __call__(
        self,
        f: np.ndarray,
        direction: np.ndarray | None = None,
        *,
        dominator: Dominator | None = None,
    ) -> tuple[np.ndarray, list[list[int]]]:
        """Sort individuals into non-dominated fronts."""
        ...


def crowding_distance(f_front: np.ndarray) -> np.ndarray:
    """
    Compute crowding distance for a single Pareto front (NSGA-II).

    Boundary solutions (minimum and maximum per objective) are assigned
    infinite crowding distance.

    Parameters
    ----------
    f_front : np.ndarray
        Objective values of individuals in the front. shape: (n, n_obj)

    Returns
    -------
    np.ndarray
        Crowding distances. shape: (n,)
    """
    n, m = f_front.shape
    cd = np.zeros(n)
    if n <= 2:
        cd[:] = np.inf
        return cd
    for obj in range(m):
        order = np.argsort(f_front[:, obj])
        cd[order[0]] = np.inf
        cd[order[-1]] = np.inf
        f_range = f_front[order[-1], obj] - f_front[order[0], obj]
        if f_range < 1e-12:
            continue
        for k in range(1, n - 1):
            cd[order[k]] += (
                f_front[order[k + 1], obj] - f_front[order[k - 1], obj]
            ) / f_range
    return cd


def crowding_distance_all_fronts(
    f: np.ndarray,
    fronts: list[list[int]],
) -> np.ndarray:
    """
    Compute crowding distance for all fronts.

    Parameters
    ----------
    f : np.ndarray
        Objective matrix. shape: (n, n_obj)
    fronts : list[list[int]]
        Output of non_dominated_sort: fronts[i] = local indices in front i.

    Returns
    -------
    np.ndarray
        Crowding distances for all individuals. shape: (n,)
    """
    cd = np.zeros(len(f))
    for front in fronts:
        if not front:
            continue
        idx = np.array(front)
        cd[idx] = crowding_distance(f[idx])
    return cd


def spea2_fitness(
    f: np.ndarray,
    direction: np.ndarray | None = None,
    dominator: Dominator | None = None,
) -> np.ndarray:
    """
    Compute SPEA2 fitness values for a set of objective vectors.

    Implements the fitness assignment procedure from Zitzler et al. (2001).
    Lower fitness is better (this deviates from the library's general
    higher-is-better score convention — do NOT pass the result directly
    to comparators that assume higher = better).

    Fitness components:

    - **Strength** ``S(i)``: number of solutions that ``i`` dominates.
    - **Raw fitness** ``R(i)``: sum of the strengths of all solutions that
      dominate ``i``.  ``R(i) = 0`` iff ``i`` is non-dominated.
    - **Density** ``D(i) = 1 / (sigma_i^k + 2)``, where ``sigma_i^k`` is the
      Euclidean distance (in objective space) from ``i`` to its k-th nearest
      neighbour. The ``+2`` keeps ``D in (0, 0.5]``.
    - **Fitness** ``F(i) = R(i) + D(i)``.

    .. note::
        **Deviation from the paper**: Zitzler et al. (2001) define
        ``k = floor(√(N + N̄))`` using the combined size of the population
        and an external archive.  This utility operates on a single set of
        ``N`` points, so it uses ``k = floor(√N)`` instead, clamped to
        ``[0, N-1]``.

    Parameters
    ----------
    f : np.ndarray
        Objective matrix.  shape: (N, n_obj).
    direction : np.ndarray or None
        Per-objective optimization direction: ``+1`` = maximize,
        ``-1`` = minimize.  ``None`` defaults to minimization for all
        objectives.  Passed directly to the dominator (same convention as
        :class:`ParetoComparator`).
    dominator : Dominator or None
        Dominance predicate used to build the dominance matrix.  ``None``
        defaults to :class:`ParetoDominator` (standard Pareto dominance).

    Returns
    -------
    np.ndarray
        SPEA2 fitness values ``F``, shape ``(N,)``.  **Lower = better.**

    References
    ----------
    .. [1] Zitzler, E., Laumanns, M., & Thiele, L. (2001). SPEA2: Improving
       the Strength Pareto Evolutionary Algorithm.  TIK-Report 103, ETH
       Zurich, Switzerland.
    """
    f = np.asarray(f, dtype=float)
    n_pts = len(f)

    if n_pts == 0:
        return np.empty(0, dtype=float)

    _dom = dominator if dominator is not None else ParetoDominator()

    # --- Strength and raw fitness ---
    # dom_mat[i, j] = True iff i dominates j
    dom_mat = _dom.dominance_matrix(f, direction)

    # strength[i] = number of solutions that i dominates (row sum)
    strength = dom_mat.sum(axis=1).astype(float)

    # raw_fitness[i] = sum of strength[j] for all j that dominate i
    # dom_mat[j, i] = True iff j dominates i; weight each by strength[j]
    raw_fitness = dom_mat.T.astype(float) @ strength  # shape (n_pts,)

    # --- Density ---
    # Pairwise Euclidean distances in objective space; shape (n_pts, n_pts)
    diff = f[:, None, :] - f[None, :, :]  # (n_pts, n_pts, n_obj)
    dist_mat = np.sqrt((diff**2).sum(axis=2))  # (n_pts, n_pts)

    # k-th nearest neighbour distance (self-distance 0 sits at index 0)
    k = int(np.floor(np.sqrt(n_pts)))
    k = min(k, n_pts - 1)  # clamp to valid index

    # Sort each row ascending and take the k-th column
    dist_sorted = np.sort(dist_mat, axis=1)  # (n_pts, n_pts)
    sigma_k = dist_sorted[:, k]  # (n_pts,)

    density = 1.0 / (sigma_k + 2.0)

    return raw_fitness + density


class ParetoComparator(Comparator):
    """
    Comparator for multi-objective optimization via non-dominated sorting only.

    Implements rank-only Pareto ordering:
    - sort_population: Pareto front rank order (no crowding distance tiebreaking).
      Infeasible individuals (cv > eps) are always ranked after feasible ones,
      ordered by ascending constraint violation.
    - compare_population: Pareto dominance (-1 = a dominates b, 1 = b
      dominates a, 0 = non-dominated). Infeasibility is handled with the
      standard constraint-domination rule.

    NaN objective values are treated as non-dominating and placed last within
    the feasible block (consistent with non_dominated_sort).

    This class is a concrete base for Pareto-based comparators. NSGA2Comparator
    inherits from it and overrides sort_population to add crowding-distance-based
    secondary ordering within each front.

    Parameters
    ----------
    direction : np.ndarray or None
        Per-objective optimization directions (+1 = maximize, -1 = minimize).
        None means all objectives are minimized (standard Pareto dominance).
    eps : float
        Epsilon tolerance for constraint violation.
    """

    def __init__(
        self,
        direction: np.ndarray | None = None,
        eps: float | None = None,
        *,
        eps_cv: float = 1e-6,
        eps_obj: float = 1e-6,
        sorter: NonDominatedSorter = non_dominated_sort,
        dominator: Dominator | None = None,
    ):
        if eps is not None:
            warn_deprecated("eps", "eps_cv", "0.1.0")
            eps_cv = eps
        direction_arr = (
            np.asarray(direction, dtype=float) if direction is not None else None
        )
        super().__init__(np.empty(0), eps_cv, eps_obj, direction=direction_arr)
        self._sorter = sorter
        # Use None-sentinel to avoid a shared mutable default.
        self._dominator: Dominator = (
            dominator if dominator is not None else ParetoDominator()
        )

    @property
    def sorter(self) -> NonDominatedSorter:
        """The non-dominated sorting callable used by this comparator."""
        return self._sorter

    @property
    def dominator(self) -> Dominator:
        """The dominance predicate used by this comparator."""
        return self._dominator

    def sort_population(self, population: Population) -> np.ndarray:
        """Sort by Pareto front rank; infeasible individuals come last."""
        f = population.get("f")
        cv = population.get("cv")
        feasible = np.where(cv <= self.eps_cv)[0]
        infeasible = np.where(cv > self.eps_cv)[0]

        sorted_feasible = np.empty(0, int)
        if len(feasible):
            ranks, _ = self._sorter(
                f[feasible], direction=self.direction, dominator=self._dominator
            )
            order = np.argsort(ranks, kind="stable")
            sorted_feasible = feasible[order]

        sorted_infeasible = np.empty(0, int)
        if len(infeasible):
            sorted_infeasible = infeasible[np.argsort(cv[infeasible])]

        return np.concatenate([sorted_feasible, sorted_infeasible]).astype(int)

    def compare_population(self, population: Population, idx_a: int, idx_b: int) -> int:
        """Compare via Pareto dominance; -1=a dominates, 1=b dominates, 0=equal."""
        f = population.get("f")
        cv = population.get("cv")
        return self.compare(f[idx_a], float(cv[idx_a]), f[idx_b], float(cv[idx_b]))

    def compare(self, fa: np.ndarray, cv_a: float, fb: np.ndarray, cv_b: float) -> int:
        """Compare two solutions directly without a Population object."""
        if cv_a > self.eps_cv and cv_b > self.eps_cv:
            if cv_a < cv_b:
                return -1
            elif cv_a > cv_b:
                return 1
            return 0
        elif cv_a > self.eps_cv:
            return 1
        elif cv_b > self.eps_cv:
            return -1

        if self._dominator.dominates(fa, fb, self.direction):
            return -1
        if self._dominator.dominates(fb, fa, self.direction):
            return 1
        return 0


class NSGA2Comparator(ParetoComparator):
    """
    Comparator for multi-objective optimization via NSGA-II style ranking.

    Extends ParetoComparator with crowding-distance-based secondary ordering:
    - sort_population: non-dominated sorting + crowding distance
    - compare_population: inherited from ParetoComparator (Pareto dominance)

    Infeasible individuals (cv > eps) are always ranked after feasible
    ones, ordered by ascending constraint violation.

    The sort result is cached in the Population cache and automatically
    invalidated when the population is modified.

    Parameters
    ----------
    direction : np.ndarray or None
        Per-objective optimization directions (+1 = maximize, -1 = minimize).
        None means all objectives are minimized.
    eps : float
        Epsilon tolerance for constraint violation.
    """

    def __init__(
        self,
        direction: np.ndarray | None = None,
        eps: float | None = None,
        *,
        eps_cv: float = 1e-6,
        eps_obj: float = 1e-6,
        sorter: NonDominatedSorter = non_dominated_sort,
        dominator: Dominator | None = None,
    ):
        if eps is not None:
            warn_deprecated("eps", "eps_cv", "0.1.0")
            eps_cv = eps
        super().__init__(
            direction,
            eps_cv=eps_cv,
            eps_obj=eps_obj,
            sorter=sorter,
            dominator=dominator,
        )

    def sort_population(self, population: Population) -> np.ndarray:
        """Sort by Pareto front rank then crowding distance (NSGA-II style)."""
        cached = population.get_cache("pareto_sort")
        if cached is not None:
            return cached

        f = population.get("f")  # (n, n_obj)
        cv = population.get("cv")  # (n,)
        feasible = np.where(cv <= self.eps_cv)[0]
        infeasible = np.where(cv > self.eps_cv)[0]

        sorted_feasible = np.empty(0, int)
        if len(feasible):
            ranks, fronts = self._sorter(
                f[feasible], direction=self.direction, dominator=self._dominator
            )
            cd = crowding_distance_all_fronts(f[feasible], fronts)
            order = np.lexsort((-cd, ranks))
            sorted_feasible = feasible[order]

        sorted_infeasible = np.empty(0, int)
        if len(infeasible):
            sorted_infeasible = infeasible[np.argsort(cv[infeasible])]

        result = np.concatenate([sorted_feasible, sorted_infeasible]).astype(int)
        population.set_cache("pareto_sort", result)
        return result


class SPEA2Comparator(Comparator):
    """
    Comparator implementing SPEA2 fitness-based ranking (Zitzler et al., 2001).

    SPEA2 fitness ``F(i) = R(i) + D(i)`` is computed over the **entire feasible
    set** and used as the ranking criterion — lower is better.  Because the
    fitness depends on the whole population, pairwise comparison of two isolated
    points is undefined.

    Ordering rules:

    - **Feasible block**: sorted by ascending SPEA2 fitness (lower = better).
      See :func:`spea2_fitness` for the ``k = √N`` density reduction note.
    - **Infeasible block**: always placed after feasible individuals, ordered by
      ascending constraint violation (Deb 2000 feasibility rule).

    .. note::
        ``compare()`` raises :exc:`NotImplementedError` because SPEA2 fitness is
        population-relative.  Components that require pairwise comparison (e.g.
        PSO pbest update, ``PairwiseComparisonSet``) should use a
        :class:`ParetoComparator` instead.  Tournament selection should call
        ``compare_population()``, which IS defined and safe to use.

    Parameters
    ----------
    direction : np.ndarray or None
        Per-objective optimization directions (+1 = maximize, -1 = minimize).
        None defaults to minimization for all objectives.
    eps_cv : float
        Feasibility threshold for constraint violation.
    eps_obj : float
        Epsilon for objective-value equality (stored for interface compatibility).
    dominator : Dominator or None
        Dominance predicate.  ``None`` defaults to :class:`ParetoDominator`.

    References
    ----------
    .. [1] Zitzler, E., Laumanns, M., & Thiele, L. (2001). SPEA2: Improving the
       Strength Pareto Evolutionary Algorithm.  TIK-Report 103, ETH Zurich.
    .. [2] Deb, K. (2000). An efficient constraint handling method for genetic
       algorithms.  Computer Methods in Applied Mechanics and Engineering,
       186(2-4), 311-338.
    """

    is_population_relative: bool = True
    """Marker indicating that ``compare()`` is unavailable for this comparator."""

    def __init__(
        self,
        direction: np.ndarray | None = None,
        *,
        eps_cv: float = 1e-6,
        eps_obj: float = 1e-6,
        dominator: Dominator | None = None,
    ):
        direction_arr = (
            np.asarray(direction, dtype=float) if direction is not None else None
        )
        super().__init__(np.empty(0), eps_cv, eps_obj, direction=direction_arr)
        self._dominator: Dominator = (
            dominator if dominator is not None else ParetoDominator()
        )

    @property
    def dominator(self) -> Dominator:
        """The dominance predicate used by this comparator."""
        return self._dominator

    def _fitness(self, population: Population) -> np.ndarray:
        """
        Return the length-N SPEA2 fitness array, computing and caching as needed.

        Infeasible individuals (cv > eps_cv) receive ``+inf`` so they naturally
        sort to the end.  Feasible rows with any NaN objective also receive
        ``+inf`` so they sort to the end of the feasible block.
        """
        cached = population.get_cache("spea2_fitness")
        if cached is not None:
            return cached

        f_arr = population.get("f")
        cv_arr = population.get("cv")
        n = len(f_arr)

        fitness_all = np.full(n, np.inf)  # infeasible -> +inf (sort last)
        feasible = np.where(cv_arr <= self.eps_cv)[0]
        if len(feasible):
            f_feasible = spea2_fitness(
                f_arr[feasible],
                direction=self.direction,
                dominator=self._dominator,
            )
            # Rows with any NaN objective -> +inf so they sort after valid feasibles
            nan_mask = np.isnan(f_arr[feasible]).any(axis=1)
            f_feasible = np.where(nan_mask, np.inf, f_feasible)
            fitness_all[feasible] = f_feasible

        population.set_cache("spea2_fitness", fitness_all)
        return fitness_all

    def sort_population(self, population: Population) -> np.ndarray:
        """
        Sort by SPEA2 fitness (ascending); infeasible individuals come last.

        Parameters
        ----------
        population : Population
            The population to sort.

        Returns
        -------
        np.ndarray
            Sorted population indices (int).
        """
        cv_arr = population.get("cv")
        fitness_all = self._fitness(population)

        feasible = np.where(cv_arr <= self.eps_cv)[0]
        infeasible = np.where(cv_arr > self.eps_cv)[0]

        sorted_feasible: np.ndarray = np.empty(0, int)
        if len(feasible):
            order = np.argsort(fitness_all[feasible], kind="stable")
            sorted_feasible = feasible[order]

        sorted_infeasible: np.ndarray = np.empty(0, int)
        if len(infeasible):
            inf_order = np.argsort(cv_arr[infeasible], kind="stable")
            sorted_infeasible = infeasible[inf_order]

        return np.concatenate([sorted_feasible, sorted_infeasible]).astype(int)

    def compare_population(self, population: Population, idx_a: int, idx_b: int) -> int:
        """
        Compare two individuals using constraint-domination then SPEA2 fitness.

        Returns ``-1`` if ``a`` is better, ``1`` if ``b`` is better, ``0`` if
        equal.  The feasibility rule (Deb 2000) is applied first: feasible
        individuals always beat infeasible ones.  Among both-infeasible pairs,
        lower constraint violation wins.  Among both-feasible pairs, lower SPEA2
        fitness wins.

        Parameters
        ----------
        population : Population
            The population containing both individuals.
        idx_a : int
            Index of the first individual.
        idx_b : int
            Index of the second individual.

        Returns
        -------
        int
            ``-1``, ``0``, or ``1``.
        """
        cv_arr = population.get("cv")
        cv_a = float(cv_arr[idx_a])
        cv_b = float(cv_arr[idx_b])

        # Both infeasible: lower cv wins
        if cv_a > self.eps_cv and cv_b > self.eps_cv:
            if cv_a < cv_b:
                return -1
            elif cv_a > cv_b:
                return 1
            return 0

        # One infeasible: feasible wins
        if cv_a > self.eps_cv:
            return 1
        if cv_b > self.eps_cv:
            return -1

        # Both feasible: lower SPEA2 fitness wins
        fitness_all = self._fitness(population)
        fa = fitness_all[idx_a]
        fb = fitness_all[idx_b]
        if fa < fb:
            return -1
        elif fa > fb:
            return 1
        return 0

    def compare(
        self,
        fa: np.ndarray,
        cv_a: float,
        fb: np.ndarray,
        cv_b: float,
    ) -> int:
        """Raise NotImplementedError — SPEA2 fitness is population-relative."""
        raise NotImplementedError(
            "SPEA2Comparator.compare() is undefined: SPEA2 fitness is "
            "population-relative and cannot be computed from two isolated "
            "points. Use compare_population() / sort_population(), or supply "
            "a ParetoComparator for components that require pairwise "
            "compare() (e.g. PSO pbest update, PairwiseComparisonSet)."
        )


class HypervolumeComparator(ParetoComparator):
    """Comparator using front rank and exclusive HV contribution (SMS-EMOA style).

    Ordering rules:

    - **Feasibility first** (Deb 2000): infeasible individuals (cv > eps_cv)
      are placed after all feasible ones, ordered by ascending constraint
      violation.
    - **Primary key**: non-dominated front rank (ascending, lower = better).
    - **Secondary key** (within a front): exclusive hypervolume contribution
      (descending, higher = better).

    .. note::
        **Generalization from SMS-EMOA.** The original SMS-EMOA
        (Beume et al., 2007) computes HV contributions only on the *last*
        (worst) front to determine the single removal candidate at each
        generation.  This comparator applies HV-contribution ordering *within
        every front* to produce a full ranking over the entire population.
        This is a deliberate generalization that enables use in standard
        survivor-selection and tournament-selection contexts.

    .. warning::
        Computing hypervolume is exponential in the number of objectives, and
        :func:`~saealib.utils.indicators.hypervolume_contributions` performs
        O(N) hypervolume evaluations per front (leave-one-out).  For large
        populations or many objectives this becomes expensive.

    .. note::
        ``compare()`` raises :exc:`NotImplementedError` because the
        hypervolume contribution of a point depends on the other points in
        the population (it is population-relative).  Components that require
        pairwise comparison (e.g. PSO pbest update,
        ``PairwiseComparisonSet``) should use a :class:`ParetoComparator`
        instead.  Tournament selection should call ``compare_population()``,
        which IS defined and safe to use.

    Parameters
    ----------
    direction : np.ndarray or None
        Per-objective optimization directions (+1 = maximize, -1 = minimize).
        None defaults to minimization for all objectives.
    eps_cv : float
        Feasibility threshold for constraint violation.
    eps_obj : float
        Epsilon for objective-value equality (stored for interface
        compatibility).
    reference_point : np.ndarray or None
        Reference point in the *original* objective space, shape
        ``(n_obj,)``.  If ``None``, it is auto-computed from the data
        with fractional padding controlled by ``margin``.
    margin : float
        Fractional padding used when auto-computing the reference point.
        Ignored when ``reference_point`` is provided.
    sorter : NonDominatedSorter
        Non-dominated sorting callable.
    dominator : Dominator or None
        Dominance predicate.  ``None`` defaults to :class:`ParetoDominator`.

    References
    ----------
    .. [1] Beume, N., Naujoks, B., & Emmerich, M. (2007).
       SMS-EMOA: Multiobjective selection based on dominated hypervolume.
       *European Journal of Operational Research*, 181(3), 1653-1669.
       https://doi.org/10.1016/j.ejor.2006.08.008
    .. [2] Deb, K. (2000). An efficient constraint handling method for genetic
       algorithms.  *Computer Methods in Applied Mechanics and Engineering*,
       186(2-4), 311-338.
    """

    is_population_relative: bool = True
    """Marker indicating that ``compare()`` is unavailable for this comparator."""

    def __init__(
        self,
        direction: np.ndarray | None = None,
        *,
        eps_cv: float = 1e-6,
        eps_obj: float = 1e-6,
        reference_point: np.ndarray | None = None,
        margin: float = 0.1,
        sorter: NonDominatedSorter = non_dominated_sort,
        dominator: Dominator | None = None,
    ):
        super().__init__(
            direction,
            eps_cv=eps_cv,
            eps_obj=eps_obj,
            sorter=sorter,
            dominator=dominator,
        )
        self._reference_point = (
            None
            if reference_point is None
            else np.asarray(reference_point, dtype=float)
        )
        self._margin = float(margin)

    @property
    def reference_point(self) -> np.ndarray | None:
        """Reference point used for hypervolume computation, or None for auto."""
        return self._reference_point

    @property
    def margin(self) -> float:
        """Fractional padding applied when auto-computing the reference point."""
        return self._margin

    def _keys(self, population: Population) -> tuple[np.ndarray, np.ndarray]:
        """Return per-individual ``(rank_all, contrib_all)`` arrays, cached.

        Arrays have length N (population size).  Infeasible individuals
        receive ``+inf`` rank and ``-inf`` contribution so they sort last.
        Results are cached under the key ``"hv_keys"``.
        """
        cached = population.get_cache("hv_keys")
        if cached is not None:
            return cached  # type: ignore[return-value]

        # Lazy import avoids a circular dependency at module load time.
        from saealib.utils.indicators import hypervolume_contributions

        f_arr = population.get("f")
        cv_arr = population.get("cv")
        n = len(f_arr)

        rank_all = np.full(n, np.inf)  # infeasible → +inf (worst rank)
        contrib_all = np.full(n, -np.inf)  # infeasible → -inf (worst contrib)

        feasible = np.where(cv_arr <= self.eps_cv)[0]
        if len(feasible):
            ranks, fronts = self._sorter(
                f_arr[feasible],
                direction=self.direction,
                dominator=self._dominator,
            )
            rank_all[feasible] = ranks

            for front in fronts:  # front = local indices into feasible subset
                local = np.asarray(front, dtype=int)
                gidx = feasible[local]
                contribs = hypervolume_contributions(
                    f_arr[gidx],
                    reference_point=self._reference_point,
                    direction=self.direction,
                    margin=self._margin,
                )
                contrib_all[gidx] = contribs

        keys = (rank_all, contrib_all)
        population.set_cache("hv_keys", keys)
        return keys

    def sort_population(self, population: Population) -> np.ndarray:
        """
        Sort by front rank then HV contribution; infeasible individuals last.

        Within the feasible block the primary sort key is non-dominated front
        rank (ascending) and the secondary key is exclusive hypervolume
        contribution (descending).  Infeasible individuals follow, sorted by
        ascending constraint violation.

        Parameters
        ----------
        population : Population
            The population to sort.

        Returns
        -------
        np.ndarray
            Sorted population indices (int).
        """
        cv_arr = population.get("cv")
        rank_all, contrib_all = self._keys(population)

        feasible = np.where(cv_arr <= self.eps_cv)[0]
        infeasible = np.where(cv_arr > self.eps_cv)[0]

        sorted_feasible: np.ndarray = np.empty(0, int)
        if len(feasible):
            # lexsort: last key is primary -> rank ascending, then -contrib ascending
            order = np.lexsort((-contrib_all[feasible], rank_all[feasible]))
            sorted_feasible = feasible[order]

        sorted_infeasible: np.ndarray = np.empty(0, int)
        if len(infeasible):
            sorted_infeasible = infeasible[np.argsort(cv_arr[infeasible])]

        return np.concatenate([sorted_feasible, sorted_infeasible]).astype(int)

    def compare_population(self, population: Population, idx_a: int, idx_b: int) -> int:
        """
        Compare two individuals using constraint-domination then HV ranking.

        Returns ``-1`` if ``a`` is better, ``1`` if ``b`` is better, ``0``
        if equal.  The feasibility rule (Deb 2000) is applied first.  Among
        both-feasible pairs, lower front rank wins; within the same front,
        higher HV contribution wins.

        Parameters
        ----------
        population : Population
            The population containing both individuals.
        idx_a : int
            Index of the first individual.
        idx_b : int
            Index of the second individual.

        Returns
        -------
        int
            ``-1``, ``0``, or ``1``.
        """
        cv_arr = population.get("cv")
        cv_a = float(cv_arr[idx_a])
        cv_b = float(cv_arr[idx_b])

        # Both infeasible: lower cv wins
        if cv_a > self.eps_cv and cv_b > self.eps_cv:
            if cv_a < cv_b:
                return -1
            elif cv_a > cv_b:
                return 1
            return 0

        # One infeasible: feasible wins
        if cv_a > self.eps_cv:
            return 1
        if cv_b > self.eps_cv:
            return -1

        # Both feasible: lower rank wins; tie-break on higher HV contribution
        rank_all, contrib_all = self._keys(population)
        ra = rank_all[idx_a]
        rb = rank_all[idx_b]
        if ra < rb:
            return -1
        elif ra > rb:
            return 1

        ca = contrib_all[idx_a]
        cb = contrib_all[idx_b]
        if ca > cb:
            return -1
        elif ca < cb:
            return 1
        return 0

    def compare(
        self,
        fa: np.ndarray,
        cv_a: float,
        fb: np.ndarray,
        cv_b: float,
    ) -> int:
        """Raise NotImplementedError — HV contribution is population-relative."""
        raise NotImplementedError(
            "HypervolumeComparator.compare() is undefined: the exclusive "
            "hypervolume contribution of a point is population-relative and "
            "cannot be computed from two isolated points.  Use "
            "compare_population() / sort_population(), or supply a "
            "ParetoComparator for components that require pairwise "
            "compare() (e.g. PSO pbest update, PairwiseComparisonSet)."
        )


class EpsilonDominanceComparator(ParetoComparator):
    """
    Comparator using ε-box dominance instead of standard Pareto dominance.

    Wraps :class:`EpsilonDominator` and injects it into the
    :class:`ParetoComparator` dominance seam via the ``dominator=`` argument.
    Front ranking, infeasibility handling, and constraint-domination logic are
    all inherited unchanged from :class:`ParetoComparator`.

    The ε-dominance relation is defined in:

        Laumanns, M., Thiele, L., Deb, K., & Zitzler, E. (2002).
        Combining convergence and diversity in evolutionary multiobjective
        optimization. *Evolutionary Computation*, 10(3), 263-282.

    Each objective axis is divided into ε-boxes of width ``eps``.  Two
    solutions that fall in the **same** ε-box are mutually non-dominating;
    a solution whose box index is strictly better in *every* objective
    dominates the other.  As ``eps → 0`` the relation recovers ordinary
    Pareto dominance.

    .. note::
        ε-box **representative selection** (the archive rule that keeps one
        solution per box — cf. Deb, Mohan & Mishra (2005), ε-MOEA) is **not**
        handled here.  That is an archive-truncation responsibility and will
        be added separately.

    Parameters
    ----------
    eps : float or np.ndarray
        Box size(s).  A scalar broadcasts to all objectives; an array of
        shape ``(n_obj,)`` sets per-objective widths.  All values must be
        strictly positive (> 0).
    mode : {"additive", "multiplicative"}
        Quantization mode passed to :class:`EpsilonDominator`.
        ``"additive"`` (default): box index = ``floor(f_i / eps_i)``.
        ``"multiplicative"``: box index = ``floor(log f_i / log(1 + eps_i))``;
        requires strictly positive objective values.
    direction : np.ndarray or None
        See :class:`ParetoComparator`.
    eps_cv : float
        See :class:`ParetoComparator`.
    eps_obj : float
        See :class:`ParetoComparator`.
    sorter : NonDominatedSorter
        See :class:`ParetoComparator`.
    """

    def __init__(
        self,
        eps: float | np.ndarray,
        mode: str = "additive",
        direction: np.ndarray | None = None,
        *,
        eps_cv: float = 1e-6,
        eps_obj: float = 1e-6,
        sorter: NonDominatedSorter = non_dominated_sort,
    ):
        super().__init__(
            direction,
            eps_cv=eps_cv,
            eps_obj=eps_obj,
            sorter=sorter,
            dominator=EpsilonDominator(eps, mode),
        )

    @property
    def eps(self) -> float | np.ndarray:
        """Box size(s) used by the underlying EpsilonDominator."""
        return self._dominator.eps  # type: ignore[attr-defined]

    @property
    def mode(self) -> str:
        """Quantization mode of the underlying EpsilonDominator."""
        return self._dominator.mode  # type: ignore[attr-defined]


def _normalize_objectives(
    f: np.ndarray,
    direction: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize objectives using NSGA-III ideal point + hyperplane intercepts.

    Deb & Jain (2014), Section IV-B.

    Parameters
    ----------
    f : np.ndarray
        Shape ``(n, n_obj)``.
    direction : np.ndarray or None
        ``+1`` maximize, ``-1`` minimize. ``None`` means all-minimize.

    Returns
    -------
    f_norm : np.ndarray
        Shape ``(n, n_obj)``. Normalized objectives.
    ideal : np.ndarray
        Shape ``(n_obj,)``. Ideal point used for translation.
    intercepts : np.ndarray
        Shape ``(n_obj,)``. Hyperplane intercepts used for scaling.
    """
    f_signed = f * direction if direction is not None else f
    ideal = f_signed.min(axis=0)
    f_trans = f_signed - ideal

    n_obj = f_trans.shape[1]
    eps_asf = 1e-6
    extreme = np.empty((n_obj, n_obj))
    for j in range(n_obj):
        w = np.full(n_obj, eps_asf)
        w[j] = 1.0
        asf = np.max(f_trans / w, axis=1)
        extreme[j] = f_trans[np.argmin(asf)]

    try:
        intercepts = np.linalg.solve(extreme, np.ones(n_obj))
        if np.any(intercepts <= 0):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        intercepts = f_trans.max(axis=0).astype(float)
        intercepts[intercepts <= 0] = 1.0

    f_norm = f_trans / intercepts
    return f_norm, ideal, intercepts


def _associate_to_reference_points(
    f_norm: np.ndarray,
    reference_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Associate each individual with the closest reference line.

    Deb & Jain (2014), Section IV-C. Uses perpendicular distance from
    ``f_norm[i]`` to the ray from the origin through ``reference_points[r]``.

    Parameters
    ----------
    f_norm : np.ndarray
        Shape ``(n, n_obj)``.
    reference_points : np.ndarray
        Shape ``(n_ref, n_obj)``.

    Returns
    -------
    assoc_idx : np.ndarray
        Shape ``(n,)`` int. Index of the nearest reference point for each individual.
    dist : np.ndarray
        Shape ``(n,)`` float. Perpendicular distance to that reference line.
    """
    ref_norms_sq = np.sum(reference_points**2, axis=1)  # (n_ref,)
    ref_norms_sq = np.maximum(ref_norms_sq, 1e-30)

    # scalar projection coefficients: (n, n_ref)
    scalars = f_norm @ reference_points.T / ref_norms_sq  # (n, n_ref)
    # projected vectors: (n, n_ref, n_obj)
    proj = scalars[:, :, None] * reference_points[None, :, :]
    # perpendicular distances: (n, n_ref)
    diff = f_norm[:, None, :] - proj
    d_perp = np.linalg.norm(diff, axis=2)

    assoc_idx = np.argmin(d_perp, axis=1)
    dist = d_perp[np.arange(len(f_norm)), assoc_idx]
    return assoc_idx, dist


def _niche_count_select(
    front_local: np.ndarray,
    assoc: np.ndarray,
    dist: np.ndarray,
    niche_count: np.ndarray,
    n_needed: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Order individuals in a front using NSGA-III niche-preservation.

    Deb & Jain (2014), Algorithm 2. Selects ``n_needed`` individuals from
    ``front_local`` in preference order, updating ``niche_count`` in-place.

    Parameters
    ----------
    front_local : np.ndarray
        Local indices (into feasible sub-array) of individuals in this front.
    assoc : np.ndarray
        Shape ``(n_feasible,)``. Reference-point association for each feasible
        individual.
    dist : np.ndarray
        Shape ``(n_feasible,)``. Perpendicular distance to associated reference line.
    niche_count : np.ndarray
        Shape ``(n_ref,)``. Accumulated niche counts; updated in-place.
    n_needed : int
        Number of individuals to select (≤ len(front_local)).
    rng : np.random.Generator
        Random number generator for tie-breaking.

    Returns
    -------
    np.ndarray
        Shape ``(n_needed,)``. Selected indices from ``front_local`` in priority order.
    """
    pool = list(front_local)
    selected = []
    for _ in range(n_needed):
        # reference points that still have candidates in the pool
        pool_refs = {int(assoc[i]) for i in pool}
        min_nc = min(niche_count[r] for r in pool_refs)
        min_refs = [r for r in pool_refs if niche_count[r] == min_nc]
        chosen_ref = int(rng.choice(min_refs))

        candidates = [i for i in pool if assoc[i] == chosen_ref]
        if niche_count[chosen_ref] == 0:
            i_star = candidates[int(np.argmin(dist[candidates]))]
        else:
            i_star = int(rng.choice(candidates))

        selected.append(i_star)
        pool.remove(i_star)
        niche_count[chosen_ref] += 1

    return np.array(selected, dtype=int)


class NSGA3Comparator(ParetoComparator):
    """
    Comparator for many-objective optimization via NSGA-III style ranking.

    Extends ParetoComparator with reference-point-based niche preservation:

    - sort_population: non-dominated sorting + normalization + niche preservation
    - compare_population: inherited from ParetoComparator (Pareto dominance)

    The NSGA-III selection mechanism (Deb & Jain 2014, Algorithm 1):

    1. Non-dominated sorting (same fronts as NSGA-II).
    2. Normalize objectives: translate by ideal point then scale by hyperplane
       intercepts computed from extreme points (ASF minimizers).
    3. Associate each individual with the nearest reference line (perpendicular
       distance from the individual to the ray from origin through each
       reference point).
    4. Order each front using niche-count-based preference: reference points
       with fewer associated individuals from earlier fronts are preferred;
       ties broken randomly.

    The total ordering returned by sort_population processes all fronts in order,
    accumulating niche counts across fronts so that earlier fronts' niche counts
    propagate to later ones—matching the selection pressure NSGA-III would apply
    when truncating to a target population size.

    Parameters
    ----------
    reference_points : np.ndarray
        Shape ``(n_ref, n_obj)``. Reference points on the unit simplex.
        Typically generated by
        :func:`~saealib.utils.weight_vectors.uniform_weight_vectors`.
    direction : np.ndarray or None
        Per-objective optimization directions (``+1`` maximize, ``-1`` minimize).
        ``None`` means all objectives are minimized.
    eps_cv : float
        Constraint-violation tolerance.
    eps_obj : float
        Objective tolerance (passed to base class).
    sorter : NonDominatedSorter
        Non-dominated sorting callable.
    dominator : Dominator or None
        Dominance predicate.
    seed : int or None
        Seed for the random number generator used in niche tie-breaking.

    References
    ----------
    .. [1] Deb, K., & Jain, H. (2014). An Evolutionary Many-Objective
       Optimization Algorithm Using Reference-Point-Based Nondominated
       Sorting Approach, Part I. IEEE Transactions on Evolutionary
       Computation, 18(4), 577-601. https://doi.org/10.1109/TEVC.2013.2281535
    """

    def __init__(
        self,
        reference_points: np.ndarray,
        direction: np.ndarray | None = None,
        *,
        eps_cv: float = 1e-6,
        eps_obj: float = 1e-6,
        sorter: NonDominatedSorter = non_dominated_sort,
        dominator: Dominator | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            direction,
            eps_cv=eps_cv,
            eps_obj=eps_obj,
            sorter=sorter,
            dominator=dominator,
        )
        self._reference_points = np.asarray(reference_points, dtype=float)
        self._rng = np.random.default_rng(seed)

    @property
    def reference_points(self) -> np.ndarray:
        """Reference points used for niche preservation."""
        return self._reference_points

    def sort_population(self, population: Population) -> np.ndarray:
        """Sort by Pareto front rank then NSGA-III niche preservation."""
        cached = population.get_cache("pareto_sort")
        if cached is not None:
            return cached

        f = population.get("f")
        cv = population.get("cv")
        feasible = np.where(cv <= self.eps_cv)[0]
        infeasible = np.where(cv > self.eps_cv)[0]

        sorted_feasible: list[int] = []
        if len(feasible):
            _ranks, fronts = self._sorter(
                f[feasible], direction=self.direction, dominator=self._dominator
            )
            f_norm, _, _ = _normalize_objectives(f[feasible], self.direction)
            assoc, dist = _associate_to_reference_points(f_norm, self._reference_points)
            niche_count = np.zeros(len(self._reference_points), dtype=int)

            for front_list in fronts:
                front_local = np.array(front_list, dtype=int)
                ordered = _niche_count_select(
                    front_local, assoc, dist, niche_count, len(front_local), self._rng
                )
                sorted_feasible.extend(feasible[ordered].tolist())

        sorted_infeasible: np.ndarray = np.empty(0, int)
        if len(infeasible):
            sorted_infeasible = infeasible[np.argsort(cv[infeasible])]

        result = np.concatenate(
            [np.array(sorted_feasible, dtype=int), sorted_infeasible]
        ).astype(int)
        population.set_cache("pareto_sort", result)
        return result


class RNSGA2Comparator(ParetoComparator):
    """
    Comparator for multi-objective optimization via R-NSGA-II style ranking.

    Extends ParetoComparator with reference-point-based secondary ordering:

    - sort_population: non-dominated sorting + reference-point proximity ordering
    - compare_population: inherited from ParetoComparator (Pareto dominance)

    The R-NSGA-II selection mechanism (Deb & Sundar 2006):

    1. Non-dominated sorting (same fronts as NSGA-II).
    2. Normalize objectives by range (min-max per objective).
    3. Associate each individual with the nearest reference point (Euclidean
       distance in normalized objective space).
    4. Within each front, sort by ``(niche_count[assoc], dist_to_nearest_ref)``
       ascending. Apply ε-clearing: solutions within ``epsilon`` of a
       higher-ranked solution with the same associated reference point are moved
       to the back of their reference-point group.

    Parameters
    ----------
    reference_points : np.ndarray
        Shape ``(n_ref, n_obj)``. User-supplied reference points (aspiration
        points) in the objective space. Unlike NSGA-III, these need not lie on
        the unit simplex.
    epsilon : float
        ε-clearing radius in normalized objective space. Solutions within
        ``epsilon`` of a better-ranked solution sharing the same reference
        point are deprioritized.
    direction : np.ndarray or None
        Per-objective optimization directions (``+1`` maximize, ``-1`` minimize).
        ``None`` means all objectives are minimized.
    eps_cv : float
        Constraint-violation tolerance.
    eps_obj : float
        Objective tolerance (passed to base class).
    sorter : NonDominatedSorter
        Non-dominated sorting callable.
    dominator : Dominator or None
        Dominance predicate.

    References
    ----------
    .. [1] Deb, K., & Sundar, J. (2006). Reference Point Based Multi-Objective
       Optimization Using Evolutionary Algorithms. Proceedings of GECCO 2006,
       635-642. https://doi.org/10.1145/1143997.1144112
    """

    def __init__(
        self,
        reference_points: np.ndarray,
        epsilon: float = 0.001,
        direction: np.ndarray | None = None,
        *,
        eps_cv: float = 1e-6,
        eps_obj: float = 1e-6,
        sorter: NonDominatedSorter = non_dominated_sort,
        dominator: Dominator | None = None,
    ) -> None:
        super().__init__(
            direction,
            eps_cv=eps_cv,
            eps_obj=eps_obj,
            sorter=sorter,
            dominator=dominator,
        )
        self._reference_points = np.asarray(reference_points, dtype=float)
        self._epsilon = float(epsilon)

    @property
    def reference_points(self) -> np.ndarray:
        """Reference points used for preference-based ordering."""
        return self._reference_points

    @property
    def epsilon(self) -> float:
        """ε-clearing radius in normalized objective space."""
        return self._epsilon

    def sort_population(self, population: Population) -> np.ndarray:
        """Sort by Pareto front rank then reference-point proximity (R-NSGA-II)."""
        cached = population.get_cache("pareto_sort")
        if cached is not None:
            return cached

        f = population.get("f")
        cv = population.get("cv")
        feasible = np.where(cv <= self.eps_cv)[0]
        infeasible = np.where(cv > self.eps_cv)[0]

        sorted_feasible: list[int] = []
        if len(feasible):
            _ranks, fronts = self._sorter(
                f[feasible], direction=self.direction, dominator=self._dominator
            )

            # Range normalization in minimize-sense
            f_signed = (
                f[feasible] * self.direction
                if self.direction is not None
                else f[feasible]
            )
            f_min = f_signed.min(axis=0)
            f_max = f_signed.max(axis=0)
            f_norm = (f_signed - f_min) / np.maximum(f_max - f_min, 1e-12)

            # Euclidean distance to each reference point
            diff = f_norm[:, None, :] - self._reference_points[None, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)  # (n_feasible, n_ref)
            assoc_idx = np.argmin(dist_matrix, axis=1)  # (n_feasible,)
            dist_min = dist_matrix[np.arange(len(feasible)), assoc_idx]

            niche_count = np.zeros(len(self._reference_points), dtype=int)

            for front_list in fronts:
                front_local = np.array(front_list, dtype=int)
                ordered = self._order_front(
                    front_local, assoc_idx, dist_min, niche_count
                )
                for idx in ordered:
                    niche_count[assoc_idx[idx]] += 1
                sorted_feasible.extend(feasible[ordered].tolist())

        sorted_infeasible: np.ndarray = np.empty(0, int)
        if len(infeasible):
            sorted_infeasible = infeasible[np.argsort(cv[infeasible])]

        result = np.concatenate(
            [np.array(sorted_feasible, dtype=int), sorted_infeasible]
        ).astype(int)
        population.set_cache("pareto_sort", result)
        return result

    def _order_front(
        self,
        front_local: np.ndarray,
        assoc_idx: np.ndarray,
        dist_min: np.ndarray,
        niche_count: np.ndarray,
    ) -> np.ndarray:
        """Order a single front by (niche_count, dist) with ε-clearing."""
        # Primary sort: (niche_count of associated ref, distance)
        nc = niche_count[assoc_idx[front_local]]
        d = dist_min[front_local]
        base_order = np.lexsort((d, nc))  # ascending both
        ordered = front_local[base_order]

        # ε-clearing: within each reference-point group, move solutions
        # that are within epsilon of a higher-ranked solution to the back.
        selected: list[int] = []
        cleared: list[int] = []
        # track the lowest dist already accepted per reference point
        best_dist: dict[int, float] = {}

        for idx in ordered:
            ref = int(assoc_idx[idx])
            d_val = float(dist_min[idx])
            if ref in best_dist and abs(d_val - best_dist[ref]) < self._epsilon:
                cleared.append(idx)
            else:
                selected.append(idx)
                if ref not in best_dist:
                    best_dist[ref] = d_val

        return np.array(selected + cleared, dtype=int)
