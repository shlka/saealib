from __future__ import annotations

from typing import Protocol

import numpy as np

from saealib.comparators.dominance import _PARETO_DOMINATOR, Dominator, ParetoDominator


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
