"""Multi-objective quality indicators."""

from __future__ import annotations

import numpy as np

from saealib.comparators import _dominance_matrix


def hypervolume(f: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Compute hypervolume indicator (minimization convention).

    Uses a recursive slicing algorithm. Complexity: O(n^(m-1) * n log n),
    where n is the number of points and m is the number of objectives.

    Parameters
    ----------
    f : np.ndarray
        Objective matrix, shape (n, n_obj).
    reference_point : np.ndarray
        Reference (nadir) point, shape (n_obj,). Each component must be
        strictly greater than the corresponding component of at least one
        point in f for a non-zero hypervolume to be returned.

    Returns
    -------
    float
        Hypervolume value.
    """
    f = np.asarray(f, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)

    if f.ndim == 1:
        f = f.reshape(1, -1)

    # Keep only points that are strictly dominated by the reference point
    mask = np.all(f < reference_point, axis=1)
    f = f[mask]

    if len(f) == 0:
        return 0.0

    return _hv(f, reference_point)


def hypervolume_contributions(
    f: np.ndarray,
    reference_point: np.ndarray | None = None,
    direction: np.ndarray | None = None,
    margin: float = 0.1,
) -> np.ndarray:
    r"""
    Compute exclusive hypervolume contributions for each point.

    The exclusive contribution of point ``i`` is defined as::

        Delta_HV(i) = HV(S) - HV(S \ {i})

    i.e. the volume exclusively dominated by point ``i`` relative to the
    reference point.  Higher is better, consistent with the library-wide
    convention.  Dominated points and points not strictly better than the
    reference point contribute 0.

    This function reuses :func:`hypervolume` and therefore inherits the
    minimisation convention internally.  Use the ``direction`` parameter to
    handle maximisation objectives.

    .. warning::
        Computing hypervolume is exponential in the number of objectives, and
        the leave-one-out loop adds a factor of N.  For many objectives or
        large N this becomes prohibitively expensive.  A future improvement
        may incorporate the WFG algorithm (While et al., 2006) for faster
        batch computation.

    Parameters
    ----------
    f : np.ndarray
        Objective matrix, shape ``(N, n_obj)``.
    reference_point : np.ndarray or None, optional
        Reference point in the *original* objective space, shape
        ``(n_obj,)``.  If ``None``, it is auto-computed from the data with
        a fractional padding controlled by ``margin``.
    direction : np.ndarray or None, optional
        Per-objective optimisation direction: ``+1`` to maximise, ``-1``
        to minimise.  ``None`` defaults to all-minimise (equivalent to
        ``np.full(n_obj, -1.0)``).  Same convention as
        :func:`saealib.comparators.spea2_fitness`.
    margin : float, optional
        Fractional padding used when auto-computing the reference point.
        For each objective axis the auto reference is placed at
        ``nadir + margin * span``.  Ignored when ``reference_point`` is
        provided.

    Returns
    -------
    np.ndarray
        Exclusive hypervolume contribution per point, shape ``(N,)``.
        All values are ``>= 0`` (dominated / out-of-reference points get 0).

    References
    ----------
    Beume, N., Naujoks, B., & Emmerich, M. (2007).
        SMS-EMOA: Multiobjective selection based on dominated hypervolume.
        *European Journal of Operational Research*, 181(3), 1653-1669.
        https://doi.org/10.1016/j.ejor.2006.08.008

    Zitzler, E., & Thiele, L. (1998).
        Multiobjective optimization using evolutionary algorithms - a
        comparative case study.  *PPSN V*, LNCS 1498, pp. 292-301.
    """
    f = np.asarray(f, dtype=float)
    if f.ndim == 1:
        f = f.reshape(1, -1)

    n, n_obj = f.shape
    if n == 0:
        return np.empty(0)

    # Build per-objective sign vector: minimise → s=+1, maximise → s=-1.
    # Multiplying f by s converts every objective to a minimisation objective.
    s = np.ones(n_obj)
    if direction is not None:
        direction = np.asarray(direction, dtype=float)
        s[direction == 1] = -1.0

    g = f * s  # transformed objective matrix (always minimise)

    # Determine reference point in g-space.
    if reference_point is not None:
        ref_g = np.asarray(reference_point, dtype=float) * s
    else:
        nadir = g.max(axis=0)
        ideal = g.min(axis=0)
        span = nadir - ideal
        ref_g = nadir.copy()
        for j in range(n_obj):
            if span[j] == 0.0:
                # Degenerate axis: use absolute pad so ref is strictly > nadir.
                ref_g[j] = nadir[j] + max(margin, 1e-12)
            else:
                ref_g[j] = nadir[j] + margin * span[j]

    total = hypervolume(g, ref_g)

    contrib = np.empty(n)
    for i in range(n):
        g_without = np.delete(g, i, axis=0)
        hv_without = hypervolume(g_without, ref_g)
        contrib[i] = total - hv_without

    # Clip floating-point noise to zero.
    return np.maximum(contrib, 0.0)


def _hv(f: np.ndarray, ref: np.ndarray) -> float:
    """Recursive hypervolume computation (internal)."""
    n, m = f.shape

    if m == 1:
        return float(ref[0] - np.min(f[:, 0]))

    # Sort ascending by last objective
    order = np.argsort(f[:, -1], kind="stable")
    f_s = f[order]

    hv = 0.0
    for i in range(n):
        upper = f_s[i + 1, -1] if i + 1 < n else ref[-1]
        height = upper - f_s[i, -1]
        if height <= 0.0:
            continue
        proj = _non_dominated(f_s[: i + 1, :-1])
        hv += height * _hv(proj, ref[:-1])

    return hv


def _non_dominated(f: np.ndarray) -> np.ndarray:
    """Return non-dominated subset (minimization).

    Parameters
    ----------
    f : np.ndarray
        Objective matrix, shape (n, n_obj).  Must contain no NaN values.

    Returns
    -------
    np.ndarray
        Rows of ``f`` that are not dominated by any other row.
    """
    n = len(f)
    if n <= 1:
        return f
    # dom[i, j] = True iff row i dominates row j; a row is dominated when any
    # column of dom is True for it (i.e. dom.any(axis=0)[j] is True).
    dom = _dominance_matrix(f)
    dominated = dom.any(axis=0)
    return f[~dominated]
