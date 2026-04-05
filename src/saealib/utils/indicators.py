"""Multi-objective quality indicators."""

from __future__ import annotations

import numpy as np


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
    """Return non-dominated subset (minimization, O(n²))."""
    n = len(f)
    if n <= 1:
        return f
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(f[j] <= f[i]) and np.any(f[j] < f[i]):
                dominated[i] = True
                break
    return f[~dominated]
