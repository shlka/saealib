"""Repair operators to clip candidates back into bounds."""

import numpy as np


def repair_clipping(
    candidates: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    Repair operator that clips candidates to the given bounds.

    Parameters
    ----------
    candidates : np.ndarray
        Candidate solutions to be repaired, shape (n, dim).
    bounds : tuple[np.ndarray, np.ndarray]
        A (lb, ub) pair of lower and upper bound arrays.

    Returns
    -------
    np.ndarray
        Repaired candidates.
    """
    lb, ub = bounds
    return np.clip(candidates, lb, ub)
