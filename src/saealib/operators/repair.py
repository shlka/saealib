"""
Repair operators module.

This module defines repair operators to fix individuals that violate problem constraints.
"""
import numpy as np


def repair_clipping(data: np.ndarray, **kwargs) -> np.ndarray:
    """
    Repair operator that clips the data to the problem's bounds.

    Parameters
    ----------
    data : np.ndarray
        Data to be repaired.

    Returns
    -------
    np.ndarray
        Repaired data.
    """
    problem = kwargs.get("optimizer", None).problem
    repaired = np.clip(data, problem.lb, problem.ub)
    return repaired
