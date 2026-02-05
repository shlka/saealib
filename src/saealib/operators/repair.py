"""
Repair operators module.

This module defines repair operators to fix individuals
that violate problem constraints.
This module defines repair operators to fix individuals
that violate problem constraints.
"""

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from saealib.context import OptimizationContext

    # from saealib.optimizer import ComponentProvider


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
    ctx: OptimizationContext = kwargs.get("ctx")
    # provider: ComponentProvider = kwargs.get("provider")
    problem = ctx.problem
    repaired = np.clip(data, problem.lb, problem.ub)
    return repaired
