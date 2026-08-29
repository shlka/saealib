"""Optimization result returned by the high-level API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.execution.history import History


@dataclass
class Result:
    """Optimization result returned by :func:`minimize` / :func:`maximize`.

    Attributes
    ----------
    x : np.ndarray
        Best design variables. Shape ``(dim,)`` for single-objective,
        ``(n_pareto, dim)`` for multi-objective.
    f : np.ndarray
        Best objective values. Shape ``(n_obj,)`` for single-objective,
        ``(n_pareto, n_obj)`` for multi-objective.
    fe : int
        Total number of true function evaluations used.
    gen : int
        Total number of generations completed.
    history : History or None
        Execution history recorded during the run.
    ctx : OptimizationState
        Full optimization context providing access to the archive and more.
    """

    x: np.ndarray
    f: np.ndarray
    fe: int
    gen: int
    ctx: OptimizationState
    history: History | None = None
