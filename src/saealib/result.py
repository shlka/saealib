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

    ``x``/``f``/``fe``/``gen`` are fixed at creation time (the arrays are
    copied).  ``ctx``/``history``/``archive``/``pareto_archive``/``population``
    are references derived from the state and may change while iteration is
    still in progress.

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

    @classmethod
    def from_state(cls, state: OptimizationState) -> Result:
        """Build a result from the final or currently observed state."""
        archive_x = state.archive.get_array("x")
        archive_f = state.archive.get_array("f")
        archive_cv = state.archive.get_array("cv")
        direction = state.problem.direction
        eps = state.problem.eps_cv

        feasible = np.where(archive_cv <= eps)[0]
        pool = feasible if len(feasible) else np.array([int(np.argmin(archive_cv))])

        if state.problem.n_obj == 1:
            scores = archive_f[pool] @ direction
            best_idx = pool[int(np.argmax(scores))]
            best_x = archive_x[best_idx]
            best_f = archive_f[best_idx]
        else:
            if len(state.pareto_archive) > 0:
                best_x = state.pareto_archive.get_array("x")
                best_f = state.pareto_archive.get_array("f")
            else:
                from saealib.comparators import non_dominated_sort

                _, fronts = non_dominated_sort(archive_f[pool], direction=direction)
                pareto_idx = pool[fronts[0]]
                best_x = archive_x[pareto_idx]
                best_f = archive_f[pareto_idx]

        return cls(
            x=np.array(best_x, copy=True),
            f=np.array(best_f, copy=True),
            fe=int(state.fe),
            gen=int(state.gen),
            history=state.history,
            ctx=state,
        )

    @property
    def problem(self):
        """Return the problem referenced by the optimization state."""
        return self.ctx.problem

    @property
    def archive(self):
        """Return the archive referenced by the optimization state."""
        return self.ctx.archive

    @property
    def pareto_archive(self):
        """Return the Pareto archive referenced by the optimization state."""
        return self.ctx.pareto_archive

    @property
    def population(self):
        """Return the population referenced by the optimization state."""
        return self.ctx.population
