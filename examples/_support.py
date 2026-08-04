"""Private support helpers shared by executable examples."""

from __future__ import annotations

import numpy as np

from saealib import Problem
from saealib.problem.constraint import InequalityConstraint


def _reference_problem(
    dim: int,
    n_obj: int,
    direction: float,
    shift: float,
    constrained: bool,
) -> Problem:
    """Build a bounded one- or two-objective quadratic problem."""

    def evaluate(x: np.ndarray) -> np.ndarray:
        values = np.array(
            [np.sum((x - shift) ** 2), np.sum((x + shift) ** 2)][:n_obj],
            dtype=np.float64,
        )
        return values if direction < 0 else -values

    constraints = (
        [InequalityConstraint(lambda x: float(np.sum(x)), threshold=0.2)]
        if constrained
        else None
    )
    return Problem(
        func=evaluate,
        dim=dim,
        n_obj=n_obj,
        direction=np.full(n_obj, direction),
        lb=[-1.0] * dim,
        ub=[1.0] * dim,
        constraints=constraints,
    )


def reference_problem(dim: int = 2, shift: float = 0.0) -> Problem:
    """Return the bounded quadratic problem used by the examples."""
    return _reference_problem(dim, 1, -1.0, shift, False)


def maximize_problem(dim: int = 2, shift: float = 0.2) -> Problem:
    """Return the maximizing counterpart of :func:`reference_problem`."""
    return _reference_problem(dim, 1, 1.0, shift, False)


def constrained_problem(
    dim: int = 2, shift: float = 0.2, maximize: bool = False
) -> Problem:
    """Return a constrained single-objective reference problem."""
    return _reference_problem(dim, 1, 1.0 if maximize else -1.0, shift, True)


def two_objective_problem(
    dim: int = 2,
    shift: float = 0.2,
    maximize: bool = False,
    constrained: bool = False,
) -> Problem:
    """Return a two-objective quadratic reference problem."""
    return _reference_problem(dim, 2, 1.0 if maximize else -1.0, shift, constrained)
