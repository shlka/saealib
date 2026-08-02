"""Private support helpers shared by executable examples."""

from __future__ import annotations

import numpy as np

from saealib import Problem


def reference_problem(dim: int = 2, shift: float = 0.0) -> Problem:
    """Return the bounded quadratic problem used by the examples."""

    def evaluate(x: np.ndarray) -> np.ndarray:
        return np.array([np.sum((x - shift) ** 2)], dtype=np.float64)

    return Problem(
        func=evaluate,
        dim=dim,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0] * dim,
        ub=[1.0] * dim,
    )
