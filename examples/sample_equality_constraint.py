import logging

import numpy as np

from saealib import (
    EqualityConstraint,
    Problem,
    minimize,
)

logging.basicConfig(level=logging.INFO)
logging.getLogger("saealib.surrogate.rbf").setLevel(logging.CRITICAL)


def main():
    """Minimize on the equality surface x1 + x2 = 1.

    Objective: f(x) = x1^2 + x2^2 (minimize)
    Constraint: h(x) = x1 + x2 - 1 = 0

    The constrained optimum lies at x1 = x2 = 0.5 with f = 0.5, the point on
    the line x1 + x2 = 1 closest to the origin. The equality constraint is
    expressed with EqualityConstraint, whose violation max(0, |h(x)| - tol)
    is aggregated through the default StaticToleranceHandler.

    The tolerance softens the equality into a feasible band
    |x1 + x2 - 1| <= tolerance; a non-trivial band is what makes the surface
    reachable for a finite-budget surrogate-assisted search.
    """
    dim = 2
    seed = 1
    lb = [-2.0] * dim
    ub = [2.0] * dim

    # tolerance defines the feasible band |x1 + x2 - 1| <= tolerance.
    equality = EqualityConstraint(lambda x: x[0] + x[1] - 1.0, tolerance=5e-2)

    problem = Problem(
        func=lambda x: float(x[0] ** 2 + x[1] ** 2),
        dim=dim,
        n_obj=1,
        weight=np.array([-1.0]),
        lb=lb,
        ub=ub,
        constraints=[equality],
    )

    result = minimize(problem, max_fe=400, seed=seed)

    x = result.X
    h = x[0] + x[1] - 1.0
    print(f"x = {x} (optimum: [0.5, 0.5])")
    print(f"f(x) = {result.F[0]:.6f} (optimum: 0.5)")
    print(f"h(x) = x1 + x2 - 1 = {h:.6e} (|h| <= {equality.tolerance:g})")


if __name__ == "__main__":
    main()
