"""
Single-objective optimization example using SA-PSO-RBF on a Sphere function.

This example demonstrates:
- Single-objective Problem setup with PSO
- IndividualBasedStrategy with RBF surrogate (surrogate-assisted PSO)
- Post-processing: retrieve the best solution from the archive
"""

import logging

import numpy as np

from saealib import (
    PSO,
    IndividualBasedStrategy,
    LHSInitializer,
    Optimizer,
    Problem,
    RBFsurrogate,
    Termination,
    gaussian_kernel,
    max_fe,
)

logging.basicConfig(level=logging.INFO)
logging.getLogger("saealib.surrogate.rbf").setLevel(logging.CRITICAL)


def sphere(x: np.ndarray) -> np.ndarray:
    """Sphere benchmark function. Minimum: f(0,...,0) = 0."""
    return np.array([np.sum(x**2)])


def main():
    """Run SA-PSO-RBF optimization on Sphere."""
    # parameters
    dim = 10
    seed = 1
    knn = 50
    rsm = 0.1
    lb = [-5.0] * dim
    ub = [5.0] * dim

    problem = Problem(
        func=sphere,
        dim=dim,
        n_obj=1,
        weight=np.array([-1.0]),
        lb=lb,
        ub=ub,
    )
    initializer = LHSInitializer(
        n_init_archive=5 * dim,
        n_init_population=4 * dim,
        seed=seed,
    )
    algorithm = PSO(w=0.7, c1=1.5, c2=1.5)
    termination = Termination(max_fe(200 * dim))
    surrogate = RBFsurrogate(gaussian_kernel, dim)
    strategy = IndividualBasedStrategy(evaluation_ratio=rsm)

    opt = (
        Optimizer(problem)
        .set_initializer(initializer)
        .set_algorithm(algorithm)
        .set_termination(termination)
        .set_surrogate(surrogate, n_neighbors=knn)
        .set_strategy(strategy)
    )
    ctx = opt.run()

    archive_f = ctx.archive.get_array("f")  # (n_archive, 1)
    archive_x = ctx.archive.get_array("x")  # (n_archive, dim)
    best_idx = int(np.argmin(archive_f[:, 0]))

    print(f"\n=== Results (FE={ctx.fe}) ===")
    print(f"Archive size : {len(archive_f)}")
    print(f"Best f       : {archive_f[best_idx, 0]:.6f}")
    print(f"Best x       : {archive_x[best_idx]}")


if __name__ == "__main__":
    main()
