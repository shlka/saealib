"""
Multi-objective optimization example using SAGA-RBF on ZDT1.

ZDT1 (Zitzler-Deb-Thiele function 1) is a 2-objective benchmark:
    f1(x) = x[0]
    g(x)  = 1 + 9 * sum(x[1:]) / (n - 1)
    f2(x) = g(x) * (1 - sqrt(x[0] / g(x)))
    domain: x in [0, 1]^n

Pareto front: f2 = 1 - sqrt(f1) for f1 in [0, 1].

This example demonstrates:
- Multi-objective Problem with automatic NSGA2Comparator selection
- IndividualBasedStrategy with per-objective RBF ensemble surrogate
- Post-processing with non_dominated_sort to extract the Pareto front
"""

import logging

import numpy as np

from saealib import (
    GA,
    CrossoverBLXAlpha,
    IndividualBasedStrategy,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    Problem,
    RBFsurrogate,
    SequentialSelection,
    Termination,
    TruncationSelection,
    gaussian_kernel,
    max_fe,
    non_dominated_sort,
)

logging.basicConfig(level=logging.INFO)
logging.getLogger("saealib.surrogate.rbf").setLevel(logging.CRITICAL)


def zdt1(x: np.ndarray) -> np.ndarray:
    """ZDT1 bi-objective benchmark function."""
    n = len(x)
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return np.array([f1, f2])


def main():
    """Run SAGA-RBF multi-objective optimization example on ZDT1."""
    # parameters
    dim = 10
    seed = 1
    knn = 50
    rsm = 0.1
    lb = [0.0] * dim
    ub = [1.0] * dim

    problem = Problem(
        func=zdt1,
        dim=dim,
        n_obj=2,
        weight=np.array([-1.0, -1.0]),
        lb=lb,
        ub=ub,
    )
    initializer = LHSInitializer(
        n_init_archive=5 * dim,
        n_init_population=4 * dim,
        seed=seed,
    )
    algorithm = GA(
        crossover=CrossoverBLXAlpha(crossover_rate=0.7, gamma=0.4),
        mutation=MutationUniform(mutation_rate=0.3),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )
    termination = Termination(max_fe(200 * dim))
    surrogate = RBFsurrogate(gaussian_kernel, dim)
    strategy = IndividualBasedStrategy(rsm=rsm)

    opt = (
        Optimizer(problem)
        .set_initializer(initializer)
        .set_algorithm(algorithm)
        .set_termination(termination)
        .set_surrogate(surrogate, n_neighbors=knn)
        .set_strategy(strategy)
    )
    ctx = opt.run()

    # --- post-processing: extract the Pareto front ---
    archive_f = ctx.archive.get("f")  # (n_archive, 2)
    _ranks, fronts = non_dominated_sort(archive_f)

    pareto_f = archive_f[fronts[0]]  # front 0 = non-dominated solutions
    pareto_f_sorted = pareto_f[np.argsort(pareto_f[:, 0])]

    print(f"\n=== Results (FE={ctx.fe}) ===")
    print(f"Archive size       : {len(archive_f)}")
    print(f"Pareto front size  : {len(pareto_f)}")
    print(f"Best f1            : {pareto_f[:, 0].min():.4f}")
    print(f"Best f2            : {pareto_f[:, 1].min():.4f}")
    print("\nPareto front (f1, f2) — sorted by f1:")
    for f1, f2 in pareto_f_sorted:
        # reference: f2_true = 1 - sqrt(f1)
        f2_ref = 1.0 - np.sqrt(f1) if f1 >= 0.0 else float("nan")
        print(f"  f1={f1:.4f}  f2={f2:.4f}  (true={f2_ref:.4f})")


if __name__ == "__main__":
    main()
