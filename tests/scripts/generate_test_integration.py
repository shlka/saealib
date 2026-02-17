"""
Generate test integration.

This script generates json files for tests/test_integration.py

If you modify test_integration.py,
please update this script to perform the same processing.
"""

import json
import os
import sys

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
)


def run_benchmark(seed: int):
    """
    Run SAGA-RBF optimization example.

    This test checks these classes:
    - Optimizer
    - Problem
    - Initializer (LHSInitializer)
    - Algorithm (GA)
    - Crossover (CrossoverBLXAlpha)
    - Mutation (MutationUniform)
    - ParentSelection (SequentialSelection)
    - SurvivorSelection (TruncationSelection)
    - Termination
    - Surrogate (RBFsurrogate)
    - Strategy (IndividualBasedStrategy)
    - Callback (logging_generation, repair_clipping)
    """
    # parameters
    dim = 10
    knn = 50
    rsm = 0.1
    ub = [5] * dim
    lb = [-5] * dim

    # benchmark function
    def sphere(x: np.ndarray) -> float:
        return np.sum(x**2)

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
    algorithm = GA(
        crossover=CrossoverBLXAlpha(crossover_rate=0.7, gamma=0.4),
        mutation=MutationUniform(mutation_rate=0.3),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )
    termination = Termination(fe=200 * dim)
    surrogate = RBFsurrogate(gaussian_kernel, dim)
    strategy = IndividualBasedStrategy(knn, rsm)

    opt = (
        Optimizer(problem)
        .set_initializer(initializer)
        .set_algorithm(algorithm)
        .set_termination(termination)
        .set_surrogate(surrogate)
        .set_strategy(strategy)
    )
    ctx = opt.run()

    return ctx


def main():
    if sys.version_info[:2] != (3, 10):
        print(
            f"Error: Snapshot must be generated with Python 3.10 "
            f"(current: {sys.version_info.major}.{sys.version_info.minor})"
        )
        sys.exit(1)

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../data/test_integration.json")
    init_seed = 42
    n_seeds = 5

    seeds = [init_seed + i for i in range(n_seeds)]
    results = {}
    for seed in seeds:
        ctx = run_benchmark(seed)
        results[str(seed)] = ctx.archive.get("f").min()
        print(f"Seed {seed}: {results[str(seed)]}")

    with open(filename, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
