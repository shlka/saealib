import logging

import numpy as np
import pytest

from saealib import (
    GA,
    CrossoverBLXAlpha,
    IndividualBasedStrategy,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    Problem,
    RBFSurrogate,
    SequentialSelection,
    Termination,
    TruncationSelection,
    gaussian_kernel,
    max_fe,
)

logging.basicConfig(level=logging.INFO)
logging.getLogger("saealib.surrogate.rbf").setLevel(logging.CRITICAL)

SEEDS = [42, 43, 44, 45, 46]


@pytest.mark.parametrize("seed", SEEDS)
def test_integration(seed: int):
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
    - Surrogate (RBFSurrogate)
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
        direction=np.array([-1.0]),
        lb=lb,
        ub=ub,
    )
    initializer = LHSInitializer(
        n_init_archive=5 * dim,
        n_init_population=4 * dim,
        seed=seed,
    )
    algorithm = GA(
        crossover=CrossoverBLXAlpha(crossover_rate=0.7, alpha=0.4),
        mutation=MutationUniform(mutation_rate=0.3),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )
    termination = Termination(max_fe(200 * dim))
    surrogate = RBFSurrogate(gaussian_kernel, dim)
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

    assert ctx is not None

    best_f = ctx.archive.get_array("f").min()
    assert best_f < 1.0


if __name__ == "__main__":
    for seed in SEEDS:
        test_integration(seed)
