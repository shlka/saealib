import logging

import numpy as np
from opfunu.cec_based import cec2015

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

logging.basicConfig(level=logging.INFO)


def main():
    """Run SAGA-RBF optimization example."""
    # parameters
    dim = 10
    seed = 1
    knn = 50
    rsm = 0.1
    ub = [100] * dim
    lb = [-100] * dim

    # benchmark function
    f1 = cec2015.F12015(ndim=10)

    problem = Problem(
        func=f1.evaluate,
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
    opt.run()


if __name__ == "__main__":
    main()
