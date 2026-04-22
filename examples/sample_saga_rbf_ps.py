import logging

import numpy as np
from opfunu.cec_based import cec2015

from saealib import (
    GA,
    CrossoverBLXAlpha,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    PreSelectionStrategy,
    Problem,
    RBFsurrogate,
    SequentialSelection,
    Termination,
    TruncationSelection,
    gaussian_kernel,
    max_fe,
)

logging.basicConfig(level=logging.INFO)
logging.getLogger("saealib.surrogate.rbf").setLevel(logging.CRITICAL)


def main():
    """Run SAGA-RBF with PreSelectionStrategy."""
    dim = 10
    seed = 1
    knn = 50
    n_candidates = 4 * dim
    n_select = dim // 2
    ub = [100] * dim
    lb = [-100] * dim

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
        crossover=CrossoverBLXAlpha(crossover_rate=0.7, alpha=0.4),
        mutation=MutationUniform(mutation_rate=0.3),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )
    termination = Termination(max_fe(200 * dim))
    surrogate = RBFsurrogate(gaussian_kernel, dim)
    # n_candidates=40: screen 40 candidates with surrogate each generation
    # n_select=5: evaluate only the top 5 with the true objective
    strategy = PreSelectionStrategy(n_candidates=n_candidates, n_select=n_select)

    opt = (
        Optimizer(problem)
        .set_initializer(initializer)
        .set_algorithm(algorithm)
        .set_termination(termination)
        .set_surrogate(surrogate, n_neighbors=knn)
        .set_strategy(strategy)
    )
    opt.run()


if __name__ == "__main__":
    main()
