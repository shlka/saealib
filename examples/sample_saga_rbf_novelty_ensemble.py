"""
SAGA-RBF + NoveltyManager ensemble example.

Demonstrates exploration-exploitation balance by combining a regression
surrogate (RBF, exploitation) with a novelty scorer (exploration) via
EnsembleSurrogateManager.

The regression surrogate is listed first so that EnsembleSurrogateManager
returns its predictions as representative, keeping offspring.f finite.
"""

import logging

import numpy as np
from opfunu.cec_based import cec2015

from saealib import (
    GA,
    CrossoverBLXAlpha,
    EnsembleSurrogateManager,
    IndividualBasedStrategy,
    LHSInitializer,
    MutationUniform,
    NoveltyManager,
    Optimizer,
    Problem,
    RBFSurrogate,
    SequentialSelection,
    Termination,
    TruncationSelection,
    gaussian_kernel,
    max_fe,
)
from saealib.acquisition import MeanPrediction
from saealib.surrogate.manager import LocalSurrogateManager

logging.basicConfig(level=logging.INFO)
logging.getLogger("saealib.surrogate.rbf").setLevel(logging.CRITICAL)


def main():
    """Run SAGA-RBF + novelty ensemble optimization example."""
    # parameters
    dim = 10
    seed = 1
    rsm = 0.1
    novelty_k = 3
    novelty_weight = 0.3
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
        crossover=CrossoverBLXAlpha(crossover_rate=0.7, alpha=0.4),
        mutation=MutationUniform(mutation_rate=0.3),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )
    termination = Termination(max_fe(200 * dim))
    strategy = IndividualBasedStrategy(evaluation_ratio=rsm)

    # Regression surrogate must be listed first so that EnsembleSurrogateManager
    # returns its predictions (finite tell_f) as representative.
    surrogate_manager = EnsembleSurrogateManager(
        managers=[
            LocalSurrogateManager(
                RBFSurrogate(gaussian_kernel, dim),
                MeanPrediction(weights=np.array([-1.0])),
            ),
            NoveltyManager(k=novelty_k),
        ],
        weights=np.array([1.0 - novelty_weight, novelty_weight]),
    )

    opt = (
        Optimizer(problem)
        .set_initializer(initializer)
        .set_algorithm(algorithm)
        .set_termination(termination)
        .set_surrogate_manager(surrogate_manager)
        .set_strategy(strategy)
    )
    opt.run()


if __name__ == "__main__":
    main()
