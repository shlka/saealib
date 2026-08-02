"""SAGA-RBF + novelty acquisition ensemble example.

Demonstrates exploration-exploitation balance by combining a regression
surrogate (RBF, exploitation) with a novelty acquisition (exploration) via
CompositeSurrogateManager and CompositeAcquisition.

The objective channel supplies the predicted objective values.
"""

import logging

import numpy as np
from opfunu.cec_based import cec2015

from saealib import (
    GA,
    CompositeSurrogateManager,
    CrossoverBLXAlpha,
    IndividualBasedStrategy,
    LHSInitializer,
    MutationUniform,
    NoveltyAcquisition,
    Optimizer,
    Problem,
    RBFSurrogate,
    SequentialSelection,
    Termination,
    TruncationSelection,
    gaussian_kernel,
    max_fe,
)
from saealib.acquisition import CompositeAcquisition, MeanPrediction
from saealib.surrogate.manager import GlobalSurrogateManager, rank_weighted_combine

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

    surrogate_manager = CompositeSurrogateManager(
        managers={
            "objective": GlobalSurrogateManager(RBFSurrogate(gaussian_kernel, dim)),
            "novelty": GlobalSurrogateManager(RBFSurrogate(gaussian_kernel, dim)),
        }
    )
    acquisition = CompositeAcquisition(
        acquisitions={
            "objective": MeanPrediction(weights=np.array([-1.0])),
            "novelty": NoveltyAcquisition(k=novelty_k),
        },
        combine_fn=rank_weighted_combine(
            np.array([1.0 - novelty_weight, novelty_weight])
        ),
    )

    opt = (
        Optimizer(problem)
        .set_initializer(initializer)
        .set_algorithm(algorithm)
        .set_termination(termination)
        .set_surrogate_manager(surrogate_manager)
        .set_acquisition(acquisition)
        .set_strategy(strategy)
    )
    opt.run()


if __name__ == "__main__":
    main()
