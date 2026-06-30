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
    SequentialSelection,
    Termination,
    TruncationSelection,
    max_fe,
)
from saealib.surrogate import SklearnGPRSurrogate

logging.basicConfig(level=logging.INFO)
logging.getLogger("saealib.surrogate.sklearn_surrogate").setLevel(logging.CRITICAL)


def test_integration():
    """
    Run SAGA-GP optimization example.

    This test checks the integration of these classes:
    - Optimizer
    - Problem
    - Initializer (LHSInitializer)
    - Algorithm (GA)
    - Crossover (CrossoverBLXAlpha)
    - Mutation (MutationUniform)
    - ParentSelection (SequentialSelection)
    - SurvivorSelection (TruncationSelection)
    - Termination
    - Surrogate (SklearnGPRSurrogate)
    - Strategy (IndividualBasedStrategy)
    """
    # Small problem configuration for fast execution in CI
    dim = 2
    knn = 10
    rsm = 0.2
    ub = [5.0] * dim
    lb = [-5.0] * dim

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
        n_init_archive=10,
        n_init_population=8,
        seed=42,
    )
    algorithm = GA(
        crossover=CrossoverBLXAlpha(prob=0.7, alpha=0.4),
        mutation=MutationUniform(prob_var=0.3),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )
    termination = Termination(max_fe(100))
    surrogate = SklearnGPRSurrogate()
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
    test_integration()
