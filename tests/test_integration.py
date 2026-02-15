import json
import logging
import os

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
    RBFsurrogate,
    SequentialSelection,
    Termination,
    TruncationSelection,
    gaussian_kernel,
)

logging.basicConfig(level=logging.INFO)
logging.getLogger("saealib.surrogate.rbf").setLevel(logging.CRITICAL)

# Load snapshot
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "data/test_integration.json")
if not os.path.exists(filename):
    pytest.skip("Snapshot file not found. Run generate_test_integration.py first.")
with open(filename) as f:
    SNAPSHOT = json.load(f)


@pytest.mark.parametrize("seed_str, expected_f", SNAPSHOT.items())
def test_integration(seed_str: str, expected_f: float):
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
    seed = int(seed_str)
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

    assert ctx is not None

    best_f = ctx.archive.get("f").min()
    assert best_f < 1.0

    assert np.isclose(best_f, expected_f, atol=1e-5)


if __name__ == "__main__":
    for seed_str, expected_f in SNAPSHOT.items():
        test_integration(seed_str, expected_f)
