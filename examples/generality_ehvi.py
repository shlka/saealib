"""Deterministic two-objective optimization with EHVI."""

from saealib import (
    PSO,
    IndividualBasedStrategy,
    LHSInitializer,
    Optimizer,
    SklearnGPRSurrogate,
    Termination,
    max_fe,
)
from saealib.acquisition import EHVIAcquisition

try:
    from examples._support import two_objective_problem
except ModuleNotFoundError:
    from _support import two_objective_problem


def main():
    """Run a small two-objective EHVI optimization."""
    seed = 5
    problem = two_objective_problem()
    return (
        Optimizer(problem, seed=seed)
        .set_initializer(LHSInitializer(6, problem.dim, seed))
        .set_algorithm(PSO())
        .set_surrogate(
            SklearnGPRSurrogate(alpha=1e-6, optimizer=None, random_state=seed)
        )
        .set_acquisition(EHVIAcquisition(n_samples=32))
        .set_strategy(IndividualBasedStrategy(0.5))
        .set_termination(Termination(max_fe(12)))
        .run()
    )


if __name__ == "__main__":
    main()
