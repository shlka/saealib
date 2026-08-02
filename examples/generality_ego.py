from saealib import (
    PSO,
    IndividualBasedStrategy,
    LHSInitializer,
    Optimizer,
    ParEGOAcquisition,
    SklearnGPRSurrogate,
    Termination,
    max_fe,
)

try:
    from examples._support import reference_problem
except ModuleNotFoundError:
    from _support import reference_problem


def main():
    """Run expected-improvement global optimization."""
    problem = reference_problem()
    return (
        Optimizer(problem, seed=23)
        .set_initializer(LHSInitializer(2, 2, 23))
        .set_algorithm(PSO())
        .set_surrogate(SklearnGPRSurrogate(alpha=1e-6, optimizer=None, random_state=23))
        .set_acquisition(ParEGOAcquisition())
        .set_strategy(IndividualBasedStrategy(0.5))
        .set_termination(Termination(max_fe(6)))
        .run()
    )


if __name__ == "__main__":
    main()
