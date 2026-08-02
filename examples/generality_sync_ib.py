from saealib import (
    PSO,
    ExpectedImprovement,
    IndividualBasedStrategy,
    LHSInitializer,
    Optimizer,
    SklearnGPRSurrogate,
    Termination,
    max_fe,
    reference_problem,
)


def main():
    """Run synchronous individual-based optimization."""
    problem = reference_problem()
    return (
        Optimizer(problem, seed=19)
        .set_initializer(LHSInitializer(2, 2, 19))
        .set_algorithm(PSO())
        .set_surrogate(SklearnGPRSurrogate(alpha=1e-6, optimizer=None, random_state=19))
        .set_acquisition(ExpectedImprovement())
        .set_strategy(IndividualBasedStrategy(0.5))
        .set_termination(Termination(max_fe(6)))
        .run()
    )


if __name__ == "__main__":
    main()
