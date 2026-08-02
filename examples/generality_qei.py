from saealib import (
    PSO,
    BatchExpectedImprovement,
    CorrelatedQuadraticSurrogate,
    GlobalSurrogateManager,
    LHSInitializer,
    Optimizer,
    PreSelectionStrategy,
    Termination,
    max_fe,
    reference_problem,
)
from saealib.surrogate import ArchiveObjectiveSet


def main():
    """Run joint batch expected improvement."""
    problem = reference_problem()
    return (
        Optimizer(problem, seed=29)
        .set_initializer(LHSInitializer(2, 2, 29))
        .set_algorithm(PSO())
        .set_surrogate_manager(
            GlobalSurrogateManager(
                CorrelatedQuadraticSurrogate(correlation=0.75),
                ArchiveObjectiveSet(),
            )
        )
        .set_acquisition(BatchExpectedImprovement(n_draws=512))
        .set_strategy(PreSelectionStrategy(4, 2))
        .set_termination(Termination(max_fe(6)))
        .run()
    )


if __name__ == "__main__":
    main()
