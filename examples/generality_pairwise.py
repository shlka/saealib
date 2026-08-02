from saealib import (
    PSO,
    IndividualBasedStrategy,
    LHSInitializer,
    Optimizer,
    PairwiseComparisonSet,
    PairwiseSurrogateManager,
    SklearnRFCClassificationSurrogate,
    Termination,
    WinRateAcquisition,
    max_fe,
    reference_problem,
)


def main():
    """Run pairwise ranking surrogate optimization."""
    problem = reference_problem()
    manager = PairwiseSurrogateManager(
        SklearnRFCClassificationSurrogate(n_estimators=8, random_state=37),
        training_set=PairwiseComparisonSet(),
    )
    return (
        Optimizer(problem, seed=37)
        .set_initializer(LHSInitializer(6, 4, 37))
        .set_algorithm(PSO())
        .set_surrogate_manager(manager)
        .set_acquisition(WinRateAcquisition())
        .set_strategy(IndividualBasedStrategy(0.5))
        .set_termination(Termination(max_fe(6)))
        .run()
    )


if __name__ == "__main__":
    main()
