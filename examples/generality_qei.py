import numpy as np

from saealib import (
    PSO,
    BatchExpectedImprovement,
    GlobalSurrogateManager,
    LHSInitializer,
    Optimizer,
    PreSelectionStrategy,
    Termination,
    max_fe,
)
from saealib.surrogate import (
    ArchiveObjectiveSet,
    PredictionChannel,
    Surrogate,
    SurrogatePrediction,
)

try:
    from examples._support import reference_problem
except ModuleNotFoundError:
    from _support import reference_problem


class CorrelatedQuadraticSurrogate(Surrogate):
    """Small deterministic joint-posterior surrogate for this example."""

    provides_uncertainty = True

    def __init__(self, correlation: float = 0.7):
        if not 0.0 <= correlation < 1.0:
            raise ValueError("correlation must be in [0, 1)")
        self.correlation = float(correlation)

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """Keep the example posterior independent of training data."""

    def predict(self, test_x: np.ndarray) -> SurrogatePrediction:
        """Return correlated objective uncertainty for the query batch."""
        x = np.atleast_2d(np.array(test_x, dtype=np.float64, copy=True))
        n = len(x)
        mean = np.sum(x * x, axis=1, keepdims=True)
        std = np.full((n, 1), 0.25, dtype=np.float64)
        covariance = np.full((n, n), self.correlation * 0.25**2)
        np.fill_diagonal(covariance, 0.25**2)
        return SurrogatePrediction(
            channels={
                "objective": PredictionChannel(
                    value=mean, std=std, covariance=covariance
                )
            },
            x=x,
        )


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
