import numpy as np

from saealib import (
    PSO,
    DirectStrategy,
    EvaluationResult,
    LHSInitializer,
    Optimizer,
    RepeatedEvaluation,
    SerialEvaluator,
    Termination,
    max_fe,
)

try:
    from examples._support import reference_problem
except ModuleNotFoundError:
    from _support import reference_problem


class SeededNoiseEvaluator(SerialEvaluator):
    """Example evaluator that adds reproducible objective noise."""

    def __init__(self, seed: int, scale: float = 0.01):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.scale = float(scale)
        self.requests = []

    def evaluate_request(self, request, problem):
        """Record the request while retaining Evaluator's lifecycle adapter."""
        self.requests.append(request)
        return super().evaluate_request(request, problem)

    def evaluate_batch(self, x, problem):
        """Evaluate through the standard serial evaluator and add noise."""
        result = super().evaluate_batch(x, problem)
        noise = self.rng.normal(0.0, self.scale, size=result.f.shape)
        return EvaluationResult(
            f=result.f + noise,
            g=result.g,
            cv=result.cv,
            cost=result.cost,
            noise=noise,
            outputs=result.outputs,
        )


def main():
    """Run repeated evaluations through the standard evaluation plan."""
    problem = reference_problem()
    evaluator = SeededNoiseEvaluator(seed=17, scale=0.02)
    state = (
        Optimizer(problem, seed=17)
        .set_initializer(LHSInitializer(2, 2, 17))
        .set_algorithm(PSO())
        .set_strategy(DirectStrategy(n_offspring=2))
        .set_evaluator(evaluator)
        .set_evaluation_planner(RepeatedEvaluation(3))
        .set_termination(Termination(max_fe(8)))
        .run()
    )
    return {"state": state, "requests": tuple(evaluator.requests)}


if __name__ == "__main__":
    main()
