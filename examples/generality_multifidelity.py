import numpy as np

from saealib import (
    PSO,
    DirectStrategy,
    EvaluationResult,
    Evaluator,
    FidelityPromotion,
    LHSInitializer,
    Optimizer,
    Termination,
    max_fe,
)

try:
    from examples._support import reference_problem
except ModuleNotFoundError:
    from _support import reference_problem


class FidelityEvaluator(Evaluator):
    """Example evaluator that reads fidelity from request metadata."""

    def __init__(self, evaluate):
        self._evaluate = evaluate
        self.requests = []

    def evaluate_batch(self, x, problem):
        """Evaluate initial candidates at the default fidelity."""
        return self._evaluate_request(x, problem, 0, np.arange(len(x), dtype=np.int64))

    def evaluate_request(self, request, problem):
        """Evaluate a planned request at its explicit fidelity."""
        self.requests.append(request)
        return self._evaluate_request(
            request.x,
            problem,
            int(request.metadata.get("fidelity", 0)),
            request.candidate_ids,
        )

    def _evaluate_request(self, x, problem, fidelity, candidate_ids):
        values = []
        constraints = []
        violations = []
        for row in x:
            g, cv = problem.evaluate_constraints(row)
            values.append(self._evaluate(row.copy(), fidelity))
            constraints.append(g)
            violations.append(cv)
        return EvaluationResult(
            f=np.asarray(values, dtype=np.float64).reshape((-1, problem.n_obj)),
            g=np.asarray(constraints, dtype=np.float64).reshape(
                (len(x), problem.n_constraints)
            ),
            cv=np.asarray(violations, dtype=np.float64),
            candidate_ids=candidate_ids,
            cost=np.full(len(x), fidelity + 1.0, dtype=np.float64),
            outputs={"fidelity": np.full(len(x), fidelity, dtype=np.float64)},
        )


def main():
    """Run an explicit fidelity plan through the standard optimizer stages."""
    problem = reference_problem()
    evaluator = FidelityEvaluator(
        lambda row, fidelity: problem.evaluate(row) + (0.5 if fidelity == 0 else 0.0)
    )
    state = (
        Optimizer(problem, seed=13)
        .set_initializer(LHSInitializer(2, 2, 13))
        .set_algorithm(PSO())
        .set_strategy(DirectStrategy(n_offspring=1))
        .set_evaluator(evaluator)
        .set_evaluation_planner(FidelityPromotion(0, 1, promotion_count=1))
        .set_termination(Termination(max_fe(3)))
        .run()
    )
    return {"state": state, "requests": tuple(evaluator.requests)}


if __name__ == "__main__":
    main()
