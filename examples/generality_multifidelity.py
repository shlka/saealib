import numpy as np

from saealib import (
    FidelityEvaluator,
    FidelityPromotion,
    FidelityPromotionRunner,
    reference_problem,
)


def main():
    """Run an explicitly labeled fidelity evaluation."""
    problem = reference_problem()
    evaluator = FidelityEvaluator(
        lambda row, fidelity: problem.evaluate(row) + (0.5 if fidelity == 0 else 0.0)
    )
    workflow = FidelityPromotionRunner(evaluator, FidelityPromotion(0, 1)).run(
        np.array([[0.25, -0.25], [0.8, 0.8]], dtype=np.float64),
        np.array([10, 11], dtype=np.int64),
        problem,
    )
    return {
        "low": workflow.low_result,
        "high": workflow.high_result,
        "promoted": workflow.high_request.metadata["fidelity"] == 1,
        "low_request": workflow.low_request,
        "high_request": workflow.high_request,
        "archive": workflow.archive,
        "fe": workflow.fe,
        "cost": workflow.cost,
    }


if __name__ == "__main__":
    main()
