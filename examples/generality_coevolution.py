import numpy as np

from saealib import CooperativeCoevolution, SerialEvaluator, reference_problem


def main():
    """Run coordinate-wise cooperative optimization."""
    problem = reference_problem(dim=4)
    coevolution = CooperativeCoevolution(4, ((0, 1), (2, 3)))
    result = coevolution.optimize(
        problem,
        (np.array([0.2, -0.2]), np.array([0.0, 0.0])),
        SerialEvaluator(),
    )
    return {
        "context": result.context,
        "history": result.objective_history,
        "ids": result.candidate_ids,
    }


if __name__ == "__main__":
    main()
