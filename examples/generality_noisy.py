import numpy as np

from saealib import RepeatedEvaluationRunner, SeededNoiseEvaluator, reference_problem


def main():
    """Run explicit repeated evaluations."""
    problem = reference_problem()
    x = np.array([[0.25, -0.25], [0.5, 0.5]], dtype=np.float64)
    result = RepeatedEvaluationRunner(SeededNoiseEvaluator(seed=17, scale=0.02), 3).run(
        x, np.array([10, 11], dtype=np.int64), problem
    )
    return {
        "summary": result.summary,
        "fe": result.fe,
        "archive": result.archive,
        "truth_archive": result.truth_archive,
        "requests": result.requests,
    }


if __name__ == "__main__":
    main()
