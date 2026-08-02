import numpy as np

from saealib import MigrationPolicy, SerialEvaluator, reference_problem


def main():
    """Run two independent islands."""
    policy = MigrationPolicy(1)
    islands = (
        np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        np.array([[-0.1, -0.2], [-0.3, -0.4]], dtype=np.float64),
    )
    final, events = policy.optimize(reference_problem(), islands, SerialEvaluator(), 2)
    return {"islands": final, "events": events}


if __name__ == "__main__":
    main()
