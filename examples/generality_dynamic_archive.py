import numpy as np

from saealib import DynamicArchiveSelector, SerialEvaluator, reference_problem


def main():
    """Run two archive snapshots under changing environments."""
    selector = DynamicArchiveSelector()
    candidates = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    snapshots = selector.optimize(
        lambda environment: reference_problem(shift=float(environment)),
        (0, 10),
        candidates,
        SerialEvaluator(),
    )
    return {"snapshots": snapshots, "selected": selector.select(9)}


if __name__ == "__main__":
    main()
