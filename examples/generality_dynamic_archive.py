from dataclasses import dataclass

from saealib import PSO, DirectStrategy, LHSInitializer, Optimizer, Termination, max_fe

try:
    from examples._support import reference_problem
except ModuleNotFoundError:
    from _support import reference_problem


@dataclass(frozen=True)
class ArchiveSnapshot:
    """An example-only archive tagged with its environment."""

    environment: int
    archive: object


def main():
    """Run standard optimizers and select the nearest environment archive."""
    snapshots = tuple(
        ArchiveSnapshot(
            environment,
            Optimizer(reference_problem(shift=float(environment)), seed=environment)
            .set_initializer(LHSInitializer(2, 2, environment))
            .set_algorithm(PSO())
            .set_strategy(DirectStrategy(n_offspring=2))
            .set_termination(Termination(max_fe(4)))
            .run()
            .archive,
        )
        for environment in (0, 10)
    )
    selected = min(snapshots, key=lambda item: abs(item.environment - 9))
    return {"snapshots": snapshots, "selected": selected}


if __name__ == "__main__":
    main()
