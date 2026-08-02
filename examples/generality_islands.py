from saealib import (
    PSO,
    DirectStrategy,
    IslandModel,
    LHSInitializer,
    Optimizer,
    Termination,
    max_fe,
)

try:
    from examples._support import reference_problem
except ModuleNotFoundError:
    from _support import reference_problem


def main():
    """Run two independent optimizers and exchange their best designs."""
    optimizers = tuple(
        Optimizer(reference_problem(), seed=seed)
        .set_initializer(LHSInitializer(2, 2, seed))
        .set_algorithm(PSO())
        .set_strategy(DirectStrategy(n_offspring=2))
        .set_termination(Termination(max_fe(4)))
        for seed in (7, 11)
    )
    states = IslandModel(optimizers).run()
    islands = tuple(state.archive.get_array("x").copy() for state in states)
    migrated = list(islands)
    migrated[1][0] = migrated[0][0]
    return {"islands": tuple(migrated), "events": ((0, 1),)}


if __name__ == "__main__":
    main()
