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
    """Run two optimizers and exchange their best designs on a ring."""
    optimizers = tuple(
        Optimizer(reference_problem(), seed=seed)
        .set_initializer(LHSInitializer(2, 2, seed))
        .set_algorithm(PSO())
        .set_strategy(DirectStrategy(n_offspring=2))
        .set_termination(Termination(max_fe(4)))
        for seed in (7, 11)
    )
    model = IslandModel(optimizers, topology="ring", migration_interval=1)
    states = model.run()
    islands = tuple(state.population.get_array("x").copy() for state in states)
    return {"islands": islands, "events": tuple(model.migration_events)}


if __name__ == "__main__":
    main()
