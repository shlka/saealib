from saealib import PSO, DirectStrategy, LHSInitializer, Optimizer, Termination, max_fe
from saealib.population import Archive

try:
    from examples._support import reference_problem
except ModuleNotFoundError:
    from _support import reference_problem


def main():
    """Run environments and select the nearest stored archive through state."""
    first_optimizer = (
        Optimizer(reference_problem(shift=0.0), seed=0)
        .set_initializer(LHSInitializer(2, 2, 0))
        .set_algorithm(PSO())
        .set_strategy(DirectStrategy(n_offspring=2))
        .set_termination(Termination(max_fe(4)))
    )
    ctx = first_optimizer.run()
    ctx.archives["env_0"] = ctx.archive

    ctx.archive = Archive(ctx.archive.attrs, duplicate_policy="keep_first")
    second_optimizer = (
        Optimizer(reference_problem(shift=10.0), seed=10)
        .set_algorithm(PSO())
        .set_strategy(DirectStrategy(n_offspring=2))
        .set_termination(Termination(max_fe(8)))
    )
    ctx = second_optimizer.run_from(ctx.replace(problem=reference_problem(shift=10.0)))
    ctx.archives["env_10"] = ctx.archive

    target_environment = 9.0
    environment_names = tuple(name for name in ctx.archives if name.startswith("env_"))
    selected_name = min(
        environment_names,
        key=lambda name: abs(float(name.removeprefix("env_")) - target_environment),
    )
    selected = ctx.archives[selected_name]
    ctx.archive = selected
    selected_size_before_third = len(selected)
    fe_before_third = ctx.fe
    third_optimizer = (
        Optimizer(reference_problem(shift=target_environment), seed=9)
        .set_algorithm(PSO())
        .set_strategy(DirectStrategy(n_offspring=2))
        .set_termination(Termination(max_fe(12)))
    )
    ctx = third_optimizer.run_from(
        ctx.replace(problem=reference_problem(shift=target_environment))
    )
    snapshots = (ctx.archives["env_0"], ctx.archives["env_10"])
    return {
        "snapshots": snapshots,
        "selected": selected,
        "selected_name": selected_name,
        "state": ctx,
        "selected_size_before_third": selected_size_before_third,
        "fe_before_third": fe_before_third,
    }


if __name__ == "__main__":
    main()
