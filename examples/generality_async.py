from saealib import (
    PSO,
    AsyncEvaluator,
    AsyncScheduler,
    LHSInitializer,
    Optimizer,
    SerialEvaluator,
    SteadyStateStrategy,
    Termination,
    reference_problem,
)


def main():
    """Run asynchronous steady-state evaluation."""
    problem = reference_problem()
    evaluator = AsyncEvaluator(SerialEvaluator(), max_workers=2)
    scheduler = AsyncScheduler(evaluator, max_pending=2)
    state = (
        Optimizer(problem, seed=31)
        .set_initializer(LHSInitializer(2, 2, 31))
        .set_algorithm(PSO())
        .set_strategy(SteadyStateStrategy())
        .set_evaluator(evaluator)
        .set_async_scheduler(scheduler)
        .set_termination(
            Termination(lambda state: state.fe + len(state.pending_evaluations) >= 6)
        )
        .run()
    )
    while state.pending_evaluations:
        state = scheduler.poll(state, wait=True)
    return state


if __name__ == "__main__":
    main()
