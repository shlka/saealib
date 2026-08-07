"""Executable access checks for the G6b instrumented-store test boundary.

Entrypoints covered here are ``Initializer.initialize``, ``Algorithm.ask`` /
``Algorithm.tell``, ``Termination.is_terminated``, and
``AsyncEvaluationScheduler.submit`` / ``poll``.  DirectStrategy's planner,
feedback builder, evaluator, and pipeline stages are intentionally not
wrapped: their compatibility-property accesses are the unowned evidence that
the boundary is limited to the declared component entrypoints.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from instrumented_store import (
    Recorder,
    attach_instrumentation,
    instrumentation_scope,
    instrumented,
    instrumented_component,
)

from saealib import (
    GA,
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    Termination,
    TruncationSelection,
    max_gen,
)
from saealib.context import OptimizationState
from saealib.core.state import (
    ARCHIVES_MAIN,
    ARCHIVES_PARETO,
    EVALUATIONS_COUNT,
    EVALUATIONS_OWNERS,
    EVALUATIONS_PLAN,
    EVALUATIONS_PLAN_STATE,
    EVALUATIONS_PLAN_UPDATES,
    FEEDBACK_RESULT,
    PENDING_EVALUATIONS,
    POPULATIONS_MAIN,
    PROPOSALS_OFFSPRING,
    RUNTIME_ASYNC_FATAL,
    SURROGATES_PREDICTIONS,
)
from saealib.execution.evaluator import EvaluationRequest, SerialEvaluator
from saealib.execution.initializer import RandomInitializer
from saealib.execution.scheduler import AsyncEvaluationScheduler
from saealib.problem import Problem
from saealib.strategies.base import OptimizationStrategy
from saealib.strategies.direct import DirectStrategy


class _Provider:
    seed = 0
    strategy: OptimizationStrategy = DirectStrategy(n_offspring=2)
    termination = Termination(max_gen(10))
    async_evaluation_scheduler = None
    feedback_builder = None
    evaluation_planner = None

    def __init__(self) -> None:
        from saealib.callback import CallbackManager

        self.algorithm = GA(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
            mutation=MutationUniform(prob_var=0.1),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        )
        self.evaluator = SerialEvaluator()
        self.cbmanager = CallbackManager()

    def dispatch(self, event: object) -> None:
        self.cbmanager.dispatch(cast(Any, event))


def _problem() -> Problem:
    return Problem(
        func=lambda x: np.array([np.sum(np.asarray(x) ** 2)]),
        dim=3,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-2.0] * 3,
        ub=[2.0] * 3,
    )


def _initialized(recorder: Recorder) -> tuple[OptimizationState, _Provider]:
    provider = _Provider()
    initializer = RandomInitializer(4, 4, seed=0)
    initialize = instrumented("initializer", recorder)(initializer.initialize)
    with instrumentation_scope(recorder):
        state = initialize(provider, _problem())
    return state, provider


def _observed(recorder: Recorder, owner: str) -> set[Any]:
    return recorder.keys(owner)


def _assert_declared(recorder: Recorder, owner: str, component: Any) -> None:
    """Fail only on under-declared reads/writes for one wrapped entrypoint."""
    contract = component.contract().state
    reads = recorder.keys(owner, "read")
    writes = recorder.keys(owner, "write")
    assert reads - set(contract.reads) == set()
    assert writes - set(contract.writes) == set()


def test_direct_ga_entrypoints_have_no_undeclared_state_access() -> None:
    """Trace one generation of built-in direct GA and allow conservative contracts."""
    recorder = Recorder()
    state, provider = _initialized(recorder)

    cast(Any, provider.algorithm).ask = instrumented("algorithm.ask", recorder)(
        provider.algorithm.ask
    )
    cast(Any, provider.algorithm).tell = instrumented("algorithm.tell", recorder)(
        provider.algorithm.tell
    )
    with instrumentation_scope(recorder):
        state = provider.strategy.step(state, cast(Any, provider))
        terminated = instrumented("termination", recorder)(
            provider.termination.is_terminated
        )
        assert not terminated(state)

    _assert_declared(recorder, "initializer", RandomInitializer(4, 4))
    _assert_declared(recorder, "algorithm.ask", provider.algorithm)
    _assert_declared(recorder, "algorithm.tell", provider.algorithm)
    _assert_declared(recorder, "termination", provider.termination)

    # A stage/policy access is deliberately outside all entrypoint wrappers.
    _ = state.population
    assert sum(recorder.unowned.values()) > 0


def test_async_scheduler_algorithm_none_matches_measured_accesses() -> None:
    """Measure scheduler submit/poll against its algorithm=None state contract."""
    recorder = Recorder()
    state, _provider = _initialized(recorder)
    offspring = state.population.empty_like(capacity=1)
    offspring._extend_internal(
        {
            "x": np.array([[0.25, 0.5, 0.75]]),
            "id": state.candidate_id_allocator.allocate(1),
        },
        preserve_ids=True,
    )
    # The candidate reservation is fixture setup, not scheduler work.  Flush
    # the pre-existing store before entering the scheduler wrapper so its
    # allocator mutation remains unowned rather than being attributed to the
    # next component call.
    recorder.flush()
    state = attach_instrumentation(state.replace(offspring=offspring), recorder)
    offspring_value = state.offspring
    assert offspring_value is not None
    ids = offspring_value.get_array("id")
    request = EvaluationRequest(np.int64(1), ids, offspring_value.get_array("x"))
    scheduler = AsyncEvaluationScheduler(SerialEvaluator(), algorithm=None)
    submit = instrumented("scheduler.submit", recorder)(scheduler.submit)
    poll = instrumented("scheduler.poll", recorder)(scheduler.poll)
    with instrumentation_scope(recorder):
        state = submit(state, [request])
        state = poll(state, wait=True)

    contract = scheduler.contract().state
    # G6a deliberately excludes stage transport from StateContract auditing.
    # G5b stores these values for checkpointing, so retain the raw trace but do
    # not turn transport reads into a new scheduler contract in this unit.
    transport = {FEEDBACK_RESULT, PROPOSALS_OFFSPRING, SURROGATES_PREDICTIONS}
    observed = _observed(recorder, "scheduler.submit") | _observed(
        recorder, "scheduler.poll"
    )
    assert observed - transport <= set(contract.reads) | set(contract.writes)
    assert PENDING_EVALUATIONS in observed
    assert transport & observed
    assert not ({RUNTIME_ASYNC_FATAL} & observed - set(contract.writes))
    assert set(contract.reads) >= {
        PENDING_EVALUATIONS,
        EVALUATIONS_OWNERS,
        EVALUATIONS_PLAN,
        EVALUATIONS_PLAN_STATE,
        EVALUATIONS_PLAN_UPDATES,
        EVALUATIONS_COUNT,
        ARCHIVES_MAIN,
        ARCHIVES_PARETO,
    }


def test_nested_wrapper_attributes_only_to_inner_component() -> None:
    recorder = Recorder()
    state, _provider = _initialized(recorder)
    attach_instrumentation(state, recorder)
    with instrumented_component("outer", recorder):
        with instrumented_component("inner", recorder):
            _ = state.population
        _ = state.archive

    assert recorder.keys("inner") == {POPULATIONS_MAIN}
    assert recorder.keys("outer") == {ARCHIVES_MAIN}
    assert not recorder.counts.get(("outer", "read", POPULATIONS_MAIN), 0)
