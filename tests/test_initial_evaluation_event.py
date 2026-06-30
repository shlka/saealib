"""
Tests for InitialEvaluationStartEvent and InitialEvaluationEndEvent (Issue #97).

Tests cover:
- Event field definitions
- Dispatch order: Start -> End, both before RunStartEvent
- candidates_x shape matches n_init_archive x dim
- Archive mutation via EndEvent handler is reflected in returned ctx
"""

from __future__ import annotations

import numpy as np

from saealib import (
    GA,
    CrossoverBLXAlpha,
    IndividualBasedStrategy,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    RBFSurrogate,
    RunStartEvent,
    SequentialSelection,
    Termination,
    TruncationSelection,
    gaussian_kernel,
    max_fe,
)
from saealib.callback import (
    Event,
    InitialEvaluationEndEvent,
    InitialEvaluationStartEvent,
)
from saealib.comparators import SingleObjectiveComparator
from saealib.population import Archive
from saealib.problem import Problem

DIM = 2
N_INIT_ARCHIVE = 10
N_INIT_POP = 6


def _make_problem() -> Problem:
    return Problem(
        func=lambda x: np.array([np.sum(x**2)]),
        dim=DIM,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0] * DIM,
        ub=[5.0] * DIM,
        comparator=SingleObjectiveComparator(),
    )


def _make_optimizer(problem: Problem | None = None) -> Optimizer:
    problem = problem or _make_problem()
    return (
        Optimizer(problem)
        .set_initializer(
            LHSInitializer(
                n_init_archive=N_INIT_ARCHIVE,
                n_init_population=N_INIT_POP,
                seed=0,
            )
        )
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.5),
                mutation=MutationUniform(mutation_rate=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_surrogate(RBFSurrogate(gaussian_kernel, DIM), n_neighbors=5)
        .set_strategy(IndividualBasedStrategy(evaluation_ratio=0.5))
        .set_termination(Termination(max_fe(50)))
    )


def _initialize(opt: Optimizer, problem: Problem):
    assert opt.initializer is not None
    return opt.initializer.initialize(opt, problem)


# ===========================================================================
# Event field tests
# ===========================================================================


class TestInitialEvaluationEventFields:
    def test_start_event_is_event_subclass(self) -> None:
        problem = _make_problem()
        opt = _make_optimizer(problem)
        ctx = _initialize(opt, problem)
        e = InitialEvaluationStartEvent(ctx=ctx)
        assert isinstance(e, Event)

    def test_end_event_is_event_subclass(self) -> None:
        problem = _make_problem()
        opt = _make_optimizer(problem)
        ctx = _initialize(opt, problem)
        e = InitialEvaluationEndEvent(ctx=ctx)
        assert isinstance(e, Event)

    def test_start_event_has_candidates_x(self) -> None:
        problem = _make_problem()
        opt = _make_optimizer(problem)
        ctx = _initialize(opt, problem)
        x = np.zeros((N_INIT_ARCHIVE, DIM))
        e = InitialEvaluationStartEvent(ctx=ctx, candidates_x=x)
        assert e.candidates_x is x

    def test_start_event_candidates_x_default_none(self) -> None:
        problem = _make_problem()
        opt = _make_optimizer(problem)
        ctx = _initialize(opt, problem)
        e = InitialEvaluationStartEvent(ctx=ctx)
        assert e.candidates_x is None

    def test_end_event_has_archive(self) -> None:
        problem = _make_problem()
        opt = _make_optimizer(problem)
        ctx = _initialize(opt, problem)
        e = InitialEvaluationEndEvent(ctx=ctx, archive=ctx.archive)
        assert e.archive is ctx.archive

    def test_end_event_archive_default_none(self) -> None:
        problem = _make_problem()
        opt = _make_optimizer(problem)
        ctx = _initialize(opt, problem)
        e = InitialEvaluationEndEvent(ctx=ctx)
        assert e.archive is None


# ===========================================================================
# Dispatch tests
# ===========================================================================


class TestInitialEvaluationEventDispatch:
    def test_start_event_is_dispatched(self) -> None:
        problem = _make_problem()
        opt = _make_optimizer(problem)

        received: list[InitialEvaluationStartEvent] = []
        opt.cbmanager.register(InitialEvaluationStartEvent, received.append)

        _initialize(opt, problem)

        assert len(received) == 1

    def test_end_event_is_dispatched(self) -> None:
        problem = _make_problem()
        opt = _make_optimizer(problem)

        received: list[InitialEvaluationEndEvent] = []
        opt.cbmanager.register(InitialEvaluationEndEvent, received.append)

        _initialize(opt, problem)

        assert len(received) == 1

    def test_dispatch_order_start_before_end(self) -> None:
        problem = _make_problem()
        opt = _make_optimizer(problem)

        order: list[type] = []
        opt.cbmanager.register(
            InitialEvaluationStartEvent,
            lambda _: order.append(InitialEvaluationStartEvent),
        )
        opt.cbmanager.register(
            InitialEvaluationEndEvent, lambda _: order.append(InitialEvaluationEndEvent)
        )

        _initialize(opt, problem)

        assert order == [InitialEvaluationStartEvent, InitialEvaluationEndEvent]

    def test_both_events_fire_before_run_start(self) -> None:
        problem = _make_problem()
        opt = _make_optimizer(problem)

        order: list[type] = []
        opt.cbmanager.register(
            InitialEvaluationStartEvent,
            lambda _: order.append(InitialEvaluationStartEvent),
        )
        opt.cbmanager.register(
            InitialEvaluationEndEvent, lambda _: order.append(InitialEvaluationEndEvent)
        )
        opt.cbmanager.register(RunStartEvent, lambda _: order.append(RunStartEvent))

        opt.run()

        assert order.index(InitialEvaluationStartEvent) < order.index(RunStartEvent)
        assert order.index(InitialEvaluationEndEvent) < order.index(RunStartEvent)

    def test_start_event_candidates_x_shape(self) -> None:
        problem = _make_problem()
        opt = _make_optimizer(problem)

        received: list[InitialEvaluationStartEvent] = []
        opt.cbmanager.register(InitialEvaluationStartEvent, received.append)

        _initialize(opt, problem)

        x = received[0].candidates_x
        assert x is not None
        assert x.shape == (N_INIT_ARCHIVE, DIM)

    def test_start_event_ctx_archive_is_empty(self) -> None:
        """ctx.archive is empty at the time StartEvent fires."""
        problem = _make_problem()
        opt = _make_optimizer(problem)

        archive_len_at_start: list[int] = []
        opt.cbmanager.register(
            InitialEvaluationStartEvent,
            lambda e: archive_len_at_start.append(len(e.ctx.archive)),
        )

        _initialize(opt, problem)

        assert archive_len_at_start[0] == 0

    def test_end_event_archive_is_populated(self) -> None:
        problem = _make_problem()
        opt = _make_optimizer(problem)

        archive_len_at_end: list[int] = []
        opt.cbmanager.register(
            InitialEvaluationEndEvent,
            lambda e: archive_len_at_end.append(len(e.ctx.archive)),
        )

        _initialize(opt, problem)

        assert archive_len_at_end[0] == N_INIT_ARCHIVE

    def test_end_event_archive_is_same_object_as_ctx_archive(self) -> None:
        problem = _make_problem()
        opt = _make_optimizer(problem)

        captured: list[tuple[Archive, Archive]] = []
        opt.cbmanager.register(
            InitialEvaluationEndEvent,
            lambda e: captured.append((e.archive, e.ctx.archive)),
        )

        _initialize(opt, problem)

        event_archive, ctx_archive = captured[0]
        assert event_archive is ctx_archive

    def test_ctx_fe_is_set_before_end_event(self) -> None:
        problem = _make_problem()
        opt = _make_optimizer(problem)

        fe_at_end: list[int] = []
        opt.cbmanager.register(
            InitialEvaluationEndEvent,
            lambda e: fe_at_end.append(e.ctx.fe),
        )

        _initialize(opt, problem)

        assert fe_at_end[0] == N_INIT_ARCHIVE


# ===========================================================================
# Archive mutation via EndEvent handler
# ===========================================================================


class TestInitialEvaluationEndEventMutation:
    def test_archive_mutation_reflected_in_returned_ctx(self) -> None:
        """Removing individuals in EndEvent handler reduces ctx.archive size."""
        problem = _make_problem()
        opt = _make_optimizer(problem)

        n_keep = 5

        def _trim_archive(event: InitialEvaluationEndEvent) -> None:
            arc = event.archive
            assert arc is not None
            kept = arc.extract(list(range(n_keep)))
            arc.clear()
            arc.extend(kept)

        opt.cbmanager.register(InitialEvaluationEndEvent, _trim_archive)

        ctx = _initialize(opt, problem)

        assert len(ctx.archive) == n_keep

    def test_population_uses_trimmed_archive(self) -> None:
        """Population is built from the archive after EndEvent, reflecting mutations."""
        problem = _make_problem()
        n_init_pop = 3
        opt = (
            Optimizer(problem)
            .set_initializer(
                LHSInitializer(
                    n_init_archive=N_INIT_ARCHIVE,
                    n_init_population=n_init_pop,
                    seed=0,
                )
            )
            .set_algorithm(
                GA(
                    crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.5),
                    mutation=MutationUniform(mutation_rate=0.1),
                    parent_selection=SequentialSelection(),
                    survivor_selection=TruncationSelection(),
                )
            )
            .set_surrogate(RBFSurrogate(gaussian_kernel, DIM), n_neighbors=5)
            .set_strategy(IndividualBasedStrategy(evaluation_ratio=0.5))
            .set_termination(Termination(max_fe(50)))
        )

        n_keep = 4

        def _trim_archive(event: InitialEvaluationEndEvent) -> None:
            arc = event.archive
            assert arc is not None
            kept = arc.extract(list(range(n_keep)))
            arc.clear()
            arc.extend(kept)

        opt.cbmanager.register(InitialEvaluationEndEvent, _trim_archive)

        ctx = _initialize(opt, problem)

        assert len(ctx.population) == min(n_init_pop, n_keep)
