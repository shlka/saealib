"""
Tests for the callback module.

Tests cover:
- CallbackManager: register, dispatch, unregister, replace
- Event hierarchy: base class and subclass field definitions
- Built-in handlers: logging_generation, logging_generation_hv
- PostEvaluationEvent dispatch from IndividualBasedStrategy
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from saealib.callback import (
    CallbackManager,
    Event,
    GenerationEndEvent,
    GenerationStartEvent,
    PostAskEvent,
    PostCrossoverEvent,
    PostEvaluationEvent,
    PostMutationEvent,
    PostSurrogateFitEvent,
    RunEndEvent,
    RunStartEvent,
    SurrogateEndEvent,
    SurrogateStartEvent,
    logging_generation,
    logging_generation_hv,
)
from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.surrogate.rbf import RBFSurrogate, gaussian_kernel

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DIM = 2
N_OBJ = 1
_ATTRS_1OBJ = [
    PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
    PopulationAttribute(name="f", dtype=np.float64, shape=(N_OBJ,)),
    PopulationAttribute(name="g", dtype=np.float64, shape=(0,)),
    PopulationAttribute(name="cv", dtype=np.float64, shape=()),
]


def _make_problem() -> Problem:
    return Problem(
        func=lambda x: np.array([np.sum(x**2)]),
        dim=DIM,
        n_obj=N_OBJ,
        direction=np.array([-1.0]),
        lb=[-5.0] * DIM,
        ub=[5.0] * DIM,
        comparator=SingleObjectiveComparator(),
    )


def _make_archive(n: int = 20, rng_seed: int = 42) -> Archive:
    arc = Archive(_ATTRS_1OBJ, init_capacity=n + 10)
    rng = np.random.default_rng(rng_seed)
    for _ in range(n):
        x = rng.uniform(-2.0, 2.0, size=DIM)
        f = np.array([np.sum(x**2)])
        arc.add(x=x, f=f)
    return arc


def _make_population(n: int = 5, rng_seed: int = 7) -> Population:
    pop = Population(_ATTRS_1OBJ, init_capacity=n + 5)
    rng = np.random.default_rng(rng_seed)
    xs = rng.uniform(-2.0, 2.0, size=(n, DIM))
    fs = np.array([[np.sum(x**2)] for x in xs])
    pop.extend({"x": xs, "f": fs})
    return pop


def _make_pareto_archive() -> ParetoArchive:
    return ParetoArchive(_ATTRS_1OBJ, init_capacity=30, direction=np.array([-1.0]))


class _MockProvider:
    """Minimal ComponentProvider that records dispatched events."""

    def __init__(self) -> None:
        self.cbmanager = CallbackManager()
        self.dispatched: list[Event] = []

    def dispatch(self, event: Event) -> None:
        self.cbmanager.dispatch(event)
        self.dispatched.append(event)


def _make_ctx(
    archive: Archive | None = None, population: Population | None = None
) -> OptimizationState:
    problem = _make_problem()
    arc = archive if archive is not None else _make_archive()
    pop = population if population is not None else _make_population()
    return OptimizationState(
        problem=problem,
        population=pop,
        archive=arc,
        pareto_archive=_make_pareto_archive(),
        rng=np.random.default_rng(0),
        fe=10,
        gen=1,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cbmanager() -> CallbackManager:
    return CallbackManager()


# ===========================================================================
# CallbackManager Tests
# ===========================================================================
class TestCallbackManager:
    """Tests for CallbackManager: register, dispatch, unregister, replace."""

    def test_register_stores_handler(self, cbmanager: CallbackManager) -> None:
        handler = MagicMock()
        cbmanager.register(RunStartEvent, handler)
        assert handler in cbmanager.handlers[RunStartEvent]

    def test_dispatch_calls_handler(self, cbmanager: CallbackManager) -> None:
        handler = MagicMock()
        cbmanager.register(RunStartEvent, handler)
        ctx = _make_ctx()
        event = RunStartEvent(ctx=ctx)
        cbmanager.dispatch(event)
        handler.assert_called_once_with(event)

    def test_dispatch_passes_exact_event_object(
        self, cbmanager: CallbackManager
    ) -> None:
        received = []
        cbmanager.register(RunStartEvent, received.append)
        ctx = _make_ctx()
        event = RunStartEvent(ctx=ctx)
        cbmanager.dispatch(event)
        assert received[0] is event

    def test_dispatch_calls_multiple_handlers_in_order(
        self, cbmanager: CallbackManager
    ) -> None:
        order: list[int] = []
        cbmanager.register(RunStartEvent, lambda _: order.append(1))
        cbmanager.register(RunStartEvent, lambda _: order.append(2))
        cbmanager.register(RunStartEvent, lambda _: order.append(3))
        ctx = _make_ctx()
        cbmanager.dispatch(RunStartEvent(ctx=ctx))
        assert order == [1, 2, 3]

    def test_dispatch_ignores_other_event_types(
        self, cbmanager: CallbackManager
    ) -> None:
        handler = MagicMock()
        cbmanager.register(RunEndEvent, handler)
        ctx = _make_ctx()
        cbmanager.dispatch(RunStartEvent(ctx=ctx))
        handler.assert_not_called()

    def test_dispatch_with_no_handlers_does_not_raise(
        self, cbmanager: CallbackManager
    ) -> None:
        ctx = _make_ctx()
        cbmanager.dispatch(RunStartEvent(ctx=ctx))  # should not raise

    def test_unregister_removes_handler(self, cbmanager: CallbackManager) -> None:
        handler = MagicMock()
        cbmanager.register(RunStartEvent, handler)
        cbmanager.unregister(RunStartEvent, handler)
        ctx = _make_ctx()
        cbmanager.dispatch(RunStartEvent(ctx=ctx))
        handler.assert_not_called()

    def test_unregister_unknown_raises_value_error(
        self, cbmanager: CallbackManager
    ) -> None:
        handler = MagicMock()
        with pytest.raises(ValueError):
            cbmanager.unregister(RunStartEvent, handler)

    def test_replace_swaps_handler(self, cbmanager: CallbackManager) -> None:
        old = MagicMock()
        new = MagicMock()
        cbmanager.register(RunStartEvent, old)
        cbmanager.replace(RunStartEvent, old, new)
        ctx = _make_ctx()
        cbmanager.dispatch(RunStartEvent(ctx=ctx))
        old.assert_not_called()
        new.assert_called_once()

    def test_replace_preserves_position(self, cbmanager: CallbackManager) -> None:
        order: list[str] = []

        def h1(_: RunStartEvent) -> None:
            order.append("h1")

        def h2(_: RunStartEvent) -> None:
            order.append("h2")

        def h3(_: RunStartEvent) -> None:
            order.append("h3")

        cbmanager.register(RunStartEvent, h1)
        cbmanager.register(RunStartEvent, h2)
        cbmanager.register(RunStartEvent, h3)

        def new(_: RunStartEvent) -> None:
            order.append("new")

        cbmanager.replace(RunStartEvent, h2, new)
        ctx = _make_ctx()
        cbmanager.dispatch(RunStartEvent(ctx=ctx))
        assert order == ["h1", "new", "h3"]

    def test_replace_unknown_old_raises_value_error(
        self, cbmanager: CallbackManager
    ) -> None:
        old = MagicMock()
        new = MagicMock()
        with pytest.raises(ValueError):
            cbmanager.replace(RunStartEvent, old, new)

    def test_register_same_handler_twice(self, cbmanager: CallbackManager) -> None:
        """Registering the same function twice results in two calls."""
        counter = [0]

        def handler(_: RunStartEvent) -> None:
            counter[0] += 1

        cbmanager.register(RunStartEvent, handler)
        cbmanager.register(RunStartEvent, handler)
        ctx = _make_ctx()
        cbmanager.dispatch(RunStartEvent(ctx=ctx))
        assert counter[0] == 2


# ===========================================================================
# Event class structure Tests
# ===========================================================================
class TestEventClasses:
    """Tests that event dataclasses expose the expected fields."""

    def test_event_has_ctx(self) -> None:
        ctx = _make_ctx()
        e = RunStartEvent(ctx=ctx)
        assert e.ctx is ctx

    @pytest.mark.parametrize(
        "cls",
        [
            RunStartEvent,
            RunEndEvent,
            GenerationStartEvent,
            GenerationEndEvent,
        ],
    )
    def test_lifecycle_events_are_event_subclasses(self, cls) -> None:
        ctx = _make_ctx()
        e = cls(ctx=ctx)
        assert isinstance(e, Event)

    def test_post_crossover_event_has_candidates(self) -> None:
        ctx = _make_ctx()
        cand = np.zeros((3, DIM))
        e = PostCrossoverEvent(ctx=ctx, candidates=cand)
        np.testing.assert_array_equal(e.candidates, cand)

    def test_post_mutation_event_has_candidates(self) -> None:
        ctx = _make_ctx()
        cand = np.ones((4, DIM))
        e = PostMutationEvent(ctx=ctx, candidates=cand)
        np.testing.assert_array_equal(e.candidates, cand)

    def test_post_ask_event_has_candidates(self) -> None:
        ctx = _make_ctx()
        cand = np.eye(DIM)
        e = PostAskEvent(ctx=ctx, candidates=cand)
        np.testing.assert_array_equal(e.candidates, cand)

    def test_surrogate_start_event_has_offspring(self) -> None:
        ctx = _make_ctx()
        pop = _make_population(3)
        e = SurrogateStartEvent(ctx=ctx, offspring=pop)
        assert e.offspring is pop

    def test_surrogate_end_event_has_offspring(self) -> None:
        ctx = _make_ctx()
        pop = _make_population(3)
        e = SurrogateEndEvent(ctx=ctx, offspring=pop)
        assert e.offspring is pop

    def test_post_surrogate_fit_event_fields(self) -> None:
        ctx = _make_ctx()
        surrogate = RBFSurrogate(gaussian_kernel, DIM)
        train_x = np.zeros((10, DIM))
        train_f = np.zeros((10, N_OBJ))
        e = PostSurrogateFitEvent(
            ctx=ctx,
            surrogate=surrogate,
            train_x=train_x,
            train_f=train_f,
        )
        assert e.surrogate is surrogate
        np.testing.assert_array_equal(e.train_x, train_x)
        np.testing.assert_array_equal(e.train_f, train_f)

    def test_post_surrogate_fit_event_defaults_none(self) -> None:
        ctx = _make_ctx()
        e = PostSurrogateFitEvent(ctx=ctx)
        assert e.surrogate is None
        assert e.train_x is None
        assert e.train_f is None

    def test_post_evaluation_event_has_offspring(self) -> None:
        ctx = _make_ctx()
        pop = _make_population(2)
        e = PostEvaluationEvent(ctx=ctx, offspring=pop)
        assert e.offspring is pop

    def test_post_evaluation_event_default_offspring_none(self) -> None:
        ctx = _make_ctx()
        e = PostEvaluationEvent(ctx=ctx)
        assert e.offspring is None


# ===========================================================================
# Built-in handler Tests
# ===========================================================================
class TestLoggingGenerationHandler:
    """Tests for logging_generation and logging_generation_hv."""

    def _make_1obj_ctx(self) -> OptimizationState:
        arc = _make_archive(20)
        return _make_ctx(archive=arc)

    def _make_2obj_ctx(self) -> OptimizationState:
        attrs = [
            PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
            PopulationAttribute(name="f", dtype=np.float64, shape=(2,)),
            PopulationAttribute(name="g", dtype=np.float64, shape=(0,)),
            PopulationAttribute(name="cv", dtype=np.float64, shape=()),
        ]
        arc = Archive(attrs, init_capacity=30)
        rng = np.random.default_rng(0)
        for _ in range(20):
            x = rng.uniform(0.0, 2.0, size=DIM)
            f = np.array([np.sum(x**2), np.sum((x - 2.0) ** 2)])
            arc.add(x=x, f=f)
        problem = Problem(
            func=lambda x: np.array([np.sum(x**2), np.sum((x - 2.0) ** 2)]),
            dim=DIM,
            n_obj=2,
            direction=np.array([-1.0, -1.0]),
            lb=[-5.0] * DIM,
            ub=[5.0] * DIM,
        )
        pop = Population(attrs, init_capacity=5)
        pareto_arc = ParetoArchive(
            attrs, init_capacity=30, direction=np.array([-1.0, -1.0])
        )
        return OptimizationState(
            problem=problem,
            population=pop,
            archive=arc,
            pareto_archive=pareto_arc,
            rng=np.random.default_rng(0),
            fe=10,
            gen=1,
        )

    def test_logging_generation_single_obj_does_not_raise(self, caplog) -> None:
        ctx = self._make_1obj_ctx()
        event = GenerationStartEvent(ctx=ctx)
        with caplog.at_level(logging.INFO, logger="saealib.callback"):
            logging_generation(event)
        assert "Generation" in caplog.text

    def test_logging_generation_single_obj_logs_best_f(self, caplog) -> None:
        ctx = self._make_1obj_ctx()
        event = GenerationStartEvent(ctx=ctx)
        with caplog.at_level(logging.INFO, logger="saealib.callback"):
            logging_generation(event)
        assert "Best f" in caplog.text

    def test_logging_generation_multi_obj_does_not_raise(self, caplog) -> None:
        ctx = self._make_2obj_ctx()
        event = GenerationStartEvent(ctx=ctx)
        with caplog.at_level(logging.INFO, logger="saealib.callback"):
            logging_generation(event)
        assert "Front1" in caplog.text

    def test_logging_generation_hv_returns_callable(self) -> None:
        ref = np.array([10.0, 10.0])
        handler = logging_generation_hv(ref)
        assert callable(handler)

    def test_logging_generation_hv_does_not_raise(self, caplog) -> None:
        ctx = self._make_2obj_ctx()
        ref = np.array([10.0, 10.0])
        handler = logging_generation_hv(ref)
        event = GenerationStartEvent(ctx=ctx)
        with caplog.at_level(logging.INFO, logger="saealib.callback"):
            handler(event)

    def test_logging_generation_registered_and_dispatched(self) -> None:
        """logging_generation fires correctly via CallbackManager."""
        cbm = CallbackManager()
        cbm.register(GenerationStartEvent, logging_generation)
        ctx = self._make_1obj_ctx()
        prov = _MockProvider()
        prov.cbmanager = cbm
        event = GenerationStartEvent(ctx=ctx)
        cbm.dispatch(event)  # should not raise


# ===========================================================================
# PostEvaluationEvent dispatch Tests
# ===========================================================================
class TestPostEvaluationDispatch:
    """
    Tests that IndividualBasedStrategy dispatches PostEvaluationEvent with the
    evaluated offspring after true objective evaluations.
    """

    def _make_strategy_provider(self, evaluation_ratio: float = 0.5):
        """Build a minimal but real optimizer setup for IndividualBasedStrategy."""
        from saealib import (
            GA,
            CrossoverBLXAlpha,
            IndividualBasedStrategy,
            LHSInitializer,
            MutationUniform,
            Optimizer,
            SequentialSelection,
            Termination,
            TruncationSelection,
            max_fe,
        )

        dim = DIM
        problem = _make_problem()
        opt = (
            Optimizer(problem)
            .set_initializer(
                LHSInitializer(n_init_archive=20, n_init_population=10, seed=0)
            )
            .set_algorithm(
                GA(
                    crossover=CrossoverBLXAlpha(crossover_rate=0.9, alpha=0.5),
                    mutation=MutationUniform(mutation_rate=0.1),
                    parent_selection=SequentialSelection(),
                    survivor_selection=TruncationSelection(),
                )
            )
            .set_surrogate(RBFSurrogate(gaussian_kernel, dim), n_neighbors=10)
            .set_strategy(IndividualBasedStrategy(evaluation_ratio=evaluation_ratio))
            .set_termination(Termination(max_fe(100)))
        )
        assert opt.initializer is not None
        ctx = opt.initializer.initialize(opt, problem)
        return ctx, opt

    def test_post_evaluation_event_is_dispatched(self) -> None:
        from saealib.strategies.ib import IndividualBasedStrategy

        ctx, opt = self._make_strategy_provider(evaluation_ratio=0.5)

        received: list[PostEvaluationEvent] = []
        opt.cbmanager.register(PostEvaluationEvent, received.append)

        strategy = IndividualBasedStrategy(evaluation_ratio=0.5)
        strategy.step(ctx, opt)

        assert len(received) == 1

    def test_post_evaluation_event_offspring_is_population(self) -> None:
        from saealib.strategies.ib import IndividualBasedStrategy

        ctx, opt = self._make_strategy_provider(evaluation_ratio=0.5)

        received: list[PostEvaluationEvent] = []
        opt.cbmanager.register(PostEvaluationEvent, received.append)

        strategy = IndividualBasedStrategy(evaluation_ratio=0.5)
        strategy.step(ctx, opt)

        event = received[0]
        assert isinstance(event.offspring, Population)

    def test_post_evaluation_offspring_has_true_f_values(self) -> None:
        from saealib.strategies.ib import IndividualBasedStrategy

        rsm = 0.3
        ctx, opt = self._make_strategy_provider(evaluation_ratio=rsm)

        received: list[PostEvaluationEvent] = []
        opt.cbmanager.register(PostEvaluationEvent, received.append)

        strategy = IndividualBasedStrategy(evaluation_ratio=rsm)
        strategy.step(ctx, opt)

        event = received[0]
        offspring = event.offspring
        assert offspring is not None
        for i in range(len(offspring)):
            x = offspring[i].x
            f = offspring[i].f
            expected = np.sum(x**2)
            assert f[0] == pytest.approx(expected, rel=1e-6)

    def test_post_evaluation_offspring_size_matches_n_eval(self) -> None:
        from saealib.strategies.ib import IndividualBasedStrategy

        rsm = 0.2
        ctx, opt = self._make_strategy_provider(evaluation_ratio=rsm)

        received: list[PostEvaluationEvent] = []
        opt.cbmanager.register(PostEvaluationEvent, received.append)

        strategy = IndividualBasedStrategy(evaluation_ratio=rsm)
        strategy.step(ctx, opt)

        popsize = len(ctx.population)
        n_eval = max(1, int(rsm * popsize))
        event = received[0]
        assert event.offspring is not None
        assert len(event.offspring) == n_eval

    def test_post_evaluation_fires_after_surrogate_end(self) -> None:
        """PostEvaluationEvent must be dispatched after SurrogateEndEvent."""
        from saealib.strategies.ib import IndividualBasedStrategy

        ctx, opt = self._make_strategy_provider(evaluation_ratio=0.5)

        order: list[type] = []
        opt.cbmanager.register(
            SurrogateEndEvent, lambda _: order.append(SurrogateEndEvent)
        )
        opt.cbmanager.register(
            PostEvaluationEvent, lambda _: order.append(PostEvaluationEvent)
        )

        strategy = IndividualBasedStrategy(evaluation_ratio=0.5)
        strategy.step(ctx, opt)

        assert order.index(SurrogateEndEvent) < order.index(PostEvaluationEvent)
