import time
from types import SimpleNamespace

import numpy as np
import pytest

from saealib.callback import CallbackManager, GenerationEndEvent, PostEvaluationEvent
from saealib.context import OptimizationState
from saealib.exceptions import (
    CheckpointError,
    EvaluationFatalError,
    EvaluationProtocolError,
)
from saealib.execution import (
    AsyncEvaluator,
    AsyncScheduler,
    EvaluationErrorInfo,
    EvaluationHandle,
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
    EvaluationUpdate,
    Evaluator,
    PendingEvaluation,
    SerialEvaluator,
)
from saealib.execution.runner import Runner
from saealib.policies.feedback import TrueOnlyFeedback
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.strategies import DirectStrategy
from saealib.surrogate.prediction import PredictionChannel, SurrogatePrediction

ATTRS = [
    PopulationAttribute("id", np.int64, (), -1),
    PopulationAttribute("x", np.float64, (1,)),
    PopulationAttribute("f", np.float64, (1,)),
    PopulationAttribute("g", np.float64, (0,)),
    PopulationAttribute("cv", np.float64, (), 0.0),
]


class SlowEvaluator(Evaluator):
    def evaluate_batch(self, x, problem):
        time.sleep(float(x[0, 0]))
        return SerialEvaluator().evaluate_batch(x, problem)


class FailingEvaluator(Evaluator):
    def evaluate_batch(self, x, problem):
        raise RuntimeError("evaluation failed")


class BareEvaluator(Evaluator):
    def evaluate_batch(self, x, problem):
        return SerialEvaluator().evaluate_batch(x, problem)


class FailSecondEvaluator(BareEvaluator):
    def __init__(self):
        self.calls = 0

    def submit(self, request, problem):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("second submission failed")
        return EvaluationHandle(request.request_id, EvaluationStatus.PENDING)


class ReattachEvaluator(Evaluator):
    def __init__(self):
        self.acknowledged = []

    def evaluate_batch(self, x, problem):
        return SerialEvaluator().evaluate_batch(x, problem)

    def submit(self, request, problem):
        return EvaluationHandle(
            request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(request, problem),
        )

    def collect(self, handle, *, wait=True):
        if handle._acknowledged_sequence >= 0:
            return []
        request, problem = handle.backend_token
        result = SerialEvaluator().evaluate_batch(request.x, problem)
        result.candidate_ids = request.candidate_ids
        result.__post_init__()
        handle._delivered_sequence = 0
        return [
            EvaluationUpdate(
                request.request_id,
                EvaluationStatus.COMPLETED,
                request.candidate_ids,
                result,
                sequence=0,
            )
        ]

    def acknowledge(self, handle, sequence):
        self.acknowledged.append((int(handle.request_id), sequence))
        handle._acknowledged_sequence = sequence

    def can_reattach(self, pending):
        return True

    def reattach(self, pending, problem):
        return EvaluationHandle(
            pending.request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(pending.request, problem),
        )


class PartialRetryEvaluator(Evaluator):
    def __init__(self):
        self.attempts = 0
        self.acks = []
        self.collected = []

    def evaluate_batch(self, x, problem):
        return SerialEvaluator().evaluate_batch(x, problem)

    def submit(self, request, problem):
        handle = EvaluationHandle(
            request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(request, problem, self.attempts),
        )
        self.attempts += 1
        return handle

    def collect(self, handle, *, wait=True):
        request, problem, attempt = handle.backend_token
        self.collected.append((attempt, request.candidate_ids.tolist()))
        if handle._acknowledged_sequence >= 0:
            return []
        if attempt == 0:
            first = SerialEvaluator().evaluate_batch(request.x[:1], problem)
            first.candidate_ids = request.candidate_ids[:1]
            first.__post_init__()
            handle._delivered_sequence = 1
            return [
                EvaluationUpdate(
                    request.request_id,
                    EvaluationStatus.PARTIAL,
                    request.candidate_ids[:1],
                    first,
                    sequence=0,
                ),
                EvaluationUpdate(
                    request.request_id,
                    EvaluationStatus.FAILED,
                    np.empty(0, dtype=np.int64),
                    error=EvaluationErrorInfo("backend", "partial failure"),
                    sequence=1,
                ),
            ]
        result = SerialEvaluator().evaluate_batch(request.x, problem)
        result.candidate_ids = request.candidate_ids
        result.__post_init__()
        handle._delivered_sequence = 0
        return [
            EvaluationUpdate(
                request.request_id,
                EvaluationStatus.COMPLETED,
                request.candidate_ids,
                result,
                sequence=0,
            )
        ]

    def acknowledge(self, handle, sequence):
        self.acks.append((handle.backend_token[2], sequence))
        handle._acknowledged_sequence = sequence


def make_state():
    problem = Problem(
        func=lambda x: np.array([x[0]]),
        dim=1,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[0.0],
        ub=[1.0],
    )
    population = Population(ATTRS, 2)
    population._extend_internal(
        {
            "id": np.array([10, 11], dtype=np.int64),
            "x": np.array([[0.2], [0.1]], dtype=np.float64),
            "f": np.full((2, 1), np.nan),
            "g": np.empty((2, 0)),
            "cv": np.zeros(2),
        },
        preserve_ids=True,
    )
    return OptimizationState(
        problem=problem,
        population=population,
        archive=Archive(ATTRS, 2),
        pareto_archive=ParetoArchive(ATTRS, 2, direction=np.array([-1.0])),
        rng=np.random.default_rng(0),
        offspring=population,
    )


def requests():
    return [
        EvaluationRequest(0, np.array([10], dtype=np.int64), np.array([[0.2]])),
        EvaluationRequest(1, np.array([11], dtype=np.int64), np.array([[0.1]])),
    ]


def test_async_out_of_order_and_nonblocking_poll():
    state = make_state()
    scheduler = AsyncScheduler(
        AsyncEvaluator(SlowEvaluator(), max_workers=2), max_pending=2
    )
    state = scheduler.submit(state, requests())
    assert scheduler.poll(state, wait=False) is state
    state = scheduler.poll(state, wait=True)
    assert state.pending_evaluations == {}
    np.testing.assert_array_equal(state.archive.id, [11, 10])
    assert state.fe == 2


def test_completed_futures_commit_by_completion_time():
    state = make_state()
    evaluator = AsyncEvaluator(SlowEvaluator(), max_workers=2)
    scheduler = AsyncScheduler(evaluator, max_pending=2)
    try:
        state = scheduler.submit(state, requests())
        time.sleep(0.25)
        state = scheduler.poll(state, wait=False)
        np.testing.assert_array_equal(state.archive.id, [11, 10])
    finally:
        evaluator.close()


def test_async_capacity_and_reserved_budget():
    state = make_state()
    evaluator = AsyncEvaluator(SlowEvaluator(), max_workers=2)
    scheduler = AsyncScheduler(evaluator, max_pending=1, max_reserved_fe=1)
    state = scheduler.submit(state, [requests()[0]])
    try:
        scheduler.submit(state, [requests()[1]])
    except Exception as exc:
        assert "capacity" in str(exc)
    else:
        raise AssertionError("capacity was not enforced")
    evaluator.close()


def test_async_checkpoint_requires_reattach(tmp_path):
    state = make_state()
    evaluator = AsyncEvaluator(SlowEvaluator())
    scheduler = AsyncScheduler(evaluator)
    state = scheduler.submit(state, [requests()[0]])
    try:
        try:
            scheduler.checkpoint(state, str(tmp_path / "pending.npz"))
        except CheckpointError as exc:
            assert "reattach" in str(exc)
        else:
            raise AssertionError("unreattachable checkpoint was accepted")
        try:
            state.save(tmp_path / "direct.npz")
        except Exception as exc:
            assert "synchronous" in str(exc)
        else:
            raise AssertionError("direct checkpoint accepted pending work")
    finally:
        evaluator.close()


def test_reattachable_pending_checkpoint_resumes_once(tmp_path):
    state = make_state()
    evaluator = ReattachEvaluator()
    scheduler = AsyncScheduler(evaluator)
    state = scheduler.submit(state, [requests()[0]])
    path = tmp_path / "pending.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)
    assert restored.pending_evaluations[0].request.candidate_ids.tolist() == [10]
    restored = scheduler.reattach(restored)
    restored = scheduler.poll(restored, wait=True)
    assert restored.pending_evaluations == {}
    assert restored.fe == 1
    assert evaluator.acknowledged == [(0, 0)]
    np.testing.assert_allclose(restored.offspring.f[0], [0.2])
    np.testing.assert_array_equal(restored.archive.id, [10])


def test_timeout_detaches_running_future_without_applying_result():
    state = make_state()
    evaluator = AsyncEvaluator(SlowEvaluator())
    scheduler = AsyncScheduler(evaluator, timeout=0.001)
    try:
        state = scheduler.submit(state, [requests()[0]])
        time.sleep(0.01)
        state = scheduler.poll(state, wait=False)
        assert state.pending_evaluations == {}
        assert state.fe == 0
        assert len(state.archive) == 0
    finally:
        evaluator.close()


def test_async_evaluation_exception_becomes_failed_update():
    state = make_state()
    evaluator = AsyncEvaluator(FailingEvaluator())
    try:
        scheduler = AsyncScheduler(evaluator)
        state = scheduler.submit(state, [requests()[0]])
        state = scheduler.poll(state, wait=True)
        assert state.pending_evaluations == {}
        assert state.fe == 0
    finally:
        evaluator.close()


def test_timeout_requires_termination_capability():
    try:
        AsyncScheduler(BareEvaluator(), timeout=0.01)
    except Exception as exc:
        assert "timeout" in str(exc)
    else:
        raise AssertionError("unsupported timeout was accepted")


def test_batch_submit_requires_rollback_capability_before_side_effects():
    evaluator = FailSecondEvaluator()
    scheduler = AsyncScheduler(evaluator, max_pending=2)
    state = make_state()
    try:
        scheduler.submit(state, requests())
    except Exception as exc:
        assert "rollback" in str(exc)
        assert evaluator.calls == 0
    else:
        raise AssertionError("non-rollback batch submission was accepted")


def test_direct_strategy_uses_scheduler_for_submit_and_poll():
    class Algorithm:
        def __init__(self):
            self.tell_ids = []

        def ask(self, state, provider, n_offspring=None):
            return state.offspring

        def tell(self, state, provider, offspring):
            self.tell_ids.extend(offspring.get_array("id").tolist())

    state = make_state()
    algorithm = Algorithm()
    evaluator = AsyncEvaluator(SlowEvaluator(), max_workers=2)
    scheduler = AsyncScheduler(evaluator, max_pending=2)
    provider = SimpleNamespace(
        algorithm=algorithm,
        evaluator=evaluator,
        evaluation_policy=None,
        feedback_policy=None,
        async_scheduler=scheduler,
        cbmanager=None,
    )
    try:
        strategy = DirectStrategy()
        pending = strategy.step(state, provider)
        assert len(pending.pending_evaluations) == 2
        while pending.pending_evaluations:
            time.sleep(0.3)
            pending = strategy.step(pending, provider)
        assert sorted(algorithm.tell_ids) == [10, 11]
        assert pending.fe == 2
    finally:
        evaluator.close()


def test_partial_failure_retries_only_unapplied_candidates():
    class Algorithm:
        def __init__(self):
            self.told = []

        def tell(self, state, provider, offspring):
            self.told.extend(offspring.get_array("id").tolist())

    state = make_state()
    request = EvaluationRequest(
        0,
        np.array([10, 11], dtype=np.int64),
        np.array([[0.2], [0.1]], dtype=np.float64),
    )
    evaluator = PartialRetryEvaluator()
    algorithm = Algorithm()
    scheduler = AsyncScheduler(
        evaluator,
        retry_limit=1,
        feedback_policy=TrueOnlyFeedback(),
        algorithm=algorithm,
    )
    state = scheduler.submit(state, [request])
    state = scheduler.poll(state, wait=True)
    assert state.pending_evaluations == {}
    assert evaluator.attempts == 2
    assert sorted(algorithm.told) == [10, 11]
    assert evaluator.collected == [(0, [10, 11]), (1, [11])]
    assert state.fe == 2
    np.testing.assert_array_equal(np.sort(state.archive.id), [10, 11])


def test_partial_retry_with_callback_keeps_applied_ids():
    class Algorithm:
        def __init__(self):
            self.told = []

        def tell(self, state, provider, offspring):
            self.told.extend(offspring.get_array("id").tolist())

    events = []
    callback = CallbackManager()
    callback.register(PostEvaluationEvent, lambda event: events.append(event))
    state = make_state()
    evaluator = PartialRetryEvaluator()
    algorithm = Algorithm()
    scheduler = AsyncScheduler(
        evaluator,
        retry_limit=1,
        feedback_policy=TrueOnlyFeedback(),
        algorithm=algorithm,
        callback_manager=callback,
    )
    state = scheduler.submit(
        state,
        [
            EvaluationRequest(
                0,
                np.array([10, 11], dtype=np.int64),
                np.array([[0.2], [0.1]], dtype=np.float64),
            )
        ],
    )
    state = scheduler.poll(state, wait=True)
    assert evaluator.collected == [(0, [10, 11]), (1, [11])]
    assert state.fe == 2
    assert sorted(algorithm.told) == [10, 11]
    assert [int(event.candidate_ids[0]) for event in events] == [10, 11]


def test_runner_drains_after_generation_termination():
    class Algorithm:
        def ask(self, state, provider, n_offspring=None):
            return state.offspring

        def tell(self, state, provider, offspring):
            pass

    state = make_state()
    evaluator = AsyncEvaluator(SlowEvaluator(), max_workers=2)
    scheduler = AsyncScheduler(evaluator, max_pending=1)
    cbmanager = CallbackManager()
    events = []
    cbmanager.register(GenerationEndEvent, lambda event: events.append(event))
    optimizer = SimpleNamespace(
        strategy=DirectStrategy(),
        evaluator=evaluator,
        evaluation_policy=None,
        feedback_policy=None,
        feedback_policy_explicit=False,
        async_scheduler=scheduler,
        algorithm=Algorithm(),
        cbmanager=cbmanager,
        surrogate_manager=None,
        termination=SimpleNamespace(is_terminated=lambda ctx: ctx.gen >= 1),
        dispatch=cbmanager.dispatch,
        problem=state.problem,
    )
    try:
        result = list(Runner(optimizer).iterate_from(state))
        final = result[-1]
        assert final.fe == 2
        assert final.pending_evaluations == {}
        assert len(events) == 1
    finally:
        evaluator.close()


def test_runner_does_not_refill_after_termination_threshold():
    class CountingEvaluator(AsyncEvaluator):
        def __init__(self):
            super().__init__(SlowEvaluator(), max_workers=2)
            self.submit_count = 0

        def submit(self, request, problem):
            self.submit_count += 1
            return super().submit(request, problem)

    class Algorithm:
        def __init__(self):
            self.asks = 0

        def ask(self, state, provider, n_offspring=None):
            self.asks += 1
            return state.offspring

        def tell(self, state, provider, offspring):
            pass

    state = make_state()
    evaluator = CountingEvaluator()
    scheduler = AsyncScheduler(evaluator, max_pending=2)
    state = scheduler.submit(state, requests())
    algorithm = Algorithm()
    optimizer = SimpleNamespace(
        strategy=DirectStrategy(),
        evaluator=evaluator,
        evaluation_policy=None,
        feedback_policy=None,
        async_scheduler=scheduler,
        algorithm=algorithm,
        cbmanager=CallbackManager(),
        surrogate_manager=None,
        termination=SimpleNamespace(is_terminated=lambda ctx: ctx.fe >= 1),
        dispatch=lambda event: None,
        problem=state.problem,
    )
    try:
        final = Runner(optimizer).run_from(state)
        assert final.fe == 2
        assert evaluator.submit_count == 2
        assert algorithm.asks == 0
    finally:
        evaluator.close()


def test_runner_reattaches_loaded_pending_state(tmp_path):
    class Algorithm:
        def ask(self, state, provider, n_offspring=None):
            return state.offspring

        def tell(self, state, provider, offspring):
            pass

    state = make_state()
    evaluator = ReattachEvaluator()
    scheduler = AsyncScheduler(evaluator)
    state = scheduler.submit(state, [requests()[0]])
    path = tmp_path / "runner.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)
    optimizer = SimpleNamespace(
        strategy=DirectStrategy(),
        evaluator=evaluator,
        evaluation_policy=None,
        feedback_policy=None,
        feedback_policy_explicit=False,
        async_scheduler=scheduler,
        algorithm=Algorithm(),
        cbmanager=CallbackManager(),
        surrogate_manager=None,
        termination=SimpleNamespace(is_terminated=lambda ctx: ctx.fe >= 1),
        dispatch=lambda event: None,
        problem=state.problem,
    )
    result = list(Runner(optimizer).iterate_from(restored))
    assert result[-1].fe == 1
    assert result[-1].pending_evaluations == {}


def test_checkpoint_replays_buffered_update_without_backend_redelivery(tmp_path):
    state = make_state()
    result = SerialEvaluator().evaluate_batch(
        np.array([[0.2]], dtype=np.float64), state.problem
    )
    result.candidate_ids = np.array([10], dtype=np.int64)
    result.__post_init__()
    request = requests()[0]
    update = EvaluationUpdate(
        request.request_id,
        EvaluationStatus.COMPLETED,
        request.candidate_ids,
        result,
        sequence=0,
    )
    pending = PendingEvaluation(
        request,
        EvaluationStatus.COMPLETED,
        np.empty(0, dtype=np.int64),
        0,
        -1,
        {0: "received"},
        (update,),
        checkpointable=True,
    )
    state = state.replace(pending_evaluations={0: pending})
    path = tmp_path / "buffered.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)
    scheduler = AsyncScheduler(ReattachEvaluator())
    restored = scheduler.reattach(restored)
    assert restored.pending_evaluations == {}
    assert restored.fe == 1
    np.testing.assert_array_equal(restored.archive.id, [10])


def test_callback_failure_is_fatal_without_losing_pending_or_fe():
    class FailingCallback:
        def __init__(self):
            self.calls = 0

        def dispatch(self, event):
            self.calls += 1
            raise RuntimeError("callback failed")

    state = make_state()
    evaluator = ReattachEvaluator()
    callback = FailingCallback()
    scheduler = AsyncScheduler(evaluator, callback_manager=callback)
    state = scheduler.submit(state, [requests()[0]])
    try:
        scheduler.poll(state, wait=True)
    except RuntimeError as exc:
        assert "callback" in str(exc)
    else:
        raise AssertionError("callback failure was not fatal")
    assert state.pending_evaluations
    assert state.fe == 0
    assert len(state.archive) == 1
    assert callback.calls == 1


def test_callback_completed_replay_only_cleans_pending(tmp_path):
    state = make_state()
    evaluator = ReattachEvaluator()
    scheduler = AsyncScheduler(evaluator)
    state = scheduler.submit(state, [requests()[0]])
    update = evaluator.collect(state.evaluation_handles[0])[0]
    pending = state.pending_evaluations[0]
    pending = PendingEvaluation(
        pending.request,
        EvaluationStatus.COMPLETED,
        np.array([10], dtype=np.int64),
        0,
        0,
        {0: "callback-completed"},
        (update,),
        pending.reserved_cost,
        0,
        True,
        pending.original_candidate_ids,
        None,
        None,
        pending.prediction,
    )
    state = state.replace(pending_evaluations={0: pending})
    path = tmp_path / "callback-completed.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)
    restored = AsyncScheduler(ReattachEvaluator()).reattach(restored)
    assert restored.pending_evaluations == {}
    assert restored.fe == 0
    assert len(restored.archive) == 0


def test_partial_callback_checkpoint_reattaches_and_finishes(tmp_path):
    class TwoStageEvaluator(Evaluator):
        def __init__(self):
            self.attempts = 0
            self.reattach_ack = []
            self.submitted_ids = []

        def evaluate_batch(self, x, problem):
            return SerialEvaluator().evaluate_batch(x, problem)

        def submit(self, request, problem):
            self.submitted_ids.append(request.candidate_ids.tolist())
            handle = EvaluationHandle(
                request.request_id,
                EvaluationStatus.PENDING,
                backend_token=(request, problem, self.attempts, -1),
            )
            self.attempts += 1
            return handle

        def collect(self, handle, *, wait=True):
            request, problem, attempt, acknowledged = handle.backend_token
            if handle._acknowledged_sequence >= 0 and acknowledged < 0:
                return []
            if attempt == 0 and acknowledged < 0:
                result = SerialEvaluator().evaluate_batch(request.x[:1], problem)
                result.candidate_ids = request.candidate_ids[:1]
                result.__post_init__()
                handle._delivered_sequence = 0
                return [
                    EvaluationUpdate(
                        request.request_id,
                        EvaluationStatus.PARTIAL,
                        request.candidate_ids[:1],
                        result,
                        sequence=0,
                    )
                ]
            if attempt == 0 and acknowledged >= 0:
                handle._delivered_sequence = 1
                return [
                    EvaluationUpdate(
                        request.request_id,
                        EvaluationStatus.FAILED,
                        np.empty(0, dtype=np.int64),
                        error=EvaluationErrorInfo("backend", "retry"),
                        sequence=1,
                    )
                ]
            result = SerialEvaluator().evaluate_batch(request.x, problem)
            result.candidate_ids = request.candidate_ids
            result.__post_init__()
            handle._delivered_sequence = 0
            return [
                EvaluationUpdate(
                    request.request_id,
                    EvaluationStatus.COMPLETED,
                    request.candidate_ids,
                    result,
                    sequence=0,
                )
            ]

        def acknowledge(self, handle, sequence):
            handle._acknowledged_sequence = sequence

        def can_reattach(self, pending):
            return True

        def reattach(self, pending, problem):
            request = pending.request
            handle = EvaluationHandle(
                request.request_id,
                EvaluationStatus.PENDING,
                backend_token=(
                    request,
                    problem,
                    pending.retry_count,
                    pending.last_acknowledged_sequence,
                ),
            )
            handle._acknowledged_sequence = pending.last_acknowledged_sequence
            self.reattach_ack.append(pending.last_acknowledged_sequence)
            return handle

    class Algorithm:
        def __init__(self):
            self.told = []

        def tell(self, state, provider, offspring):
            self.told.extend(offspring.get_array("id").tolist())

    callback = CallbackManager()
    callback_ids = []
    callback.register(
        PostEvaluationEvent,
        lambda event: callback_ids.extend(event.candidate_ids.tolist()),
    )
    algorithm = Algorithm()
    evaluator = TwoStageEvaluator()
    scheduler = AsyncScheduler(
        evaluator,
        retry_limit=1,
        feedback_policy=TrueOnlyFeedback(),
        algorithm=algorithm,
        callback_manager=callback,
    )
    request = EvaluationRequest(
        0,
        np.array([10, 11], dtype=np.int64),
        np.array([[0.2], [0.1]]),
    )
    state = scheduler.submit(make_state(), [request])
    state = scheduler.poll(state, wait=False)
    assert state.pending_evaluations[0].last_acknowledged_sequence == 0
    path = tmp_path / "partial-callback.npz"
    scheduler.checkpoint(state, path)
    restored = OptimizationState.load(path, state.problem)
    resumed = AsyncScheduler(
        evaluator,
        retry_limit=1,
        feedback_policy=TrueOnlyFeedback(),
        algorithm=algorithm,
        callback_manager=callback,
    ).reattach(restored)
    resumed = AsyncScheduler(
        evaluator,
        retry_limit=1,
        feedback_policy=TrueOnlyFeedback(),
        algorithm=algorithm,
        callback_manager=callback,
    ).poll(resumed, wait=True)
    assert resumed.pending_evaluations == {}
    assert resumed.evaluation_handles == {}
    assert resumed.evaluation_owners == {}
    assert evaluator.reattach_ack == [0]
    assert evaluator.submitted_ids == [[10, 11], [11]]
    assert callback_ids == [10, 11]
    assert sorted(algorithm.told) == [10, 11]
    assert sorted(resumed.archive.id.tolist()) == [10, 11]
    assert resumed.fe == 2


def test_fatal_tombstone_roundtrip_raises_typed_error(tmp_path):
    class FailingAlgorithm:
        def tell(self, state, provider, offspring):
            raise RuntimeError("tell failed")

    state = make_state()
    evaluator = AsyncEvaluator(SlowEvaluator())
    scheduler = AsyncScheduler(
        evaluator,
        algorithm=FailingAlgorithm(),
        feedback_policy=TrueOnlyFeedback(),
    )
    state = scheduler.submit(state, [requests()[0]])
    try:
        scheduler.poll(state, wait=True)
    except EvaluationFatalError as exc:
        fatal_state = exc.state
    else:
        raise AssertionError("tell failure was not fatal")
    finally:
        evaluator.close()
    path = tmp_path / "fatal.npz"
    fatal_state.save(path)
    restored = OptimizationState.load(path, state.problem)
    with pytest.raises(EvaluationFatalError) as caught:
        AsyncScheduler(AsyncEvaluator(SlowEvaluator())).reattach(restored)
    assert caught.value.state is restored


def test_tell_failure_cannot_be_retried_after_tell_started():
    class FailingAlgorithm:
        def __init__(self):
            self.calls = 0

        def tell(self, state, provider, offspring):
            self.calls += 1
            raise RuntimeError("tell failed")

    state = make_state()
    evaluator = ReattachEvaluator()
    algorithm = FailingAlgorithm()
    scheduler = AsyncScheduler(
        evaluator, algorithm=algorithm, feedback_policy=TrueOnlyFeedback()
    )
    state = scheduler.submit(state, [requests()[0]])
    try:
        scheduler.poll(state, wait=True)
    except RuntimeError as exc:
        assert "tell" in str(exc)
    else:
        raise AssertionError("tell failure was not fatal")
    try:
        scheduler.poll(state, wait=True)
    except Exception as exc:
        assert "fatal" in str(exc) or "retried" in str(exc)
    else:
        raise AssertionError("tell failure was retried")
    assert algorithm.calls == 1


def test_async_archive_updates_main_and_pareto_once():
    state = make_state()
    scheduler = AsyncScheduler(ReattachEvaluator())
    request = EvaluationRequest(
        0, np.array([10, 11], dtype=np.int64), state.offspring.x.copy()
    )
    state = scheduler.submit(state, [request])
    state = scheduler.poll(state, wait=True)
    assert len(state.archive) == 2
    assert len(state.pareto_archive) == 1
    assert state.archive.id.tolist() == [10, 11]
    assert state.pareto_archive.id.tolist() == [11]


def test_archive_re_evaluation_uses_latest_observation():
    class ReevaluateEvaluator(ReattachEvaluator):
        def collect(self, handle, *, wait=True):
            if handle._acknowledged_sequence >= 0:
                return []
            request, problem = handle.backend_token
            base = SerialEvaluator().evaluate_batch(request.x, problem)
            result = EvaluationResult(
                np.full_like(base.f, float(request.metadata["value"])),
                base.g,
                base.cv,
                request.candidate_ids,
            )
            handle._delivered_sequence = 0
            return [
                EvaluationUpdate(
                    request.request_id,
                    EvaluationStatus.COMPLETED,
                    request.candidate_ids,
                    result,
                    sequence=0,
                )
            ]

    archive = Archive(ATTRS, 2, duplicate_policy="replace")
    state = make_state().replace(archive=archive)
    evaluator = ReevaluateEvaluator()
    scheduler = AsyncScheduler(evaluator)
    first = EvaluationRequest(
        0,
        np.array([10], dtype=np.int64),
        np.array([[0.2]]),
        metadata={"value": 0.2},
    )
    second = EvaluationRequest(
        1,
        np.array([10], dtype=np.int64),
        np.array([[0.2]]),
        metadata={"value": 0.8},
    )
    state = scheduler.poll(scheduler.submit(state, [first]), wait=True)
    state = scheduler.poll(scheduler.submit(state, [second]), wait=True)
    assert len(state.archive) == 1
    np.testing.assert_allclose(state.archive.f[0], [0.8])
    assert len(state.pareto_archive) == 1
    np.testing.assert_allclose(state.pareto_archive.f[0], [0.8])


def test_append_archive_keeps_distinct_request_observations():
    attrs = [*ATTRS, PopulationAttribute("request_id", np.int64, (), -1)]
    state = make_state().replace(archive=Archive(attrs, 2, duplicate_policy="append"))
    evaluator = ReattachEvaluator()
    scheduler = AsyncScheduler(evaluator)
    for request_id in (0, 1):
        request = EvaluationRequest(
            request_id,
            np.array([10], dtype=np.int64),
            np.array([[0.2]]),
        )
        state = scheduler.poll(scheduler.submit(state, [request]), wait=True)
    assert len(state.archive) == 2
    assert sorted(
        zip(state.archive.request_id.tolist(), state.archive.id.tolist())
    ) == [(0, 10), (1, 10)]


def test_chunk_cost_budget_uses_fsum_boundary():
    population = Population(ATTRS, 8)
    population._extend_internal(
        {
            "id": np.arange(8, dtype=np.int64),
            "x": np.arange(8, dtype=np.float64).reshape(-1, 1),
            "f": np.full((8, 1), np.nan),
            "g": np.empty((8, 0)),
            "cv": np.zeros(8),
        },
        preserve_ids=True,
    )
    state = make_state().replace(population=population, offspring=population)
    scheduler = AsyncScheduler(
        ReattachEvaluator(), max_pending=8, max_reserved_cost=0.1
    )
    for index in range(8):
        request = EvaluationRequest(
            index,
            np.array([index], dtype=np.int64),
            population.x[index : index + 1],
            metadata={"estimated_cost": 0.1 / 8},
        )
        state = scheduler.submit(state, [request])
    assert scheduler.reserved_cost(state) <= 0.1


def test_pending_prediction_snapshot_survives_wave_overwrite(tmp_path):
    first = SurrogatePrediction(
        {"objective": PredictionChannel(np.array([[1.0]]))},
        x=np.array([[0.2]]),
    )
    second = SurrogatePrediction(
        {"objective": PredictionChannel(np.array([[9.0]]))},
        x=np.array([[0.2]]),
    )
    state = make_state().replace(predictions=first)
    scheduler = AsyncScheduler(ReattachEvaluator())
    state = scheduler.submit(state, [requests()[0]])
    state = state.replace(predictions=second)
    assert state.pending_evaluations[0].prediction.value[0, 0] == 1.0
    path = tmp_path / "prediction-owner.npz"
    state.save(path)
    restored = OptimizationState.load(path, state.problem)
    assert restored.pending_evaluations[0].prediction.value[0, 0] == 1.0


def test_batch_submit_requires_declared_rollback_capability():
    class FalseRollbackEvaluator(BareEvaluator):
        def cancel(self, handle):
            return False

        def detach(self, handle):
            return False

    state = make_state()
    scheduler = AsyncScheduler(FalseRollbackEvaluator(), max_pending=2)
    with pytest.raises(EvaluationProtocolError, match="rollback"):
        scheduler.submit(state, requests())


def test_timeout_without_runtime_termination_keeps_fatal_tombstone():
    class UnstoppableEvaluator(ReattachEvaluator):
        def cancel(self, handle):
            return False

        def detach(self, handle):
            return False

    state = make_state()
    scheduler = AsyncScheduler(UnstoppableEvaluator(), timeout=0)
    state = scheduler.submit(state, [requests()[0]])
    state = scheduler.poll(state, wait=False)
    assert state.pending_evaluations[0].fatal_error is not None
    assert scheduler.pending_candidate_ids(state).tolist() == [10]
    assert state.data["async_fatal"]["request_id"] == 0
