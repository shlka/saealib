import importlib.util
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from saealib import (
    PSO,
    BatchExpectedImprovement,
    CooperativeCoevolution,
    CorrelatedQuadraticSurrogate,
    DynamicArchiveSelector,
    EvaluationRequest,
    EvaluationResult,
    Evaluator,
    FidelityEvaluator,
    FidelityPromotion,
    GlobalSurrogateManager,
    LHSInitializer,
    MigrationPolicy,
    Optimizer,
    PredictionChannel,
    PreSelectionStrategy,
    ReplicateSummary,
    SteadyStateStrategy,
    SurrogatePrediction,
    Termination,
    aggregate_replicates,
    max_fe,
    reference_problem,
)
from saealib.surrogate import ArchiveObjectiveSet


def _load_example(name):
    path = Path(__file__).parents[2] / "examples" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_qei_is_joint_and_uses_candidate_covariance():
    x = np.array([[0.0], [1.0]], dtype=np.float64)
    prediction = SurrogatePrediction(
        channels={
            "objective": PredictionChannel(
                value=np.array([[0.2], [0.2]]),
                std=np.array([[1.0], [1.0]]),
                covariance=np.array([[1.0, 0.95], [0.95, 1.0]]),
            )
        }
    )
    archive = SimpleNamespace(f=np.array([[0.0]], dtype=np.float64))
    result = BatchExpectedImprovement(n_draws=2048).evaluate(x, prediction, archive)
    assert result.artifacts["joint"] is True
    assert result.artifacts["order"].shape == (2,)
    assert np.array_equal(result.artifacts["order"], np.array([1, 0]))


def test_qei_direction_none_and_minimize_direction_are_equivalent():
    prediction = SurrogatePrediction(
        channels={
            "objective": PredictionChannel(
                value=np.array([[0.2], [0.4]]),
                std=np.ones((2, 1)),
                covariance=np.eye(2),
            )
        }
    )
    archive = SimpleNamespace(f=np.array([[0.0]], dtype=np.float64))
    none = BatchExpectedImprovement(n_draws=32).evaluate(
        np.zeros((2, 1)), prediction, archive
    )
    minimum = BatchExpectedImprovement(n_draws=32, direction=np.array([-1.0])).evaluate(
        np.zeros((2, 1)), prediction, archive
    )
    assert np.array_equal(none.scores, minimum.scores)


def test_qei_reference_surrogate_exposes_correlated_posterior():
    prediction = CorrelatedQuadraticSurrogate(0.8).predict(
        np.array([[0.0], [1.0]], dtype=np.float64)
    )
    assert prediction.channels["objective"].covariance[0, 1] > 0


def test_qei_correlation_changes_known_selection_order():
    archive = SimpleNamespace(f=np.array([[0.0]], dtype=np.float64))
    values = np.zeros((2, 1), dtype=np.float64)
    std = np.ones((2, 1), dtype=np.float64)

    def evaluate(covariance):
        prediction = SurrogatePrediction(
            channels={
                "objective": PredictionChannel(
                    value=values, std=std, covariance=covariance
                )
            }
        )
        return BatchExpectedImprovement(n_draws=8192).evaluate(
            np.zeros((2, 1)),
            prediction,
            archive,
            ctx=SimpleNamespace(rng=np.random.default_rng(7)),
        )

    diagonal = evaluate(np.eye(2))
    correlated = evaluate(np.full((2, 2), 0.99) + np.diag([0.01, 0.01]))
    assert correlated.artifacts["qei"][1] < diagonal.artifacts["qei"][1]
    assert not np.allclose(diagonal.artifacts["qei"], correlated.artifacts["qei"])


def test_qei_optimizer_path_preserves_joint_covariance():
    class RecordingQEI(BatchExpectedImprovement):
        def __init__(self):
            super().__init__(n_draws=64)
            self.covariances = []

        def evaluate(
            self, candidates_x, prediction, archive, ctx=None, *, prepared=None
        ):
            self.covariances.append(
                np.array(prediction.channels["objective"].covariance, copy=True)
            )
            return super().evaluate(
                candidates_x, prediction, archive, ctx, prepared=prepared
            )

    acquisition = RecordingQEI()
    optimizer = (
        Optimizer(reference_problem(), seed=41)
        .set_initializer(LHSInitializer(2, 2, 41))
        .set_algorithm(PSO())
        .set_surrogate_manager(
            GlobalSurrogateManager(
                CorrelatedQuadraticSurrogate(0.8), ArchiveObjectiveSet()
            )
        )
        .set_acquisition(acquisition)
        .set_strategy(PreSelectionStrategy(4, 2))
        .set_termination(Termination(max_fe(6)))
    )
    optimizer.run()
    assert any(
        covariance.shape[0] > 1
        and np.any(np.abs(covariance - np.diag(np.diag(covariance))) > 0)
        for covariance in acquisition.covariances
    )


def test_repeated_observations_preserve_ids_and_aggregate_noise():
    ids = np.array([10, 11], dtype=np.int64)
    observations = np.array(
        [[[1.0], [2.0]], [[1.2], [1.8]], [[0.8], [2.2]]], dtype=np.float64
    )
    summary = aggregate_replicates(ids, observations)
    assert isinstance(summary, ReplicateSummary)
    assert np.array_equal(summary.candidate_ids, ids)
    assert np.array_equal(summary.count, np.array([3, 3]))
    assert np.allclose(summary.mean[:, 0], [1.0, 2.0])
    assert np.all(summary.std > 0)


def test_repeated_policy_creates_distinct_requests_for_same_candidates():
    from saealib import RepeatedEvaluation

    class Allocator:
        def __init__(self):
            self.next = 10

        def allocate(self, count):
            values = np.arange(self.next, self.next + count, dtype=np.int64)
            self.next += count
            return values

    class Candidates(SimpleNamespace):
        def __len__(self):
            return len(self.x)

    candidates = Candidates(x=np.array([[0.1], [0.2]], dtype=np.float64), schema={})
    requests = RepeatedEvaluation(3).plan_replicates(
        candidates, SimpleNamespace(request_id_allocator=Allocator())
    )
    assert [request.metadata["replicate"] for request in requests] == [0, 1, 2]
    assert len({int(request.request_id) for request in requests}) == 3
    assert all(np.array_equal(request.candidate_ids, [0, 1]) for request in requests)


def test_fidelity_promotion_creates_new_request_with_same_candidates():
    allocator = SimpleNamespace(allocate=lambda count: np.array([99], dtype=np.int64))
    ctx = SimpleNamespace(request_id_allocator=allocator)
    request = EvaluationRequest(np.int64(4), np.array([10]), np.array([[0.2]]))
    promoted = FidelityPromotion(0, 1).promote(request, ctx)
    assert promoted.request_id == 99
    assert np.array_equal(promoted.candidate_ids, [10])
    assert promoted.metadata["fidelity"] == 1


def test_fidelity_evaluator_passes_explicit_level_to_problem_boundary():
    evaluator = FidelityEvaluator(lambda row, level: np.array([row[0] + level]))
    assert evaluator.evaluate(np.array([0.5]), 0)[0] == 0.5
    assert evaluator.evaluate(np.array([0.5]), 1)[0] == 1.5


def test_workflow_examples_use_evaluator_histories():
    noisy = _load_example("generality_noisy.py").main()
    assert len(noisy["archive"]) == 6
    assert len({int(request.request_id) for request in noisy["requests"]}) == 3
    assert np.array_equal(noisy["summary"].count, [3, 3])
    fidelity = _load_example("generality_multifidelity.py").main()
    assert fidelity["low_request"].metadata["fidelity"] == 0
    assert fidelity["high_request"].metadata["fidelity"] == 1
    assert fidelity["low_request"].candidate_ids.tolist() == [10, 11]
    assert fidelity["high_request"].candidate_ids.tolist() == [10]


def test_coevolution_updates_one_block_in_shared_context():
    component = CooperativeCoevolution(4, ((0, 1), (2, 3)))
    component.update(0, np.array([1.0, 2.0]))
    context = component.update(1, np.array([3.0, 4.0]))
    assert np.array_equal(context, [1.0, 2.0, 3.0, 4.0])
    result = component.optimize(
        reference_problem(4),
        (np.array([0.2, -0.2]), np.array([0.0, 0.0])),
    )
    assert result.candidate_ids.tolist() == [0, 1]
    assert result.objective_history[-1] <= result.objective_history[0]


def test_migration_copies_rows_without_aliasing():
    source = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.zeros((2, 2))
    migrated = MigrationPolicy(1).migrate(source, target)
    assert np.array_equal(migrated, [[1.0, 2.0], [0.0, 0.0]])
    assert not np.shares_memory(migrated, source)
    final, events = MigrationPolicy(1).optimize(
        reference_problem(),
        (source, target),
        rounds=2,
    )
    assert len(events) == 2
    assert final[1].shape == target.shape


def test_dynamic_archive_selects_nearest_environment_snapshot():
    selector = DynamicArchiveSelector()
    selector.add(0, np.array([[0.0]]), np.array([[0.0]]))
    selector.add(10, np.array([[1.0]]), np.array([[1.0]]))
    assert selector.select(8).environment == 10
    workflow_selector = DynamicArchiveSelector()
    snapshots = workflow_selector.optimize(
        lambda environment: reference_problem(shift=float(environment)),
        (0, 10),
        np.array([[0.0, 0.0], [1.0, 1.0]]),
    )
    assert len(snapshots) == 2


def test_async_example_contract_has_independent_candidate_futures():
    from saealib import AsyncEvaluator, Problem

    class Delayed(Evaluator):
        def evaluate_batch(self, x, problem):
            time.sleep(0.02 if x[0, 0] < 0.5 else 0.001)
            return EvaluationResult(
                f=np.array([[x[0, 0]]], dtype=np.float64),
                g=np.empty((1, 0), dtype=np.float64),
                cv=np.zeros(1, dtype=np.float64),
            )

    problem = Problem(
        lambda row: np.array([row[0]]),
        dim=1,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[0.0],
        ub=[1.0],
    )
    evaluator = AsyncEvaluator(Delayed(), max_workers=2)
    slow = EvaluationRequest(np.int64(1), np.array([10]), np.array([[0.1]]))
    fast = EvaluationRequest(np.int64(2), np.array([11]), np.array([[0.9]]))
    slow_handle = evaluator.submit(slow, problem)
    fast_handle = evaluator.submit(fast, problem)
    fast_update = evaluator.collect(fast_handle, wait=True)[0]
    slow_update = evaluator.collect(slow_handle, wait=True)[0]
    assert fast_update.candidate_ids.tolist() == [11]
    assert slow_update.candidate_ids.tolist() == [10]


def test_async_optimizer_uses_single_candidate_refill_and_drains():
    from saealib import AsyncEvaluator, AsyncScheduler, SerialEvaluator

    class DelayedSerial(SerialEvaluator):
        def __init__(self):
            super().__init__()
            self.requests = []
            self.completions = []
            self.active = 0
            self.peak_active = 0
            self.lock = threading.Lock()
            self.first_started = threading.Event()
            self.refill_started = threading.Event()
            self.release_first = threading.Event()
            self.events = []

        def evaluate_request(self, request, problem):
            with self.lock:
                self.requests.append(request)
                self.active += 1
                self.peak_active = max(self.peak_active, self.active)
                single_count = sum(
                    len(item.candidate_ids) == 1 for item in self.requests
                )
                if len(request.candidate_ids) == 1:
                    self.events.append(f"start-{single_count}")
            if len(request.candidate_ids) == 1:
                if single_count == 1:
                    self.first_started.set()
                    assert self.release_first.wait(5)
                elif single_count == 2:
                    assert self.first_started.wait(5)
                else:
                    self.refill_started.set()
                    self.release_first.set()
            result = super().evaluate_request(request, problem)
            with self.lock:
                self.completions.append(request.candidate_ids.copy())
                if len(request.candidate_ids) == 1:
                    self.events.append(f"complete-{single_count}")
                self.active -= 1
            return result

    class RecordingPSO(PSO):
        def __init__(self):
            super().__init__()
            self.told_ids = []

        def tell(self, ctx, provider, offspring):
            self.told_ids.extend(offspring.get_array("id").tolist())
            return super().tell(ctx, provider, offspring)

    recorder = DelayedSerial()
    evaluator = AsyncEvaluator(recorder, max_workers=2)
    scheduler = AsyncScheduler(evaluator, max_pending=2, max_reserved_fe=6)
    algorithm = RecordingPSO()
    optimizer = (
        Optimizer(reference_problem(), seed=53)
        .set_initializer(LHSInitializer(2, 2, 53))
        .set_algorithm(algorithm)
        .set_strategy(SteadyStateStrategy())
        .set_evaluator(evaluator)
        .set_async_scheduler(scheduler)
        .set_termination(
            Termination(lambda state: state.fe + len(state.pending_evaluations) >= 6)
        )
    )
    state = optimizer.run()
    while state.pending_evaluations:
        state = scheduler.poll(state, wait=True)
    assert all(len(request.candidate_ids) == 1 for request in recorder.requests[1:])
    assert len(recorder.requests) <= 6
    assert recorder.peak_active == 2
    assert recorder.refill_started.is_set()
    assert recorder.events.index("complete-2") < recorder.events.index("start-3")
    assert recorder.events.index("start-3") < recorder.events.index("complete-1")
    submitted = [int(request.candidate_ids[0]) for request in recorder.requests[1:]]
    assert len(submitted) >= 2
    completed = [int(ids[0]) for ids in recorder.completions if len(ids) == 1]
    assert completed != submitted
    archive_ids = set(state.archive.get_array("id"))
    completion_ids = set(np.concatenate(recorder.completions))
    initial_ids = archive_ids - completion_ids
    assert len(initial_ids) == 2
    assert completion_ids == set(algorithm.told_ids)
    assert archive_ids - initial_ids <= set(algorithm.told_ids)
    assert state.fe <= 6
    assert len(state.pending_evaluations) == 0
    assert len(state.evaluation_handles) == 0
    assert len(state.evaluation_owners) == 0


def test_reference_examples_use_their_actual_public_components():
    examples = {
        "generality_sync_ib.py": "archive",
        "generality_ego.py": "archive",
        "generality_qei.py": "archive",
        "generality_async.py": "archive",
        "generality_pairwise.py": "archive",
    }
    for name, attribute in examples.items():
        result = _load_example(name).main()
        assert len(getattr(result, attribute)) > 0
    async_state = _load_example("generality_async.py").main()
    assert async_state.fe == 6
    assert len(async_state.pending_evaluations) == 0
    assert len(async_state.evaluation_handles) == 0
    assert len(async_state.evaluation_owners) == 0
    assert len(async_state.archive) == len(
        np.unique(async_state.archive.get_array("id"))
    )

    noisy = _load_example("generality_noisy.py").main()
    assert noisy["summary"].count.tolist() == [3, 3]
    assert np.all(noisy["summary"].std > 0)
    assert noisy["fe"] == 6
    assert len(noisy["archive"]) == 6
    assert len(noisy["truth_archive"]) == 2
    assert np.array_equal(noisy["truth_archive"].get_array("id"), [10, 11])
    noisy_again = _load_example("generality_noisy.py").main()
    assert np.array_equal(noisy["summary"].mean, noisy_again["summary"].mean)
    assert np.array_equal(noisy["summary"].std, noisy_again["summary"].std)
    fidelity = _load_example("generality_multifidelity.py").main()
    assert fidelity["promoted"] is True
    assert fidelity["fe"] == 3
    assert fidelity["cost"] == 4.0
    assert len(fidelity["archive"]) == 3
    coevolution = _load_example("generality_coevolution.py").main()
    assert np.array_equal(coevolution["context"], [0.2, -0.2, 0.0, 0.0])
    assert len(coevolution["history"]) == 2
    islands = _load_example("generality_islands.py").main()
    assert len(islands["events"]) == 2
    dynamic = _load_example("generality_dynamic_archive.py").main()
    assert len(dynamic["snapshots"]) == 2
    assert dynamic["selected"].environment == 10
