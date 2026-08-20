"""Integration regressions for CORS beta cadence across execution modes."""

from __future__ import annotations

import warnings
from itertools import pairwise
from typing import Any

import numpy as np

from saealib import (
    GA,
    CrossoverBLXAlpha,
    GaussianKernel,
    IndividualBasedStrategy,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    RBFSurrogate,
    RepeatedEvaluation,
    SequentialSelection,
    Termination,
    TruncationSelection,
    max_gen,
)
from saealib.acquisition import CompositeAcquisition
from saealib.acquisition.mean import CORSDistance
from saealib.callback import AcquisitionEndEvent, PostEvaluationEvent
from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.core.compiler.cors_diagnostics import CORS_NONSEQUENTIAL_MESSAGE
from saealib.core.state import PROPOSALS_CURRENT
from saealib.execution import (
    AsyncEvaluationScheduler,
    EvaluationHandle,
    EvaluationRequest,
    EvaluationStatus,
    EvaluationUpdate,
    Evaluator,
    SerialEvaluator,
)
from saealib.policies.evaluation import (
    EvaluateAll,
    EvaluationPlan,
    EvaluationPlanner,
    TopKEvaluation,
)
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.stages import (
    AcquisitionStage,
    AsyncEvaluationSubmitStage,
    EvaluationPlanStage,
)
from saealib.strategies.ps import PreSelectionStrategy
from saealib.surrogate.prediction import SurrogatePrediction

SEARCH_PATTERN = (0.9, 0.4, 0.0)
DIM = 2


class _RecordingCORSDistance(CORSDistance):
    """Expose the beta resolved by the real CORS implementation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.records: list[tuple[int, int, float]] = []

    def prepare(self, archive, ctx=None):
        reference = super().prepare(archive, ctx)
        assert ctx is not None
        self.records.append((ctx.gen, ctx.decision_count, reference.beta))
        return reference


class _ImmediateEvaluator(Evaluator):
    """Deterministic evaluator used behind the real async scheduler seam."""

    def evaluate_batch(self, x, problem):
        return SerialEvaluator().evaluate_batch(x, problem)

    def submit(self, request, problem):
        return EvaluationHandle(
            request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(request, problem),
        )

    def collect(self, handle, *, wait=True):
        request, problem = handle.backend_token
        result = self.evaluate_batch(request.x, problem)
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

    def supports_batch_rollback(self):
        return True

    def can_reattach(self, pending):
        return True

    def reattach(self, pending, problem):
        return EvaluationHandle(
            pending.request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(pending.request, problem),
        )


class _OpaqueBatchPlanner(EvaluationPlanner):
    """Planner whose batch behavior is only observable at runtime."""

    def plan(self, candidates, acquisition, ctx):
        del candidates, acquisition, ctx
        return EvaluationPlan(
            (
                EvaluationRequest(
                    np.int64(100),
                    np.array([10], dtype=np.int64),
                    np.array([[0.2]], dtype=np.float64),
                ),
                EvaluationRequest(
                    np.int64(101),
                    np.array([11], dtype=np.int64),
                    np.array([[0.3]], dtype=np.float64),
                ),
            )
        )


class _OpaqueEvaluateAllPlanner(EvaluationPlanner):
    """Planner whose multi-candidate behavior is only observable at runtime."""

    def plan(self, candidates, acquisition, ctx):
        return EvaluateAll().plan(candidates, acquisition, ctx)


_ASYNC_ATTRS = [
    PopulationAttribute("id", np.int64, (), -1),
    PopulationAttribute("x", np.float64, (1,)),
    PopulationAttribute("f", np.float64, (1,)),
    PopulationAttribute("g", np.float64, (0,)),
    PopulationAttribute("cv", np.float64, (), 0.0),
]


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


def _make_optimizer(
    problem: Problem, acquisition: CORSDistance, n_gen: int
) -> Optimizer:
    return (
        Optimizer(problem, seed=7)
        .set_initializer(LHSInitializer(6, 4))
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.5),
                mutation=MutationUniform(prob_var=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_surrogate(RBFSurrogate(GaussianKernel()), n_neighbors=5)
        .set_acquisition(acquisition)
        .set_strategy(IndividualBasedStrategy(evaluation_ratio=0.5))
        .set_evaluation_planner(RepeatedEvaluation(2))
        .set_termination(Termination(max_gen(n_gen)))
    )


def _make_candidate(candidate_id: int, x_value: float) -> Population:
    population = Population(_ASYNC_ATTRS, init_capacity=1)
    population._extend_internal(
        {
            "id": np.array([candidate_id], dtype=np.int64),
            "x": np.array([[x_value]], dtype=np.float64),
            "f": np.full((1, 1), np.nan),
            "g": np.empty((1, 0), dtype=np.float64),
            "cv": np.zeros(1, dtype=np.float64),
        },
        preserve_ids=True,
    )
    return population


def _make_async_state() -> OptimizationState:
    problem = Problem(
        func=lambda x: np.array([x[0]]),
        dim=1,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[0.0],
        ub=[1.0],
        comparator=SingleObjectiveComparator(),
    )
    initial_population = _make_candidate(10, 0.2)
    archive = Archive(_ASYNC_ATTRS, init_capacity=8)
    archive._extend_internal(
        {
            "id": np.array([0], dtype=np.int64),
            "x": np.array([[0.0]], dtype=np.float64),
            "f": np.array([[0.0]], dtype=np.float64),
            "g": np.empty((1, 0), dtype=np.float64),
            "cv": np.zeros(1, dtype=np.float64),
        },
        preserve_ids=True,
    )
    state = OptimizationState(
        problem=problem,
        population=initial_population,
        archive=archive,
        pareto_archive=ParetoArchive(
            _ASYNC_ATTRS, init_capacity=8, direction=np.array([-1.0])
        ),
        rng=np.random.default_rng(0),
        gen=1,
        offspring=initial_population,
    )
    state.set_state(PROPOSALS_CURRENT, 0)
    return state


def test_repeated_plan_keeps_one_cors_decision_per_generation():
    """Repeated requests do not advance CORS beta within one generation."""
    acquisition = _RecordingCORSDistance(delta=1.0, search_pattern=SEARCH_PATTERN)
    optimizer = _make_optimizer(_make_problem(), acquisition, n_gen=3)
    score_calls: list[tuple[int, int]] = []

    def record_score(event: AcquisitionEndEvent) -> None:
        score_calls.append(
            (int(event.ctx.gen), int(getattr(event.ctx, "decision_count")))
        )

    optimizer.cbmanager.register(AcquisitionEndEvent, record_score)
    states = list(optimizer.iterate())

    assert [(state.gen, state.decision_count) for state in states] == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    ]
    assert score_calls == [(1, 0), (2, 1), (3, 2)]
    assert [(gen, count) for gen, count, _ in acquisition.records] == score_calls
    np.testing.assert_allclose(
        [beta for _, _, beta in acquisition.records], SEARCH_PATTERN
    )
    assert states[-1].decision_count == 3


def test_canonical_cors_evaluates_one_candidate_per_decision():
    acquisition = _RecordingCORSDistance(delta=1.0, search_pattern=SEARCH_PATTERN)
    optimizer = _make_optimizer(_make_problem(), acquisition, n_gen=3)
    optimizer.set_strategy(PreSelectionStrategy(n_candidates=4, n_select=1))
    optimizer.set_evaluation_planner(TopKEvaluation(1, sanitize_nonfinite=True))
    score_calls: list[tuple[int, int]] = []
    optimizer.cbmanager.register(
        AcquisitionEndEvent,
        lambda event: score_calls.append(
            (int(event.ctx.gen), int(event.ctx.decision_count))
        ),
    )

    states = list(optimizer.iterate())

    assert [(state.gen, state.decision_count) for state in states] == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    ]
    assert score_calls == [(1, 0), (2, 1), (3, 2)]
    assert [(gen, count) for gen, count, _ in acquisition.records] == score_calls
    np.testing.assert_allclose(
        [beta for _, _, beta in acquisition.records], SEARCH_PATTERN
    )
    assert [current.fe - previous.fe for previous, current in pairwise(states)] == [
        1,
        1,
        1,
    ]


def test_async_refill_reads_advanced_decision_count_within_generation():
    """Async scheduler refill follows CORS phase without generation changes.

    The scheduler and async submission stage are real.  The evaluator is a
    deterministic in-process adapter that completes immediately; repeated
    stage entry models steady-state refill without wall-clock/thread flakiness.
    """
    state = _make_async_state()
    acquisition = _RecordingCORSDistance(delta=1.0, search_pattern=SEARCH_PATTERN)
    acquisition_stage = AcquisitionStage(acquisition)
    evaluator = _ImmediateEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator, max_pending=1)
    submit_stage = AsyncEvaluationSubmitStage(scheduler, EvaluateAll())
    completed: list[tuple[int, int, int]] = []

    for index in range(3):
        population = _make_candidate(10 + index, 0.2 + 0.1 * index)
        state = state.replace(
            offspring=population,
            predictions=SurrogatePrediction.objective(
                value=np.array([[1.0]], dtype=np.float64), x=population.x
            ),
        )
        state = acquisition_stage.execute(state)
        state = submit_stage.execute(state)
        assert state.decision_count == index + 1

        state = scheduler.poll(state, wait=True)
        completed.append((state.gen, state.fe, state.decision_count))
        assert state.pending_evaluations == {}

    assert completed == [(1, 1, 1), (1, 2, 2), (1, 3, 3)]
    assert [(gen, count) for gen, count, _ in acquisition.records] == [
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    np.testing.assert_allclose(
        [beta for _, _, beta in acquisition.records], SEARCH_PATTERN
    )


def test_npz_resume_preserves_cors_search_phase(tmp_path):
    """Portable state resume uses the saved decision count for CORS beta."""
    problem = _make_problem()
    midpoint_acquisition = _RecordingCORSDistance(
        delta=1.0, search_pattern=SEARCH_PATTERN
    )
    midpoint = _make_optimizer(problem, midpoint_acquisition, n_gen=2).run()
    checkpoint = tmp_path / "cors-phase.npz"
    midpoint.save(checkpoint)

    saved_decision_count = midpoint.decision_count
    loaded = OptimizationState.load(checkpoint, problem)
    assert saved_decision_count == 2
    assert loaded.decision_count == saved_decision_count

    resumed_acquisition = _RecordingCORSDistance(
        delta=1.0, search_pattern=SEARCH_PATTERN
    )
    resumed_optimizer = _make_optimizer(problem, resumed_acquisition, n_gen=4)
    resumed = resumed_optimizer.run_from(loaded)

    assert resumed_acquisition.records
    first_resume_prepare = resumed_acquisition.records[0]
    assert first_resume_prepare[0] == midpoint.gen + 1
    assert first_resume_prepare[1] == saved_decision_count
    assert (
        first_resume_prepare[2]
        == SEARCH_PATTERN[saved_decision_count % len(SEARCH_PATTERN)]
    )
    assert resumed.decision_count == 4


def test_repeated_evaluation_uses_unique_candidate_ids_for_batch_warning():
    events: list[tuple[int, bool]] = []
    state = _make_async_state()
    stage = EvaluationPlanStage(
        RepeatedEvaluation(2),
        cors_runtime_warning=lambda candidate_count, overlap: events.append(
            (candidate_count, overlap)
        ),
    )

    state = stage.execute(state)

    assert state.decision_count == 1
    assert events == []


def test_async_sequential_cors_submission_does_not_warn():
    state = _make_async_state()
    acquisition = _RecordingCORSDistance(delta=1.0, search_pattern=SEARCH_PATTERN)
    acquisition_stage = AcquisitionStage(acquisition)
    evaluator = _ImmediateEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator, max_pending=1)
    events: list[tuple[int, bool]] = []
    submit_stage = AsyncEvaluationSubmitStage(
        scheduler,
        EvaluateAll(),
        cors_runtime_warning=lambda candidate_count, overlap: events.append(
            (candidate_count, overlap)
        ),
    )

    for index in range(3):
        population = _make_candidate(10 + index, 0.2 + 0.1 * index)
        state = state.replace(
            offspring=population,
            predictions=SurrogatePrediction.objective(
                value=np.array([[1.0]], dtype=np.float64), x=population.x
            ),
        )
        state = acquisition_stage.execute(state)
        state = submit_stage.execute(state)
        state = scheduler.poll(state, wait=True)

    assert events == []


def test_async_overlapping_cors_decisions_warn_at_runtime():
    state = _make_async_state()
    evaluator = _ImmediateEvaluator()
    scheduler = AsyncEvaluationScheduler(evaluator, max_pending=2)
    events: list[tuple[int, bool]] = []
    submit_stage = AsyncEvaluationSubmitStage(
        scheduler,
        EvaluateAll(),
        cors_runtime_warning=lambda candidate_count, overlap: events.append(
            (candidate_count, overlap)
        ),
    )

    state = submit_stage.execute(state)
    state = state.replace(
        offspring=_make_candidate(11, 0.3),
        evaluation_plan=None,
        evaluation_plan_state=None,
        evaluation_plan_updates={},
        evaluation_request=None,
    )
    state = submit_stage.execute(state)

    assert events == [(1, True)]
    assert len(state.pending_evaluations) == 2


def test_optimizer_emits_one_runtime_warning_for_a_true_evaluation_batch():
    problem = _make_problem()
    optimizer = (
        Optimizer(problem, seed=7)
        .set_initializer(LHSInitializer(6, 4))
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.5),
                mutation=MutationUniform(prob_var=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_surrogate(RBFSurrogate(GaussianKernel()), n_neighbors=5)
        .set_acquisition(CORSDistance(delta=1.0, search_pattern=SEARCH_PATTERN))
        .set_strategy(PreSelectionStrategy(n_candidates=4, n_select=2))
        .set_termination(Termination(max_gen(2)))
    )

    evaluated_batch_sizes: list[int] = []
    optimizer.cbmanager.register(
        PostEvaluationEvent,
        lambda event: evaluated_batch_sizes.append(
            0 if event.candidate_ids is None else len(event.candidate_ids)
        ),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        list(optimizer.iterate())

    cors_warnings = [
        item
        for item in caught
        if "CORSDistance is used outside the source-faithful sequential one-candidate "
        in str(item.message)
    ]
    assert len(cors_warnings) == 1
    assert evaluated_batch_sizes == [2, 2]


def test_cors_runtime_warning_is_once_per_run():
    optimizer = (
        Optimizer(_make_problem(), seed=7)
        .set_initializer(LHSInitializer(6, 4))
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.5),
                mutation=MutationUniform(prob_var=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_surrogate(RBFSurrogate(GaussianKernel()), n_neighbors=5)
        .set_acquisition(CORSDistance(delta=1.0, search_pattern=SEARCH_PATTERN))
        .set_strategy(PreSelectionStrategy(n_candidates=4, n_select=2))
        .set_evaluation_planner(_OpaqueEvaluateAllPlanner())
        .set_termination(Termination(max_gen(1)))
    )

    for _ in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            optimizer.run()

        cors_warnings = [
            item for item in caught if str(item.message) == CORS_NONSEQUENTIAL_MESSAGE
        ]
        assert len(cors_warnings) == 1


def test_nested_composite_cors_warns_for_opaque_runtime_batch():
    inner = CompositeAcquisition(
        {
            "cors": CORSDistance(delta=1.0, search_pattern=SEARCH_PATTERN),
        },
        combine_fn=lambda scores: scores[0],
    )
    acquisition = CompositeAcquisition(
        {"inner": inner},
        combine_fn=lambda scores: scores[0],
    )
    optimizer = Optimizer(_make_problem(), seed=7).set_acquisition(acquisition)
    stage = EvaluationPlanStage(
        planner=_OpaqueBatchPlanner(),
        cors_runtime_warning=optimizer._cors_runtime_warning,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stage.execute(_make_async_state())

    cors_warnings = [
        item
        for item in caught
        if "CORSDistance is used outside the source-faithful sequential one-candidate "
        in str(item.message)
    ]
    assert len(cors_warnings) == 1
