"""Integration regressions for CORS beta cadence across execution modes."""

from __future__ import annotations

import warnings
from dataclasses import replace
from itertools import pairwise
from typing import Any

import numpy as np
import pytest

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
from saealib.acquisition.mean import CORSDistance, MeanPrediction
from saealib.callback import AcquisitionEndEvent, PostEvaluationEvent
from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.core.compiler.cors_diagnostics import CORS_NONSEQUENTIAL_MESSAGE
from saealib.core.compiler.graph import DataEdge, NodeRef
from saealib.core.graph_builder import (
    NodeAdapterSpec,
    StageContractNodeAdapter,
    build_decomposed_component_graph_from_specs,
)
from saealib.core.state import PROPOSALS_CURRENT
from saealib.exceptions import ValidationError
from saealib.execution import (
    AsyncEvaluationScheduler,
    EvaluationErrorInfo,
    EvaluationHandle,
    EvaluationRequest,
    EvaluationStatus,
    EvaluationUpdate,
    Evaluator,
    SerialEvaluator,
)
from saealib.pipeline import Stage
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
    ArchiveUpdateStage,
    AsyncEvaluationSubmitStage,
    CountGenerationStage,
    EvaluationAcknowledgeStage,
    EvaluationApplyStage,
    EvaluationCollectStage,
    EvaluationPlanStage,
    EvaluationSubmitStage,
)
from saealib.strategies.base import OptimizationStrategy
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


class _StatusEvaluator(_ImmediateEvaluator):
    """Return scripted terminal statuses through both evaluator seams."""

    def __init__(self, statuses: list[EvaluationStatus]) -> None:
        self._statuses = list(statuses)

    def collect(self, handle, *, wait=True):
        status = self._statuses.pop(0)
        if status is EvaluationStatus.COMPLETED:
            return super().collect(handle, wait=wait)
        request, _ = handle.backend_token
        handle._delivered_sequence = 0
        return [
            EvaluationUpdate(
                request.request_id,
                status,
                np.empty(0, dtype=np.int64),
                error=EvaluationErrorInfo("scripted", status.value),
                sequence=0,
            )
        ]


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


class _CustomPreSelectionStrategy(OptimizationStrategy):
    """Custom graph strategy that does not know about CORS diagnostics."""

    requires_surrogate = True

    def build_graph(self, provider):
        return PreSelectionStrategy(n_candidates=4, n_select=1).build_graph(provider)


class _PrepareOffspringStage(Stage):
    """Provide a deterministic candidate batch to the branch planner stages."""

    name = "prepare_offspring"

    def execute(self, state: OptimizationState) -> OptimizationState:
        assert state.population is not None
        return state.replace(
            offspring=state.population,
            scores=np.zeros(len(state.population), dtype=np.float64),
        )


class _NoopAcquisitionStage(AcquisitionStage):
    """Expose an acquisition branch without needing surrogate predictions."""

    def execute(self, state: OptimizationState) -> OptimizationState:
        return state


class _ClearingEvaluationPlanStage(EvaluationPlanStage):
    """Run one real planner call, then close its request for the next branch."""

    def __init__(self, planner, candidate_counts: list[int]) -> None:
        super().__init__(planner)
        self.candidate_counts = candidate_counts

    def execute(self, state: OptimizationState) -> OptimizationState:
        result = super().execute(state)
        plan = result.evaluation_plan
        assert plan is not None
        self.candidate_counts.append(
            sum(len(request.candidate_ids) for request in plan.requests)
        )
        return result.replace(
            evaluation_request=None,
            evaluation_plan=None,
            evaluation_plan_state=None,
            pending_evaluations={},
            evaluation_updates=[],
            evaluation_update_new_ids=[],
            evaluation_plan_updates={},
            evaluation_new_ids=np.empty(0, dtype=np.int64),
        )


class _IndependentRuntimeBranchesStrategy(OptimizationStrategy):
    """Sequentially execute two data-flow-independent acquisition branches."""

    requires_surrogate = False

    def __init__(self) -> None:
        self._graph = None
        self.cors_candidate_counts: list[int] = []
        self.batch_candidate_counts: list[int] = []

    def build_graph(self, provider):
        del provider
        if self._graph is not None:
            return self._graph

        cors_acquisition = _NoopAcquisitionStage(
            CORSDistance(delta=1.0, search_pattern=SEARCH_PATTERN)
        )
        cors_acquisition.name = "cors_acquisition"
        cors_planner = _ClearingEvaluationPlanStage(
            TopKEvaluation(1), candidate_counts=self.cors_candidate_counts
        )
        cors_planner.name = "cors_planner"

        other_acquisition = _NoopAcquisitionStage(
            MeanPrediction(direction=np.array([-1.0]))
        )
        other_acquisition.name = "other_acquisition"
        other_planner = _ClearingEvaluationPlanStage(
            EvaluateAll(), candidate_counts=self.batch_candidate_counts
        )
        other_planner.name = "other_planner"

        stages = (
            CountGenerationStage(),
            _PrepareOffspringStage(),
            cors_acquisition,
            cors_planner,
            other_acquisition,
            other_planner,
        )
        specs = tuple(
            NodeAdapterSpec(
                component_id=stage.name,
                adapter=StageContractNodeAdapter(stage, node_path=stage.name),
            )
            for stage in stages
        )
        graph = build_decomposed_component_graph_from_specs(specs)
        branch_edges = (
            DataEdge(
                source=NodeRef(
                    component_id="cors_acquisition___acquisition", role="acquisition"
                ),
                target=NodeRef(
                    component_id="cors_planner___planner", role="evaluation_planner"
                ),
                source_port="scores",
                target_port="acquisition",
            ),
            DataEdge(
                source=NodeRef(
                    component_id="other_acquisition___acquisition", role="acquisition"
                ),
                target=NodeRef(
                    component_id="other_planner___planner", role="evaluation_planner"
                ),
                source_port="scores",
                target_port="acquisition",
            ),
        )
        self._graph = replace(graph, data_edges=(*graph.data_edges, *branch_edges))
        return self._graph


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
    assert states[-1].completed_decision_count == states[-1].decision_count == 3


def test_canonical_cors_evaluates_one_candidate_per_decision():
    acquisition = _RecordingCORSDistance(delta=1.0, search_pattern=SEARCH_PATTERN)
    optimizer = _make_optimizer(_make_problem(), acquisition, n_gen=3)
    optimizer.set_strategy(PreSelectionStrategy(n_candidates=4, n_select=1))
    optimizer.set_evaluation_planner(TopKEvaluation(1, sanitize_nonfinite=True))
    score_calls: list[int] = []
    optimizer.cbmanager.register(
        AcquisitionEndEvent,
        lambda event: score_calls.append(int(event.ctx.gen)),
    )

    states = list(optimizer.iterate())

    assert [(state.gen, state.decision_count) for state in states] == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    ]
    assert score_calls == [1, 2, 3]
    assert [gen for gen, _, _ in acquisition.records] == score_calls
    np.testing.assert_allclose(
        [beta for _, _, beta in acquisition.records], SEARCH_PATTERN
    )
    assert [current.fe - previous.fe for previous, current in pairwise(states)] == [
        1,
        1,
        1,
    ]
    assert states[-1].completed_decision_count == states[-1].decision_count == 3


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
    assert state.completed_decision_count == state.decision_count == 3
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
    assert resumed.completed_decision_count == 4


def test_repeated_evaluation_uses_unique_candidate_ids_for_batch_warning():
    events: list[tuple[int, bool]] = []
    state = _make_async_state()
    stage = EvaluationPlanStage(
        RepeatedEvaluation(2),
        semantic_warning=lambda candidate_count, overlap: events.append(
            (candidate_count, overlap)
        ),
    )

    state = stage.execute(state)

    assert state.decision_count == 1
    assert events == []


def test_zero_candidate_top_k_plan_is_rejected_before_decision_count_advances():
    with pytest.raises(ValidationError, match="positive"):
        TopKEvaluation(0)
    with pytest.raises(ValidationError, match="n_candidates"):
        PreSelectionStrategy(n_candidates=0, n_select=1)
    with pytest.raises(ValidationError, match="n_select"):
        PreSelectionStrategy(n_candidates=1, n_select=0)

    state = _make_async_state().replace(scores=np.array([1.0], dtype=np.float64))
    stage = EvaluationPlanStage(TopKEvaluation(1))
    planned = stage.execute(state)

    assert planned.decision_count == 1


def _run_sync_status_plan(state, evaluator):
    for stage in (
        EvaluationPlanStage(EvaluateAll()),
        EvaluationSubmitStage(evaluator),
        EvaluationCollectStage(evaluator),
        EvaluationApplyStage(),
        ArchiveUpdateStage(),
        EvaluationAcknowledgeStage(evaluator),
    ):
        state = stage.execute(state)
    return state


@pytest.mark.parametrize(
    "terminal_status",
    (EvaluationStatus.FAILED, EvaluationStatus.CANCELLED),
)
def test_sync_failed_or_cancelled_plan_does_not_advance_cors_beta(terminal_status):
    acquisition = _RecordingCORSDistance(delta=1.0, search_pattern=SEARCH_PATTERN)
    acquisition_stage = AcquisitionStage(acquisition)
    evaluator = _StatusEvaluator([terminal_status, EvaluationStatus.COMPLETED])
    state = _make_async_state()
    assert state.offspring is not None
    prediction = SurrogatePrediction.objective(
        value=np.array([[1.0]], dtype=np.float64), x=state.offspring.x
    )

    state = state.replace(predictions=prediction)
    state = acquisition_stage.execute(state)
    state = _run_sync_status_plan(state, evaluator)
    assert state.decision_count == 1
    assert state.completed_decision_count == 0

    state = state.replace(predictions=prediction)
    state = acquisition_stage.execute(state)
    state = _run_sync_status_plan(state, evaluator)
    assert state.decision_count == 2
    assert state.completed_decision_count == 1

    state = state.replace(predictions=prediction)
    acquisition_stage.execute(state)
    assert [beta for _, _, beta in acquisition.records] == [0.9, 0.9, 0.4]


@pytest.mark.parametrize(
    "terminal_status",
    (EvaluationStatus.FAILED, EvaluationStatus.CANCELLED),
)
def test_async_failed_or_cancelled_plan_does_not_advance_cors_beta(terminal_status):
    acquisition = _RecordingCORSDistance(delta=1.0, search_pattern=SEARCH_PATTERN)
    acquisition_stage = AcquisitionStage(acquisition)
    evaluator = _StatusEvaluator([terminal_status, EvaluationStatus.COMPLETED])
    scheduler = AsyncEvaluationScheduler(evaluator, max_pending=1)
    submit_stage = AsyncEvaluationSubmitStage(scheduler, EvaluateAll())
    state = _make_async_state()

    for index in range(2):
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

    assert state.decision_count == 2
    assert state.completed_decision_count == 1
    population = _make_candidate(12, 0.4)
    state = state.replace(
        offspring=population,
        predictions=SurrogatePrediction.objective(
            value=np.array([[1.0]], dtype=np.float64), x=population.x
        ),
    )
    acquisition_stage.execute(state)
    assert [beta for _, _, beta in acquisition.records] == [0.9, 0.9, 0.4]


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
        semantic_warning=lambda candidate_count, overlap: events.append(
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
        semantic_warning=lambda candidate_count, overlap: events.append(
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


def test_custom_strategy_and_opaque_planner_warn_at_runtime_without_cors_wiring():
    optimizer = _make_optimizer(_make_problem(), CORSDistance(delta=1.0), n_gen=1)
    optimizer.set_strategy(_CustomPreSelectionStrategy())
    optimizer.set_evaluation_planner(_OpaqueEvaluateAllPlanner())

    plan = optimizer._compile_plan()

    assert plan is not None
    assert not any(
        diagnostic.code == "cors_nonsequential_evaluation"
        for diagnostic in plan.diagnostics
    )
    assert not hasattr(optimizer, "_cors_runtime_warning")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        optimizer.run()

    cors_warnings = [
        item for item in caught if str(item.message) == CORS_NONSEQUENTIAL_MESSAGE
    ]
    assert len(cors_warnings) == 1


def test_cors_runtime_ignores_batch_planner_on_an_independent_branch():
    strategy = _IndependentRuntimeBranchesStrategy()
    optimizer = (
        _make_optimizer(_make_problem(), CORSDistance(delta=1.0), n_gen=1)
        .set_initializer(LHSInitializer(10, 10))
        .set_strategy(strategy)
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        optimizer.run()

    assert strategy.cors_candidate_counts == [1]
    assert strategy.batch_candidate_counts == [10]
    cors_warnings = [
        item for item in caught if str(item.message) == CORS_NONSEQUENTIAL_MESSAGE
    ]
    assert cors_warnings == []


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
    # This direct stage test bypasses graph compilation; model the compiled
    # graph's semantic requirement explicitly before invoking the generic hook.
    optimizer._requires_sequential_decisions_in_plan = True
    stage = EvaluationPlanStage(
        planner=_OpaqueBatchPlanner(),
        semantic_warning=optimizer._semantic_runtime_warning,
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
