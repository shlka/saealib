from typing import ClassVar, cast

import numpy as np

from saealib.algorithms import LegacyPopulationAlgorithmAdapter, ProposalRequest
from saealib.algorithms.base import Algorithm
from saealib.context import OptimizationState
from saealib.core.contracts import (
    FeedbackBatch,
    FeedbackRequirement,
    ObservationBatch,
    ObservationSchema,
    ProposalBatch,
    ProposalRelations,
)
from saealib.core.state import (
    POPULATIONS_MAIN,
    PROPOSALS_CURRENT,
    RUNTIME_RNG,
    LegacyAlgorithmStateView,
    StateKey,
    StatePatch,
    StateView,
)
from saealib.execution import (
    AsyncEvaluationScheduler,
    EvaluationHandle,
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
    EvaluationUpdate,
    Evaluator,
)
from saealib.identity import IDAllocator
from saealib.policies.feedback import TrueOnlyFeedback
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.stages import AskStage, TellStage

ATTRS = [
    PopulationAttribute("id", np.int64, (), default=-1),
    PopulationAttribute("x", np.float64, (1,)),
    PopulationAttribute("f", np.float64, (1,)),
    PopulationAttribute("g", np.float64, (0,)),
    PopulationAttribute("cv", np.float64, (), default=0.0),
]


def _state(*, proposal_id: int | None = None) -> OptimizationState:
    problem = Problem(
        func=lambda x: np.array([x[0]]),
        dim=1,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0],
        ub=[1.0],
    )
    population = Population(ATTRS, 2)
    population._extend_internal(
        {
            "id": np.array([10, 11], dtype=np.int64),
            "x": np.array([[0.1], [0.2]], dtype=np.float64),
            "f": np.full((2, 1), np.nan),
            "g": np.empty((2, 0)),
            "cv": np.zeros(2),
        },
        preserve_ids=True,
    )
    custom: dict[StateKey[object], object] = {}
    if proposal_id is not None:
        custom[PROPOSALS_CURRENT] = proposal_id
    return OptimizationState(
        problem=problem,
        population=population,
        archive=Archive(ATTRS, 2),
        pareto_archive=ParetoArchive(ATTRS, 2, direction=np.array([-1.0])),
        rng=np.random.default_rng(0),
        candidate_id_allocator=IDAllocator(100),
        request_id_allocator=IDAllocator(200),
        offspring=population,
        _custom_state=custom,
    )


def _observations(value: float = 1.0) -> ObservationBatch:
    return ObservationBatch.from_dense(
        ObservationSchema(objective_count=1),
        np.array([10], dtype=np.int64),
        np.array([[value]], dtype=np.float64),
        np.empty((1, 0), dtype=np.float64),
    )


class _Legacy(Algorithm):
    ask_notation: ClassVar[list[str]] = []

    def __init__(self):
        self.told: list[Population] = []

    def contract(self):
        return super().contract()

    def get_required_attrs(self, problem):
        return ATTRS

    @property
    def population_class(self):
        return Population

    @property
    def archive_class(self):
        return Archive

    def ask(self, ctx, provider, n_offspring=None):
        return ctx.population.extract(np.array([0, 1]))

    def tell(self, ctx, provider, offspring):
        self.told.append(offspring)
        offspring.update_rows(np.array([0]), {"f": np.array([[99.0]])})


def test_j7b_sync_tell_passes_legacy_equivalent_population():
    state = _state(proposal_id=1)
    state = state.replace(
        feedback_result=TrueOnlyFeedback().build(
            state.offspring,
            None,
            EvaluationResult(
                np.array([[1.0]]),
                np.empty((1, 0)),
                np.zeros(1),
                candidate_ids=np.array([10], dtype=np.int64),
            ),
            [10],
            state,
        )
    )
    legacy = _Legacy()

    TellStage(legacy).execute(state)

    assert len(legacy.told) == 1
    np.testing.assert_array_equal(legacy.told[0].get_array("id"), [10])
    np.testing.assert_array_equal(legacy.told[0].get_array("x"), [[0.1]])
    np.testing.assert_array_equal(legacy.told[0].get_array("f"), [[99.0]])


class _RecordingConsumer:
    def __init__(self):
        self.feedback: list[FeedbackBatch] = []

    def tell(self, feedback: FeedbackBatch, state: StateView) -> StatePatch:
        self.feedback.append(feedback)
        return StatePatch(writes={})


class _PartialEvaluator(Evaluator):
    def evaluate_batch(self, x, problem):
        raise AssertionError("scheduler must use submit/collect")

    def submit(self, request, problem):
        return EvaluationHandle(
            request.request_id,
            EvaluationStatus.PENDING,
            backend_token=(request, problem),
        )

    def collect(self, handle, *, wait=True):
        request, _problem = handle.backend_token
        if handle._delivered_sequence < 0:
            result = EvaluationResult(
                np.array([[1.0]]),
                np.empty((1, 0)),
                np.zeros(1),
                request.candidate_ids[:1],
            )
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
        if handle._delivered_sequence == 0:
            result = EvaluationResult(
                np.array([[2.0]]),
                np.empty((1, 0)),
                np.zeros(1),
                request.candidate_ids[1:],
            )
            handle._delivered_sequence = 1
            return [
                EvaluationUpdate(
                    request.request_id,
                    EvaluationStatus.COMPLETED,
                    request.candidate_ids[1:],
                    result,
                    sequence=1,
                )
            ]
        return []


def test_j7b_async_tells_each_partial_and_marks_only_final_terminal():
    state = _state(proposal_id=55)
    consumer = _RecordingConsumer()
    offspring = state.offspring
    assert offspring is not None
    request = EvaluationRequest(
        np.int64(0), np.array([10, 11], dtype=np.int64), offspring.x
    )
    scheduler = AsyncEvaluationScheduler(
        _PartialEvaluator(),
        algorithm=consumer,
        feedback_builder=TrueOnlyFeedback(),
    )

    scheduler.poll(scheduler.submit(state, [request]), wait=True)

    assert [batch.final for batch in consumer.feedback] == [False, True]
    assert [batch.sequence for batch in consumer.feedback] == [0, 1]


class _Proposer:
    def ask(self, request: ProposalRequest, state: StateView) -> ProposalBatch:
        population = cast(Population, state.get(POPULATIONS_MAIN))
        candidates = population.extract(np.array([0]))
        return ProposalBatch(
            proposal_id=777,
            candidates=candidates,
            relations=ProposalRelations({}, row_count=1),
            requirements=FeedbackRequirement(quantities=()),
        )


def test_j7b_ask_stores_proposal_id_used_by_feedback_batch():
    state = AskStage(_Proposer()).execute(_state())
    state = state.replace(
        feedback_result=TrueOnlyFeedback().build(
            state.offspring,
            None,
            EvaluationResult(
                np.array([[1.0]]),
                np.empty((1, 0)),
                np.zeros(1),
                candidate_ids=np.array([10], dtype=np.int64),
            ),
            [10],
            state,
        )
    )
    consumer = _RecordingConsumer()
    TellStage(consumer).execute(state)

    assert state._store.get(PROPOSALS_CURRENT) == 777
    assert [batch.proposal_id for batch in consumer.feedback] == [777]
    assert getattr(consumer.feedback[0].observations, "_dense_inputs", None) is not None


def test_j7b_v3_proposal_key_roundtrips_and_old_checkpoint_loads(tmp_path):
    state = _state(proposal_id=321)
    current = tmp_path / "current.npz"
    state.save(current)
    restored = OptimizationState.load(current, state.problem)
    assert restored._store.get(PROPOSALS_CURRENT) == 321

    old = tmp_path / "old.npz"
    state._save_v2(old)
    legacy = OptimizationState.load(old, state.problem)
    assert not legacy._store.contains(PROPOSALS_CURRENT)


def test_j7b_legacy_adapter_tell_mutates_population_and_returns_empty_patch():
    state = _state()
    legacy = _Legacy()
    adapter = LegacyPopulationAlgorithmAdapter(legacy)
    view = LegacyAlgorithmStateView(
        state._store, (POPULATIONS_MAIN, RUNTIME_RNG), state
    )
    batch = FeedbackBatch(
        proposal_id=1,
        observations=_observations(),
        channel="true",
        final=True,
        sequence=0,
    )

    patch = adapter.tell(batch, view)

    assert patch == StatePatch(writes={})
    assert len(legacy.told) == 1
    np.testing.assert_array_equal(legacy.told[0].get_array("f"), [[99.0]])
