from typing import ClassVar, cast

import numpy as np
import pytest

import saealib
from saealib.algorithms import (
    GA,
    PSO,
    FeedbackConsumer,
    LegacyPopulationAlgorithmAdapter,
    ProposalRequest,
    Proposer,
)
from saealib.algorithms.base import Algorithm
from saealib.algorithms.pymoo_algorithm import PymooAlgorithm
from saealib.callback import PostAskEvent
from saealib.context import OptimizationState
from saealib.core.contracts import (
    COMPLETE_BATCH,
    FeedbackRequirement,
    ProposalBatch,
    ProposalRelations,
    QuantityRef,
    QuantityRequirement,
)
from saealib.core.state import (
    POPULATIONS_MAIN,
    RUNTIME_RNG,
    LegacyAlgorithmStateView,
    StateView,
)
from saealib.exceptions import ValidationError
from saealib.identity import IDAllocator
from saealib.operators import (
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
)
from saealib.optimizer import Optimizer
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.stages import AskStage

ATTRS = [
    PopulationAttribute("id", np.int64, (), default=-1),
    PopulationAttribute("x", np.float64, (2,)),
    PopulationAttribute("f", np.float64, (1,)),
    PopulationAttribute("g", np.float64, (0,)),
    PopulationAttribute("cv", np.float64, (), default=0.0),
]


def _state() -> OptimizationState:
    problem = Problem(
        func=lambda x: np.array([np.sum(x**2)]),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )
    population = Population(ATTRS, 2)
    population._extend_internal(
        {
            "id": np.array([40, 41], dtype=np.int64),
            "x": np.array([[0.1, 0.2], [0.3, 0.4]]),
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
        candidate_id_allocator=IDAllocator(100),
        request_id_allocator=IDAllocator(200),
    )


class _LegacyAlgorithm(Algorithm):
    """A small legacy-shaped algorithm used to exercise the adapter boundary."""

    ask_notation: ClassVar[list[str]] = [r"$Q \leftarrow P$"]

    def __init__(self):
        self.calls = 0

    def contract(self):
        return PSO().contract()

    def get_required_attrs(self, problem):
        return ATTRS

    @property
    def population_class(self):
        return Population

    @property
    def archive_class(self):
        return Archive

    def ask(self, ctx, provider, n_offspring=None):
        self.calls += 1
        assert provider is not None
        assert ctx.population is ctx.populations["main"]
        return ctx.population.extract(np.array([0, 1]))

    def tell(self, ctx, provider, offspring):
        return None


class _NewProposer:
    def __init__(self):
        self.request = None
        self.view = None

    def ask(self, request: ProposalRequest, state: StateView) -> ProposalBatch:
        self.request = request
        self.view = state
        candidates = cast(Population, state.get(POPULATIONS_MAIN)).extract(
            np.array([0, 1])
        )
        return ProposalBatch(
            proposal_id=77,
            candidates=candidates,
            relations=ProposalRelations({}, row_count=len(candidates)),
            requirements=FeedbackRequirement(quantities=()),
        )


class _DispatchingLegacyAlgorithm(_LegacyAlgorithm):
    def ask(self, ctx, provider, n_offspring=None):
        candidates = super().ask(ctx, provider, n_offspring)
        provider.dispatch(PostAskEvent(ctx=ctx, candidates=candidates))
        return candidates


def test_j7a_protocols_are_split_and_legacy_adapter_is_both_roles():
    adapter = LegacyPopulationAlgorithmAdapter(_LegacyAlgorithm())
    assert isinstance(adapter, Proposer)
    assert isinstance(adapter, FeedbackConsumer)
    assert hasattr(adapter, "tell")


def test_j7a_migration_types_are_namespace_only():
    for name in (
        "FeedbackConsumer",
        "LegacyPopulationAlgorithmAdapter",
        "ProposalRequest",
        "Proposer",
    ):
        assert not hasattr(saealib, name)


def test_j7a_legacy_adapter_calls_ask_once_and_preserves_ids():
    state = _state()
    legacy = _LegacyAlgorithm()
    result = AskStage(legacy, n_offspring=2).execute(state)

    assert legacy.calls == 1
    assert result.offspring is not None
    np.testing.assert_array_equal(result.offspring.get_array("id"), [40, 41])


def test_j7a_legacy_adapter_reuses_derived_values_when_stages_are_rebuilt():
    legacy = _LegacyAlgorithm()

    first = AskStage(legacy)
    second = AskStage(legacy)

    first_adapter = cast(LegacyPopulationAlgorithmAdapter, first._algorithm)
    second_adapter = cast(LegacyPopulationAlgorithmAdapter, second._algorithm)
    assert first_adapter is not second_adapter
    assert first_adapter.requirements is second_adapter.requirements


def test_j7a_shared_algorithm_keeps_optimizer_callbacks_isolated():
    shared = _DispatchingLegacyAlgorithm()
    state1 = _state()
    state2 = _state()
    optimizer1 = Optimizer(state1.problem).set_algorithm(shared)
    optimizer2 = Optimizer(state2.problem).set_algorithm(shared)
    received1: list[PostAskEvent] = []
    received2: list[PostAskEvent] = []
    optimizer1.cbmanager.register(PostAskEvent, received1.append)
    optimizer2.cbmanager.register(PostAskEvent, received2.append)

    stage1 = AskStage(optimizer1.algorithm, cbmanager=optimizer1.cbmanager)
    stage2 = AskStage(optimizer2.algorithm, cbmanager=optimizer2.cbmanager)
    state1 = stage1.execute(state1)
    state2 = stage2.execute(state2)
    state1 = stage1.execute(state1)
    state2 = stage2.execute(state2)

    assert len(received1) == 2
    assert len(received2) == 2


def test_j7a_proposal_id_and_relations_match_the_legacy_candidates():
    state = _state()
    legacy = _LegacyAlgorithm()
    adapter = LegacyPopulationAlgorithmAdapter(legacy)
    state_view = LegacyAlgorithmStateView(
        state._store, (POPULATIONS_MAIN, RUNTIME_RNG), state
    )

    proposal = adapter.ask(ProposalRequest(n_offspring=2), state_view)

    assert proposal.proposal_id == 200
    assert state.request_id_allocator.next_value == 201
    np.testing.assert_array_equal(proposal.candidates.get_array("id"), [40, 41])
    assert proposal.relations.row_count == len(proposal.candidates)


def test_j7a_new_proposer_receives_only_request_and_state_view():
    state = _state()
    proposer = _NewProposer()
    result = AskStage(proposer, n_offspring=1).execute(state)

    assert proposer.request == ProposalRequest(n_offspring=1)
    assert isinstance(proposer.view, StateView)
    assert result.offspring is not None
    np.testing.assert_array_equal(result.offspring.get_array("id"), [40, 41])


def test_j7a_legacy_requirements_must_be_contained():
    too_wide = FeedbackRequirement(
        quantities=(
            QuantityRequirement(
                quantity=QuantityRef.from_value(("objective", 0)),
                sources=frozenset({"surrogate"}),
            ),
        ),
    )
    with pytest.raises(ValidationError):
        LegacyPopulationAlgorithmAdapter(_LegacyAlgorithm(), requirements=too_wide)

    adapter = LegacyPopulationAlgorithmAdapter(_LegacyAlgorithm())
    assert adapter.requirements.quantities == ()
    assert adapter.requirements.completion == COMPLETE_BATCH


@pytest.mark.parametrize(
    "algorithm",
    [
        GA(
            crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
            mutation=MutationUniform(prob_var=0.1),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        ),
        PSO(),
    ],
)
def test_j7a_builtin_legacy_algorithms_are_auto_adapted(algorithm):
    assert isinstance(AskStage(algorithm)._algorithm, LegacyPopulationAlgorithmAdapter)


def test_j7a_pymoo_algorithm_uses_the_same_legacy_path():
    try:
        from pymoo.algorithms.soo.nonconvex.ga import GA as PymooGA  # noqa: N811
    except ImportError:
        pytest.skip("pymoo is not installed")
    algorithm = PymooAlgorithm(PymooGA(pop_size=2))
    assert isinstance(AskStage(algorithm)._algorithm, LegacyPopulationAlgorithmAdapter)
