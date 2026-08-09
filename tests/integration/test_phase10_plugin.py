"""Phase 10 external plugin foundation and U10-1 boundaries."""

import numpy as np
from phase10_plugin import (
    COMPLETE_BATCH,
    DE_TARGET_IDS,
    DE_TRIAL_IDS,
    MF_CANDIDATE_IDS,
    MF_HIGH_FIDELITY,
    PARTIAL_ALLOWED,
    DifferentialEvolutionPlugin,
    MultiFidelityPlugin,
    Phase10PluginFixture,
    build_plugin_graph,
)

from saealib.core.compiler import Compiler
from saealib.core.compiler.diagnostics import Severity


def test_u10_0_external_plugin_proposes_evaluates_and_delivers_feedback() -> None:
    fixture = Phase10PluginFixture()
    proposal, candidate_ids = fixture.propose()
    observations = fixture.evaluate(proposal, candidate_ids)
    feedback = fixture.feedback(proposal, observations)

    assert len(proposal.candidates) == len(candidate_ids) == 3
    assert observations.candidate_ids.tolist() == candidate_ids.tolist()
    assert feedback.observations is observations
    assert fixture.feedback_state[proposal.proposal_id] is feedback


def test_u10_0_plugin_component_compiles_without_error_diagnostics() -> None:
    plan = Compiler().compile(build_plugin_graph())

    assert not [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.severity is Severity.ERROR
    ]
    assert plan.graph.node_by_id("phase10_plugin").component is not None


def test_u10_1_differential_evolution_preserves_targets_and_replaces_improvements() -> (
    None
):
    plugin = DifferentialEvolutionPlugin()
    proposal = plugin.propose()
    target_ids = np.asarray(proposal.relations["target_ids"], dtype=np.int64)
    assert target_ids.tolist() == DE_TARGET_IDS.tolist()
    feedback = plugin.evaluate(proposal)
    assert feedback.observations.candidate_ids.tolist() == DE_TRIAL_IDS.tolist()
    before = {
        candidate_id: value[0].copy()
        for candidate_id, value in plugin.target_state.items()
    }
    plugin.apply_feedback(proposal, feedback)

    assert np.array_equal(plugin.target_state[8201][0], proposal.candidates.x[0])
    assert np.array_equal(plugin.target_state[8202][0], before[8202])
    assert np.array_equal(plugin.target_state[8203][0], before[8203])


def test_u10_1_multifidelity_accepts_sparse_low_and_selects_high_promotion() -> None:
    plugin = MultiFidelityPlugin()
    low_proposal = plugin.low_proposal()
    low_feedback = plugin.low_feedback(low_proposal)
    assert len(low_proposal.candidates) == len(MF_CANDIDATE_IDS) == 3
    assert low_proposal.requirements.quantities[0].fidelity == 1
    assert low_proposal.requirements.completion == PARTIAL_ALLOWED
    assert len(low_feedback.observations.candidate_ids) == 2
    assert low_feedback.observations.candidate_ids.tolist() == [0, 1]
    assert len(low_feedback.observations.candidate_ids) < len(MF_CANDIDATE_IDS)
    assert set(low_feedback.observations.records.column("fidelity")) == {1}

    high_proposal, promoted_id = plugin.promote(low_feedback)
    assert high_proposal.requirements.quantities[0].fidelity == MF_HIGH_FIDELITY
    assert high_proposal.requirements.completion == COMPLETE_BATCH
    high_feedback = plugin.high_feedback(high_proposal, promoted_id)
    record = high_feedback.observations.records[0]
    assert promoted_id == 0
    assert record.fidelity == MF_HIGH_FIDELITY
    assert record.cost == 1.0
    assert dict(record.provenance) == {"plugin": "phase10", "stage": "high"}
    assert plugin.feedback_state["high"] is high_feedback
