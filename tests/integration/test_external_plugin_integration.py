"""External plugin foundation and boundary tests."""

from dataclasses import replace
from typing import Any, cast

import numpy as np
from external_plugin_fixture import (
    CMAES_CANDIDATE_IDS,
    COMPLETE_BATCH,
    DE_TARGET_IDS,
    DE_TRIAL_IDS,
    MAPELITES_CANDIDATE_IDS,
    MF_CANDIDATE_IDS,
    MF_HIGH_FIDELITY,
    MOEAD_CANDIDATE_IDS,
    PARTIAL_ALLOWED,
    CMAESPlugin,
    CooperativeCoevolutionPlugin,
    DifferentialEvolutionPlugin,
    ExternalPluginFixture,
    MAPElitesPlugin,
    MOEADPlugin,
    MultiFidelityPlugin,
    build_plugin_graph,
)

from saealib.core.compiler import Compiler
from saealib.core.compiler.diagnostics import Severity
from saealib.core.contracts import ObservationBatch
from saealib.population import Population


def _population_candidates(proposal: Any) -> Population[Any]:
    return cast(Population[Any], proposal.candidates)


def test_external_plugin_proposes_evaluates_and_delivers_feedback() -> None:
    fixture = ExternalPluginFixture()
    proposal, candidate_ids = fixture.propose()
    observations = fixture.evaluate(proposal, candidate_ids)
    feedback = fixture.feedback(proposal, observations)

    assert len(proposal.candidates) == len(candidate_ids) == 3
    assert observations.candidate_ids.tolist() == candidate_ids.tolist()
    assert feedback.observations is observations
    assert fixture.feedback_state[proposal.proposal_id] is feedback


def test_external_plugin_component_compiles_without_error_diagnostics() -> None:
    plan = Compiler().compile(build_plugin_graph())

    assert not [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.severity is Severity.ERROR
    ]
    assert plan.graph.node_by_id("external_plugin").component is not None


def test_differential_evolution_preserves_targets_and_replaces_improvements() -> None:
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

    assert np.array_equal(
        plugin.target_state[8201][0], _population_candidates(proposal).x[0]
    )
    assert np.array_equal(plugin.target_state[8202][0], before[8202])
    assert np.array_equal(plugin.target_state[8203][0], before[8203])


def test_multifidelity_accepts_sparse_low_and_selects_high_promotion() -> None:
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
    assert dict(record.provenance) == {"plugin": "external_plugin", "stage": "high"}
    assert plugin.feedback_state["high"] is high_feedback


def test_cmaes_updates_only_on_complete_generation_feedback() -> None:
    plugin = CMAESPlugin()
    proposal = plugin.propose()
    feedback = plugin.evaluate(proposal)
    before = plugin.snapshot()
    restored = CMAESPlugin()
    restored.restore(before)
    restored_snapshot = restored.snapshot()
    assert np.array_equal(restored_snapshot["mean"], before["mean"])
    assert np.array_equal(restored_snapshot["covariance"], before["covariance"])
    assert restored_snapshot["step_size"] == before["step_size"]
    assert restored_snapshot["generation"] == before["generation"]

    partial_observations = ObservationBatch(
        schema=feedback.observations.schema,
        records=feedback.observations.records.take(np.array([0, 1], dtype=np.intp)),
    )
    partial_feedback = replace(feedback, observations=partial_observations, final=False)
    assert len(partial_feedback.observations.records) == 2
    assert len(feedback.observations.records) == 4
    assert partial_feedback.observations.candidate_ids.tolist() == [8601, 8602]
    plugin.apply_feedback(proposal, partial_feedback)
    unchanged = plugin.snapshot()
    assert np.array_equal(unchanged["mean"], before["mean"])
    assert np.array_equal(unchanged["covariance"], before["covariance"])
    assert unchanged["step_size"] == before["step_size"]
    assert unchanged["generation"] == before["generation"]

    plugin.apply_feedback(proposal, feedback)
    assert plugin.generation == before["generation"] + 1
    assert plugin.step_size != before["step_size"]
    assert not np.array_equal(plugin.covariance, before["covariance"])
    assert feedback.observations.candidate_ids.tolist() == CMAES_CANDIDATE_IDS.tolist()


def test_moead_updates_only_the_child_subproblem_neighborhood() -> None:
    plugin = MOEADPlugin()
    proposal = plugin.propose()
    assert np.asarray(proposal.relations["subproblem_ids"]).tolist() == [0, 1, 2]
    assert np.array_equal(plugin.weights[0], [1.0, 0.0])
    assert plugin.neighbors[0] == (0, 1)
    before = {
        subproblem: state[0].copy()
        for subproblem, state in plugin.subproblem_state.items()
    }

    feedback = plugin.evaluate(proposal)
    assert feedback.observations.candidate_ids.tolist() == MOEAD_CANDIDATE_IDS.tolist()
    plugin.apply_feedback(proposal, feedback)

    assert np.array_equal(
        plugin.subproblem_state[0][0], _population_candidates(proposal).x[0]
    )
    assert np.array_equal(
        plugin.subproblem_state[1][0], _population_candidates(proposal).x[0]
    )
    assert np.array_equal(plugin.subproblem_state[2][0], before[2])


def test_map_elites_uses_behavior_cells_and_quality_replacement() -> None:
    plugin = MAPElitesPlugin()
    proposal = plugin.propose()
    assert plugin.emitter_calls == 1
    feedback = plugin.evaluate(proposal)
    assert (
        feedback.observations.candidate_ids.tolist() == MAPELITES_CANDIDATE_IDS.tolist()
    )
    assert len(feedback.observations.records) == 6
    plugin.apply_feedback(feedback)

    assert set(plugin.archive) == {(0,), (1,)}
    assert np.isclose(plugin.archive[(0,)][1], 0.02)
    assert np.array_equal(
        plugin.archive[(0,)][0], _population_candidates(proposal).x[1]
    )
    assert np.isclose(plugin.archive[(1,)][1], 1.28)


def test_map_elites_archive_is_independent_of_record_order() -> None:
    plugin = MAPElitesPlugin()
    proposal = plugin.propose()
    feedback = plugin.evaluate(proposal)
    shuffled = replace(
        feedback,
        observations=replace(
            feedback.observations,
            records=feedback.observations.records.take(
                np.array([1, 0, 3, 2, 5, 4], dtype=np.intp)
            ),
        ),
    )

    plugin.apply_feedback(shuffled)

    assert set(plugin.archive) == {(0,), (1,)}
    assert np.isclose(plugin.archive[(0,)][1], 0.02)
    assert np.array_equal(
        plugin.archive[(0,)][0], _population_candidates(proposal).x[1]
    )
    assert np.isclose(plugin.archive[(1,)][1], 1.28)


def test_coevolution_joins_named_blocks_and_updates_one_block() -> None:
    plugin = CooperativeCoevolutionPlugin()
    plugin.named_populations["block_a"].update_rows(
        [1], {"x": np.array([[0.3]], dtype=np.float64)}
    )
    proposal = plugin.coordinator_join()
    assert set(plugin.named_populations) == {"block_a", "block_b"}
    assert _population_candidates(proposal).x.shape == (1, 2)
    assert np.array_equal(_population_candidates(proposal).x[0], [0.3, 0.2])
    feedback = plugin.evaluate(proposal)
    assert feedback.observations.f.shape == (1, 1)
    assert np.isclose(feedback.observations.f[0, 0], 0.13)

    plugin.apply_feedback(feedback)
    assert np.array_equal(plugin.named_populations["block_a"].x[0], [0.3])
    assert np.array_equal(plugin.named_populations["block_b"].x[0], [0.2])
    next_proposal = plugin.coordinator_join()
    assert np.array_equal(_population_candidates(next_proposal).x[0], [0.3, 0.2])
