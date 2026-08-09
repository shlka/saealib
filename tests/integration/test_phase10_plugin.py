"""U10-0: external plugin foundation boundary."""

from phase10_plugin import Phase10PluginFixture, build_plugin_graph

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
