"""Small external-plugin-shaped fixture used by Phase 10 integration tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from saealib.core.compiler import (
    ComponentBindings,
    ComponentGraph,
    ComponentNode,
    GraphTemplate,
    NodeRef,
)
from saealib.core.contracts import (
    COMPLETE_BATCH,
    MANY,
    OBJECTIVE,
    TRUE,
    ComponentContract,
    DataSpec,
    FeedbackBatch,
    FeedbackRequirement,
    ObservationBatch,
    ObservationSchema,
    PortContract,
    PortDirection,
    PortSpec,
    ProposalBatch,
    ProposalRelations,
    QuantityRef,
    QuantityRequirement,
)
from saealib.execution import SerialEvaluator
from saealib.execution.evaluator import (
    EvaluationAdapter,
    EvaluationQuery,
    EvaluationRequest,
)
from saealib.population import Population, PopulationAttribute
from saealib.problem import Problem

PLUGIN_SEED = 1042
PLUGIN_PROPOSAL_ID = 7001
PLUGIN_CANDIDATE_IDS = np.array([8101, 8102, 8103], dtype=np.int64)


class PluginEvaluationAdapter(EvaluationAdapter):
    """Convert the plugin's population genomes into the problem payload."""

    def transform(self, genomes: Any, request: EvaluationQuery) -> np.ndarray:
        values = np.asarray(genomes.array, dtype=np.float64)
        assert np.array_equal(request.candidate_ids, PLUGIN_CANDIDATE_IDS)
        return values


class PluginCandidateComponent:
    """A contract-only plugin component; execution stays in the fixture seam."""

    def contract(self) -> ComponentContract:
        return ComponentContract(
            ports={
                "proposal": PortContract(
                    outputs=(
                        PortSpec(
                            name="proposals",
                            direction=PortDirection.OUTPUT,
                            data=DataSpec(kind="ProposalBatch"),
                            cardinality=MANY,
                        ),
                    )
                )
            }
        )


class PluginGraphTemplate(GraphTemplate):
    """Register one external component through the public graph seam."""

    def build_graph(self, bindings: ComponentBindings) -> ComponentGraph:
        return ComponentGraph(
            nodes=(
                ComponentNode(
                    component_id="phase10_plugin",
                    component=bindings.components["phase10_plugin"],
                ),
            ),
            entry_points=(NodeRef(component_id="phase10_plugin"),),
        )


@dataclass
class Phase10PluginFixture:
    """Deterministic propose/evaluate/feedback flow for later Phase 10 units."""

    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(PLUGIN_SEED)
    )
    feedback_state: dict[int, FeedbackBatch] = field(default_factory=dict)

    def problem(self) -> Problem:
        return Problem(
            func=lambda x: np.asarray([np.sum(np.asarray(x, dtype=float) ** 2)]),
            dim=2,
            n_obj=1,
            direction=np.array([-1.0]),
            lb=[-1.0, -1.0],
            ub=[1.0, 1.0],
            evaluation_adapter=PluginEvaluationAdapter(),
        )

    def evaluator(self) -> SerialEvaluator:
        return SerialEvaluator()

    def propose_population(self) -> Population:
        genomes = self.rng.uniform(-1.0, 1.0, size=(len(PLUGIN_CANDIDATE_IDS), 2))
        population = Population(
            attrs=[PopulationAttribute(name="x", dtype=np.float64, shape=(2,))],
            init_capacity=len(genomes),
        )
        for genome in genomes:
            population.append(x=genome)
        return population

    def propose(self) -> tuple[ProposalBatch, np.ndarray]:
        candidates = self.propose_population()
        requirements = FeedbackRequirement(
            quantities=(
                QuantityRequirement(quantity=QuantityRef(kind=OBJECTIVE, index=0)),
            ),
            completion=COMPLETE_BATCH,
        )
        proposal = ProposalBatch(
            proposal_id=PLUGIN_PROPOSAL_ID,
            candidates=candidates,
            relations=ProposalRelations(row_count=len(candidates)),
            requirements=requirements,
        )
        return proposal, PLUGIN_CANDIDATE_IDS.copy()

    def observations_from_result(
        self, candidate_ids: np.ndarray, result: Any
    ) -> ObservationBatch:
        return ObservationBatch.from_dense(
            ObservationSchema(objective_count=1, constraint_count=0),
            candidate_ids,
            result.f,
            result.g,
            result.cv,
            source=TRUE,
        )

    def evaluate(
        self, proposal: ProposalBatch, candidate_ids: np.ndarray
    ) -> ObservationBatch:
        request = EvaluationRequest(
            request_id=np.int64(1),
            candidate_ids=candidate_ids,
            payload=proposal.candidates.genomes,
        )
        result = self.evaluator().evaluate_request(request, self.problem())
        return self.observations_from_result(candidate_ids, result)

    def feedback(
        self, proposal: ProposalBatch, observations: ObservationBatch
    ) -> FeedbackBatch:
        batch = FeedbackBatch(
            proposal_id=proposal.proposal_id,
            observations=observations,
            channel=TRUE,
            final=True,
            sequence=0,
        )
        self.feedback_state[proposal.proposal_id] = batch
        return batch


def build_plugin_graph() -> ComponentGraph:
    """Build the graph used by the focused compiler assertion."""
    return PluginGraphTemplate().build_graph(
        ComponentBindings(components={"phase10_plugin": PluginCandidateComponent()})
    )
