"""Small external-plugin-shaped fixture used by Phase 10 integration tests."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from saealib.core.adapters import FeedbackAccumulator
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
    PARTIAL_ALLOWED,
    TRUE,
    ComponentContract,
    DataSpec,
    FeedbackBatch,
    FeedbackContract,
    FeedbackRequirement,
    ObservationBatch,
    ObservationRecord,
    ObservationRecords,
    ObservationSchema,
    PortContract,
    PortDirection,
    PortSpec,
    ProposalBatch,
    ProposalRelations,
    QuantityRef,
    QuantityRequirement,
)
from saealib.core.contracts.observation import OK
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
DE_TARGET_IDS = np.array([8201, 8202, 8203], dtype=np.int64)
DE_TRIAL_IDS = np.array([8301, 8302, 8303], dtype=np.int64)
MF_CANDIDATE_IDS = np.array([0, 1, 2], dtype=np.int64)
MF_LOW_FIDELITY = 1
MF_HIGH_FIDELITY = 2


class PluginEvaluationAdapter(EvaluationAdapter):
    """Convert the plugin's population genomes into the problem payload."""

    def transform(self, genomes: Any, request: EvaluationQuery) -> np.ndarray:
        values = np.asarray(genomes.array, dtype=np.float64)
        assert len(request.candidate_ids) == len(np.unique(request.candidate_ids))
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


@dataclass
class DifferentialEvolutionPlugin:
    """Minimal DE plugin with explicit target/trial relation handling."""

    target_state: dict[int, tuple[np.ndarray, float]] = field(default_factory=dict)

    def problem(self) -> Problem:
        return Phase10PluginFixture().problem()

    def evaluator(self) -> SerialEvaluator:
        return SerialEvaluator()

    def target_genomes(self) -> np.ndarray:
        return np.array([[0.9, 0.9], [0.1, 0.1], [0.2, 0.2]], dtype=np.float64)

    def target_population(self) -> Population:
        targets = self.target_genomes()
        population = Population(
            attrs=[PopulationAttribute(name="x", dtype=np.float64, shape=(2,))],
            init_capacity=len(targets),
        )
        for genome in targets:
            population.append(x=genome)
        return population

    def make_trial_genomes(self) -> np.ndarray:
        targets = self.target_genomes()
        donors_a = targets[[1, 0, 1]]
        donors_b = targets[[2, 2, 0]]
        return targets + 0.5 * (donors_a - donors_b)

    def propose(self) -> ProposalBatch:
        targets = self.target_genomes()
        self.target_state = {
            int(candidate_id): (
                targets[index].copy(),
                float(np.sum(targets[index] ** 2)),
            )
            for index, candidate_id in enumerate(DE_TARGET_IDS)
        }
        trials = self.make_trial_genomes()
        candidates = Population(
            attrs=[PopulationAttribute(name="x", dtype=np.float64, shape=(2,))],
            init_capacity=len(trials),
        )
        for genome in trials:
            candidates.append(x=genome)
        return ProposalBatch(
            proposal_id=PLUGIN_PROPOSAL_ID + 1,
            candidates=candidates,
            relations=ProposalRelations({"target_ids": DE_TARGET_IDS.copy()}),
            requirements=objective_requirement(),
        )

    def evaluate(self, proposal: ProposalBatch) -> FeedbackBatch:
        request = EvaluationRequest(
            request_id=np.int64(2),
            candidate_ids=DE_TRIAL_IDS,
            payload=proposal.candidates.genomes,
        )
        result = self.evaluator().evaluate_request(request, self.problem())
        observations = Phase10PluginFixture().observations_from_result(
            DE_TRIAL_IDS, result
        )
        return FeedbackBatch(
            proposal_id=proposal.proposal_id,
            observations=observations,
            channel=TRUE,
            final=True,
            sequence=0,
        )

    def apply_feedback(self, proposal: ProposalBatch, feedback: FeedbackBatch) -> None:
        target_ids = np.asarray(proposal.relations["target_ids"], dtype=np.int64)
        trial_ids = feedback.observations.candidate_ids
        trial_scores = np.asarray(feedback.observations.f[:, 0], dtype=float)
        trial_by_id = {
            int(candidate_id): float(score)
            for candidate_id, score in zip(trial_ids, trial_scores)
        }
        trials = proposal.candidates.x
        for row, target_id in enumerate(target_ids):
            target_key = int(target_id)
            score = trial_by_id[int(DE_TRIAL_IDS[row])]
            _, current_score = self.target_state[target_key]
            if score < current_score:
                self.target_state[target_key] = (trials[row].copy(), score)


def objective_requirement(
    *, fidelity: int | None = None, completion: str = COMPLETE_BATCH
) -> FeedbackRequirement:
    return FeedbackRequirement(
        quantities=(
            QuantityRequirement(
                quantity=QuantityRef(kind=OBJECTIVE, index=0),
                fidelity=fidelity,
            ),
        ),
        completion=completion,
    )


@dataclass
class MultiFidelityPlugin:
    """Sparse low-fidelity feedback followed by one high-fidelity promotion."""

    feedback_state: dict[str, FeedbackBatch] = field(default_factory=dict)

    def population(self, count: int) -> Population:
        population = Population(
            attrs=[PopulationAttribute(name="x", dtype=np.float64, shape=(2,))],
            init_capacity=count,
        )
        for index in range(count):
            population.append(x=np.array([0.2 * (index + 1), 0.0]))
        return population

    def low_proposal(self) -> ProposalBatch:
        return ProposalBatch(
            proposal_id=PLUGIN_PROPOSAL_ID + 2,
            candidates=self.population(len(MF_CANDIDATE_IDS)),
            relations=ProposalRelations(row_count=len(MF_CANDIDATE_IDS)),
            requirements=objective_requirement(
                fidelity=MF_LOW_FIDELITY, completion=PARTIAL_ALLOWED
            ),
        )

    def observation_record(
        self,
        candidate_id: int,
        value: float,
        *,
        fidelity: int,
        cost: float,
        stage: str,
    ) -> ObservationRecord:
        return ObservationRecord(
            subject=("candidate", np.array([candidate_id], dtype=np.int64)),
            quantity=(OBJECTIVE, 0),
            value=value,
            status=OK,
            source=TRUE,
            fidelity=fidelity,
            cost=cost,
            provenance={"plugin": "phase10", "stage": stage},
        )

    def feedback_from_records(
        self,
        proposal_id: int,
        records: tuple[ObservationRecord, ...],
        *,
        sequence: int,
        final: bool,
    ) -> FeedbackBatch:
        return FeedbackBatch(
            proposal_id=proposal_id,
            observations=ObservationBatch(
                schema=ObservationSchema(objective_count=1),
                records=ObservationRecords.from_records(records),
            ),
            channel=TRUE,
            final=final,
            sequence=sequence,
        )

    def low_feedback(self, proposal: ProposalBatch) -> FeedbackBatch:
        records = (
            self.observation_record(
                0, 0.40, fidelity=MF_LOW_FIDELITY, cost=0.1, stage="low"
            ),
            self.observation_record(
                1, 0.90, fidelity=MF_LOW_FIDELITY, cost=0.1, stage="low"
            ),
        )
        batch = self.feedback_from_records(
            proposal.proposal_id, records, sequence=0, final=True
        )
        accumulator = FeedbackAccumulator(
            FeedbackContract(accepted_channels=frozenset({TRUE}))
        )
        observed_proposal = replace(
            proposal.take([0, 1]),
            requirements=objective_requirement(fidelity=MF_LOW_FIDELITY),
        )
        accumulator.register(observed_proposal)
        accumulator.add(batch)
        completed = accumulator.pop_ready()
        assert completed is not None
        self.feedback_state["low"] = completed
        return completed

    def promote(self, low_feedback: FeedbackBatch) -> tuple[ProposalBatch, int]:
        records = low_feedback.observations.records
        candidate_ids = low_feedback.observations.candidate_ids
        values = np.asarray(records.column("value"), dtype=float)
        promoted_id = int(candidate_ids[int(np.argmin(values))])
        return (
            ProposalBatch(
                proposal_id=PLUGIN_PROPOSAL_ID + 3,
                candidates=self.population(1),
                relations=ProposalRelations(row_count=1),
                requirements=objective_requirement(fidelity=MF_HIGH_FIDELITY),
            ),
            promoted_id,
        )

    def high_feedback(self, proposal: ProposalBatch, promoted_id: int) -> FeedbackBatch:
        batch = self.feedback_from_records(
            proposal.proposal_id,
            (
                self.observation_record(
                    promoted_id,
                    0.10,
                    fidelity=MF_HIGH_FIDELITY,
                    cost=1.0,
                    stage="high",
                ),
            ),
            sequence=0,
            final=True,
        )
        accumulator = FeedbackAccumulator(
            FeedbackContract(accepted_channels=frozenset({TRUE}))
        )
        accumulator.register(proposal)
        accumulator.add(batch)
        completed = accumulator.pop_ready()
        assert completed is not None
        self.feedback_state["high"] = completed
        return completed


def build_plugin_graph() -> ComponentGraph:
    """Build the graph used by the focused compiler assertion."""
    return PluginGraphTemplate().build_graph(
        ComponentBindings(components={"phase10_plugin": PluginCandidateComponent()})
    )
