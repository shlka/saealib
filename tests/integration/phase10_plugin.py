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
    BEHAVIOR,
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
    ObservationSubject,
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
CMAES_CANDIDATE_IDS = np.array([8601, 8602, 8603, 8604], dtype=np.int64)
MOEAD_CANDIDATE_IDS = np.array([8701, 8702, 8703], dtype=np.int64)
MAPELITES_CANDIDATE_IDS = np.array([8801, 8802, 8803], dtype=np.int64)
COEVOLUTION_CANDIDATE_ID = np.int64(8901)


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
        self,
        candidate_ids: np.ndarray,
        result: Any,
        *,
        objective_count: int = 1,
        include_cv: bool = True,
    ) -> ObservationBatch:
        return ObservationBatch.from_dense(
            ObservationSchema(objective_count=objective_count, constraint_count=0),
            candidate_ids,
            result.f,
            result.g,
            result.cv if include_cv else None,
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
class CMAESPlugin:
    """Deterministic CMA-ES-shaped plugin with plugin-owned algorithm state."""

    mean: np.ndarray = field(
        default_factory=lambda: np.array([0.25, -0.25], dtype=np.float64)
    )
    covariance: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=np.float64))
    step_size: float = 0.2
    generation: int = 0

    def problem(self) -> Problem:
        return Phase10PluginFixture().problem()

    def evaluator(self) -> SerialEvaluator:
        return SerialEvaluator()

    def snapshot(self) -> dict[str, Any]:
        return {
            "mean": self.mean.copy(),
            "covariance": self.covariance.copy(),
            "step_size": self.step_size,
            "generation": self.generation,
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.mean = np.asarray(snapshot["mean"], dtype=np.float64).copy()
        self.covariance = np.asarray(snapshot["covariance"], dtype=np.float64).copy()
        self.step_size = float(snapshot["step_size"])
        self.generation = int(snapshot["generation"])

    def propose(self) -> ProposalBatch:
        offsets = np.array(
            [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            dtype=np.float64,
        )
        genomes = self.mean + self.step_size * offsets
        candidates = Population(
            attrs=[PopulationAttribute(name="x", dtype=np.float64, shape=(2,))],
            init_capacity=len(genomes),
        )
        for genome in genomes:
            candidates.append(x=genome)
        return ProposalBatch(
            proposal_id=PLUGIN_PROPOSAL_ID + 4 + self.generation,
            candidates=candidates,
            relations=ProposalRelations(row_count=len(genomes)),
            requirements=objective_requirement(),
        )

    def evaluate(self, proposal: ProposalBatch) -> FeedbackBatch:
        request = EvaluationRequest(
            request_id=np.int64(3),
            candidate_ids=CMAES_CANDIDATE_IDS,
            payload=proposal.candidates.genomes,
        )
        result = self.evaluator().evaluate_request(request, self.problem())
        observations = Phase10PluginFixture().observations_from_result(
            CMAES_CANDIDATE_IDS, result, include_cv=False
        )
        return FeedbackBatch(
            proposal_id=proposal.proposal_id,
            observations=observations,
            channel=TRUE,
            final=True,
            sequence=0,
        )

    def apply_feedback(self, proposal: ProposalBatch, feedback: FeedbackBatch) -> None:
        if not feedback.final or not np.array_equal(
            feedback.observations.candidate_ids, CMAES_CANDIDATE_IDS
        ):
            return
        samples = proposal.candidates.x
        self.mean = np.mean(samples, axis=0)
        centered = samples - self.mean
        self.covariance = centered.T @ centered / len(samples)
        self.step_size *= 0.9
        self.generation += 1


@dataclass
class MOEADPlugin:
    """Minimal MOEA/D plugin with explicit subproblem-local replacement."""

    weights: np.ndarray = field(
        default_factory=lambda: np.array(
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float64
        )
    )
    neighbors: dict[int, tuple[int, ...]] = field(
        default_factory=lambda: {0: (0, 1), 1: (1,), 2: (2,)}
    )
    subproblem_state: dict[int, tuple[np.ndarray, float]] = field(default_factory=dict)

    def problem(self) -> Problem:
        return Problem(
            func=lambda x: np.asarray(
                [
                    float(np.asarray(x, dtype=float)[0] ** 2),
                    float(np.asarray(x, dtype=float)[1] ** 2),
                ]
            ),
            dim=2,
            n_obj=2,
            direction=np.array([-1.0, -1.0]),
            lb=[-1.0, -1.0],
            ub=[1.0, 1.0],
            evaluation_adapter=PluginEvaluationAdapter(),
        )

    def evaluator(self) -> SerialEvaluator:
        return SerialEvaluator()

    def initial_genomes(self) -> np.ndarray:
        return np.array([[0.8, 0.8], [0.6, 0.6], [0.2, 0.2]], dtype=np.float64)

    def propose(self) -> ProposalBatch:
        initial = self.initial_genomes()
        self.subproblem_state = {
            index: (genome.copy(), float(np.dot(self.weights[index], genome**2)))
            for index, genome in enumerate(initial)
        }
        children = np.array([[0.1, 0.1], [0.9, 0.9], [0.5, 0.5]], dtype=np.float64)
        candidates = Population(
            attrs=[
                PopulationAttribute(name="x", dtype=np.float64, shape=(2,)),
                PopulationAttribute(name="subproblem_id", dtype=np.int64, default=-1),
            ],
            init_capacity=len(children),
        )
        for subproblem_id, genome in enumerate(children):
            candidates.append(x=genome, subproblem_id=subproblem_id)
        return ProposalBatch(
            proposal_id=PLUGIN_PROPOSAL_ID + 5,
            candidates=candidates,
            relations=ProposalRelations(row_count=len(children)),
            requirements=objective_requirement(),
        )

    def evaluate(self, proposal: ProposalBatch) -> FeedbackBatch:
        request = EvaluationRequest(
            request_id=np.int64(4),
            candidate_ids=MOEAD_CANDIDATE_IDS,
            payload=proposal.candidates.genomes,
        )
        result = self.evaluator().evaluate_request(request, self.problem())
        observations = Phase10PluginFixture().observations_from_result(
            MOEAD_CANDIDATE_IDS, result, objective_count=2
        )
        return FeedbackBatch(
            proposal_id=proposal.proposal_id,
            observations=observations,
            channel=TRUE,
            final=True,
            sequence=0,
        )

    def apply_feedback(self, proposal: ProposalBatch, feedback: FeedbackBatch) -> None:
        if not feedback.final:
            return
        subproblem_ids = proposal.candidates.get_array("subproblem_id")
        objectives = feedback.observations.f
        for row, subproblem_id in enumerate(subproblem_ids):
            child_score = float(
                np.dot(self.weights[int(subproblem_id)], objectives[row])
            )
            child = proposal.candidates.x[row].copy()
            for neighbor in self.neighbors[int(subproblem_id)]:
                _, current_score = self.subproblem_state[neighbor]
                if child_score < current_score:
                    self.subproblem_state[neighbor] = (child.copy(), child_score)


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


@dataclass
class MAPElitesPlugin:
    """Minimal MAP-Elites emitter and behavior-indexed archive."""

    archive: dict[tuple[int, ...], tuple[np.ndarray, float]] = field(
        default_factory=dict
    )
    emitter_calls: int = 0
    pending_genomes: dict[int, np.ndarray] = field(default_factory=dict)

    def emitter(self) -> Population:
        self.emitter_calls += 1
        genomes = np.array([[0.4, 0.4], [0.1, 0.1], [0.8, 0.8]], dtype=np.float64)
        population = Population(
            attrs=[PopulationAttribute(name="x", dtype=np.float64, shape=(2,))],
            init_capacity=len(genomes),
        )
        for genome in genomes:
            population.append(x=genome)
        return population

    def propose(self) -> ProposalBatch:
        candidates = self.emitter()
        self.pending_genomes = {
            int(candidate_id): candidates.x[row].copy()
            for row, candidate_id in enumerate(MAPELITES_CANDIDATE_IDS)
        }
        return ProposalBatch(
            proposal_id=PLUGIN_PROPOSAL_ID + 6,
            candidates=candidates,
            relations=ProposalRelations(row_count=len(candidates)),
            requirements=objective_requirement(),
        )

    def behavior(self, row: int) -> np.ndarray:
        return np.array([0.25, 0.25, 0.75][row], dtype=np.float64)

    def evaluate(self, proposal: ProposalBatch) -> FeedbackBatch:
        records: list[ObservationRecord] = []
        for row, candidate_id in enumerate(MAPELITES_CANDIDATE_IDS):
            genome = proposal.candidates.x[row]
            records.append(
                ObservationRecord(
                    subject=("candidate", np.array([candidate_id], dtype=np.int64)),
                    quantity=(OBJECTIVE, 0),
                    value=float(np.sum(genome**2)),
                    status=OK,
                    source=TRUE,
                )
            )
            records.append(
                ObservationRecord(
                    subject=("candidate", np.array([candidate_id], dtype=np.int64)),
                    quantity=(BEHAVIOR, 0),
                    value=self.behavior(row),
                    status=OK,
                    source=TRUE,
                )
            )
        return FeedbackBatch(
            proposal_id=proposal.proposal_id,
            observations=ObservationBatch(
                schema=ObservationSchema(objective_count=1, quantities={BEHAVIOR: 1}),
                records=ObservationRecords.from_records(records),
            ),
            channel=TRUE,
            final=True,
            sequence=0,
        )

    def cell_key(self, behavior: np.ndarray) -> tuple[int, ...]:
        values = np.floor(np.asarray(behavior, dtype=float) * 2).astype(int).reshape(-1)
        return tuple(values)

    def apply_feedback(self, feedback: FeedbackBatch) -> None:
        records = feedback.observations.records
        candidate_ids = feedback.observations.candidate_ids
        values = records.column("value")
        record_ids = np.asarray(
            [
                int(ObservationSubject.from_value(records[index].subject).payload[0])
                for index in range(len(records))
            ],
            dtype=np.int64,
        )
        for candidate_id in candidate_ids:
            rows = np.flatnonzero(record_ids == int(candidate_id))
            objective_row, behavior_row = rows[:2]
            score = float(values[objective_row])
            cell = self.cell_key(np.asarray(values[behavior_row], dtype=float))
            current = self.archive.get(cell)
            if current is None or score < current[1]:
                self.archive[cell] = (
                    self.pending_genomes[int(candidate_id)].copy(),
                    score,
                )


@dataclass
class CooperativeCoevolutionPlugin:
    """Two named populations joined into one evaluated cooperative vector."""

    named_populations: dict[str, Population] = field(default_factory=dict)
    pending_blocks: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.named_populations = {
            "block_a": self._population(0.8, 0.1),
            "block_b": self._population(0.2),
        }

    def _population(self, *values: float) -> Population:
        population = Population(
            attrs=[PopulationAttribute(name="x", dtype=np.float64, shape=(1,))],
            init_capacity=len(values),
        )
        for value in values:
            population.append(x=np.array([value], dtype=np.float64))
        return population

    def coordinator_join(self) -> ProposalBatch:
        self.pending_blocks = {
            "block_a": self.named_populations["block_a"].x[1].copy(),
            "block_b": self.named_populations["block_b"].x[0].copy(),
        }
        joint = np.concatenate(
            (self.pending_blocks["block_a"], self.pending_blocks["block_b"])
        )
        candidates = Population(
            attrs=[PopulationAttribute(name="x", dtype=np.float64, shape=(2,))],
            init_capacity=1,
        )
        candidates.append(x=joint)
        return ProposalBatch(
            proposal_id=PLUGIN_PROPOSAL_ID + 7,
            candidates=candidates,
            relations=ProposalRelations(row_count=1),
            requirements=objective_requirement(),
        )

    def problem(self) -> Problem:
        return Phase10PluginFixture().problem()

    def evaluator(self) -> SerialEvaluator:
        return SerialEvaluator()

    def evaluate(self, proposal: ProposalBatch) -> FeedbackBatch:
        request = EvaluationRequest(
            request_id=np.int64(5),
            candidate_ids=np.array([COEVOLUTION_CANDIDATE_ID], dtype=np.int64),
            payload=proposal.candidates.genomes,
        )
        result = self.evaluator().evaluate_request(request, self.problem())
        observations = Phase10PluginFixture().observations_from_result(
            np.array([COEVOLUTION_CANDIDATE_ID], dtype=np.int64), result
        )
        return FeedbackBatch(
            proposal_id=proposal.proposal_id,
            observations=observations,
            channel=TRUE,
            final=True,
            sequence=0,
        )

    def apply_feedback(self, feedback: FeedbackBatch) -> None:
        score = float(feedback.observations.f[0, 0])
        current = self.named_populations["block_a"].x[0]
        baseline = float(
            np.sum(
                np.concatenate((current, self.named_populations["block_b"].x[0])) ** 2
            )
        )
        if score < baseline:
            self.named_populations["block_a"].update_rows(
                [0], {"x": self.pending_blocks["block_a"].reshape(1, 1)}
            )


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
