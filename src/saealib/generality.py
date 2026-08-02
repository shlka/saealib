"""Reusable orchestration components for structured optimization workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from saealib.exceptions import ValidationError
from saealib.execution.evaluator import (
    EvaluationRequest,
    EvaluationResult,
    Evaluator,
    SerialEvaluator,
)
from saealib.policies.evaluation import (
    RepeatedEvaluation,
    ReplicateSummary,
    aggregate_replicates,
)
from saealib.population import Archive, PopulationAttribute
from saealib.problem import Problem
from saealib.surrogate import PredictionChannel, Surrogate, SurrogatePrediction


def _history_archive(dim: int) -> Archive:
    attrs = [
        PopulationAttribute("x", np.float64, (dim,)),
        PopulationAttribute("f", np.float64, (1,)),
        PopulationAttribute("g", np.float64, (0,)),
        PopulationAttribute("cv", np.float64, (), 0.0),
        PopulationAttribute("id", np.int64, (), -1),
        PopulationAttribute("request_id", np.int64, (), -1),
        PopulationAttribute("fidelity", np.int64, (), 0),
    ]
    return Archive(attrs, duplicate_policy="append", init_capacity=16)


class _RequestAllocator:
    def __init__(self):
        self.next = 0

    def allocate(self, count: int) -> np.ndarray:
        values = np.arange(self.next, self.next + count, dtype=np.int64)
        self.next += count
        return values


class _CandidateBatch:
    def __init__(self, x: np.ndarray, ids: np.ndarray):
        self.x = np.array(x, dtype=np.float64, copy=True)
        self.schema = {"id": object()}
        self._ids = np.array(ids, dtype=np.int64, copy=True)

    def __len__(self):
        return len(self.x)

    def get_array(self, name):
        return self._ids if name == "id" else self.x


@dataclass(frozen=True)
class EvaluationWorkflowResult:
    """Results and append history from a finite public evaluation workflow."""

    archive: Archive
    truth_archive: Archive
    requests: tuple[EvaluationRequest, ...]
    summary: ReplicateSummary
    fe: int


class RepeatedEvaluationRunner:
    """Execute replicate requests through an Evaluator and append history."""

    def __init__(self, evaluator: Evaluator, replicates: int = 3):
        self.evaluator = evaluator
        self.policy = RepeatedEvaluation(replicates)

    def run(self, x: np.ndarray, candidate_ids: np.ndarray, problem: Problem):
        """Run all replicates and return candidate-aligned archive history."""
        batch = _CandidateBatch(x, candidate_ids)
        ctx = SimpleNamespace(request_id_allocator=_RequestAllocator())
        requests = self.policy.plan_replicates(batch, ctx)
        archive = _history_archive(problem.dim)
        observations: list[np.ndarray] = []
        for request in requests:
            handle = self.evaluator.submit(request, problem)
            update = self.evaluator.collect(handle, wait=True)[0]
            if update.result is None:
                raise ValidationError("replicate evaluation returned no result")
            observations.append(np.array(update.result.f, copy=True))
            for row, candidate_id in enumerate(request.candidate_ids):
                archive.add(
                    x=request.x[row],
                    f=update.result.f[row],
                    g=update.result.g[row],
                    cv=update.result.cv[row],
                    id=np.int64(candidate_id),
                    request_id=np.int64(request.request_id),
                    fidelity=np.int64(0),
                )
            self.evaluator.acknowledge(handle, update.sequence)
        summary = aggregate_replicates(candidate_ids, np.stack(observations))
        truth_archive = Archive(
            [
                PopulationAttribute("x", np.float64, (problem.dim,)),
                PopulationAttribute("f", np.float64, (problem.n_obj,)),
                PopulationAttribute("std", np.float64, (problem.n_obj,)),
                PopulationAttribute("count", np.int64, (), 0),
                PopulationAttribute("id", np.int64, (), -1),
            ],
            duplicate_policy="keep_first",
        )
        for row, candidate_id in enumerate(candidate_ids):
            truth_archive.add(
                x=x[row],
                f=summary.mean[row],
                std=summary.std[row],
                count=summary.count[row],
                id=np.int64(candidate_id),
            )
        return EvaluationWorkflowResult(
            archive, truth_archive, requests, summary, len(requests) * len(x)
        )


class SeededNoiseEvaluator(SerialEvaluator):
    """Serial evaluator that adds reproducible objective noise."""

    def __init__(self, seed: int, scale: float = 0.01):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.scale = float(scale)

    def evaluate_request(self, request, problem):
        """Evaluate and add independent seeded noise per objective."""
        result = super().evaluate_request(request, problem)
        noise = self.rng.normal(0.0, self.scale, size=result.f.shape)
        return EvaluationResult(
            f=result.f + noise,
            g=result.g,
            cv=result.cv,
            candidate_ids=result.candidate_ids,
            cost=result.cost,
            noise=noise,
            outputs=result.outputs,
        )


class FidelityEvaluator(Evaluator):
    """Evaluator that consumes request metadata at the problem boundary."""

    def __init__(self, evaluate: Callable[[np.ndarray, int], np.ndarray]):
        self._evaluate = evaluate

    def evaluate_batch(self, x: np.ndarray, problem: Problem) -> EvaluationResult:
        """Evaluate a request with its default fidelity."""
        return self.evaluate_request(
            EvaluationRequest(np.int64(0), np.arange(len(x), dtype=np.int64), x),
            problem,
        )

    def evaluate_request(self, request, problem):
        """Evaluate candidates using the request fidelity metadata."""
        fidelity = int(request.metadata.get("fidelity", 0))
        f = np.stack([self._evaluate(row.copy(), fidelity) for row in request.x])
        g = np.empty((len(f), problem.n_constraints), dtype=np.float64)
        cv = np.zeros(len(f), dtype=np.float64)
        return EvaluationResult(
            f=f,
            g=g,
            cv=cv,
            candidate_ids=request.candidate_ids,
            cost=np.full(len(f), fidelity + 1.0, dtype=np.float64),
            outputs={"fidelity": np.full(len(f), fidelity, dtype=np.float64)},
        )

    def evaluate(self, x: np.ndarray, fidelity: int) -> np.ndarray:
        """Evaluate one candidate at the requested fidelity."""
        return np.array(
            self._evaluate(np.array(x, dtype=np.float64, copy=True), fidelity)
        )


@dataclass(frozen=True)
class FidelityWorkflowResult:
    """Low- and high-fidelity observations from a promotion run."""

    low_request: EvaluationRequest
    high_request: EvaluationRequest
    low_result: EvaluationResult
    high_result: EvaluationResult
    archive: Archive
    fe: int
    cost: float


class FidelityPromotionRunner:
    """Evaluate a low-fidelity batch and promote its best candidate."""

    def __init__(self, evaluator: FidelityEvaluator, policy):
        self.evaluator = evaluator
        self.policy = policy

    def run(self, x: np.ndarray, candidate_ids: np.ndarray, problem: Problem):
        """Return low/high history with promotion selected by low objective."""
        allocator = _RequestAllocator()
        low = EvaluationRequest(
            allocator.allocate(1)[0],
            candidate_ids,
            x,
            metadata={"fidelity": self.policy.fidelity},
        )
        low_result = self.evaluator.submit(low, problem)
        low_update = self.evaluator.collect(low_result, wait=True)[0]
        if low_update.result is None:
            raise ValidationError("low-fidelity evaluation returned no result")
        best = int(np.argmin(low_update.result.f[:, 0]))
        ctx = SimpleNamespace(request_id_allocator=allocator)
        promoted = self.policy.promote(
            EvaluationRequest(
                low.request_id, np.array([candidate_ids[best]]), x[[best]]
            ),
            ctx,
        )
        high_handle = self.evaluator.submit(promoted, problem)
        high_update = self.evaluator.collect(high_handle, wait=True)[0]
        if high_update.result is None:
            raise ValidationError("high-fidelity evaluation returned no result")
        self.evaluator.acknowledge(low_result, low_update.sequence)
        self.evaluator.acknowledge(high_handle, high_update.sequence)
        archive = _history_archive(problem.dim)
        for result, request in (
            (low_update.result, low),
            (high_update.result, promoted),
        ):
            if result.candidate_ids is None:
                raise ValidationError("fidelity result has no candidate IDs")
            for row, candidate_id in enumerate(result.candidate_ids):
                archive.add(
                    x=request.x[
                        np.flatnonzero(request.candidate_ids == candidate_id)[0]
                    ],
                    f=result.f[row],
                    g=result.g[row],
                    cv=result.cv[row],
                    id=np.int64(candidate_id),
                    request_id=request.request_id,
                    fidelity=np.int64(request.metadata["fidelity"]),
                )
        if low_update.result.cost is None or high_update.result.cost is None:
            raise ValidationError("fidelity result has no cost")
        total_cost = float(
            np.sum(low_update.result.cost) + np.sum(high_update.result.cost)
        )
        return FidelityWorkflowResult(
            low,
            promoted,
            low_update.result,
            high_update.result,
            archive,
            len(low.candidate_ids) + len(promoted.candidate_ids),
            total_cost,
        )


class CorrelatedQuadraticSurrogate(Surrogate):
    """Deterministic regression surrogate with a joint posterior."""

    provides_uncertainty = True

    def __init__(self, correlation: float = 0.7):
        if not 0.0 <= correlation < 1.0:
            raise ValidationError("correlation must be in [0, 1)")
        self.correlation = float(correlation)

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """Keep the deterministic posterior independent of training data."""
        return None

    def predict(self, test_x: np.ndarray) -> SurrogatePrediction:
        """Return correlated objective uncertainty for the query batch."""
        x = np.array(test_x, dtype=np.float64, order="C", copy=True)
        x = np.atleast_2d(x)
        n = len(x)
        mean = np.sum(x * x, axis=1, keepdims=True)
        std = np.full((n, 1), 0.25, dtype=np.float64)
        covariance = np.full((n, n), self.correlation * 0.25**2, dtype=np.float64)
        np.fill_diagonal(covariance, 0.25**2)
        return SurrogatePrediction(
            channels={
                "objective": PredictionChannel(
                    value=mean, std=std, covariance=covariance
                )
            },
            x=x,
        )


@dataclass(frozen=True)
class CooperativeResult:
    """Result of block-wise evaluations against a shared context."""

    context: np.ndarray
    objective_history: np.ndarray
    candidate_ids: np.ndarray


def _evaluate_one(
    evaluator: Evaluator, problem: Problem, request_id: int, x: np.ndarray
):
    request = EvaluationRequest(
        np.int64(request_id), np.array([request_id], dtype=np.int64), x[None, :]
    )
    handle = evaluator.submit(request, problem)
    update = evaluator.collect(handle, wait=True)[0]
    evaluator.acknowledge(handle, update.sequence)
    if update.result is None:
        raise ValidationError("evaluation returned no result")
    return request, update.result


def reference_problem(dim: int = 2, shift: float = 0.0) -> Problem:
    """Return a deterministic bounded single-objective problem."""

    def evaluate(x: np.ndarray) -> np.ndarray:
        return np.array([np.sum((x - shift) ** 2)], dtype=np.float64)

    return Problem(
        func=evaluate,
        dim=dim,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0] * dim,
        ub=[1.0] * dim,
    )


class CooperativeCoevolution:
    """Maintain a shared vector while updating disjoint coordinate blocks."""

    def __init__(self, dim: int, blocks: tuple[tuple[int, ...], ...]):
        if dim < 1 or not blocks:
            raise ValidationError("dim and blocks must be non-empty")
        flat = [index for block in blocks for index in block]
        if sorted(flat) != list(range(dim)):
            raise ValidationError("blocks must partition all coordinates")
        self.dim = dim
        self.blocks = blocks
        self.context = np.zeros(dim, dtype=np.float64)

    def assemble(self, block: int, values: np.ndarray) -> np.ndarray:
        """Combine one block with the current shared context."""
        indices = self.blocks[block]
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (len(indices),):
            raise ValidationError("block values have an invalid shape")
        candidate = self.context.copy()
        candidate[list(indices)] = values
        return candidate

    def update(self, block: int, values: np.ndarray) -> np.ndarray:
        """Update one block and return an owned full candidate."""
        self.context = self.assemble(block, values)
        return self.context.copy()

    def optimize(
        self,
        problem: Problem,
        proposals: tuple[np.ndarray, ...],
        evaluator: Evaluator | None = None,
    ) -> CooperativeResult:
        """Evaluate and accept improving block proposals."""
        evaluator = evaluator or SerialEvaluator()
        history: list[float] = []
        ids: list[int] = []
        current = np.inf
        for block, values in enumerate(proposals):
            candidate = self.assemble(block, values)
            _, result = _evaluate_one(evaluator, problem, block, candidate)
            score = float(result.f[0, 0])
            if score <= current:
                self.context = candidate
                current = score
            history.append(current)
            ids.append(block)
        return CooperativeResult(
            self.context.copy(),
            np.asarray(history, dtype=np.float64),
            np.asarray(ids, dtype=np.int64),
        )


class MigrationPolicy:
    """Deterministically copy selected island rows between numeric arrays."""

    def __init__(self, count: int = 1):
        if count < 1:
            raise ValidationError("count must be positive")
        self.count = count

    def migrate(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return a target copy with selected source rows."""
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
            raise ValidationError("island arrays must be two-dimensional and aligned")
        result = target.copy()
        count = min(self.count, len(source), len(result))
        result[:count] = source[:count]
        return result

    def optimize(
        self,
        problem: Problem,
        islands: tuple[np.ndarray, ...],
        evaluator: Evaluator | None = None,
        rounds: int = 2,
    ) -> tuple[tuple[np.ndarray, ...], tuple[tuple[int, ...], ...]]:
        """Run island evaluations and scheduled migration."""
        evaluator = evaluator or SerialEvaluator()
        states = [np.array(island, dtype=np.float64, copy=True) for island in islands]
        events: list[tuple[int, ...]] = []
        request_id = 0
        for _ in range(rounds):
            winners: list[np.ndarray] = []
            for island in states:
                values = []
                for row in island:
                    _, result = _evaluate_one(evaluator, problem, request_id, row)
                    values.append(float(result.f[0, 0]))
                    request_id += 1
                winners.append(island[int(np.argmin(values))].copy())
            for index in range(1, len(states)):
                states[index] = self.migrate(winners[index - 1][None, :], states[index])
            events.append(tuple(range(len(states))))
        return tuple(states), tuple(events)


@dataclass(frozen=True)
class ArchiveSnapshot:
    """Archive values tagged with the environment that produced them."""

    environment: int
    x: np.ndarray
    f: np.ndarray


class DynamicArchiveSelector:
    """Select the snapshot whose objective best matches the current environment."""

    def __init__(self):
        self.snapshots: list[ArchiveSnapshot] = []

    def add(self, environment: int, x: np.ndarray, f: np.ndarray) -> None:
        """Store an owned archive snapshot."""
        self.snapshots.append(
            ArchiveSnapshot(
                environment,
                np.array(x, dtype=np.float64, copy=True),
                np.array(f, dtype=np.float64, copy=True),
            )
        )

    def select(self, environment: int) -> ArchiveSnapshot:
        """Return the snapshot nearest to the requested environment."""
        if not self.snapshots:
            raise ValidationError("no archive snapshots are available")
        return min(
            self.snapshots,
            key=lambda snapshot: abs(snapshot.environment - environment),
        )

    def optimize(
        self,
        problem_factory: Callable[[int], Problem],
        environments: tuple[int, ...],
        x: np.ndarray,
        evaluator: Evaluator | None = None,
    ) -> tuple[ArchiveSnapshot, ...]:
        """Evaluate candidates in each environment and store snapshots."""
        evaluator = evaluator or SerialEvaluator()
        x = np.array(x, dtype=np.float64, copy=True)
        for environment in environments:
            problem = problem_factory(environment)
            request = EvaluationRequest(
                np.int64(environment),
                np.arange(len(x), dtype=np.int64),
                x,
            )
            handle = evaluator.submit(request, problem)
            update = evaluator.collect(handle, wait=True)[0]
            evaluator.acknowledge(handle, update.sequence)
            if update.result is None:
                raise ValidationError("environment evaluation returned no result")
            self.add(environment, x, update.result.f)
        return tuple(self.snapshots)
