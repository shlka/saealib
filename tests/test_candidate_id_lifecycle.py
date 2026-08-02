"""Integration tests for stable candidate IDs across a real Optimizer run."""

from __future__ import annotations

import numpy as np
import pytest

from saealib import (
    GA,
    CrossoverBLXAlpha,
    IndividualBasedStrategy,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    RBFSurrogate,
    SequentialSelection,
    Termination,
    TruncationSelection,
    gaussian_kernel,
    max_gen,
)
from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.exceptions import ValidationError
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.stages import AskStage

DIM = 2
N_INIT_ARCHIVE = 10
N_INIT_POP = 6


def _sphere(x: np.ndarray) -> np.ndarray:
    return np.array([np.sum(x**2)])


def _make_problem() -> Problem:
    return Problem(
        func=_sphere,
        dim=DIM,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0] * DIM,
        ub=[5.0] * DIM,
        comparator=SingleObjectiveComparator(),
    )


def _make_optimizer(problem: Problem, seed: int, n_gen: int) -> Optimizer:
    return (
        Optimizer(problem, seed=seed)
        .set_initializer(LHSInitializer(N_INIT_ARCHIVE, N_INIT_POP))
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.5),
                mutation=MutationUniform(prob_var=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_surrogate(RBFSurrogate(gaussian_kernel, DIM), n_neighbors=5)
        .set_strategy(IndividualBasedStrategy(evaluation_ratio=0.5))
        .set_termination(Termination(max_gen(n_gen)))
    )


def _assert_real_unique_ids(ids: np.ndarray) -> None:
    assert np.all(ids != -1)
    assert len(ids) == len(np.unique(ids))


def test_every_row_has_real_unique_id_after_generations():
    problem = _make_problem()
    ctx = _make_optimizer(problem, seed=0, n_gen=3).run()

    _assert_real_unique_ids(ctx.population.get_array("id"))
    _assert_real_unique_ids(ctx.archive.get_array("id"))
    _assert_real_unique_ids(ctx.pareto_archive.get_array("id"))


def test_same_logical_candidate_shares_id_across_collections():
    problem = _make_problem()
    ctx = _make_optimizer(problem, seed=0, n_gen=3).run()

    archive_ids = set(ctx.archive.get_array("id").tolist())
    pareto_ids = set(ctx.pareto_archive.get_array("id").tolist())
    assert pareto_ids <= archive_ids

    archive_x = ctx.archive.get_array("x")
    archive_id = ctx.archive.get_array("id")
    pop_x = ctx.population.get_array("x")
    pop_id = ctx.population.get_array("id")
    n_matched = 0
    for i in range(len(pop_x)):
        matches = np.where(np.all(np.isclose(archive_x, pop_x[i]), axis=1))[0]
        if len(matches) == 0:
            continue
        n_matched += 1
        assert pop_id[i] in archive_id[matches]
    assert n_matched > 0, (
        "expected at least one population row to trace to an archive row"
    )


def test_save_load_roundtrips_ids_and_allocator_state(tmp_path):
    problem = _make_problem()
    ctx = _make_optimizer(problem, seed=0, n_gen=3).run()

    next_candidate_before = ctx.candidate_id_allocator.next_value
    next_request_before = ctx.request_id_allocator.next_value

    p = tmp_path / "ckpt.npz"
    ctx.save(p)
    loaded = OptimizationState.load(p, problem)

    np.testing.assert_array_equal(
        loaded.population.get_array("id"), ctx.population.get_array("id")
    )
    np.testing.assert_array_equal(
        loaded.archive.get_array("id"), ctx.archive.get_array("id")
    )
    assert loaded.candidate_id_allocator.next_value == next_candidate_before
    assert loaded.request_id_allocator.next_value == next_request_before


class _DuplicateRealIdAlgorithm:
    def ask(self, ctx, provider, n_offspring=None):
        attrs = [
            PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
            PopulationAttribute(name="id", dtype=np.int64, shape=(), default=-1),
        ]
        offspring = Population(attrs, init_capacity=2)
        offspring._extend_internal(
            {
                "x": np.array([[0.0, 0.0], [1.0, 1.0]]),
                "id": np.array([42, 42], dtype=np.int64),
            },
            preserve_ids=True,
        )
        return offspring


def test_ask_stage_rejects_all_real_duplicate_id_batch():
    problem = _make_problem()
    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
        PopulationAttribute(name="id", dtype=np.int64, shape=(), default=-1),
    ]
    pop = Population(attrs, init_capacity=1)
    archive = Archive(attrs, init_capacity=1)
    pareto = ParetoArchive(attrs, init_capacity=1, direction=problem.direction)
    state = OptimizationState(
        problem=problem,
        population=pop,
        archive=archive,
        pareto_archive=pareto,
        rng=np.random.default_rng(0),
    )
    stage = AskStage(_DuplicateRealIdAlgorithm(), cbmanager=None)
    with pytest.raises(ValidationError):
        stage.execute(state)
