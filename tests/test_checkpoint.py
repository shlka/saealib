"""Tests for seed, checkpoint, and resume features (Issue #95)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from saealib import (
    GA,
    CheckpointCallback,
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
from saealib.problem import Problem

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


def _make_optimizer(
    problem: Problem, seed: int | None = None, n_gen: int = 4
) -> Optimizer:
    return (
        Optimizer(problem, seed=seed)
        .set_initializer(LHSInitializer(N_INIT_ARCHIVE, N_INIT_POP))
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.5),
                mutation=MutationUniform(mutation_rate=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_surrogate(RBFSurrogate(gaussian_kernel, DIM), n_neighbors=5)
        .set_strategy(IndividualBasedStrategy(evaluation_ratio=0.5))
        .set_termination(Termination(max_gen(n_gen)))
    )


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def test_seed_reproducibility():
    problem = _make_problem()
    ctx1 = _make_optimizer(problem, seed=42).run()
    ctx2 = _make_optimizer(problem, seed=42).run()
    np.testing.assert_array_equal(ctx1.archive.x, ctx2.archive.x)
    np.testing.assert_array_equal(ctx1.archive.f, ctx2.archive.f)


def test_set_seed_method():
    problem = _make_problem()
    opt = _make_optimizer(problem)
    opt.set_seed(99)
    assert opt.seed == 99


def test_seed_none_stored():
    problem = _make_problem()
    opt = Optimizer(problem)
    assert opt.seed is None


def test_lhs_uses_optimizer_seed_over_own():
    """Optimizer.seed takes priority over LHSInitializer.seed."""
    problem = _make_problem()

    opt_a = (
        Optimizer(problem, seed=7)
        .set_initializer(LHSInitializer(N_INIT_ARCHIVE, N_INIT_POP, seed=999))
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.5),
                mutation=MutationUniform(mutation_rate=0.1),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_surrogate(RBFSurrogate(gaussian_kernel, DIM), n_neighbors=5)
        .set_strategy(IndividualBasedStrategy(evaluation_ratio=0.5))
        .set_termination(Termination(max_gen(2)))
    )
    opt_b = _make_optimizer(problem, seed=7, n_gen=2)

    ctx_a = opt_a.run()
    ctx_b = opt_b.run()
    np.testing.assert_array_equal(ctx_a.archive.x, ctx_b.archive.x)


# ---------------------------------------------------------------------------
# OptimizationState.save / load (npz)
# ---------------------------------------------------------------------------


def test_npz_roundtrip(tmp_path):
    problem = _make_problem()
    ctx = _make_optimizer(problem, seed=0, n_gen=2).run()

    p = tmp_path / "ckpt.npz"
    ctx.save(p)
    assert p.exists()

    loaded = OptimizationState.load(p, problem)
    np.testing.assert_array_equal(loaded.archive.x, ctx.archive.x)
    np.testing.assert_array_equal(loaded.archive.f, ctx.archive.f)
    np.testing.assert_array_equal(loaded.population.x, ctx.population.x)
    assert loaded.fe == ctx.fe
    assert loaded.gen == ctx.gen


def test_npz_extension_added(tmp_path):
    problem = _make_problem()
    ctx = _make_optimizer(problem, seed=0, n_gen=2).run()
    ctx.save(tmp_path / "ckpt")
    assert (tmp_path / "ckpt.npz").exists()


def test_npz_rng_state_preserved(tmp_path):
    problem = _make_problem()
    ctx = _make_optimizer(problem, seed=0, n_gen=2).run()
    state_before = ctx.rng.bit_generator.state

    p = tmp_path / "ckpt.npz"
    ctx.save(p)
    loaded = OptimizationState.load(p, problem)
    assert loaded.rng.bit_generator.state == state_before


def test_resumed_flag_npz(tmp_path):
    problem = _make_problem()
    ctx = _make_optimizer(problem, seed=0, n_gen=2).run()
    assert ctx.data.get("resumed", False) is False

    p = tmp_path / "ckpt.npz"
    ctx.save(p)
    loaded = OptimizationState.load(p, problem)
    assert loaded.data.get("resumed") is True


# ---------------------------------------------------------------------------
# Optimizer.save_pickle / load_pickle
# ---------------------------------------------------------------------------


def test_pickle_roundtrip(tmp_path):
    problem = _make_problem()
    opt = _make_optimizer(problem, seed=0, n_gen=2)
    ctx = opt.run()

    p = tmp_path / "ckpt.pkl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        opt.save_pickle(ctx, p)
    assert p.exists()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _opt2, ctx2 = Optimizer.load_pickle(p)

    np.testing.assert_array_equal(ctx2.archive.x, ctx.archive.x)
    assert ctx2.gen == ctx.gen


def test_resumed_flag_pickle(tmp_path):
    problem = _make_problem()
    opt = _make_optimizer(problem, seed=0, n_gen=2)
    ctx = opt.run()

    p = tmp_path / "ckpt.pkl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        opt.save_pickle(ctx, p)
        _, ctx2 = Optimizer.load_pickle(p)
    assert ctx2.data.get("resumed") is True


def test_pickle_extension_added(tmp_path):
    problem = _make_problem()
    opt = _make_optimizer(problem, seed=0, n_gen=2)
    ctx = opt.run()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        opt.save_pickle(ctx, tmp_path / "ckpt")
    assert (tmp_path / "ckpt.pkl").exists()


# ---------------------------------------------------------------------------
# run_from / iterate_from
# ---------------------------------------------------------------------------


def test_run_from_npz_resumed_flag(tmp_path):
    problem = _make_problem()
    ctx_mid = _make_optimizer(problem, seed=42, n_gen=2).run()
    p = tmp_path / "ckpt.npz"
    ctx_mid.save(p)

    loaded = OptimizationState.load(p, problem)
    ctx_final = _make_optimizer(problem, seed=42, n_gen=4).run_from(loaded)
    assert ctx_final.data.get("resumed") is True
    assert ctx_final.gen == 4


def test_run_from_npz_matches_full_run(tmp_path):
    """Resume from npz checkpoint produces identical results to uninterrupted run."""
    problem = _make_problem()

    # Full run: 4 generations
    ctx_full = _make_optimizer(problem, seed=42, n_gen=4).run()

    # Split: 2 + 2 generations
    ctx_mid = _make_optimizer(problem, seed=42, n_gen=2).run()
    p = tmp_path / "ckpt.npz"
    ctx_mid.save(p)

    ctx_loaded = OptimizationState.load(p, problem)
    ctx_resumed = _make_optimizer(problem, seed=42, n_gen=4).run_from(ctx_loaded)

    np.testing.assert_array_equal(ctx_full.archive.x, ctx_resumed.archive.x)
    np.testing.assert_array_equal(ctx_full.archive.f, ctx_resumed.archive.f)


def test_run_from_pickle_matches_full_run(tmp_path):
    """Resume from pickle checkpoint produces identical results to uninterrupted run."""
    problem = _make_problem()

    ctx_full = _make_optimizer(problem, seed=42, n_gen=4).run()

    opt_mid = _make_optimizer(problem, seed=42, n_gen=2)
    ctx_mid = opt_mid.run()
    p = tmp_path / "ckpt.pkl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        opt_mid.save_pickle(ctx_mid, p)
        opt_resume, ctx_loaded = Optimizer.load_pickle(p)

    opt_resume.set_termination(Termination(max_gen(4)))
    ctx_resumed = opt_resume.run_from(ctx_loaded)

    np.testing.assert_array_equal(ctx_full.archive.x, ctx_resumed.archive.x)
    np.testing.assert_array_equal(ctx_full.archive.f, ctx_resumed.archive.f)


def test_iterate_from_yields_correct_gen(tmp_path):
    problem = _make_problem()
    ctx_mid = _make_optimizer(problem, seed=0, n_gen=2).run()
    p = tmp_path / "ckpt.npz"
    ctx_mid.save(p)

    loaded = OptimizationState.load(p, problem)
    gens = [ctx.gen for ctx in _make_optimizer(problem, n_gen=4).iterate_from(loaded)]
    assert gens[-1] == 4


# ---------------------------------------------------------------------------
# CheckpointCallback
# ---------------------------------------------------------------------------


def test_checkpoint_callback_files_created(tmp_path):
    problem = _make_problem()
    cb = CheckpointCallback(tmp_path / "ckpts", interval=1, format="npz")
    opt = _make_optimizer(problem, seed=0, n_gen=3)
    cb.register(opt.cbmanager)
    opt.run()

    npz_files = sorted((tmp_path / "ckpts").glob("checkpoint_*.npz"))
    assert len(npz_files) == 3


def test_checkpoint_callback_interval(tmp_path):
    problem = _make_problem()
    cb = CheckpointCallback(tmp_path / "ckpts", interval=2, format="npz")
    opt = _make_optimizer(problem, seed=0, n_gen=4)
    cb.register(opt.cbmanager)
    opt.run()

    npz_files = sorted((tmp_path / "ckpts").glob("checkpoint_*.npz"))
    assert len(npz_files) == 2


def test_checkpoint_callback_delete_on_success(tmp_path):
    problem = _make_problem()
    cb = CheckpointCallback(
        tmp_path / "ckpts", interval=1, format="npz", delete_on_success=True
    )
    opt = _make_optimizer(problem, seed=0, n_gen=3)
    cb.register(opt.cbmanager)
    opt.run()

    npz_files = list((tmp_path / "ckpts").glob("checkpoint_*.npz"))
    assert len(npz_files) == 0


def test_checkpoint_callback_both_formats(tmp_path):
    problem = _make_problem()
    opt = _make_optimizer(problem, seed=0, n_gen=2)
    cb = CheckpointCallback(
        tmp_path / "ckpts", interval=1, format="both", optimizer=opt
    )
    cb.register(opt.cbmanager)
    opt.run()

    npz_files = list((tmp_path / "ckpts").glob("checkpoint_*.npz"))
    pkl_files = list((tmp_path / "ckpts").glob("checkpoint_*.pkl"))
    assert len(npz_files) == 2
    assert len(pkl_files) == 2


def test_run_with_checkpoint_path(tmp_path):
    problem = _make_problem()
    opt = _make_optimizer(problem, seed=0, n_gen=3)
    opt.run(checkpoint_path=tmp_path / "ckpts", checkpoint_interval=1)

    npz_files = sorted((tmp_path / "ckpts").glob("checkpoint_*.npz"))
    assert len(npz_files) == 3


def test_checkpoint_callback_invalid_format():
    with pytest.raises(ValueError, match="format must be"):
        CheckpointCallback("/tmp", format="json")


def test_checkpoint_callback_pickle_requires_optimizer():
    with pytest.raises(ValueError, match="optimizer must be provided"):
        CheckpointCallback("/tmp", format="pickle")
