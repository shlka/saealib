import csv
import json

import numpy as np

from saealib import Optimizer, Problem
from saealib.algorithms import GenomeGA
from saealib.execution.initializer import GenomeInitializer
from saealib.experiment import RunResult, run_trial
from saealib.operators import OrderCrossover, SequentialSelection, SwapMutation
from saealib.operators.selection import TruncationSelection
from saealib.space import PermutationSpace
from saealib.strategies import DirectStrategy
from saealib.termination import Termination, max_fe, max_gen


def _sphere_optimizer(channels=("summary",), seed=None):
    problem = Problem(
        func=lambda x: float(np.sum(x**2)),
        dim=3,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-2.0] * 3,
        ub=[2.0] * 3,
    )
    return (
        Optimizer(problem, seed=seed)
        .set_termination(Termination(max_fe(60)))
        .set_history(list(channels))
    )


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def test_run_trial_writes_archive_history_and_meta(tmp_path):
    out = tmp_path / "run_seed3"

    result = run_trial(_sphere_optimizer(), seed=3, output_dir=out)

    assert isinstance(result, RunResult)
    assert result.output_dir == out
    assert result.seed == 3
    assert result.wall_time > 0.0

    archive = _read_csv(out / "archive.csv")
    assert archive[0] == ["id", "x0", "x1", "x2", "f0", "cv"]

    header, *rows = _read_csv(out / "history.csv")
    assert "best_f" in header and "fe" in header
    assert len(rows) == result.gen + 1

    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 3
    assert meta["fe"] == result.fe
    assert meta["gen"] == result.gen
    assert meta["history_channels"] == ["summary"]
    assert len(archive) - 1 == meta["archive_size"]


def test_run_trial_carries_labels_onto_the_result_and_meta(tmp_path):
    out = tmp_path / "run"

    result = run_trial(
        _sphere_optimizer(),
        seed=1,
        output_dir=out,
        labels={"problem": "sphere", "algorithm": "ga"},
    )

    assert result.labels == {"problem": "sphere", "algorithm": "ga"}
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["labels"] == {"problem": "sphere", "algorithm": "ga"}


def test_run_trial_defaults_to_empty_labels():
    assert run_trial(_sphere_optimizer(), seed=1).labels == {}


def test_run_trial_without_output_dir_writes_nothing(tmp_path):
    result = run_trial(_sphere_optimizer(), seed=1)

    assert result.output_dir is None
    assert list(tmp_path.iterdir()) == []


def test_run_trial_omits_history_file_when_the_channel_is_off(tmp_path):
    out = tmp_path / "run"

    run_trial(_sphere_optimizer(channels=()), seed=1, output_dir=out)

    assert (out / "archive.csv").exists()
    assert (out / "meta.json").exists()
    assert not (out / "history.csv").exists()
    assert (
        json.loads((out / "meta.json").read_text(encoding="utf-8"))["history_channels"]
        == []
    )


def test_run_trial_writes_checkpoints_at_the_requested_interval(tmp_path):
    out = tmp_path / "run"

    result = run_trial(
        _sphere_optimizer(), seed=1, output_dir=out, checkpoint_interval=2
    )

    checkpoints = sorted((out / "checkpoint").glob("checkpoint_*.npz"))
    generations = [int(path.stem.rsplit("_", 1)[1]) for path in checkpoints]
    assert generations == list(range(2, result.gen + 1, 2))


def test_run_trial_seed_argument_drives_reproducibility(tmp_path):
    first = run_trial(_sphere_optimizer(), seed=7)
    same = run_trial(_sphere_optimizer(), seed=7)
    other = run_trial(_sphere_optimizer(), seed=8)

    np.testing.assert_allclose(first.best_f, same.best_f)
    assert not np.allclose(first.best_f, other.best_f)


def test_run_trial_records_the_optimizers_own_seed_when_none_is_given(tmp_path):
    out = tmp_path / "run"

    result = run_trial(_sphere_optimizer(seed=5), output_dir=out)

    assert result.seed == 5
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 5


def test_run_trial_on_a_non_dense_space_writes_an_archive_without_x(tmp_path):
    space = PermutationSpace(8)
    problem = Problem(
        func=lambda x: np.asarray([float(sum(i * v for i, v in enumerate(x)))]),
        dim=space.dim,
        n_obj=1,
        direction=np.array([-1.0]),
        space=space,
    )
    optimizer = (
        Optimizer(problem, seed=1)
        .set_algorithm(
            GenomeGA(
                OrderCrossover(),
                SwapMutation(),
                SequentialSelection(),
                TruncationSelection(),
            )
        )
        .set_strategy(DirectStrategy())
        .set_initializer(GenomeInitializer(24, 24))
        .set_termination(Termination(max_gen(3)))
        .set_history(["summary"])
    )
    out = tmp_path / "permutation"

    result = run_trial(optimizer, output_dir=out)

    assert result.best_x is None
    header = _read_csv(out / "archive.csv")[0]
    assert not any(name.startswith("x") for name in header)
    assert "f0" in header
