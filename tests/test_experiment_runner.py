import dataclasses
import json

import pytest

from saealib.exceptions import ValidationError
from saealib.experiment import AlgorithmEntry, ExperimentConfig, ExperimentRunner

PROBLEM_SOURCE = """
import numpy as np
from saealib import Problem

problem = Problem(
    func=lambda x: float(np.sum(x**2)),
    dim=3,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=[-2.0] * 3,
    ub=[2.0] * 3,
)
"""

BROKEN_SOURCE = "raise RuntimeError('problem file is broken')\n"


def _config(
    tmp_path, *, sources=(("sphere", PROBLEM_SOURCE),), seeds=(0, 1), workers=1
):
    problems = tmp_path / "problems"
    problems.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, source in sources:
        path = problems / f"{name}.py"
        path.write_text(source, encoding="utf-8")
        paths.append(path)
    return ExperimentConfig(
        problems=paths,
        algorithms=[AlgorithmEntry("default")],
        seeds=seeds,
        output_dir=tmp_path / "results",
        termination={"max_fe": 60},
        n_workers=workers,
    )


def test_trials_expand_the_product_in_problem_major_order(tmp_path):
    config = dataclasses.replace(
        _config(
            tmp_path,
            sources=(("a", PROBLEM_SOURCE), ("b", PROBLEM_SOURCE)),
        ),
        algorithms=[AlgorithmEntry("x"), AlgorithmEntry("y")],
        seeds=(0, 1),
    )

    coordinates = [
        (trial.problem.stem, trial.algorithm.name, trial.seed)
        for trial in config.trials()
    ]

    assert coordinates == [
        ("a", "x", 0),
        ("a", "x", 1),
        ("a", "y", 0),
        ("a", "y", 1),
        ("b", "x", 0),
        ("b", "x", 1),
        ("b", "y", 0),
        ("b", "y", 1),
    ]


def test_run_writes_the_layout_and_a_config_snapshot(tmp_path):
    config = _config(tmp_path)

    results = ExperimentRunner(config).run()

    assert [result.labels["seed"] for result in results] == ["0", "1"]
    root = tmp_path / "results"
    assert (root / "config.yaml").exists()
    for seed in (0, 1):
        trial_dir = root / "sphere" / "default" / f"seed{seed}"
        assert (trial_dir / "archive.csv").exists()
        assert (trial_dir / "history.csv").exists()
        meta = json.loads((trial_dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["labels"]["problem"] == "sphere"
        assert meta["seed"] == seed


def test_the_config_snapshot_reloads(tmp_path):
    config = _config(tmp_path)

    ExperimentRunner(config).run()

    reloaded = ExperimentConfig.from_yaml(tmp_path / "results" / "config.yaml")
    assert [path.stem for path in reloaded.problems] == ["sphere"]
    assert reloaded.seeds == (0, 1)


def test_parallel_execution_matches_sequential(tmp_path):
    sequential = ExperimentRunner(_config(tmp_path / "seq")).run()
    parallel = ExperimentRunner(_config(tmp_path / "par", workers=2)).run()

    assert [result.labels for result in parallel] == [
        result.labels for result in sequential
    ]
    for expected, actual in zip(sequential, parallel, strict=True):
        assert expected.best_f.tolist() == actual.best_f.tolist()
        assert expected.fe == actual.fe


def test_a_failing_trial_propagates(tmp_path):
    config = _config(tmp_path, sources=(("broken", BROKEN_SOURCE),), seeds=(0,))

    with pytest.raises(RuntimeError, match="broken"):
        ExperimentRunner(config).run()


@pytest.mark.parametrize("name", ["../../escape", "/etc/evil"])
def test_trial_paths_cannot_escape_experiment_root(tmp_path, name):
    with pytest.raises(ValidationError, match=name.replace("/", r"/")) as exc_info:
        dataclasses.replace(
            _config(tmp_path, seeds=(0,)),
            algorithms=[AlgorithmEntry(name)],
        )

    assert "algorithm" in str(exc_info.value)
    assert not (tmp_path / "results").exists()


def test_trial_path_validation_precedes_every_trial(tmp_path):
    output_dir = tmp_path / "results"

    with pytest.raises(ValidationError):
        dataclasses.replace(
            _config(tmp_path, seeds=(0,)),
            algorithms=[AlgorithmEntry("default"), AlgorithmEntry("../../escape")],
        )

    assert not output_dir.exists()


def test_existing_trials_require_overwrite_and_overwrite_removes_old_files(tmp_path):
    config = _config(tmp_path, seeds=(0,))
    ExperimentRunner(config).run()
    trial_dir = config.output_dir / "sphere" / "default" / "seed0"
    stale = trial_dir / "checkpoint" / "checkpoint_999999.npz"
    stale.parent.mkdir()
    stale.write_bytes(b"stale")

    with pytest.raises(ValidationError, match=r"1 existing.*--overwrite"):
        ExperimentRunner(config).run()
    assert stale.exists()

    ExperimentRunner(config, overwrite=True).run()
    assert not stale.exists()


def test_existing_trial_detection_precedes_every_trial(tmp_path):
    config = _config(tmp_path, seeds=(0, 1))
    existing = config.output_dir / "sphere" / "default" / "seed1"
    existing.mkdir(parents=True)
    (existing / "meta.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValidationError, match=r"1 existing.*--overwrite"):
        ExperimentRunner(config).run()

    assert not (config.output_dir / "sphere" / "default" / "seed0").exists()


def test_empty_trial_directory_is_allowed(tmp_path):
    config = _config(tmp_path, seeds=(0,))
    trial_dir = config.output_dir / "sphere" / "default" / "seed0"
    trial_dir.mkdir(parents=True)

    ExperimentRunner(config).run()

    assert (trial_dir / "meta.json").exists()


def test_nonempty_trial_directory_requires_overwrite_even_without_meta(tmp_path):
    config = _config(tmp_path, seeds=(0,))
    trial_dir = config.output_dir / "sphere" / "default" / "seed0"
    trial_dir.mkdir(parents=True)
    stale = trial_dir / "stale.txt"
    stale.write_text("stale", encoding="utf-8")

    with pytest.raises(ValidationError, match=r"1 existing.*--overwrite"):
        ExperimentRunner(config).run()
    assert stale.exists()

    ExperimentRunner(config, overwrite=True).run()
    assert not stale.exists()
