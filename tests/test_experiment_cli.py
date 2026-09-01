import json
import subprocess
import sys

import pytest

from saealib.exceptions import ValidationError
from saealib.experiment import ExperimentConfig
from saealib.experiment._cli import main
from saealib.experiment._runner import ExperimentRunner

PROBLEM = """
import numpy as np
from saealib import Problem
problem = Problem(func=lambda x: float(np.sum(x**2)), dim=2, n_obj=1,
                  direction=np.array([-1.0]), lb=[-2.0, -2.0], ub=[2.0, 2.0])
"""


def _write_config(tmp_path, max_fe=60, workers=1):
    problem = tmp_path / "sphere.py"
    problem.write_text(PROBLEM, encoding="utf-8")
    config = tmp_path / "experiment.yaml"
    config.write_text(
        f"problems: [sphere.py]\nalgorithms: [{{name: default}}]\nseeds: [3]\n"
        f"output_dir: results\ntermination: {{max_fe: {max_fe}}}\n"
        f"n_workers: {workers}\ncheckpoint_interval: 1\n",
        encoding="utf-8",
    )
    return config


def test_run_cli_writes_summary_and_trial_files(tmp_path):
    config = _write_config(tmp_path)

    assert main(["run", "--config", str(config)]) == 0
    root = tmp_path / "results"
    assert (root / "summary.csv").exists()
    assert (root / "sphere" / "default" / "seed3" / "meta.json").exists()


def test_run_cli_workers_override(tmp_path):
    config = _write_config(tmp_path, workers=1)

    assert main(["run", "--config", str(config), "--workers", "2"]) == 0
    assert (
        ExperimentConfig.from_yaml(tmp_path / "results" / "config.yaml").n_workers == 2
    )


def test_run_cli_overwrite_allows_explicit_rerun(tmp_path):
    config = _write_config(tmp_path)

    assert main(["run", "--config", str(config)]) == 0
    assert main(["run", "--config", str(config)]) == 2
    assert main(["run", "--config", str(config), "--overwrite"]) == 0


def test_run_cli_configuration_errors_return_two(tmp_path):
    assert main(["run", "--config", str(tmp_path / "missing.yaml")]) == 2
    config = _write_config(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8") + "unknown: true\n", encoding="utf-8"
    )
    assert main(["run", "--config", str(config)]) == 2


def test_run_cli_trial_errors_return_one(tmp_path, monkeypatch):
    config = _write_config(tmp_path)

    def fail_run(self):
        raise ValueError("trial failed")

    monkeypatch.setattr(ExperimentRunner, "run", fail_run)

    assert main(["run", "--config", str(config)]) == 1


def test_resume_cli_continues_from_checkpoint(tmp_path):
    config = _write_config(tmp_path, max_fe=20)
    assert main(["run", "--config", str(config)]) == 0
    trial_dir = tmp_path / "results" / "sphere" / "default" / "seed3"
    before = json.loads((trial_dir / "meta.json").read_text(encoding="utf-8"))["fe"]
    snapshot = tmp_path / "results" / "config.yaml"
    text = snapshot.read_text(encoding="utf-8").replace("max_fe: 20", "max_fe: 60")
    snapshot.write_text(text, encoding="utf-8")

    assert main(["resume", "--run", str(trial_dir)]) == 0
    after = json.loads((trial_dir / "meta.json").read_text(encoding="utf-8"))["fe"]
    assert after > before


def test_resume_cli_works_in_a_subprocess(tmp_path):
    config = _write_config(tmp_path)
    trial_dir = tmp_path / "results" / "sphere" / "default" / "seed3"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "saealib.experiment._cli",
            "run",
            "--config",
            str(config),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert (trial_dir / "archive.csv").exists()


def test_checkpoint_interval_requires_output_dir():
    import numpy as np

    from saealib import Optimizer, Problem

    optimizer = Optimizer(
        Problem(
            func=lambda x: float(np.sum(x**2)),
            dim=1,
            n_obj=1,
            direction=np.array([-1.0]),
            lb=[-1.0],
            ub=[1.0],
        )
    )
    from saealib.experiment import run_trial

    try:
        run_trial(optimizer, checkpoint_interval=1)
    except ValidationError:
        pass
    else:
        raise AssertionError("expected ValidationError")


def test_latest_checkpoint_orders_by_generation_not_string_length(tmp_path):
    from saealib.experiment._runner import latest_checkpoint

    for generation in (5, 45, 123, 999999, 1000000):
        (tmp_path / f"checkpoint_{generation:06d}.npz").write_bytes(b"")

    assert latest_checkpoint(tmp_path).name == "checkpoint_1000000.npz"

    with pytest.raises(ValidationError, match="No checkpoint"):
        latest_checkpoint(tmp_path / "empty")
