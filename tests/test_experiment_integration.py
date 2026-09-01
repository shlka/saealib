import csv
import json
import statistics
from pathlib import Path

import numpy as np
import pytest

from saealib.algorithms import GA, PSO
from saealib.experiment import (
    AlgorithmEntry,
    ExperimentConfig,
    ExperimentRunner,
    resume_trial,
)

PROBLEM1_SOURCE = """
import numpy as np
from saealib import Problem

problem = Problem(
    func=lambda x: float(np.sum(x**2)),
    dim=2,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=[-2.0] * 2,
    ub=[2.0] * 2,
)
"""

PROBLEM2_SOURCE = """
import numpy as np
from saealib import Problem

problem = Problem(
    func=lambda x: float(np.sum((x - 1.0) ** 2)),
    dim=2,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=[-2.0] * 2,
    ub=[2.0] * 2,
)
"""

YAML_SOURCE_TEMPLATE = """
problems:
  - ./problems/sphere.py
  - ./problems/shifted.py
algorithms:
  - name: ga
  - name: pso
    preset:
      algorithm:
        type: PSO
seeds: [0, 1, 2]
termination: {{max_fe: 20}}
output_dir: {output_dir}
n_workers: {workers}
"""


def _read_csv(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def _assert_experiment_layout_and_contents(
    config: ExperimentConfig, results, root: Path
) -> None:
    expected_trials = config.trials()
    assert len(results) == 12
    assert len(expected_trials) == 12

    assert (root / "config.yaml").exists()
    assert (root / "summary.csv").exists()
    assert (root / "aggregate.json").exists()

    expected_trial_dirs = {trial.relative_dir for trial in expected_trials}
    actual_trial_dirs = {
        path.parent.relative_to(root) for path in root.rglob("meta.json")
    }
    assert actual_trial_dirs == expected_trial_dirs
    assert len(actual_trial_dirs) == 12
    assert {path.as_posix() for path in actual_trial_dirs} == {
        f"{problem}/{algorithm}/seed{seed}"
        for problem in ("sphere", "shifted")
        for algorithm in ("ga", "pso")
        for seed in (0, 1, 2)
    }

    for trial, result in zip(expected_trials, results, strict=True):
        assert result.labels == trial.labels
        assert result.seed == trial.seed
        trial_dir = root / trial.relative_dir
        assert trial_dir.is_dir()
        assert (trial_dir / "archive.csv").exists()
        assert (trial_dir / "history.csv").exists()
        assert (trial_dir / "meta.json").exists()

        meta = json.loads((trial_dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["labels"] == trial.labels
        assert meta["seed"] == trial.seed
        assert "fe" in meta
        assert "wall_time" in meta

    summary_rows = _read_csv(root / "summary.csv")
    assert summary_rows[0] == [
        "problem",
        "algorithm",
        "seed",
        "fe",
        "gen",
        "wall_time",
        "best_f",
    ]
    data_rows = summary_rows[1:]
    assert len(data_rows) == 12

    summary_tuples = [(row[0], row[1], int(row[2])) for row in data_rows]
    expected_tuples = [
        (trial.problem.stem, trial.algorithm.name, trial.seed)
        for trial in expected_trials
    ]
    assert summary_tuples == expected_tuples
    assert len(set(summary_tuples)) == 12

    aggregate = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
    assert aggregate["metric"] == "best_f"
    assert "orientation" not in aggregate
    groups = aggregate["groups"]
    assert len(groups) == 4
    assert {(group["problem"], group["algorithm"]) for group in groups} == {
        (trial.problem.stem, trial.algorithm.name) for trial in expected_trials
    }

    for group in groups:
        assert group["orientation"] == "min"
        assert group["n_trials"] == 3
        prob_name = group["problem"]
        algo_name = group["algorithm"]
        matching_values = [
            float(row[6])
            for row in data_rows
            if row[0] == prob_name and row[1] == algo_name
        ]
        assert len(matching_values) == 3
        expected_median = statistics.median(matching_values)
        assert np.isclose(group["median"], expected_median)
        assert np.isclose(group["best"], min(matching_values))
        assert np.isclose(group["worst"], max(matching_values))


def test_experiment_integration(tmp_path: Path) -> None:
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir(parents=True, exist_ok=True)
    (problems_dir / "sphere.py").write_text(PROBLEM1_SOURCE, encoding="utf-8")
    (problems_dir / "shifted.py").write_text(PROBLEM2_SOURCE, encoding="utf-8")

    seq_root = tmp_path / "results_seq"
    seq_yaml_path = tmp_path / "config_seq.yaml"
    seq_yaml_path.write_text(
        YAML_SOURCE_TEMPLATE.format(
            output_dir="./results_seq",
            workers=1,
        ),
        encoding="utf-8",
    )

    par_root = tmp_path / "results_par"
    par_yaml_path = tmp_path / "config_par.yaml"
    par_yaml_path.write_text(
        YAML_SOURCE_TEMPLATE.format(
            output_dir="./results_par",
            workers=2,
        ),
        encoding="utf-8",
    )

    seq_config = ExperimentConfig.from_yaml(seq_yaml_path)
    par_config = ExperimentConfig.from_yaml(par_yaml_path)

    trials = seq_config.trials()
    ga_trial = next(t for t in trials if t.algorithm.name == "ga")
    pso_trial = next(t for t in trials if t.algorithm.name == "pso")

    opt_ga = seq_config.build_optimizer(ga_trial)
    opt_pso = seq_config.build_optimizer(pso_trial)
    opt_ga.resolve_defaults()
    opt_pso.resolve_defaults()

    assert isinstance(opt_ga.algorithm, GA)
    assert isinstance(opt_pso.algorithm, PSO)
    assert type(opt_ga.algorithm) is not type(opt_pso.algorithm)

    seq_results = ExperimentRunner(seq_config).run()
    _assert_experiment_layout_and_contents(seq_config, seq_results, seq_root)

    par_results = ExperimentRunner(par_config).run()
    _assert_experiment_layout_and_contents(par_config, par_results, par_root)

    seq_summary = _read_csv(seq_root / "summary.csv")
    par_summary = _read_csv(par_root / "summary.csv")

    assert len(seq_summary) == len(par_summary) == 13
    assert seq_summary[0] == par_summary[0]

    for seq_row, par_row in zip(seq_summary[1:], par_summary[1:], strict=True):
        assert seq_row[:5] == par_row[:5]
        assert float(seq_row[6]) == pytest.approx(float(par_row[6]))


def test_relative_config_snapshot_can_resume_a_trial(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    problems_dir = Path("problems")
    problems_dir.mkdir()
    problem_path = problems_dir / "sphere.py"
    problem_path.write_text(PROBLEM1_SOURCE, encoding="utf-8")

    config = ExperimentConfig(
        problems=[Path("problems/sphere.py")],
        algorithms=[AlgorithmEntry("default")],
        seeds=[0],
        output_dir=Path("results"),
        termination={"max_fe": 20},
        checkpoint_interval=1,
    )

    results = ExperimentRunner(config).run()
    snapshot = ExperimentConfig.from_yaml(Path("results/config.yaml"))

    assert len(results) == 1
    assert snapshot.problems == config.problems
    assert snapshot.output_dir == config.output_dir
    resumed = resume_trial(snapshot, snapshot.trials()[0])

    assert resumed.output_dir == (tmp_path / "results/sphere/default/seed0").resolve()


def test_mixed_minimize_maximize_sweep_writes_group_orientations(
    tmp_path: Path,
) -> None:
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    minimize_path = problems_dir / "minimize.py"
    maximize_path = problems_dir / "maximize.py"
    minimize_path.write_text(PROBLEM1_SOURCE, encoding="utf-8")
    maximize_path.write_text(
        PROBLEM1_SOURCE.replace(
            "direction=np.array([-1.0])", "direction=np.array([1.0])"
        ),
        encoding="utf-8",
    )
    config = ExperimentConfig(
        problems=(minimize_path, maximize_path),
        algorithms=(AlgorithmEntry("ga"),),
        seeds=(0, 1),
        output_dir=tmp_path / "results",
        termination={"max_fe": 20},
    )

    results = ExperimentRunner(config).run()

    aggregate = json.loads(
        (config.output_dir / "aggregate.json").read_text(encoding="utf-8")
    )
    assert len(results) == 4
    assert "orientation" not in aggregate
    assert {
        (group["problem"], group["orientation"]) for group in aggregate["groups"]
    } == {("minimize", "min"), ("maximize", "max")}
