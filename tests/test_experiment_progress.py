from __future__ import annotations

import sys

import pytest

from saealib.experiment import (
    AlgorithmEntry,
    ExperimentConfig,
    ExperimentRunner,
    ProgressReporter,
    RichProgress,
    SilentProgress,
    TqdmProgress,
)

PROBLEM_SOURCE = """
import numpy as np
from saealib import Problem

problem = Problem(
    func=lambda x: float(np.sum(x**2)),
    dim=2,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=[-1.0] * 2,
    ub=[1.0] * 2,
)
"""


def _config(tmp_path, *, workers=1, seeds=(0, 1)):
    tmp_path.mkdir(parents=True, exist_ok=True)
    problem = tmp_path / "sphere.py"
    problem.write_text(PROBLEM_SOURCE, encoding="utf-8")
    return ExperimentConfig(
        problems=[problem],
        algorithms=[AlgorithmEntry("default")],
        seeds=seeds,
        output_dir=tmp_path / "results",
        termination={"max_fe": 10},
        n_workers=workers,
    )


class RecordingProgress(ProgressReporter):
    def __init__(self):
        self.started = []
        self.advanced = []
        self.finished = 0

    def start(self, total):
        self.started.append(total)

    def advance(self, result):
        self.advanced.append(result)

    def finish(self):
        self.finished += 1


def test_default_progress_is_silent(tmp_path, capsys):
    ExperimentRunner(_config(tmp_path, seeds=(0,))).run()

    assert capsys.readouterr().out == ""
    assert isinstance(
        ExperimentRunner(_config(tmp_path / "other")).progress, SilentProgress
    )


def test_custom_progress_receives_each_trial(tmp_path):
    progress = RecordingProgress()

    results = ExperimentRunner(_config(tmp_path, seeds=(0, 1, 2)), progress).run()

    assert progress.started == [3]
    assert len(progress.advanced) == len(results) == 3
    assert progress.finished == 1


def test_progress_finishes_when_a_trial_raises(tmp_path):
    problem = tmp_path / "broken.py"
    problem.write_text("raise RuntimeError('broken')\n", encoding="utf-8")
    config = ExperimentConfig(
        problems=[problem],
        algorithms=[AlgorithmEntry("default")],
        seeds=[0],
        output_dir=tmp_path / "results",
    )
    progress = RecordingProgress()

    with pytest.raises(RuntimeError, match="broken"):
        ExperimentRunner(config, progress).run()

    assert progress.finished == 1


def test_parallel_results_remain_in_sweep_order(tmp_path):
    results = ExperimentRunner(_config(tmp_path, workers=2, seeds=(2, 0, 1))).run()

    assert [result.labels["seed"] for result in results] == ["2", "0", "1"]


@pytest.mark.parametrize(
    ("progress_type", "module_name", "extra"),
    [
        (TqdmProgress, "tqdm", "tqdm"),
        (RichProgress, "rich", "rich"),
    ],
)
def test_optional_progress_reports_actionable_import_error(
    monkeypatch, progress_type, module_name, extra
):
    monkeypatch.setitem(sys.modules, module_name, None)

    with pytest.raises(ImportError, match=f"pip install saealib\\[{extra}\\]"):
        progress_type()


def test_progress_reporter_is_abstract():
    with pytest.raises(TypeError):
        ProgressReporter()


SLOW_PROBLEM_SOURCE = """
import time

import numpy as np
from saealib import Problem


def slow(x):
    time.sleep(0.02)
    return float(np.sum(x**2))


problem = Problem(
    func=slow,
    dim=3,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=[-2.0] * 3,
    ub=[2.0] * 3,
)
"""

FAST_PROBLEM_SOURCE = """
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


def test_parallel_results_keep_sweep_order_when_trials_finish_out_of_order(tmp_path):
    problems = tmp_path / "problems"
    problems.mkdir()
    (problems / "aslow.py").write_text(SLOW_PROBLEM_SOURCE, encoding="utf-8")
    (problems / "bfast.py").write_text(FAST_PROBLEM_SOURCE, encoding="utf-8")
    config = ExperimentConfig(
        problems=[problems / "aslow.py", problems / "bfast.py"],
        algorithms=[AlgorithmEntry("d")],
        seeds=[0],
        output_dir=tmp_path / "out",
        termination={"max_fe": 60},
        n_workers=2,
    )
    recorder = RecordingProgress()

    results = ExperimentRunner(config, recorder).run()

    sweep_order = [(trial.problem.stem, str(trial.seed)) for trial in config.trials()]
    assert [
        (result.labels["problem"], result.labels["seed"]) for result in results
    ] == sweep_order
    completion_order = [
        (result.labels["problem"], result.labels["seed"])
        for result in recorder.advanced
    ]
    assert completion_order != sweep_order
