import builtins
import importlib.util

import numpy as np
import pytest

from saealib import Optimizer, Problem
from saealib.exceptions import ValidationError
from saealib.experiment import AlgorithmEntry, ExperimentConfig, run_trial
from saealib.experiment._hdf5 import _require_h5py, read_hdf5_trial, write_trial
from saealib.space import PermutationSpace
from saealib.termination import Termination, max_fe

requires_h5py = pytest.mark.skipif(
    importlib.util.find_spec("h5py") is None, reason="h5py is not installed"
)


def test_experiment_exports_only_stable_public_api():
    import saealib.experiment as experiment

    assert set(experiment.__all__) == {
        "AlgorithmEntry",
        "ExperimentConfig",
        "ExperimentRunner",
        "build_termination",
        "execute_trial",
        "latest_checkpoint",
        "ProgressReporter",
        "read_hdf5_trial",
        "RichProgress",
        "RunResult",
        "SilentProgress",
        "TqdmProgress",
        "TrialSpec",
        "resume_trial",
        "run_trial",
        "write_aggregate",
        "write_summary",
    }
    assert not hasattr(experiment, "read_trial")


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


@requires_h5py
def test_hdf5_trial_writes_hdf5_and_meta_only(tmp_path):
    out = tmp_path / "run"

    run_trial(_sphere_optimizer(), seed=3, output_dir=out, result_format="hdf5")

    assert (out / "trial.h5").exists()
    assert (out / "meta.json").exists()
    assert not (out / "archive.csv").exists()
    assert not (out / "history.csv").exists()


@requires_h5py
def test_hdf5_trial_records_the_optimizers_own_seed_when_none_is_given(tmp_path):
    out = tmp_path / "run"

    result = run_trial(_sphere_optimizer(seed=5), output_dir=out, result_format="hdf5")

    assert result.seed == 5
    assert read_hdf5_trial(out / "trial.h5")["meta"]["seed"] == 5


@requires_h5py
def test_hdf5_preserves_variable_length_history_blocks(tmp_path):
    out = tmp_path / "run"
    optimizer = _sphere_optimizer(channels=("summary", "front", "population"))
    optimizer.set_seed(3)
    state = optimizer.run()
    write_trial(out / "trial.h5", state, seed=3, wall_time=0.0, labels={})
    loaded = read_hdf5_trial(out / "trial.h5")
    for channel, column in (("front", "f"), ("population", "x")):
        expected_blocks = state.history.blocks(channel, column)
        actual_blocks = loaded["history"][channel][column]
        assert len(actual_blocks) == len(expected_blocks)
        for expected, actual in zip(
            expected_blocks,
            actual_blocks,
        ):
            np.testing.assert_array_equal(expected, actual)


@requires_h5py
def test_hdf5_omits_x_for_non_dense_space(tmp_path):
    space = PermutationSpace(8)
    from saealib import Optimizer, Problem
    from saealib.algorithms import GenomeGA
    from saealib.execution.initializer import GenomeInitializer
    from saealib.operators import OrderCrossover, SequentialSelection, SwapMutation
    from saealib.operators.selection import TruncationSelection
    from saealib.strategies import DirectStrategy
    from saealib.termination import Termination, max_gen

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
    run_trial(optimizer, output_dir=tmp_path / "run", result_format="hdf5")
    assert "x" not in read_hdf5_trial(tmp_path / "run" / "trial.h5")["archive"]


def test_hdf5_missing_dependency_message(monkeypatch):
    original_import = builtins.__import__

    def fail_h5py(name, *args, **kwargs):
        if name == "h5py":
            raise ModuleNotFoundError("h5py")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_h5py)
    with pytest.raises(ImportError, match=r"pip install saealib\[hdf5\]"):
        _require_h5py()


def test_invalid_result_format_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="result_format"):
        ExperimentConfig(
            problems=(tmp_path / "a.py",),
            algorithms=(AlgorithmEntry("ga"),),
            seeds=(0,),
            output_dir=tmp_path / "out",
            result_format="json",
        )


@requires_h5py
def test_hdf5_metadata_round_trips_as_json_types(tmp_path):
    import json

    out = tmp_path / "run"
    run_trial(
        _sphere_optimizer(),
        seed=3,
        output_dir=out,
        labels={"problem": "sphere", "algorithm": "ga"},
        result_format="hdf5",
    )

    attributes = read_hdf5_trial(out / "trial.h5")["meta"]
    sidecar = json.loads((out / "meta.json").read_text(encoding="utf-8"))

    assert attributes["labels"] == {"problem": "sphere", "algorithm": "ga"}
    assert list(attributes["direction"]) == [-1.0]
    assert list(attributes["history_channels"]) == ["summary"]
    for name in set(sidecar) - {"finished_at"}:
        value = attributes[name]
        assert (list(value) if isinstance(value, (list, tuple)) else value) == sidecar[
            name
        ], name


def test_run_trial_rejects_an_unknown_result_format_before_running(tmp_path):
    calls = []

    def counted(x):
        calls.append(1)
        return float(np.sum(x**2))

    problem = Problem(
        func=counted,
        dim=3,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-2.0] * 3,
        ub=[2.0] * 3,
    )
    optimizer = Optimizer(problem, seed=0).set_termination(Termination(max_fe(20)))

    with pytest.raises(ValidationError, match="result_format"):
        run_trial(optimizer, output_dir=tmp_path, result_format="parquet")

    assert calls == []
