from pathlib import Path

import numpy as np
import pytest

from saealib.exceptions import ValidationError
from saealib.experiment import (
    AlgorithmEntry,
    ExperimentConfig,
    TrialSpec,
    build_termination,
)

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

CONFIG_SOURCE = """
problems:
  - ./problems/sphere.py
algorithms:
  - {name: default}
seeds: [0, 1]
termination: {max_fe: 60}
output_dir: ./results
history_channels: [summary]
"""


@pytest.fixture
def experiment_dir(tmp_path):
    (tmp_path / "problems").mkdir()
    (tmp_path / "problems" / "sphere.py").write_text(PROBLEM_SOURCE, encoding="utf-8")
    (tmp_path / "experiment.yaml").write_text(CONFIG_SOURCE, encoding="utf-8")
    return tmp_path


def test_from_yaml_resolves_paths_against_the_config_file(experiment_dir):
    config = ExperimentConfig.from_yaml(experiment_dir / "experiment.yaml")

    assert config.problems == (experiment_dir / "problems" / "sphere.py",)
    assert config.output_dir == experiment_dir / "results"
    assert config.algorithms == (AlgorithmEntry(name="default", preset=None),)
    assert config.seeds == (0, 1)
    assert config.termination == {"max_fe": 60}
    assert config.n_workers == 1


def test_to_yaml_round_trips(experiment_dir):
    config = ExperimentConfig.from_yaml(experiment_dir / "experiment.yaml")

    written = config.to_yaml(experiment_dir / "copy.yaml")

    assert ExperimentConfig.from_yaml(written) == config


def test_to_yaml_adds_the_suffix_and_writes_relative_paths(experiment_dir):
    config = ExperimentConfig.from_yaml(experiment_dir / "experiment.yaml")

    written = config.to_yaml(experiment_dir / "copy")

    assert written.name == "copy.yaml"
    assert "problems/sphere.py" in written.read_text(encoding="utf-8")


def test_relative_paths_are_absolute_and_round_trip_from_a_snapshot(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(
        problems=[Path("problems/sphere.py")],
        algorithms=[AlgorithmEntry(name="default")],
        seeds=[0],
        output_dir=Path("results"),
        termination={"max_fe": 40},
    )

    assert config.problems == ((tmp_path / "problems/sphere.py").resolve(),)
    assert config.output_dir == (tmp_path / "results").resolve()

    loaded = ExperimentConfig.from_yaml(config.to_yaml(Path("results/config.yaml")))

    assert loaded.problems == config.problems
    assert loaded.output_dir == config.output_dir


def test_relative_path_preset_is_absolute_and_round_trips(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = ExperimentConfig(
        problems=[Path("problems/sphere.py")],
        algorithms=[
            AlgorithmEntry(
                name="custom",
                preset=Path("presets/custom.yaml"),
            )
        ],
        seeds=[0],
        output_dir=Path("results"),
    )

    expected_preset = (tmp_path / "presets/custom.yaml").resolve()
    assert config.algorithms[0].preset == expected_preset

    loaded = ExperimentConfig.from_yaml(config.to_yaml(tmp_path / "config.yaml"))

    assert loaded.algorithms[0].preset == expected_preset
    assert loaded.algorithms == config.algorithms


def test_checkpoint_interval_is_loaded_and_round_trips(experiment_dir):
    source = experiment_dir / "experiment.yaml"
    source.write_text(
        source.read_text(encoding="utf-8") + "checkpoint_interval: 7\n",
        encoding="utf-8",
    )

    config = ExperimentConfig.from_yaml(source)
    written = config.to_yaml(experiment_dir / "copy.yaml")

    assert config.checkpoint_interval == 7
    assert ExperimentConfig.from_yaml(written).checkpoint_interval == 7


def test_a_bare_string_algorithm_names_itself_after_its_preset(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "problems: [a.py]\nalgorithms: [ga_rbf_ib]\nseeds: [0]\noutput_dir: ./o\n",
        encoding="utf-8",
    )

    config = ExperimentConfig.from_yaml(path)

    assert config.algorithms == (AlgorithmEntry(name="ga_rbf_ib", preset="ga_rbf_ib"),)


def test_sequences_are_normalized_to_tuples(tmp_path):
    config = ExperimentConfig(
        problems=[tmp_path / "a.py"],
        algorithms=[AlgorithmEntry("ga")],
        seeds=[0],
        output_dir=tmp_path / "out",
    )

    assert config.problems == (tmp_path / "a.py",)
    assert config.seeds == (0,)


@pytest.mark.parametrize(
    "text, message",
    [
        ("algorithms: []\nseeds: [0]\noutput_dir: ./o\n", "problems"),
        ("problems: [a.py]\nseeds: [0]\noutput_dir: ./o\n", "algorithms"),
        ("problems: [a.py]\nalgorithms: [x]\noutput_dir: ./o\n", "seeds"),
        ("problems: [a.py]\nalgorithms: [x]\nseeds: [0]\n", "output_dir"),
    ],
)
def test_missing_required_keys_are_reported(tmp_path, text, message):
    path = tmp_path / "c.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValidationError, match=message):
        ExperimentConfig.from_yaml(path)


def test_unknown_keys_are_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "problems: [a.py]\nalgorithms: [x]\nseeds: [0]\noutput_dir: ./o\nworkers: 4\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="workers"):
        ExperimentConfig.from_yaml(path)


def test_duplicate_algorithm_names_and_problem_stems_are_rejected(tmp_path):
    with pytest.raises(ValidationError, match="unique"):
        ExperimentConfig(
            problems=(tmp_path / "a.py",),
            algorithms=(AlgorithmEntry("ga"), AlgorithmEntry("ga")),
            seeds=(0,),
            output_dir=tmp_path / "out",
        )
    with pytest.raises(ValidationError, match="unique"):
        ExperimentConfig(
            problems=(tmp_path / "x" / "a.py", tmp_path / "y" / "a.py"),
            algorithms=(AlgorithmEntry("ga"),),
            seeds=(0,),
            output_dir=tmp_path / "out",
        )


def test_resolved_trial_directory_collisions_are_rejected(tmp_path):
    output_dir = tmp_path / "results"

    with pytest.raises(ValidationError, match="collision"):
        ExperimentConfig(
            problems=(tmp_path / "problem.py",),
            algorithms=(AlgorithmEntry("a/../b"), AlgorithmEntry("b")),
            seeds=(0,),
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_algorithm_path_cannot_escape_its_problem_directory(tmp_path):
    with pytest.raises(ValidationError, match="algorithm"):
        ExperimentConfig(
            problems=(tmp_path / "problem.py",),
            algorithms=(AlgorithmEntry("../escape"),),
            seeds=(0,),
            output_dir=tmp_path / "results",
        )


def test_invalid_channel_and_worker_count_are_rejected(tmp_path):
    with pytest.raises(ValidationError, match="channel"):
        ExperimentConfig(
            problems=(tmp_path / "a.py",),
            algorithms=(AlgorithmEntry("ga"),),
            seeds=(0,),
            output_dir=tmp_path / "out",
            history_channels=("nope",),
        )
    with pytest.raises(ValidationError, match="n_workers"):
        ExperimentConfig(
            problems=(tmp_path / "a.py",),
            algorithms=(AlgorithmEntry("ga"),),
            seeds=(0,),
            output_dir=tmp_path / "out",
            n_workers=0,
        )


@pytest.mark.parametrize(
    "indicator",
    [
        "invalid",
        {"type": "not_an_indicator"},
        {"type": "hypervolume", "params": {}},
        {"type": "gd", "params": {}},
    ],
)
def test_invalid_indicator_specs_are_rejected_during_config_validation(
    tmp_path, indicator
):
    with pytest.raises(ValidationError, match=r"indicator|requires"):
        ExperimentConfig(
            problems=(tmp_path / "a.py",),
            algorithms=(AlgorithmEntry("ga"),),
            seeds=(0,),
            output_dir=tmp_path / "out",
            indicator=indicator,
        )


def test_build_termination_combines_conditions_with_or():
    termination = build_termination({"max_fe": 10, "max_gen": 100})

    assert len(termination.conditions) == 2
    with pytest.raises(ValidationError, match="at least one"):
        build_termination({})


def test_build_optimizer_applies_seed_termination_and_channels(experiment_dir):
    config = ExperimentConfig.from_yaml(experiment_dir / "experiment.yaml")
    trial = TrialSpec(config.problems[0], config.algorithms[0], 4)

    optimizer = config.build_optimizer(trial)

    assert optimizer.seed == 4
    assert tuple(optimizer.history_channels) == ("summary",)
    state = optimizer.run()
    assert state.fe == 60


def test_trial_labels_and_directory_describe_the_sweep_point(tmp_path):
    trial = TrialSpec(tmp_path / "zdt1.py", AlgorithmEntry("nsga2"), 7)

    assert trial.labels == {
        "problem": "zdt1",
        "algorithm": "nsga2",
        "seed": "7",
    }
    assert trial.relative_dir.as_posix() == "zdt1/nsga2/seed7"


def test_config_without_termination_leaves_the_problem_files_budget(experiment_dir):
    config = ExperimentConfig(
        problems=(experiment_dir / "problems" / "sphere.py",),
        algorithms=(AlgorithmEntry("default"),),
        seeds=(0,),
        output_dir=experiment_dir / "results",
    )

    optimizer = config.build_optimizer(
        TrialSpec(config.problems[0], config.algorithms[0], 0)
    )

    assert optimizer.seed == 0
    assert np.asarray(optimizer.problem.direction).tolist() == [-1.0]
