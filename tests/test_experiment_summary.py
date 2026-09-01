import csv
import json

import numpy as np
import pytest

from saealib.exceptions import ValidationError
from saealib.experiment import RunResult
from saealib.experiment._summary import write_aggregate, write_summary
from saealib.utils import hypervolume


def _result(seed, value, direction=(-1.0,), labels=None):
    return RunResult(
        seed=seed,
        fe=10,
        gen=2,
        best_f=np.asarray(value, dtype=float),
        best_x=None,
        wall_time=0.25,
        output_dir=None,
        labels=labels
        or {"problem": "sphere", "algorithm": "default", "seed": str(seed)},
        direction=np.asarray(direction, dtype=float),
    )


def _rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def test_single_objective_summary_and_aggregate(tmp_path):
    results = [_result(seed, [value]) for seed, value in enumerate((1.0, 3.0, 2.0))]

    summary = write_summary(results, tmp_path)
    aggregate = write_aggregate(summary, results)

    rows = _rows(summary)
    assert rows[0] == [
        "problem",
        "algorithm",
        "seed",
        "fe",
        "gen",
        "wall_time",
        "best_f",
    ]
    assert rows[1:] == [
        ["sphere", "default", "0", "10", "2", "0.25", "1.0"],
        ["sphere", "default", "1", "10", "2", "0.25", "3.0"],
        ["sphere", "default", "2", "10", "2", "0.25", "2.0"],
    ]
    assert aggregate is not None
    data = json.loads(aggregate.read_text(encoding="utf-8"))
    assert data["metric"] == "best_f"
    assert "orientation" not in data
    assert data["groups"][0]["orientation"] == "min"
    group = data["groups"][0]
    assert (group["median"], group["iqr"], group["best"], group["worst"]) == (
        2.0,
        1.0,
        1.0,
        3.0,
    )


def test_maximization_uses_max_as_best(tmp_path):
    results = [
        _result(seed, [value], direction=(1.0,))
        for seed, value in enumerate((2.0, 9.0, 4.0))
    ]

    summary = write_summary(results, tmp_path)
    aggregate = write_aggregate(summary, results)
    assert aggregate is not None
    data = json.loads(aggregate.read_text(encoding="utf-8"))

    assert "orientation" not in data
    assert data["groups"][0]["orientation"] == "max"
    assert data["groups"][0]["best"] == 9.0
    assert data["groups"][0]["worst"] == 2.0


def test_hypervolume_uses_minimize_space_for_front_and_reference(tmp_path):
    front = np.array([[2.0, 8.0], [6.0, 3.0]])
    reference = np.array([0.0, 10.0])
    result = _result(0, front, direction=(1.0, -1.0))
    indicator = {
        "type": "hypervolume",
        "params": {"reference_point": reference.tolist()},
    }

    summary = write_summary([result], tmp_path, indicator)
    rows = _rows(summary)

    assert rows[0][-1] == "hypervolume"
    assert float(rows[1][-1]) == hypervolume(
        front * np.array([-1.0, 1.0]), reference * np.array([-1.0, 1.0])
    )


def test_multi_objective_without_indicator_has_no_metric_or_aggregate(tmp_path):
    result = _result(0, [[1.0, 2.0], [2.0, 1.0]], direction=(-1.0, -1.0))

    summary = write_summary([result], tmp_path)

    assert _rows(summary)[0] == [
        "problem",
        "algorithm",
        "seed",
        "fe",
        "gen",
        "wall_time",
    ]
    assert write_aggregate(summary, [result]) is None
    assert not (tmp_path / "aggregate.json").exists()


def test_spacing_passes_non_reference_params_to_indicator(tmp_path):
    from saealib.utils import spacing

    result = _result(
        0,
        [[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]],
        direction=(-1.0, -1.0),
    )
    indicator = {"type": "spacing", "params": {"squared": True}}

    summary = write_summary([result], tmp_path, indicator)
    value = float(_rows(summary)[1][-1])

    assert value == spacing(result.best_f, squared=True)


def test_aggregate_orientation_is_stored_per_group(tmp_path):
    results = [
        _result(
            0,
            [1.0],
            direction=(-1.0,),
            labels={"problem": "min", "algorithm": "a", "seed": "0"},
        ),
        _result(
            0,
            [9.0],
            direction=(1.0,),
            labels={"problem": "max", "algorithm": "a", "seed": "0"},
        ),
    ]

    summary = write_summary(results, tmp_path)
    aggregate = write_aggregate(summary, results)
    assert aggregate is not None
    data = json.loads(aggregate.read_text(encoding="utf-8"))

    assert "orientation" not in data
    assert {group["problem"]: group["orientation"] for group in data["groups"]} == {
        "min": "min",
        "max": "max",
    }


def test_aggregate_rejects_orientation_mismatch_inside_group(tmp_path):
    results = [
        _result(0, [1.0], direction=(-1.0,)),
        _result(1, [2.0], direction=(1.0,)),
    ]
    summary = write_summary(results, tmp_path)

    with pytest.raises(ValidationError, match="orientation"):
        write_aggregate(summary, results)
