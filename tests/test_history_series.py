"""Tests for history analysis series and built-in values."""

import subprocess
import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from saealib.acquisition.base import direction_to_minimize_sign
from saealib.exceptions import ValidationError
from saealib.execution.history import History
from saealib.result import Result
from saealib.utils import gd, gd_plus, hypervolume, igd, igd_plus, spread


def _result(channels=("summary",), n_obj=1):
    history = History(channels)
    problem = SimpleNamespace(n_obj=n_obj, direction=np.full(n_obj, -1.0))
    ctx = SimpleNamespace(problem=problem)
    return Result(np.array([0.0]), np.array([0.0]), 0, 0, ctx, history), history  # ty: ignore[invalid-argument-type]


def _population_result(services):
    history = History(("population",))
    problem = SimpleNamespace(
        n_obj=1,
        direction=np.array([-1.0]),
        space=SimpleNamespace(services=services),
    )
    ctx = SimpleNamespace(problem=problem)
    return Result(np.array([0.0]), np.array([0.0]), 0, 0, ctx, history), history  # ty: ignore[invalid-argument-type]


def test_summary_values_and_generation_axis():
    result, history = _result()
    history.append(
        "summary", gen=1, fe=4, best_f=3.0, min_cv=0.0, feasible_ratio=1.0, front_size=2
    )
    series = result.history_series("best", x="gen")
    assert series.x_name == "Generations"
    assert series.y.tolist() == [3.0]


def test_history_series_is_frozen_and_summary_builtins_are_available():
    result, history = _result()
    history.append(
        "summary",
        gen=1,
        fe=4,
        best_f=3.0,
        min_cv=0.25,
        feasible_ratio=0.5,
        front_size=2,
    )

    for name, expected in {
        "best": 3.0,
        "min_cv": 0.25,
        "feasible_ratio": 0.5,
        "front_size": 2.0,
    }.items():
        series = result.history_series(name)
        assert series.y.tolist() == [expected]
        assert series.x_name == "Function evaluations"
        assert series.y_name == name

    with pytest.raises(FrozenInstanceError):
        series.x_name = "changed"


def test_unknown_and_multi_objective_best_values_raise():
    result, _ = _result(n_obj=2)
    with pytest.raises(ValidationError, match="Unknown history value"):
        result.history_series("unknown")

    history = result.history
    assert history is not None
    history.append(
        "summary",
        gen=1,
        fe=4,
        best_f=np.nan,
        min_cv=0.0,
        feasible_ratio=1.0,
        front_size=1,
    )
    with pytest.raises(ValidationError, match="single-objective"):
        result.history_series("best")


def test_removed_diversity_name_is_unknown():
    result, _ = _result()
    with pytest.raises(ValidationError) as exc_info:
        result.history_series("diversity")

    message = str(exc_info.value)
    assert "Unknown history value 'diversity'. Choose one of:" in message
    candidates = message.split("Choose one of: ", 1)[1].rstrip(".").split(", ")
    assert "mean_normalized_pairwise_distance" in candidates
    assert "diversity" not in candidates


def test_front_channel_error_is_actionable():
    result, _ = _result()
    with pytest.raises(ValidationError, match=r"front.*history_channels.*set_history"):
        result.history_series("spacing")


def test_population_channel_error_is_actionable():
    result, _ = _result()
    with pytest.raises(
        ValidationError,
        match=r"mean_normalized_pairwise_distance.*population.*history_channels.*set_history",
    ):
        result.history_series("mean_normalized_pairwise_distance")


def test_callable_uses_records():
    result, history = _result(("front",))
    history.append_block("front", {"f": np.array([[1.0], [2.0]])}, gen=2, fe=8)
    series = result.history_series(lambda record: record["f"].mean(), channel="front")
    assert series.x.tolist() == [8.0]
    assert series.y.tolist() == [1.5]


def test_callable_requires_channel_and_converts_invalid_returns():
    result, history = _result(("front",))
    history.append_block("front", {"f": np.array([[1.0]])}, gen=2, fe=8)

    with pytest.raises(ValidationError, match="requires channel"):
        result.history_series(lambda record: record["f"].mean())
    with pytest.raises(ValidationError, match="scalar number"):
        result.history_series(lambda record: "not a number", channel="front")


def test_invalid_x_and_missing_history_raise():
    result, _ = _result()
    with pytest.raises(ValidationError, match=r"fe.*gen"):
        result.history_series("min_cv", x="invalid")

    no_history = Result(
        np.array([0.0]),
        np.array([0.0]),
        0,
        0,
        result.ctx,
        history=None,
    )
    with pytest.raises(ValidationError, match=r"history_channels.*set_history"):
        no_history.history_series("min_cv")


def test_front_metric_and_reference_validation():
    result, history = _result(("front",))
    history.append_block("front", {"f": np.array([[1.0], [2.0]])}, gen=1, fe=2)
    series = result.history_series("spacing")
    assert np.isfinite(series.y[0])
    with pytest.raises(ValidationError):
        result.history_series("gd")
    with pytest.raises(ValidationError):
        result.history_series("gd", reference_front=np.empty((0, 1)))


def test_front_metrics_match_minimization_space_values():
    result, history = _result(("front",), n_obj=2)
    front = np.array([[1.0, 3.0], [2.0, 2.0]])
    reference = np.array([[1.0, 2.0], [2.0, 3.0]])
    history.append_block("front", {"f": front}, gen=1, fe=2)
    sign = np.asarray(direction_to_minimize_sign(result.problem.direction))
    expected = {
        "gd": gd(front * sign, reference * sign),
        "gd_plus": gd_plus(front * sign, reference * sign),
        "igd": igd(front * sign, reference * sign),
        "igd_plus": igd_plus(front * sign, reference * sign),
        "spread": spread(front * sign, reference * sign),
    }
    for name, value in expected.items():
        np.testing.assert_allclose(
            result.history_series(name, reference_front=reference).y, [value]
        )

    reference_point = np.array([3.0, 4.0])
    np.testing.assert_allclose(
        result.history_series("hypervolume", reference_point=reference_point).y,
        [hypervolume(front * sign, reference_point * sign)],
    )


def test_reference_shapes_and_empty_fronts_are_validated():
    result, history = _result(("front",), n_obj=2)
    history.append_block("front", {"f": np.empty((0, 2))}, gen=1, fe=2)
    assert np.isnan(result.history_series("spacing").y[0])
    assert np.isnan(
        result.history_series("hypervolume", reference_point=[1.0, 1.0]).y[0]
    )

    with pytest.raises(ValidationError):
        result.history_series("hypervolume", reference_point=[1.0])
    with pytest.raises(ValidationError):
        result.history_series("gd", reference_front=np.ones((2, 3)))


def test_mean_normalized_pairwise_distance_uses_bounds_and_population_x_block():
    dense = object()
    bounds = SimpleNamespace(bounds=(np.array([0.0, 10.0]), np.array([1.0, 10.0])))
    services = SimpleNamespace(
        get=lambda name: {"DenseNumericView": dense, "BoundsService": bounds}.get(name)
    )
    result, history = _population_result(services)
    history.append_block(
        "population",
        {
            "f": np.zeros((3, 1)),
            "x": np.array([[0.0, 10.0], [1.0, 10.0], [0.5, 10.0]]),
        },
        gen=1,
        fe=2,
        size=3,
    )
    np.testing.assert_allclose(
        result.history_series("mean_normalized_pairwise_distance").y, [2.0 / 3.0]
    )


def test_mean_normalized_pairwise_distance_requires_services_and_x_block():
    result, history = _population_result(SimpleNamespace(get=lambda name: None))
    history.append_block(
        "population",
        {"f": np.zeros((2, 1)), "x": np.zeros((2, 2))},
        gen=1,
        fe=2,
        size=2,
    )
    with pytest.raises(ValidationError, match="DenseNumericView"):
        result.history_series("mean_normalized_pairwise_distance")

    bounds = SimpleNamespace(bounds=(np.zeros(2), np.ones(2)))
    result, history = _population_result(
        SimpleNamespace(
            get=lambda name: bounds if name == "BoundsService" else object()
        )
    )
    history.append_block(
        "population",
        {"f": np.zeros((2, 1))},
        gen=1,
        fe=2,
        size=2,
    )
    with pytest.raises(ValidationError, match=r"population.*x"):
        result.history_series("mean_normalized_pairwise_distance")


def test_series_modules_do_not_import_matplotlib():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import saealib.result; import saealib._series_values; "
                "assert not any(name == 'matplotlib' or name.startswith('matplotlib.') "
                "for name in sys.modules)"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
