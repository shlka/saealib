"""Tests for multi-run convergence plots."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from saealib import minimize
from saealib.api import Result
from saealib.exceptions import ValidationError
from saealib.execution.history import History
from saealib.viz import plot_convergence


def _result(fe: list[int], values: list[float], direction: float = -1.0) -> Result:
    history = History(("summary",))
    for index, (function_evaluations, value) in enumerate(zip(fe, values)):
        history.append(
            "summary",
            gen=index,
            fe=function_evaluations,
            f_min_0=value,
            f_max_0=value,
            min_cv=0.0,
            feasible_ratio=1.0,
        )
    problem = SimpleNamespace(n_obj=1, direction=np.array([direction]))
    return cast(
        Result,
        SimpleNamespace(
            f=np.array([values[-1]]) if values else np.empty(0),
            history=history,
            ctx=SimpleNamespace(problem=problem),
        ),
    )


def _empty_result() -> Result:
    history = History(("summary",))
    return cast(
        Result,
        SimpleNamespace(
            f=np.array([0.0]),
            history=history,
            ctx=SimpleNamespace(
                problem=SimpleNamespace(n_obj=1, direction=np.array([-1.0]))
            ),
        ),
    )


def test_plot_convergence_aggregates_median_and_iqr_on_fe_union() -> None:
    results = [
        _result([1, 3, 5], [5.0, 2.0, 4.0]),
        _result([1, 4, 5], [3.0, 4.0, 1.0]),
        _result([2, 3, 5], [4.0, 6.0, 0.0]),
    ]

    fig = plot_convergence(results)

    ax = fig.axes[0]
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(
        np.asarray(ax.lines[0].get_ydata()), [4.0, 3.0, 3.0, 1.0]
    )
    assert len(ax.collections) == 1
    vertices = np.asarray(ax.collections[0].get_paths()[0].vertices)
    lower = np.array([3.5, 2.5, 2.5, 0.5])
    upper = np.array([4.5, 3.5, 3.5, 1.5])
    for x, expected_lower, expected_upper in zip([2.0, 3.0, 4.0, 5.0], lower, upper):
        ys = vertices[np.isclose(vertices[:, 0], x), 1]
        assert np.any(np.isclose(ys, expected_lower))
        assert np.any(np.isclose(ys, expected_upper))


def test_plot_convergence_converts_each_run_to_best_so_far() -> None:
    results = [
        _result([1, 3], [5.0, 1.0]),
        _result([1, 3], [3.0, 2.0]),
    ]

    fig = plot_convergence(results)

    np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), [1.0, 3.0])
    np.testing.assert_allclose(np.asarray(fig.axes[0].lines[0].get_ydata()), [4.0, 1.5])


def test_plot_convergence_uses_maximize_best_so_far_values() -> None:
    results = [
        _result([1, 3], [1.0, 3.0], direction=1.0),
        _result([1, 3], [2.0, 1.0], direction=1.0),
    ]

    fig = plot_convergence(results)

    np.testing.assert_allclose(np.asarray(fig.axes[0].lines[0].get_ydata()), [1.5, 2.5])


def test_plot_convergence_common_and_full_ranges_hold_short_run_value() -> None:
    results = [_result([1, 3], [5.0, 2.0]), _result([2, 4, 6], [8.0, 6.0, 4.0])]

    common = plot_convergence(results, fe_range="common")
    full = plot_convergence(results, fe_range="full")

    np.testing.assert_array_equal(common.axes[0].lines[0].get_xdata(), [2.0, 3.0])
    np.testing.assert_array_equal(
        full.axes[0].lines[0].get_xdata(), [2.0, 3.0, 4.0, 6.0]
    )
    np.testing.assert_allclose(
        np.asarray(full.axes[0].lines[0].get_ydata()), [6.5, 5.0, 4.0, 3.0]
    )


def test_plot_convergence_rejects_non_overlapping_common_range() -> None:
    results = [_result([1, 2], [2.0, 1.0]), _result([3, 4], [4.0, 3.0])]

    with pytest.raises(ValidationError, match=r'fe_range="full"'):
        plot_convergence(results)


def test_plot_convergence_groups_and_labels() -> None:
    results = [
        _result([1, 2], [4.0, 2.0]),
        _result([1, 2], [6.0, 3.0]),
        _result([1, 2], [8.0, 5.0]),
        _result([1, 2], [10.0, 7.0]),
    ]

    fig = plot_convergence(
        results,
        groups=["first", "first", "second", "second"],
        labels={"first": "A", "second": "B"},
    )

    ax = fig.axes[0]
    assert len(ax.lines) == 2
    assert len(ax.collections) == 2
    assert [line.get_label() for line in ax.lines] == ["A", "B"]
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["A", "B"]


def test_plot_convergence_single_result_has_one_unaggregated_line() -> None:
    result = _result([1, 2], [5.0, 3.0])

    fig = plot_convergence(result, labels="run A")

    ax = fig.axes[0]
    assert isinstance(fig, Figure)
    assert len(ax.lines) == 1
    assert len(ax.collections) == 0
    assert ax.lines[0].get_label() == "run A"
    np.testing.assert_array_equal(np.asarray(ax.lines[0].get_ydata()), [5.0, 3.0])


def test_plot_convergence_single_result_sequence_has_no_band() -> None:
    result = _result([1, 2], [5.0, 3.0])

    fig = plot_convergence([result])

    ax = fig.axes[0]
    assert len(ax.lines) == 1
    assert len(ax.collections) == 0
    np.testing.assert_array_equal(np.asarray(ax.lines[0].get_xdata()), [1.0, 2.0])


def test_plot_convergence_validates_multi_run_inputs() -> None:
    result = _result([1, 2], [2.0, 1.0])

    with pytest.raises(ValidationError, match="at least one"):
        plot_convergence([])
    with pytest.raises(ValidationError, match=r'zero-row|"fe" column'):
        plot_convergence(_empty_result())
    with pytest.raises(ValidationError, match="one key per result"):
        plot_convergence([result, result], groups=["only"])
    with pytest.raises(ValidationError, match="string when groups"):
        plot_convergence([result, result], labels={"group": "label"})
    with pytest.raises(ValidationError, match="mapping when groups"):
        plot_convergence([result, result], groups=["a", "a"], labels="label")
    with pytest.raises(ValidationError, match="keys must match"):
        plot_convergence([result, result], groups=["a", "b"], labels={"a": "A"})
    with pytest.raises(ValidationError, match="fe_range"):
        plot_convergence([result, result], fe_range="invalid")


def test_plot_convergence_rejects_mixed_directions_and_multi_objective() -> None:
    minimize_result = _result([1, 2], [2.0, 1.0], direction=-1.0)
    maximize_result = _result([1, 2], [2.0, 3.0], direction=1.0)
    multi_result = cast(
        Result,
        SimpleNamespace(
            f=np.ones((2, 2)),
            history=minimize_result.history,
            ctx=SimpleNamespace(
                problem=SimpleNamespace(n_obj=2, direction=np.array([-1.0, -1.0]))
            ),
        ),
    )

    with pytest.raises(ValidationError, match=r"same.*direction"):
        plot_convergence([minimize_result, maximize_result])
    with pytest.raises(ValidationError, match="single objective"):
        plot_convergence([minimize_result, multi_result])


def test_plot_convergence_real_runs_and_preserves_history() -> None:
    results = [
        minimize(
            lambda x: np.sum(x**2),
            dim=2,
            lb=[-1.0, -1.0],
            ub=[1.0, 1.0],
            max_fe=20,
            pop_size=4,
            seed=seed,
            history_channels=["summary"],
            verbose=False,
        )
        for seed in (0, 1)
    ]
    before = []
    for result in results:
        history = result.history
        assert history is not None
        before.append(
            {name: values.copy() for name, values in history.channel("summary").items()}
        )

    fig = plot_convergence(results, labels="runs")

    assert isinstance(fig, Figure)
    assert len(fig.axes[0].lines) == 1
    for result, expected in zip(results, before):
        history = result.history
        assert history is not None
        actual = history.channel("summary")
        for name, values in expected.items():
            np.testing.assert_array_equal(actual[name], values)
