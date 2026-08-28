"""Tests for optimization progress plots."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from saealib import InequalityConstraint, Problem, minimize
from saealib.api import Result
from saealib.exceptions import ValidationError
from saealib.execution.history import History
from saealib.utils import hypervolume
from saealib.viz import (
    plot_constraint_violation,
    plot_convergence,
    plot_hypervolume,
    plot_indicator,
    plot_objective_heatmap,
    plot_pareto,
    plot_pareto_evolution,
    plot_pcp,
    plot_running_metric,
)


def _result(f: np.ndarray, direction: np.ndarray, history: History) -> Result:
    problem = SimpleNamespace(n_obj=len(direction), direction=direction)
    return cast(
        Result,
        SimpleNamespace(
            f=f,
            history=history,
            ctx=SimpleNamespace(problem=problem),
        ),
    )


def _summary_result(direction: np.ndarray | None = None) -> Result:
    if direction is None:
        direction = np.array([-1.0])
    history = History(("summary",))
    for gen, fe, f_min, f_max, min_cv, feasible_ratio in (
        (0, 4, 5.0, 8.0, 2.0, 0.25),
        (1, 8, 3.0, 6.0, 0.5, 0.75),
    ):
        history.append(
            "summary",
            gen=gen,
            fe=fe,
            f_min_0=f_min,
            f_max_0=f_max,
            min_cv=min_cv,
            feasible_ratio=feasible_ratio,
        )
    return _result(np.array([3.0]), direction, history)


def _front_result(
    fronts: tuple[np.ndarray, ...],
    direction: np.ndarray | None = None,
) -> Result:
    if direction is None:
        direction = np.full(fronts[0].shape[1], -1.0)
    history = History(("front",))
    for gen, front in enumerate(fronts):
        history.append_block(
            "front",
            {"f": front},
            gen=gen,
            fe=gen + 1,
        )
    return _result(fronts[-1], direction, history)


def test_plot_convergence_uses_summary_fe_and_minimum_value() -> None:
    result = _summary_result()

    fig = plot_convergence(result, labels="run")

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [4, 8])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [5.0, 3.0])
    assert ax.get_xlabel() == "Function evaluations"
    assert ax.get_ylabel() == "Objective value"
    assert ax.get_legend() is not None


def test_plot_convergence_uses_maximum_value_for_maximize_direction() -> None:
    result = _summary_result(np.array([1.0]))

    fig = plot_convergence(result)

    np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), [8.0, 6.0])


def test_plot_convergence_rejects_multi_objective_result() -> None:
    result = _summary_result(np.array([-1.0, -1.0]))
    result.f = np.array([[1.0, 2.0]])

    with pytest.raises(ValidationError, match=r"plot_hypervolume.*plot_indicator"):
        plot_convergence(result)


def test_plot_convergence_drops_empty_front_rows() -> None:
    history = History(("summary",))
    history.append(
        "summary",
        gen=0,
        fe=1,
        f_min_0=np.nan,
        f_max_0=np.nan,
        min_cv=np.nan,
        feasible_ratio=np.nan,
    )
    history.append(
        "summary",
        gen=1,
        fe=2,
        f_min_0=1.0,
        f_max_0=1.0,
        min_cv=0.0,
        feasible_ratio=1.0,
    )
    result = _result(np.array([1.0]), np.array([-1.0]), history)

    fig = plot_convergence(result)

    np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), [2])
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), [1.0])


def test_plot_hypervolume_handles_direction_and_empty_front() -> None:
    fronts = (
        np.array([[1.0, 2.0], [2.0, 1.0]]),
        np.empty((0, 2)),
    )
    result = _front_result(fronts, np.array([-1.0, 1.0]))

    fig = plot_hypervolume(result, np.array([3.0, 0.0]))

    expected = hypervolume(
        fronts[0] * np.array([1.0, -1.0]), np.array([3.0, 0.0]) * [1.0, -1.0]
    )
    ydata = np.asarray(fig.axes[0].lines[0].get_ydata())
    np.testing.assert_allclose(ydata[0], expected)
    assert np.isnan(ydata[1])
    assert fig.axes[0].get_ylabel() == "Hypervolume"


def test_plot_hypervolume_rejects_reference_shape() -> None:
    result = _front_result((np.ones((2, 2)),))

    with pytest.raises(ValidationError, match="reference_point"):
        plot_hypervolume(result, np.ones(3))


def test_plot_indicator_validates_name_and_reference_front() -> None:
    result = _front_result((np.ones((2, 2)),))

    with pytest.raises(ValidationError, match=r"gd.*reference_front"):
        plot_indicator(result, "gd")
    with pytest.raises(ValidationError, match=r"gd_plus.*igd"):
        plot_indicator(result, "unknown")


def test_plot_indicator_spacing_returns_figure() -> None:
    result = _front_result((np.array([[0.0, 1.0], [1.0, 0.0]]),))

    fig = plot_indicator(result, "spacing")

    assert isinstance(fig, Figure)
    assert fig.axes[0].get_ylabel() == "spacing"


def test_plot_constraint_violation_uses_two_axes_and_combined_legend() -> None:
    result = _summary_result()

    fig = plot_constraint_violation(result)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    left, right = fig.axes
    np.testing.assert_array_equal(left.lines[0].get_ydata(), [2.0, 0.5])
    np.testing.assert_array_equal(right.lines[0].get_ydata(), [0.25, 0.75])
    legend = left.get_legend()
    assert legend is not None
    assert len(legend.get_lines()) == 2


def test_plot_running_metric_marks_significant_movement_and_ends_at_zero() -> None:
    fronts = (
        np.array([[1.0, 4.0], [4.0, 1.0]]),
        np.array([[0.0, 3.0], [3.0, 0.0]]),
    )
    result = _front_result(fronts)

    fig = plot_running_metric(result, significance=0.1)

    ax = fig.axes[0]
    assert ax.get_yscale() == "symlog"
    assert ax.get_xlabel() == "Generation"
    assert ax.get_ylabel() == "$\\Delta f$"
    assert np.asarray(ax.lines[0].get_ydata())[-1] == 0.0
    marker_lines = [
        line for line in ax.lines if line.get_marker() not in (None, "None")
    ]
    assert len(marker_lines) == 1


def test_plot_running_metric_ignores_empty_fronts_and_accepts_single_front() -> None:
    result = _front_result((np.empty((0, 2)), np.array([[1.0, 2.0]])))

    fig = plot_running_metric(result)

    np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), [1])
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), [0.0])


def test_plot_preserves_history_and_returns_root_for_existing_subfigure_axes() -> None:
    fronts = (np.array([[1.0, 2.0], [2.0, 1.0]]),)
    result = _front_result(fronts)
    history = result.history
    assert history is not None
    before = tuple(array.copy() for array in history.blocks("front", "f"))
    from matplotlib.figure import Figure, SubFigure

    root = Figure()
    subfig = root.subfigures(1)
    assert isinstance(subfig, SubFigure)
    subax = subfig.add_subplot(111)

    fig = plot_hypervolume(result, np.array([3.0, 3.0]), ax=subax)

    assert fig is root
    for actual, expected in zip(history.blocks("front", "f"), before):
        np.testing.assert_array_equal(actual, expected)


def test_real_single_objective_run_supports_summary_plots() -> None:
    result = minimize(
        lambda x: np.sum(x**2),
        dim=2,
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
        max_fe=24,
        pop_size=4,
        seed=0,
        history_channels=["summary"],
        verbose=False,
    )

    assert isinstance(plot_convergence(result), Figure)
    assert isinstance(plot_constraint_violation(result), Figure)


def test_real_multi_objective_run_supports_summary_and_front_plots() -> None:
    result = minimize(
        lambda x: np.array([np.sum(x**2), np.sum((x - 1.0) ** 2)]),
        dim=2,
        n_obj=2,
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
        max_fe=24,
        pop_size=4,
        seed=1,
        history_channels=["summary", "front"],
        verbose=False,
    )

    figures = (
        plot_hypervolume(result, np.array([10.0, 10.0])),
        plot_running_metric(result),
        plot_pareto(result),
        plot_pareto_evolution(result),
        plot_pcp(result),
        plot_objective_heatmap(result),
    )
    assert all(isinstance(fig, Figure) for fig in figures)


def test_real_constrained_run_allows_upward_convergence_curve() -> None:
    problem = Problem(
        func=lambda x: np.sum(x**2),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0, -5.0],
        ub=[5.0, 5.0],
        constraints=[InequalityConstraint(lambda x: 9.9 - np.sum(x))],
    )
    result = minimize(
        problem,
        max_fe=200,
        seed=7,
        history_channels=["summary"],
        verbose=False,
    )

    fig = plot_convergence(result)

    assert isinstance(fig, Figure)
    assert len(fig.axes[0].lines) == 1
    assert np.any(np.diff(np.asarray(fig.axes[0].lines[0].get_ydata())) > 0)
