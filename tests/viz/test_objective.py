"""Tests for objective-space plots."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from saealib.api import Result
from saealib.exceptions import ValidationError
from saealib.execution.history import History
from saealib.viz import (
    plot_objective_heatmap,
    plot_pareto,
    plot_pareto_evolution,
    plot_pcp,
    plot_radar,
)


def _result(
    f: np.ndarray,
    direction: np.ndarray | None = None,
    history: History | None = None,
) -> Result:
    if direction is None:
        direction = np.full(f.shape[-1], -1.0)
    problem = SimpleNamespace(n_obj=len(direction), direction=direction)
    return cast(
        Result,
        SimpleNamespace(
            f=f,
            history=history,
            ctx=SimpleNamespace(problem=problem),
        ),
    )


def _evolution_result() -> Result:
    history = History(("front",))
    history.append_block(
        "front",
        {"f": np.empty((0, 2))},
        gen=0,
        fe=1,
    )
    history.append_block(
        "front",
        {"f": np.array([[1.0, 3.0], [3.0, 1.0]])},
        gen=1,
        fe=2,
    )
    history.append_block(
        "front",
        {"f": np.array([[0.5, 2.5], [2.5, 0.5]])},
        gen=2,
        fe=3,
    )
    return _result(history.blocks("front", "f")[-1], history=history)


def test_plot_pareto_two_objectives_has_scatter_and_labels() -> None:
    result = _result(np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]))

    fig = plot_pareto(result)

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert len(ax.collections) == 1
    assert ax.get_xlabel() == "f0"
    assert ax.get_ylabel() == "f1"


def test_plot_pareto_three_objectives_uses_3d_axes() -> None:
    result = _result(np.arange(9, dtype=float).reshape(3, 3))

    fig = plot_pareto(result)

    ax = fig.axes[0]
    assert ax.name == "3d"
    assert len(ax.collections) == 1
    assert cast(Any, ax).get_zlabel() == "f2"


def test_plot_pareto_three_objectives_rejects_rectilinear_axes() -> None:
    result = _result(np.arange(9, dtype=float).reshape(3, 3))
    fig = Figure()
    ax = fig.add_subplot(111)

    with pytest.raises(ValidationError, match=r'projection="3d"') as excinfo:
        plot_pareto(result, ax=ax)
    assert 'fig.add_subplot(111, projection="3d")' in str(excinfo.value)


def test_plot_pareto_existing_subfigure_axes_returns_root() -> None:
    result = _result(np.array([[0.0, 1.0], [1.0, 0.0]]))
    from matplotlib.figure import SubFigure

    root = Figure()
    subfig = root.subfigures(1)
    assert isinstance(subfig, SubFigure)
    ax = subfig.add_subplot(111)

    fig = plot_pareto(result, ax=ax)

    assert fig is root
    assert ax.collections


def test_plot_pareto_requires_explicit_objectives_for_four_objectives() -> None:
    result = _result(np.arange(20, dtype=float).reshape(4, 5))

    with pytest.raises(ValidationError, match="objectives"):
        plot_pareto(result)
    with pytest.raises(ValidationError, match="outside"):
        plot_pareto(result, objectives=(0, 5))


def test_plot_pareto_evolution_has_scatter_collections_and_colorbar() -> None:
    result = _evolution_result()

    fig = plot_pareto_evolution(result, cmap="plasma")

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    assert len(fig.axes[0].collections) == 2
    assert fig.axes[1].get_ylabel() == "Generation"


def test_plot_pareto_evolution_requires_objectives_for_four_objectives() -> None:
    history = History(("front",))
    history.append_block(
        "front",
        {"f": np.ones((2, 4))},
        gen=0,
        fe=1,
    )
    result = _result(np.ones((2, 4)), history=history)

    with pytest.raises(ValidationError, match="objectives"):
        plot_pareto_evolution(result)


def test_plot_pcp_normalizes_each_column_and_labels_axes() -> None:
    result = _result(np.array([[0.0, 5.0], [2.0, 5.0], [1.0, 5.0]]))

    fig = plot_pcp(result)

    ax = fig.axes[0]
    assert len(ax.lines) == 3
    np.testing.assert_allclose(np.asarray(ax.lines[0].get_ydata()), [0.0, 0.5])
    np.testing.assert_allclose(np.asarray(ax.lines[1].get_ydata()), [1.0, 0.5])
    assert [label.get_text() for label in ax.get_xticklabels()] == ["f0", "f1"]


def test_plot_radar_draws_closed_polygons() -> None:
    result = _result(np.arange(12, dtype=float).reshape(4, 3))

    fig = plot_radar(result)

    ax = fig.axes[0]
    assert ax.name == "polar"
    assert len(ax.lines) == 4
    for line in ax.lines:
        xdata = np.asarray(line.get_xdata())
        ydata = np.asarray(line.get_ydata())
        assert xdata[0] == xdata[-1]
        assert ydata[0] == ydata[-1]


def test_plot_radar_rejects_two_objectives_with_alternatives() -> None:
    result = _result(np.ones((2, 2)))

    with pytest.raises(ValidationError, match=r"plot_pareto.*plot_pcp"):
        plot_radar(result)


def test_plot_radar_rejects_rectilinear_axes() -> None:
    result = _result(np.arange(9, dtype=float).reshape(3, 3))
    fig = Figure()
    ax = fig.add_subplot(111)

    with pytest.raises(ValidationError, match=r'projection="polar"') as excinfo:
        plot_radar(result, ax=ax)
    assert 'fig.add_subplot(111, projection="polar")' in str(excinfo.value)


def test_plot_objective_heatmap_sorts_by_minimize_objective_zero() -> None:
    values = np.array([[1.0, 5.0], [3.0, 2.0], [2.0, 4.0]])
    result = _result(values, direction=np.array([1.0, -1.0]))

    fig = plot_objective_heatmap(result, cmap="viridis")

    ax = fig.axes[0]
    assert len(ax.images) == 1
    assert len(fig.axes) == 2


@pytest.mark.parametrize(
    "plot",
    [plot_pareto, plot_pareto_evolution, plot_pcp, plot_radar, plot_objective_heatmap],
)
def test_objective_plots_reject_single_objective(plot: Any) -> None:
    result = _result(np.array([1.0]))

    with pytest.raises(ValidationError, match="plot_convergence"):
        plot(result)


def test_objective_plot_does_not_change_front_history() -> None:
    result = _evolution_result()
    history = result.history
    assert history is not None
    before = tuple(array.copy() for array in history.blocks("front", "f"))

    plot_pareto_evolution(result)

    for actual, expected in zip(history.blocks("front", "f"), before):
        np.testing.assert_array_equal(actual, expected)
