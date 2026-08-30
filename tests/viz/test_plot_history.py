"""Tests for history plots."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from saealib import (
    DirectStrategy,
    GenomeInitializer,
    Optimizer,
    Problem,
    Termination,
    max_gen,
)
from saealib.algorithms import GenomeGA
from saealib.operators import (
    OrderCrossover,
    SequentialSelection,
    SwapMutation,
    TruncationSelection,
)
from saealib.result import HistorySeries, Result
from saealib.space import PermutationSpace
from saealib.viz import plot_history


def _result() -> Result:
    return Result(
        x=np.array([0.0]),
        f=np.array([0.0]),
        fe=0,
        gen=0,
        ctx=SimpleNamespace(),  # ty: ignore[invalid-argument-type]
    )


def test_plot_history_delegates_and_labels_axes() -> None:
    result = _result()
    result.history_series = Mock(
        return_value=HistorySeries(
            x=np.array([1.0, 2.0]),
            y=np.array([3.0, 4.0]),
            x_name="Generations",
            y_name="best",
        )
    )
    fig = Figure()
    ax = fig.add_subplot(111)

    returned = plot_history(
        result,
        "best",
        x="gen",
        channel="summary",
        ax=ax,
        reference_point=np.array([1.0]),
    )

    assert returned is fig
    result.history_series.assert_called_once_with(
        "best",
        x="gen",
        channel="summary",
        reference_point=np.array([1.0]),
    )
    assert ax.get_xlabel() == "Generations"
    assert ax.get_ylabel() == "best"
    np.testing.assert_allclose(
        np.asarray(ax.lines[0].get_xdata(), dtype=float), [1.0, 2.0]
    )
    np.testing.assert_allclose(
        np.asarray(ax.lines[0].get_ydata(), dtype=float), [3.0, 4.0]
    )


def test_plot_history_converts_state_to_result(monkeypatch) -> None:
    result = _result()
    result.history_series = Mock(
        return_value=HistorySeries(
            x=np.array([1.0]),
            y=np.array([2.0]),
            x_name="Function evaluations",
            y_name="metric",
        )
    )
    state = object()
    from_state = Mock(return_value=result)
    monkeypatch.setattr(Result, "from_state", from_state)

    plot_history(state, "metric")  # ty: ignore[invalid-argument-type]

    from_state.assert_called_once_with(state)
    result.history_series.assert_called_once_with("metric", x="fe", channel=None)


def test_non_dense_state_best_history_plot_succeeds():
    space = PermutationSpace(8)
    problem = Problem(
        func=lambda x: np.asarray([float(sum(i * v for i, v in enumerate(x)))]),
        dim=space.dim,
        n_obj=1,
        direction=np.array([-1.0]),
        space=space,
    )
    ga = GenomeGA(
        OrderCrossover(),
        SwapMutation(),
        SequentialSelection(),
        TruncationSelection(),
    )
    state = (
        Optimizer(problem, seed=1)
        .set_algorithm(ga)
        .set_strategy(DirectStrategy())
        .set_initializer(GenomeInitializer(24, 24))
        .set_termination(Termination(max_gen(3)))
        .set_history(["summary", "front", "population"])
        .run()
    )

    assert isinstance(plot_history(state, "best"), Figure)
