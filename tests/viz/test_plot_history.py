"""Tests for history plots."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from saealib.result import HistorySeries, Result
from saealib.viz import plot_history


def _result() -> Result:
    return Result(
        x=np.array([0.0]),
        f=np.array([0.0]),
        fe=0,
        gen=0,
        ctx=SimpleNamespace(),
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
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [1.0, 2.0])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [3.0, 4.0])


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

    plot_history(state, "metric")

    from_state.assert_called_once_with(state)
    result.history_series.assert_called_once_with("metric", x="fe", channel=None)
