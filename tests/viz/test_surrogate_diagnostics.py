"""Tests for surrogate diagnostics and prescreening plots."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from saealib import minimize
from saealib.api import Result
from saealib.exceptions import ValidationError
from saealib.execution.history import History
from saealib.viz import (
    plot_prescreening,
    plot_surrogate_accuracy,
    plot_surrogate_error_history,
)


def _result(history: History, n_obj: int = 1) -> Result:
    return cast(
        Result,
        SimpleNamespace(
            history=history,
            ctx=SimpleNamespace(
                problem=SimpleNamespace(n_obj=n_obj, direction=np.full(n_obj, -1.0))
            ),
        ),
    )


def _accuracy_history() -> History:
    history = History(("surrogate_accuracy",))
    history.append_block(
        "surrogate_accuracy",
        {"predicted": np.array([[2.0], [4.0]]), "true": np.array([[1.0], [3.0]])},
        size=2,
        gen=0,
        fe_after=2,
    )
    history.append_block(
        "surrogate_accuracy",
        {"predicted": np.empty((0, 1)), "true": np.empty((0, 1))},
        size=0,
        gen=1,
        fe_after=3,
    )
    history.append_block(
        "surrogate_accuracy",
        {"predicted": np.array([[5.0]]), "true": np.array([[3.0]])},
        size=1,
        gen=1,
        fe_after=4,
    )
    return history


def _mixed_finite_accuracy_history() -> History:
    history = History(("surrogate_accuracy",))
    history.append_block(
        "surrogate_accuracy",
        {
            "predicted": np.array([[2.0], [np.nan], [4.0]]),
            "true": np.array([[1.0], [3.0], [np.inf]]),
        },
        size=3,
        gen=0,
        fe_after=2,
    )
    history.append_block(
        "surrogate_accuracy",
        {
            "predicted": np.array([[5.0], [np.inf]]),
            "true": np.array([[3.0], [4.0]]),
        },
        size=2,
        gen=1,
        fe_after=4,
    )
    return history


def test_accuracy_pools_pairs_and_skips_empty_rows() -> None:
    fig = plot_surrogate_accuracy(_result(_accuracy_history()))

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert len(cast(Any, ax.collections[0]).get_offsets()) == 3
    assert "RMSE = 1.41421" in ax.texts[0].get_text()
    assert "R² = -1.25" in ax.texts[0].get_text()


def test_accuracy_and_error_history_filter_nonfinite_pairs() -> None:
    result = _result(_mixed_finite_accuracy_history())

    accuracy = plot_surrogate_accuracy(result)
    error = plot_surrogate_error_history(result)
    moving = plot_surrogate_error_history(result, window=2)

    assert len(cast(Any, accuracy.axes[0].collections[0]).get_offsets()) == 2
    assert "RMSE = 1.58114" in accuracy.axes[0].texts[0].get_text()
    np.testing.assert_allclose(
        cast(Any, error.axes[0].lines[0]).get_ydata(), [1.0, 2.0]
    )
    np.testing.assert_allclose(
        cast(Any, moving.axes[0].lines[0]).get_ydata(), [1.0, np.sqrt(2.5)]
    )


def test_error_history_groups_and_window() -> None:
    result = _result(_accuracy_history())
    grouped = plot_surrogate_error_history(result)
    moving = plot_surrogate_error_history(result, window=2)

    assert len(cast(Any, grouped.axes[0].lines[0]).get_xdata()) == 2
    np.testing.assert_allclose(
        cast(Any, moving.axes[0].lines[0]).get_ydata(), [1.0, np.sqrt(2.5)]
    )
    for window in (0, -1, 1.5):
        with pytest.raises(ValidationError, match="positive integer"):
            plot_surrogate_error_history(result, window=cast(Any, window))


def test_prescreening_dense_and_opaque_fallback() -> None:
    dense = History(("decision_candidates",))
    dense.append_block(
        "decision_candidates",
        {
            "candidates": np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]),
            "selected": np.array([[True], [False], [True]]),
            "acquisition_scores": np.array([[0.9], [0.2], [0.5]]),
        },
        size=3,
    )
    fig = plot_prescreening(_result(dense))
    assert len(fig.axes[0].collections) == 2
    assert len(fig.axes) == 2

    opaque = History(("decision_candidates",))
    opaque.append_block(
        "decision_candidates",
        {
            "selected": np.array([[True], [False]]),
            "acquisition_scores": np.array([[0.1], [0.8]]),
        },
        size=2,
    )
    fallback = plot_prescreening(_result(opaque))
    assert fallback.axes[0].get_xlabel() == "Candidate rank by acquisition score"
    assert len(fallback.axes[0].collections) == 2


def test_prescreening_applies_requested_cmap_to_both_candidate_groups() -> None:
    history = History(("decision_candidates",))
    history.append_block(
        "decision_candidates",
        {
            "candidates": np.array([[0.0, 1.0], [1.0, 0.0]]),
            "selected": np.array([[True], [False]]),
            "acquisition_scores": np.array([[0.9], [0.2]]),
        },
        size=2,
    )

    fig = plot_prescreening(_result(history), cmap="plasma")

    assert [collection.get_cmap().name for collection in fig.axes[0].collections] == [
        "plasma",
        "plasma",
    ]


def test_prescreening_nan_scores_and_variable_selection() -> None:
    history = History(("decision_candidates",))
    history.append_block(
        "decision_candidates",
        {
            "candidates": np.array([[0.0, 1.0], [1.0, 0.0]]),
            "selected": np.array([[True], [False]]),
            "acquisition_scores": np.full((2, 1), np.nan),
        },
        size=2,
    )
    fig = plot_prescreening(_result(history))
    assert len(fig.axes) == 1
    assert len(fig.axes[0].collections) == 2

    three_dimensional = History(("decision_candidates",))
    three_dimensional.append_block(
        "decision_candidates",
        {
            "candidates": np.zeros((2, 3)),
            "selected": np.array([[True], [False]]),
            "acquisition_scores": np.array([[0.8], [0.2]]),
        },
        size=2,
    )
    with pytest.raises(ValidationError, match="variables"):
        plot_prescreening(_result(three_dimensional))


def test_prescreening_decision_index_and_opaque_nan_validation() -> None:
    history = History(("decision_candidates",))
    for selected in (True, False):
        history.append_block(
            "decision_candidates",
            {
                "selected": np.array([[selected]]),
                "acquisition_scores": np.array([[0.5]]),
            },
            size=1,
        )
    plot_prescreening(_result(history), decision=-1)
    with pytest.raises(ValidationError, match="2 recorded decisions"):
        plot_prescreening(_result(history), decision=2)
    with pytest.raises(ValidationError, match="2 recorded decisions"):
        plot_prescreening(_result(history), decision=-3)

    nan_history = History(("decision_candidates",))
    nan_history.append_block(
        "decision_candidates",
        {
            "selected": np.array([[True], [False]]),
            "acquisition_scores": np.full((2, 1), np.nan),
        },
        size=2,
    )
    with pytest.raises(ValidationError, match="scores and candidates"):
        plot_prescreening(_result(nan_history))


def test_multi_objective_requires_objective_and_empty_pairs_error() -> None:
    multi = History(("surrogate_accuracy",))
    multi.append_block(
        "surrogate_accuracy",
        {"predicted": np.ones((1, 2)), "true": np.ones((1, 2))},
        size=1,
        gen=0,
        fe_after=1,
    )
    with pytest.raises(ValidationError, match="objective"):
        plot_surrogate_accuracy(_result(multi, n_obj=2))
    empty = History(("surrogate_accuracy",))
    empty.append_block(
        "surrogate_accuracy",
        {"predicted": np.empty((0, 1)), "true": np.empty((0, 1))},
        size=0,
        gen=0,
        fe_after=0,
    )
    with pytest.raises(ValidationError, match="no valid"):
        plot_surrogate_accuracy(_result(empty))

    with pytest.raises(ValidationError, match="no valid"):
        plot_surrogate_error_history(_result(empty))


def test_all_nonfinite_pairs_are_rejected() -> None:
    history = History(("surrogate_accuracy",))
    history.append_block(
        "surrogate_accuracy",
        {
            "predicted": np.array([[np.nan], [np.inf]]),
            "true": np.array([[1.0], [2.0]]),
        },
        size=2,
        gen=0,
        fe_after=1,
    )
    result = _result(history)

    with pytest.raises(ValidationError, match="no valid"):
        plot_surrogate_accuracy(result)
    with pytest.raises(ValidationError, match="no valid"):
        plot_surrogate_error_history(result)


def test_multi_objective_uses_only_selected_objective() -> None:
    history = History(("surrogate_accuracy",))
    history.append_block(
        "surrogate_accuracy",
        {
            "predicted": np.array([[2.0, 20.0], [4.0, 40.0]]),
            "true": np.array([[1.0, 10.0], [3.0, 30.0]]),
        },
        size=2,
        gen=0,
        fe_after=2,
    )
    fig = plot_surrogate_accuracy(_result(history, n_obj=2), objective=1)
    np.testing.assert_allclose(
        cast(Any, fig.axes[0].collections[0]).get_offsets(), [[10, 20], [30, 40]]
    )


def test_existing_axes_and_history_are_unchanged() -> None:
    history = _accuracy_history()
    scalar_before = {
        name: np.array(values, copy=True)
        for name, values in history.channel("surrogate_accuracy").items()
    }
    blocks_before = {
        name: tuple(
            np.array(block, copy=True)
            for block in history.blocks("surrogate_accuracy", name)
        )
        for name in ("predicted", "true")
    }
    fig = Figure()
    ax = fig.add_subplot(111)
    assert plot_surrogate_accuracy(_result(history), ax=ax) is fig
    for name, values in scalar_before.items():
        np.testing.assert_array_equal(
            history.channel("surrogate_accuracy")[name], values
        )
    for name, expected in blocks_before.items():
        for actual, before in zip(
            history.blocks("surrogate_accuracy", name), expected, strict=True
        ):
            np.testing.assert_array_equal(actual, before)


def test_short_real_saea_run_produces_all_unit_four_figures() -> None:
    result = minimize(
        lambda x: np.sum(x**2),
        dim=3,
        lb=[-1.0] * 3,
        ub=[1.0] * 3,
        surrogate="rbf",
        max_fe=24,
        pop_size=6,
        seed=0,
        verbose=False,
        history_channels=["summary", "surrogate_accuracy", "decision_candidates"],
    )
    assert result.history is not None
    recorded_history = result.history
    rng_state = deepcopy(result.ctx.rng.bit_generator.state)
    history_state = {
        "accuracy": {
            name: tuple(
                np.array(block, copy=True)
                for block in recorded_history.blocks("surrogate_accuracy", name)
            )
            for name in ("predicted", "true")
        },
        "decisions": {
            name: tuple(
                np.array(block, copy=True)
                for block in recorded_history.blocks("decision_candidates", name)
            )
            for name in ("selected", "acquisition_scores", "candidates")
        },
    }
    assert isinstance(plot_surrogate_accuracy(result), Figure)
    assert isinstance(plot_surrogate_error_history(result), Figure)
    assert isinstance(plot_prescreening(result, variables=(0, 1)), Figure)
    assert result.ctx.rng.bit_generator.state == rng_state
    for channel, blocks in history_state.items():
        channel_name = (
            "surrogate_accuracy" if channel == "accuracy" else "decision_candidates"
        )
        for name, expected in blocks.items():
            for actual, before in zip(
                recorded_history.blocks(channel_name, name), expected, strict=True
            ):
                np.testing.assert_array_equal(actual, before)
