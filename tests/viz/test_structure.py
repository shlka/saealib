"""Tests for decision-space and island-structure plots."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from saealib import (
    PSO,
    DirectStrategy,
    IslandModel,
    LHSInitializer,
    Optimizer,
    Problem,
    Termination,
    max_fe,
)
from saealib.context import OptimizationState
from saealib.exceptions import ValidationError
from saealib.execution.history import History
from saealib.utils import uniform_weight_vectors
from saealib.viz import plot_island_migration, plot_weight_vectors


def _summary_history(
    rows: Sequence[tuple[int, float, float]],
) -> History:
    history = History(("summary",))
    for gen, (fe, f_min, f_max) in enumerate(rows):
        history.append(
            "summary",
            gen=gen,
            fe=fe,
            f_min_0=f_min,
            f_max_0=f_max,
            feasible_ratio=1.0,
            min_cv=0.0,
        )
    return history


def _state(
    rows: Sequence[tuple[int, float, float]],
    direction: float | Sequence[float] = -1.0,
    n_obj: int = 1,
) -> SimpleNamespace:
    if isinstance(direction, (int, float)):
        directions = np.full(n_obj, direction)
    else:
        directions = np.asarray(direction, dtype=float)
    return SimpleNamespace(
        problem=SimpleNamespace(n_obj=n_obj, direction=directions),
        history=_summary_history(rows),
    )


def _model(
    n_islands: int = 3,
    events: Sequence[tuple[int, int]] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        optimizers=tuple(SimpleNamespace() for _ in range(n_islands)),
        migration_events=list(events),
    )


def _plot_stub(
    model: SimpleNamespace,
    states: Sequence[SimpleNamespace],
    axes: Sequence[Axes] | None = None,
) -> Figure:
    return plot_island_migration(
        cast(IslandModel, model),
        cast(Sequence[OptimizationState], states),
        axes=axes,
    )


def test_plot_weight_vectors_selects_projection_and_labels_axes() -> None:
    fig_2d = plot_weight_vectors(uniform_weight_vectors(2, 4))

    assert isinstance(fig_2d, Figure)
    assert fig_2d.axes[0].name == "rectilinear"
    assert [fig_2d.axes[0].get_xlabel(), fig_2d.axes[0].get_ylabel()] == [
        "w0",
        "w1",
    ]

    fig_3d = plot_weight_vectors(uniform_weight_vectors(3, 4))

    assert isinstance(fig_3d, Figure)
    assert fig_3d.axes[0].name == "3d"
    axis_3d = cast(Any, fig_3d.axes[0])
    assert [axis_3d.get_xlabel(), axis_3d.get_ylabel(), axis_3d.get_zlabel()] == [
        "w0",
        "w1",
        "w2",
    ]


def test_plot_weight_vectors_requires_objectives_for_four_dimensions() -> None:
    vectors = np.eye(4)

    with pytest.raises(ValidationError, match="objectives"):
        plot_weight_vectors(vectors)

    fig = plot_weight_vectors(vectors, objectives=(0, 1, 3))

    assert isinstance(fig, Figure)
    assert fig.axes[0].name == "3d"
    axis_3d = cast(Any, fig.axes[0])
    assert [axis_3d.get_xlabel(), axis_3d.get_ylabel(), axis_3d.get_zlabel()] == [
        "w0",
        "w1",
        "w3",
    ]


def test_plot_weight_vectors_treats_one_dimensional_input_as_one_point() -> None:
    fig = plot_weight_vectors(np.array([0.25, 0.75]))

    offsets = np.asarray(fig.axes[0].collections[0].get_offsets())
    assert offsets.shape == (1, 2)
    np.testing.assert_allclose(offsets[0], [0.25, 0.75])


@pytest.mark.parametrize(
    "vectors",
    [np.empty((0, 2)), np.ones((2, 2, 2))],
    ids=["empty", "not-two-dimensional"],
)
def test_plot_weight_vectors_rejects_invalid_shapes(vectors: np.ndarray) -> None:
    with pytest.raises(ValidationError):
        plot_weight_vectors(vectors)


def test_plot_weight_vectors_rejects_out_of_range_objective() -> None:
    with pytest.raises(ValidationError, match="outside"):
        plot_weight_vectors(np.ones((2, 2)), objectives=(0, 2))


def test_plot_island_migration_aggregates_events_and_uses_summary_direction() -> None:
    model = _model(events=[(0, 1), (0, 1), (1, 0)])
    states = (
        _state(((10, 5.0, 50.0), (20, 3.0, 40.0))),
        _state(((10, 60.0, 7.0), (20, 50.0, 9.0)), direction=1.0),
        _state(((10, 8.0, 80.0), (20, 6.0, 70.0))),
    )

    fig = _plot_stub(model, states)

    migration_ax, convergence_ax = fig.axes
    offsets = np.asarray(migration_ax.collections[0].get_offsets())
    assert offsets.shape == (3, 2)
    assert [text.get_text() for text in migration_ax.texts[:3]] == ["0", "1", "2"]
    arrows = [
        getattr(text, "arrow_patch")
        for text in migration_ax.texts
        if getattr(text, "arrow_patch", None) is not None
    ]
    assert [arrow.get_linewidth() for arrow in arrows] == [3.0, 2.0]
    assert [text.get_text() for text in migration_ax.texts[3:] if text.get_text()] == [
        "2",
        "1",
    ]

    assert len(convergence_ax.lines) == 3
    assert [line.get_label() for line in convergence_ax.lines] == [
        "island 0",
        "island 1",
        "island 2",
    ]
    np.testing.assert_array_equal(convergence_ax.lines[0].get_xdata(), [10, 20])
    np.testing.assert_array_equal(convergence_ax.lines[0].get_ydata(), [5.0, 3.0])
    np.testing.assert_array_equal(convergence_ax.lines[1].get_xdata(), [10, 20])
    np.testing.assert_array_equal(convergence_ax.lines[1].get_ydata(), [7.0, 9.0])
    np.testing.assert_array_equal(convergence_ax.lines[2].get_xdata(), [10, 20])
    np.testing.assert_array_equal(convergence_ax.lines[2].get_ydata(), [8.0, 6.0])
    legend = convergence_ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "island 0",
        "island 1",
        "island 2",
    ]


def test_plot_island_migration_with_no_events_draws_nodes_only() -> None:
    model = _model(events=())
    states = (_state(((1, 2.0, 3.0),)),) * 3

    fig = _plot_stub(model, states)

    migration_ax = fig.axes[0]
    offsets = np.asarray(migration_ax.collections[0].get_offsets())
    assert offsets.shape == (3, 2)
    assert not any(
        getattr(text, "arrow_patch", None) is not None for text in migration_ax.texts
    )


def test_plot_island_migration_reuses_axes_and_returns_root_figure() -> None:
    model = _model(events=())
    states = (_state(((1, 2.0, 3.0),)),) * 3
    root = Figure()
    subfigure = root.subfigures(1)
    assert isinstance(subfigure, SubFigure)
    ax1 = subfigure.add_subplot(121)
    ax2 = subfigure.add_subplot(122)

    fig = _plot_stub(model, states, axes=(ax1, ax2))

    assert fig is root
    assert ax1.collections
    assert ax2.lines


def test_plot_island_migration_rejects_invalid_axes() -> None:
    model = _model(events=())
    states = (_state(((1, 2.0, 3.0),)),) * 3
    fig = Figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    other_fig = Figure()
    other_ax = other_fig.add_subplot(111)

    with pytest.raises(ValidationError):
        _plot_stub(model, states, axes=(ax1,))
    with pytest.raises(ValidationError):
        _plot_stub(model, states, axes=(ax1, ax2, other_ax))
    with pytest.raises(ValidationError, match="same Figure"):
        _plot_stub(model, states, axes=(ax1, other_ax))


def test_plot_island_migration_rejects_invalid_state_and_event_data() -> None:
    model = _model(events=())
    states = (_state(((1, 2.0, 3.0),)),) * 3

    with pytest.raises(ValidationError, match="one state"):
        _plot_stub(model, states[:2])

    with pytest.raises(ValidationError, match="outside"):
        _plot_stub(_model(events=[(0, 3)]), states)

    multiobjective = _state(((1, 2.0, 3.0),), direction=(-1.0, -1.0), n_obj=2)
    with pytest.raises(ValidationError, match="single-objective"):
        _plot_stub(_model(1), (multiobjective,))

    no_history = SimpleNamespace(
        problem=SimpleNamespace(n_obj=1, direction=np.array([-1.0])),
        history=None,
    )
    with pytest.raises(ValidationError, match="history"):
        _plot_stub(_model(1), (no_history,))

    invalid_history = History(("summary",))
    invalid_summary = SimpleNamespace(
        problem=SimpleNamespace(n_obj=1, direction=np.array([-1.0])),
        history=invalid_history,
    )
    with pytest.raises(ValidationError, match="summary"):
        _plot_stub(_model(1), (invalid_summary,))


def _island_optimizer(seed: int) -> Optimizer:
    def evaluate(x: np.ndarray) -> np.ndarray:
        return np.array([np.sum(x**2)], dtype=np.float64)

    return (
        Optimizer(
            Problem(
                func=evaluate,
                dim=2,
                n_obj=1,
                direction=np.array([-1.0]),
                lb=[-1.0, -1.0],
                ub=[1.0, 1.0],
            ),
            seed=seed,
        )
        .set_initializer(LHSInitializer(4, 4, seed))
        .set_algorithm(PSO())
        .set_strategy(DirectStrategy())
        .set_termination(Termination(max_fe(12)))
    )


def _history_snapshot(state: OptimizationState) -> dict[str, dict[str, np.ndarray]]:
    history = state.history
    assert history is not None
    return {
        channel: {
            name: np.asarray(values).copy()
            for name, values in history.channel(channel).items()
        }
        for channel in history.enabled
    }


def test_plot_island_migration_supports_real_model_and_preserves_histories() -> None:
    model = IslandModel(
        tuple(_island_optimizer(seed) for seed in (1, 2, 3)),
        topology="ring",
        migration_interval=2,
    )
    states = model.run()
    before = tuple(_history_snapshot(state) for state in states)

    fig = plot_island_migration(model, states)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    assert len(states) == 3
    for state, expected in zip(states, before):
        actual = _history_snapshot(state)
        assert actual.keys() == expected.keys()
        for channel, columns in expected.items():
            assert actual[channel].keys() == columns.keys()
            for name, values in columns.items():
                np.testing.assert_array_equal(actual[channel][name], values)
