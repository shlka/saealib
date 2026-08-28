from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from saealib import minimize
from saealib.exceptions import ValidationError
from saealib.execution.history import History
from saealib.viz import (
    plot_archive,
    plot_design_pcp,
    plot_diversity,
    plot_variable_distribution,
)


class _Services:
    def __init__(
        self,
        bounds: tuple[list[float], list[float]] | None = None,
        dense: bool = True,
    ) -> None:
        self._bounds = bounds
        self._dense = dense

    def get(self, name: str) -> object | None:
        if name == "DenseNumericView" and self._dense:
            return SimpleNamespace(get_view=lambda genomes: genomes)
        if name == "BoundsService" and self._bounds is not None:
            return SimpleNamespace(bounds=tuple(np.asarray(v) for v in self._bounds))
        return None


def _result(
    values: np.ndarray,
    objectives: np.ndarray | None = None,
    history: History | None = None,
    bounds: tuple[list[float], list[float]] | None = None,
    dense: bool = True,
):
    if objectives is None:
        objectives = np.sum(values**2, axis=1, keepdims=True)
    services = _Services(bounds, dense=dense)

    class _Archive:
        genomes = values

        def __len__(self) -> int:
            return len(values)

        def get_array(self, name: str) -> np.ndarray:
            return values if name == "x" else objectives

    archive = _Archive()
    space = SimpleNamespace(services=services)
    ctx = SimpleNamespace(archive=archive, problem=SimpleNamespace(space=space))
    return SimpleNamespace(ctx=ctx, history=history, f=objectives)


def _population_result(rows: list[tuple[int, int, np.ndarray]], dim: int = 2):
    history = History(("population",))
    for gen, fe, values in rows:
        history.append_block(
            "population", {"x": values}, gen=gen, fe=fe, size=len(values)
        )
    return _result(
        np.zeros((1, dim)),
        history=history,
        bounds=([-1.0] * dim, [1.0] * dim),
    )


def test_plot_diversity_uses_bounds_and_excludes_fixed_variables() -> None:
    rows = [
        (0, 2, np.array([[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]])),
        (1, 3, np.array([[2.0, 1.0], [3.0, 1.0]])),
        (2, 4, np.array([[0.0, 1.0]])),
    ]
    result = _population_result(rows)
    result.ctx.problem.space.services = _Services(([-1.0, 1.0], [3.0, 1.0]))

    fig = plot_diversity(result)

    assert isinstance(fig, Figure)
    np.testing.assert_allclose(
        np.asarray(fig.axes[0].lines[0].get_ydata(), dtype=float),
        np.array([1 / 6, 0.25, np.nan]),
        equal_nan=True,
    )


def test_plot_diversity_is_invariant_to_absolute_position() -> None:
    rows = [
        (0, 2, np.array([[1.0, 2.0], [3.0, 4.0]])),
        (1, 4, np.array([[5.0, 6.0], [7.0, 8.0]])),
    ]
    result = _population_result(rows)
    result.ctx.problem.space.services = _Services(([0.0, 0.0], [10.0, 10.0]))

    fig = plot_diversity(result)

    np.testing.assert_allclose(
        np.asarray(fig.axes[0].lines[0].get_ydata(), dtype=float),
        np.array([0.2, 0.2]),
    )


def test_plot_design_pcp_uses_bounds_not_observed_range() -> None:
    values = np.array([[2.0, 10.0], [4.0, 20.0]])
    result = _result(values, bounds=([0.0, 0.0], [10.0, 100.0]))

    fig = plot_design_pcp(result)

    np.testing.assert_allclose(
        np.asarray(fig.axes[0].lines[0].get_ydata(), dtype=float),
        np.array([0.2, 0.1]),
    )


def test_plot_archive_requires_explicit_variables_for_three_dimensions() -> None:
    result = _result(np.zeros((2, 3)))
    with pytest.raises(ValidationError, match="variables"):
        plot_archive(result)
    fig = plot_archive(result, variables=(0, 2))
    assert len(fig.axes) == 2


def test_plot_archive_requires_objective_for_multiobjective() -> None:
    result = _result(np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValidationError, match="objective"):
        plot_archive(result)


def test_plot_variable_distribution_selection_and_default_cap() -> None:
    result = _population_result(
        [
            (gen, gen, np.array([[float(gen), 0.0], [float(gen) + 1, 1.0]]))
            for gen in range(25)
        ]
    )
    fig = plot_variable_distribution(result, variable=0)
    assert len(fig.axes[0].get_xticklabels()) == 20
    explicit = plot_variable_distribution(result, variable=1, generations=[2, 7])
    assert [text.get_text() for text in explicit.axes[0].get_xticklabels()] == [
        "2",
        "7",
    ]
    with pytest.raises(ValidationError, match="unrecorded"):
        plot_variable_distribution(result, variable=0, generations=[99])


def test_plot_variable_distribution_requires_variable_for_multiple_dimensions() -> None:
    result = _population_result([(0, 1, np.array([[0.0, 0.0], [1.0, 1.0]]))])

    with pytest.raises(ValidationError, match="requires variable"):
        plot_variable_distribution(result)


def test_design_plots_do_not_mutate_history() -> None:
    history = History(("population",))
    history.append_block(
        "population",
        {"x": np.array([[0.0, 0.0], [1.0, 1.0]])},
        gen=0,
        fe=2,
        size=2,
    )
    result = _result(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        history=history,
        bounds=([0.0, 0.0], [1.0, 1.0]),
    )
    before_columns = {
        name: values.copy() for name, values in history.channel("population").items()
    }
    before_blocks = tuple(block.copy() for block in history.blocks("population", "x"))

    plot_archive(result)
    plot_design_pcp(result)
    plot_variable_distribution(result, variable=0)
    plot_diversity(result)

    after_columns = history.channel("population")
    for name, values in before_columns.items():
        np.testing.assert_array_equal(after_columns[name], values)
    for before, after in zip(before_blocks, history.blocks("population", "x")):
        np.testing.assert_array_equal(after, before)


def test_design_plots_reject_spaces_without_dense_numeric_view() -> None:
    values = np.zeros((2, 2))
    result = _result(values, history=History(("population",)), dense=False)
    for plot in (
        plot_archive,
        plot_design_pcp,
        plot_variable_distribution,
        plot_diversity,
    ):
        with pytest.raises(ValidationError, match="dense numeric view"):
            plot(result)


def test_design_plots_work_with_a_short_real_run() -> None:
    result = minimize(
        lambda x: np.sum(x**2),
        dim=2,
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
        max_fe=16,
        pop_size=4,
        seed=0,
        history_channels=["summary", "population"],
        verbose=False,
    )
    assert isinstance(plot_archive(result), Figure)
    assert isinstance(plot_design_pcp(result), Figure)
    assert isinstance(plot_variable_distribution(result, variable=0), Figure)
    assert isinstance(plot_diversity(result), Figure)
