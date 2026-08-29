"""Tests for the unified result plot."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from saealib.exceptions import ValidationError
from saealib.result import Result
from saealib.viz import plot_result


class _Collection:
    def __init__(self, x: np.ndarray, f: np.ndarray) -> None:
        self.genomes = x
        self._f = f

    def __len__(self) -> int:
        return len(self.genomes)

    def get_array(self, name: str) -> np.ndarray:
        if name == "cv":
            return np.zeros(len(self.genomes))
        return self.genomes if name == "x" else self._f


def _result(n_obj: int = 2, dim: int = 3) -> Result:
    x = np.arange(12, dtype=float).reshape(4, 3)[:, :dim]
    f = np.arange(4 * n_obj, dtype=float).reshape(4, n_obj)
    archive = _Collection(x, f)
    population = _Collection(x[::-1], f[::-1])
    services = SimpleNamespace(get=lambda name: SimpleNamespace(get_view=lambda v: v))
    problem = SimpleNamespace(
        n_obj=n_obj,
        direction=np.full(n_obj, -1.0),
        eps_cv=0.0,
        space=SimpleNamespace(services=services),
    )
    ctx = SimpleNamespace(
        problem=problem,
        archive=archive,
        population=population,
        pareto_archive=archive,
    )
    return Result(x=x[0], f=f[0], fe=1, gen=1, ctx=ctx)


@pytest.mark.parametrize("source", ["result", "archive", "population"])
def test_objective_sources_plot(source: str) -> None:
    fig = plot_result(_result(2), source=source)

    assert isinstance(fig, Figure)
    assert len(fig.axes[0].collections) == 1


@pytest.mark.parametrize("source", ["result", "archive", "population"])
def test_decision_sources_plot(source: str) -> None:
    fig = plot_result(_result(2), space="decision", source=source, dimensions=(0, 1))

    assert isinstance(fig, Figure)
    assert len(fig.axes[0].collections) == 1
    assert fig.axes[0].get_xlabel() == "x0"


def test_auto_selects_scatter_or_parallel() -> None:
    assert plot_result(_result(2)).axes[0].name == "rectilinear"
    assert plot_result(_result(3)).axes[0].name == "3d"
    parallel = plot_result(_result(4), source="archive")
    assert len(parallel.axes[0].lines) == 4


def test_dimensions_can_reduce_parallel_to_scatter() -> None:
    fig = plot_result(_result(4), dimensions=(0, 2))

    assert len(fig.axes[0].collections) == 1


def test_one_dimension_is_not_a_supported_plot() -> None:
    with pytest.raises(ValidationError):
        plot_result(_result(4), dimensions=(0,), kind="parallel")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "scatter"},
        {"dimensions": ()},
        {"dimensions": (True, 1)},
        {"dimensions": (4,)},
    ],
)
def test_invalid_plot_options_raise(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        plot_result(_result(4), **kwargs)


@pytest.mark.parametrize(
    "key,value",
    [("space", "bad"), ("source", "bad"), ("kind", "bad")],
)
def test_unknown_options_raise(key: str, value: str) -> None:
    with pytest.raises(ValidationError):
        plot_result(_result(), **{key: value})


def test_existing_axes_and_subfigure_return_root_figure() -> None:
    root = Figure()
    subfigure = root.subfigures(1)
    ax = subfigure.add_subplot(111)

    assert plot_result(_result(), ax=ax) is root
    assert len(ax.collections) == 1


def test_state_is_accepted() -> None:
    result = _result()
    state = SimpleNamespace(
        archive=result.archive,
        pareto_archive=result.pareto_archive,
        problem=SimpleNamespace(
            n_obj=2,
            direction=np.array([-1.0, -1.0]),
            eps_cv=0.0,
        ),
        fe=1,
        gen=1,
        history=None,
    )
    fig = plot_result(state, space="objective")

    assert isinstance(fig, Figure) is True


def test_decision_requires_dense_numeric_view() -> None:
    result = _result()
    result.ctx.problem.space.services = SimpleNamespace(get=lambda name: None)

    with pytest.raises(ValidationError, match="dense numeric view"):
        plot_result(result, space="decision", source="archive")
