"""Cross-cutting contracts shared by every public visualization function."""

from __future__ import annotations

import copy
import importlib
import inspect
import pkgutil
import subprocess
import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

import saealib.viz as viz
from saealib import (
    PSO,
    DirectStrategy,
    IslandModel,
    LHSInitializer,
    Optimizer,
    Problem,
    Termination,
    max_fe,
    minimize,
)
from saealib.acquisition import EHVIAcquisition
from saealib.exceptions import ValidationError
from saealib.execution.history import History
from saealib.surrogate import SklearnGPRSurrogate
from saealib.utils import uniform_weight_vectors

_CHANNELS = (
    "summary",
    "front",
    "population",
    "evaluation",
    "surrogate_accuracy",
    "decision_candidates",
)
_BLOCKS = {
    "front": ("f",),
    "population": ("f", "x"),
    "evaluation": ("candidate_ids", "f", "cv", "cost"),
    "surrogate_accuracy": ("predicted", "true"),
    "decision_candidates": (
        "candidate_ids",
        "selected",
        "acquisition_scores",
        "prediction_mean",
        "prediction_std",
        "candidates",
    ),
}


def _objective(x: np.ndarray) -> np.ndarray:
    return np.array([np.sum(x**2), np.sum((x - 0.5) ** 2), np.sum((x + 0.25) ** 2)])


def _single_objective(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def _island_optimizer(seed: int) -> Optimizer:
    problem = Problem(
        _single_objective,
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )
    optimizer = Optimizer(problem, seed=seed)
    return (
        optimizer.set_initializer(LHSInitializer(4, 4, seed))
        .set_algorithm(PSO())
        .set_strategy(DirectStrategy(n_offspring=2))
        .set_termination(Termination(max_fe(12)))
        .set_history(_CHANNELS)
    )


@pytest.fixture(scope="module")
def single_result() -> Any:
    return minimize(
        _single_objective,
        dim=2,
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
        max_fe=16,
        pop_size=4,
        seed=1,
        history_channels=_CHANNELS,
        verbose=False,
    )


@pytest.fixture(scope="module")
def multi_result() -> Any:
    return minimize(
        _objective,
        dim=2,
        n_obj=3,
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
        max_fe=16,
        pop_size=4,
        seed=2,
        history_channels=_CHANNELS,
        verbose=False,
    )


@pytest.fixture(scope="module")
def saea_result() -> Any:
    return minimize(
        _objective,
        dim=2,
        n_obj=3,
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
        surrogate="rbf",
        max_fe=20,
        pop_size=4,
        seed=3,
        history_channels=_CHANNELS,
        verbose=False,
    )


@pytest.fixture(scope="module")
def island_run() -> tuple[IslandModel, tuple[Any, ...]]:
    model = IslandModel(
        tuple(_island_optimizer(seed) for seed in (4, 5, 6)),
        topology="ring",
        migration_interval=2,
    )
    return model, model.run()


@pytest.fixture(scope="module")
def fitted_surrogate(saea_result: Any) -> SklearnGPRSurrogate:
    surrogate = SklearnGPRSurrogate(alpha=1e-6, optimizer=None, random_state=3)
    surrogate.fit(
        saea_result.ctx.archive.get_array("x"), saea_result.ctx.archive.get_array("f")
    )
    return surrogate


def _plot_cases() -> tuple[tuple[str, str, dict[str, Any]], ...]:
    return (
        (
            "plot_acquisition",
            "saea",
            {"resolution": 4, "variables": (0, 1), "rng": 0},
        ),
        ("plot_archive", "single", {}),
        ("plot_constraint_violation", "single", {}),
        ("plot_convergence", "single", {}),
        ("plot_design_pcp", "single", {}),
        ("plot_diversity", "single", {}),
        ("plot_hypervolume", "multi", {"reference_point": np.full(3, 10.0)}),
        ("plot_indicator", "multi", {"indicator": "spacing"}),
        ("plot_island_migration", "island", {}),
        ("plot_objective_heatmap", "multi", {}),
        ("plot_pareto", "multi", {"objectives": (0, 1, 2)}),
        (
            "plot_pareto_evolution",
            "multi",
            {"objectives": (0, 1, 2)},
        ),
        ("plot_pcp", "multi", {}),
        ("plot_prescreening", "saea", {"variables": (0, 1)}),
        ("plot_radar", "multi", {}),
        ("plot_running_metric", "multi", {}),
        (
            "plot_surrogate",
            "saea",
            {"resolution": 4, "variables": (0, 1), "objective": 0},
        ),
        (
            "plot_surrogate_accuracy",
            "saea",
            {"objective": 0},
        ),
        (
            "plot_surrogate_error_history",
            "saea",
            {"objective": 0},
        ),
        (
            "plot_surrogate_uncertainty",
            "saea",
            {"resolution": 4, "variables": (0, 1), "objective": 0},
        ),
        (
            "plot_variable_distribution",
            "single",
            {"variable": 0},
        ),
        (
            "plot_weight_vectors",
            "vectors",
            {"objectives": (0, 1, 2)},
        ),
    )


def _call_case(
    name: str,
    source: str,
    kwargs: dict[str, Any],
    results: dict[str, Any],
    fitted_surrogate: SklearnGPRSurrogate | None = None,
    ax: Axes | None = None,
) -> Figure:
    function = cast(Callable[..., Figure], getattr(viz, name))
    if source == "island":
        model, states = results[source]
        axes = kwargs.pop("axes", None)
        return function(model, states, **kwargs, axes=axes)
    if source == "vectors":
        return function(uniform_weight_vectors(3, 4), **kwargs, ax=ax)
    result = results[source]
    if name in {"plot_surrogate", "plot_surrogate_uncertainty", "plot_acquisition"}:
        assert fitted_surrogate is not None
        if name == "plot_acquisition":
            return function(
                result, fitted_surrogate, EHVIAcquisition(n_samples=4), **kwargs, ax=ax
            )
        return function(result, fitted_surrogate, **kwargs, ax=ax)
    if name == "plot_hypervolume":
        return function(result, **kwargs, ax=ax)
    return function(result, **kwargs, ax=ax)


def _snapshot(result: Any) -> dict[str, Any]:
    history = result.history
    assert history is not None
    channels: dict[str, Any] = {}
    for channel in history.enabled:
        channels[channel] = {
            "columns": {
                name: np.array(values, copy=True)
                for name, values in history.channel(channel).items()
            },
            "blocks": {
                name: tuple(
                    np.array(block, copy=True)
                    for block in history.blocks(channel, name)
                )
                for name in _BLOCKS.get(channel, ())
            },
        }
    ctx = getattr(result, "ctx", result)
    return {
        "rng": copy.deepcopy(ctx.rng.bit_generator.state),
        "fe": ctx.fe,
        "gen": ctx.gen,
        "archive": (ctx.archive.value_version, ctx.archive.structure_version),
        "population": (ctx.population.value_version, ctx.population.structure_version),
        "history": channels,
    }


def _assert_snapshot(result: Any, before: dict[str, Any]) -> None:
    after = _snapshot(result)
    assert after["rng"] == before["rng"]
    assert after["fe"] == before["fe"]
    assert after["gen"] == before["gen"]
    assert after["archive"] == before["archive"]
    assert after["population"] == before["population"]
    for channel, expected in before["history"].items():
        actual = after["history"][channel]
        assert actual["columns"].keys() == expected["columns"].keys()
        for name, values in expected["columns"].items():
            np.testing.assert_array_equal(actual["columns"][name], values)
        assert actual["blocks"].keys() == expected["blocks"].keys()
        for name, blocks in expected["blocks"].items():
            for actual_block, expected_block in zip(
                actual["blocks"][name], blocks, strict=True
            ):
                np.testing.assert_array_equal(actual_block, expected_block)


@pytest.mark.parametrize(
    "name,source,kwargs", _plot_cases(), ids=[name for name, _, _ in _plot_cases()]
)
def test_all_public_plots_return_root_figures_and_preserve_state(
    name: str,
    source: str,
    kwargs: dict[str, Any],
    single_result: Any,
    multi_result: Any,
    saea_result: Any,
    island_run: tuple[IslandModel, tuple[Any, ...]],
    fitted_surrogate: SklearnGPRSurrogate,
) -> None:
    results = {
        "single": single_result,
        "multi": multi_result,
        "saea": saea_result,
        "island": island_run,
    }
    snapshots = {
        key: _snapshot(value)
        for key, value in results.items()
        if key not in {"island", "vectors"}
    }
    if source == "island":
        island_before = tuple(_snapshot(state) for state in island_run[1])
        root = Figure()
        subfig = root.subfigures(1)
        assert isinstance(subfig, SubFigure)
        axes = (subfig.add_subplot(121), subfig.add_subplot(122))
        figure = _call_case(
            name, source, {**kwargs, "axes": axes}, results, fitted_surrogate
        )
        assert figure is root
        assert len(figure.axes) == 2
        for state, before in zip(island_run[1], island_before, strict=True):
            _assert_snapshot(state, before)
        return
    projection = (
        "3d"
        if name in {"plot_pareto", "plot_pareto_evolution", "plot_weight_vectors"}
        else "polar"
        if name == "plot_radar"
        else None
    )
    root = Figure()
    subfig = root.subfigures(1)
    assert isinstance(subfig, SubFigure)
    if projection == "3d":
        importlib.import_module("mpl_toolkits.mplot3d")
        ax = subfig.add_subplot(111, projection="3d")
    elif projection == "polar":
        ax = subfig.add_subplot(111, projection="polar")
    else:
        ax = subfig.add_subplot(111)
    figure = _call_case(name, source, kwargs, results, fitted_surrogate, ax=ax)
    assert isinstance(figure, Figure)
    assert not isinstance(figure, SubFigure)
    assert figure is root
    if source != "vectors":
        _assert_snapshot(results[source], snapshots[source])


@pytest.mark.parametrize("name", viz.__all__)
def test_public_plots_fail_actionably_without_matplotlib(name: str) -> None:
    saved_modules = sys.modules.copy()
    saved_meta_path = sys.meta_path.copy()

    class MatplotlibBlocker:
        def find_spec(self, fullname: str, path: object = None, target: object = None):
            if fullname == "matplotlib" or fullname.startswith("matplotlib."):
                raise ModuleNotFoundError(f"No module named {fullname!r}")
            return None

    blocker = MatplotlibBlocker()
    for key in list(sys.modules):
        if key == "matplotlib" or key.startswith("matplotlib."):
            sys.modules.pop(key, None)
    sys.meta_path.insert(0, blocker)
    try:
        with pytest.raises(ImportError, match=r"pip install saealib\[viz\]"):
            _call_without_matplotlib(name)
    finally:
        sys.meta_path[:] = saved_meta_path
        sys.modules.clear()
        sys.modules.update(saved_modules)
    assert sys.meta_path == saved_meta_path
    assert sys.modules == saved_modules


def _call_without_matplotlib(name: str) -> Any:
    function = getattr(viz, name)
    if name == "plot_weight_vectors":
        return function(np.ones((2, 2)))
    if name == "plot_island_migration":
        return function(None, None)
    if name == "plot_hypervolume":
        return function(None, None)
    if name == "plot_indicator":
        return function(None, "spacing")
    if name in {"plot_surrogate", "plot_surrogate_uncertainty"}:
        return function(None, None)
    if name == "plot_acquisition":
        return function(None, None, None)
    return function(None)


def test_public_surface_is_complete_and_import_safe() -> None:
    assert len(viz.__all__) == 22
    assert all(callable(getattr(viz, name)) for name in viz.__all__)
    assert {name for name, _, _ in _plot_cases()} == set(viz.__all__)
    private_modules = tuple(
        importlib.import_module(module.name)
        for module in pkgutil.iter_modules(viz.__path__, "saealib.viz.")
        if module.name.rsplit(".", 1)[-1].startswith("_")
    )
    defined = {
        name
        for module in private_modules
        for name, value in inspect.getmembers(module, inspect.isfunction)
        if name.startswith("plot_") and value.__module__ == module.__name__
    }
    assert defined == set(viz.__all__)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, saealib; assert 'saealib.viz' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_projection_contracts_reject_wrong_existing_axes(
    multi_result: Any,
) -> None:
    root = Figure()
    ax = root.add_subplot(111)
    with pytest.raises(ValidationError):
        viz.plot_pareto(multi_result, objectives=(0, 1, 2), ax=ax)
    with pytest.raises(ValidationError):
        viz.plot_radar(multi_result, ax=ax)


def test_degenerate_histories_are_handled_explicitly() -> None:
    empty: Any = SimpleNamespace(
        f=np.array([1.0]),
        history=History(("summary",)),
        ctx=SimpleNamespace(
            problem=SimpleNamespace(
                n_obj=1,
                direction=np.array([-1.0]),
            )
        ),
    )
    with pytest.raises(ValidationError):
        viz.plot_convergence(empty)

    one_generation = History(("summary",))
    one_generation.append(
        "summary",
        gen=0,
        fe=1,
        f_min_0=1.0,
        f_max_0=1.0,
        min_cv=0.0,
        feasible_ratio=1.0,
    )
    one: Any = SimpleNamespace(
        f=np.array([1.0]),
        history=one_generation,
        ctx=empty.ctx,
    )
    assert isinstance(viz.plot_convergence(one), Figure)

    empty_front = History(("front",))
    for gen in (0, 1):
        empty_front.append_block(
            "front",
            {"f": np.empty((0, 2))},
            gen=gen,
            fe=gen,
        )
    front: Any = SimpleNamespace(
        f=np.ones((1, 2)),
        history=empty_front,
        ctx=SimpleNamespace(
            problem=SimpleNamespace(
                n_obj=2,
                direction=np.full(2, -1.0),
            )
        ),
    )
    try:
        figure = viz.plot_pareto_evolution(front)
    except ValidationError:
        return
    assert isinstance(figure, Figure)
