from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from saealib import Optimizer, Problem, Termination, max_fe
from saealib.acquisition import EHVIAcquisition
from saealib.exceptions import ValidationError
from saealib.execution.history import History
from saealib.surrogate import SklearnGPRSurrogate, Surrogate
from saealib.surrogate.manager import LocalSurrogateManager
from saealib.surrogate.prediction import SurrogatePrediction
from saealib.viz import (
    plot_acquisition,
    plot_surrogate,
    plot_surrogate_uncertainty,
)


class _Model(Surrogate):
    def __init__(self, uncertainty: bool = False) -> None:
        self.provides_uncertainty = uncertainty

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        pass

    def predict(self, test_x: np.ndarray) -> SurrogatePrediction:
        value = np.sum(test_x**2, axis=1, keepdims=True)
        std = np.ones_like(value) * 0.1 if self.provides_uncertainty else None
        return SurrogatePrediction.objective(value, std=std)


class _Acquisition:
    def evaluate(self, candidates_x, prediction, archive, ctx):
        return SimpleNamespace(scores=ctx.rng.standard_normal(len(candidates_x)))


def _result(dim: int = 2):
    values = np.linspace(-1.0, 1.0, dim * 4).reshape(4, dim)
    services = SimpleNamespace(
        get=lambda name: {
            "DenseNumericView": SimpleNamespace(get_view=lambda genomes: genomes),
            "BoundsService": SimpleNamespace(
                bounds=(np.full(dim, -1.0), np.full(dim, 1.0))
            ),
            "FeatureEncoder": SimpleNamespace(encode=lambda genomes: genomes.array),
        }.get(name)
    )
    archive = SimpleNamespace(
        genomes=values,
        f=np.sum(values**2, axis=1, keepdims=True),
    )
    problem = SimpleNamespace(space=SimpleNamespace(services=services), n_obj=1)

    class _Context(SimpleNamespace):
        def replace(self, **changes):
            return _Context(**{**self.__dict__, **changes})

    history = History(("population",))
    history.append_block(
        "population", {"x": values}, gen=0, fe=len(values), size=len(values)
    )
    ctx = _Context(problem=problem, archive=archive, rng=np.random.default_rng(7))
    return SimpleNamespace(ctx=ctx, history=history)


def _history_snapshot(history: History):
    columns = {
        name: values.copy() for name, values in history.channel("population").items()
    }
    blocks = tuple(block.copy() for block in history.blocks("population", "x"))
    return columns, blocks


def test_surrogate_landscape_image_colorbar_and_archive() -> None:
    fig = plot_surrogate(_result(), _Model(), resolution=7)
    assert np.asarray(fig.axes[0].images[0].get_array()).shape == (7, 7)
    assert len(fig.axes[0].collections) == 1
    assert len(fig.axes) == 2


def test_uncertainty_requires_and_accepts_std() -> None:
    with pytest.raises(ValidationError, match="does not provide uncertainty"):
        plot_surrogate_uncertainty(_result(), _Model(), resolution=3)
    fig = plot_surrogate_uncertainty(_result(), _Model(True), resolution=3)
    assert np.asarray(fig.axes[0].images[0].get_array()).shape == (3, 3)


def test_uncertainty_works_with_sklearn_gpr() -> None:
    result = _result()
    train_x = result.ctx.archive.genomes
    train_y = result.ctx.archive.f
    surrogate = SklearnGPRSurrogate(alpha=1e-6, n_restarts_optimizer=0)
    surrogate.fit(train_x, train_y)

    fig = plot_surrogate_uncertainty(result, surrogate, resolution=3)

    assert np.asarray(fig.axes[0].images[0].get_array()).shape == (3, 3)


def test_dim3_requires_fixed_and_validates_resolution() -> None:
    result = _result(3)
    with pytest.raises(ValidationError, match="variables"):
        plot_surrogate(result, _Model())
    with pytest.raises(ValidationError, match="fixed"):
        plot_surrogate(result, _Model(), variables=(0, 1))
    with pytest.raises(ValidationError, match="outside"):
        plot_surrogate(result, _Model(), variables=(0, 1), fixed={2: 2.0})
    with pytest.raises(ValidationError, match="resolution"):
        plot_surrogate(result, _Model(), variables=(0, 1), fixed={2: 0.0}, resolution=1)
    with pytest.raises(ValidationError, match="resolution"):
        plot_surrogate(
            result,
            _Model(),
            variables=(0, 1),
            fixed={2: 0.0},
            resolution=cast(Any, 1.5),
        )
    with pytest.raises(ValidationError, match="invalid indices"):
        plot_surrogate(result, _Model(), variables=(0, 0), fixed={2: 0.0})
    with pytest.raises(ValidationError, match="overlaps"):
        plot_surrogate(result, _Model(), variables=(0, 1), fixed={1: 0.0, 2: 0.0})
    with pytest.raises(ValidationError, match="outside"):
        plot_surrogate(result, _Model(), variables=(0, 1), fixed={3: 0.0})


def test_acquisition_detaches_rng() -> None:
    result = _result()
    before = copy.deepcopy(result.ctx.rng.bit_generator.state)
    plot_acquisition(
        result,
        _Model(True),
        EHVIAcquisition(n_samples=4),
        resolution=4,
        rng=0,
    )
    assert result.ctx.rng.bit_generator.state == before


def test_landscapes_do_not_mutate_manager_or_history() -> None:
    result = _result()
    manager = LocalSurrogateManager(_Model(True))
    marker = object()
    manager.last_accuracy = cast(Any, marker)
    before_columns, before_blocks = _history_snapshot(result.history)

    plot_surrogate(result, manager.surrogate, resolution=3)
    plot_surrogate_uncertainty(result, manager.surrogate, resolution=3)
    plot_acquisition(
        result,
        manager.surrogate,
        EHVIAcquisition(n_samples=4),
        resolution=3,
        rng=0,
    )

    assert manager.last_accuracy is marker
    after_columns, after_blocks = _history_snapshot(result.history)
    for name, values in before_columns.items():
        np.testing.assert_array_equal(after_columns[name], values)
    for before, after in zip(before_blocks, after_blocks):
        np.testing.assert_array_equal(after, before)


def test_landscapes_work_with_a_short_optimizer_run() -> None:
    problem = Problem(
        lambda x: np.sum(np.asarray(x) ** 2),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )
    optimizer = Optimizer(problem, seed=4).set_termination(Termination(max_fe(24)))
    ctx = optimizer.run()
    result = cast(Any, SimpleNamespace(ctx=ctx, history=ctx.history))
    manager = cast(Any, optimizer.surrogate_manager)
    surrogate = manager.surrogate
    acquisition = cast(Any, optimizer.acquisition)

    assert isinstance(plot_surrogate(result, surrogate, resolution=4), Figure)
    assert isinstance(
        plot_acquisition(result, surrogate, acquisition, resolution=4, rng=0),
        Figure,
    )


def test_real_ehvi_landscape_does_not_mutate_context_rng() -> None:
    problem = Problem(
        lambda x: np.array(
            [np.sum(np.asarray(x) ** 2), np.sum((np.asarray(x) - 0.5) ** 2)]
        ),
        dim=2,
        n_obj=2,
        direction=np.array([-1.0, -1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )
    optimizer = (
        Optimizer(problem, seed=4)
        .set_surrogate(SklearnGPRSurrogate(alpha=1e-6, optimizer=None, random_state=4))
        .set_acquisition(EHVIAcquisition(n_samples=4))
        .set_termination(Termination(max_fe(16)))
    )
    ctx = optimizer.run()
    result = cast(Any, SimpleNamespace(ctx=ctx, history=ctx.history))
    train_x = ctx.archive.get_array("x")
    train_y = ctx.archive.get_array("f")
    surrogate = SklearnGPRSurrogate(alpha=1e-6, optimizer=None, random_state=4)
    surrogate.fit(train_x, train_y)
    acquisition = EHVIAcquisition(n_samples=4)

    before = copy.deepcopy(ctx.rng.bit_generator.state)
    plot_acquisition(result, surrogate, acquisition, resolution=3, rng=0)
    assert ctx.rng.bit_generator.state == before

    plot_acquisition(result, surrogate, acquisition, resolution=3, rng=None)
    assert ctx.rng.bit_generator.state == before

    candidates_x = train_x[:1]
    direct_before = copy.deepcopy(ctx.rng.bit_generator.state)
    acquisition.evaluate(
        candidates_x,
        surrogate.predict(candidates_x),
        ctx.archive,
        ctx,
    )
    assert ctx.rng.bit_generator.state != direct_before


def test_missing_dense_view_is_rejected() -> None:
    result = _result()
    result.ctx.problem.space.services = SimpleNamespace(get=lambda name: None)
    for function, args in (
        (cast(Any, plot_surrogate), (_Model(),)),
        (cast(Any, plot_surrogate_uncertainty), (_Model(True),)),
        (cast(Any, plot_acquisition), (_Model(True), _Acquisition())),
    ):
        with pytest.raises(ValidationError, match="dense numeric view"):
            function(result, *args, resolution=3)


def test_missing_bounds_service_is_rejected() -> None:
    result = _result()
    result.ctx.problem.space.services = SimpleNamespace(
        get=lambda name: (
            SimpleNamespace(get_view=lambda genomes: genomes)
            if name == "DenseNumericView"
            else None
        )
    )

    with pytest.raises(ValidationError, match="BoundsService"):
        plot_surrogate(result, _Model(), resolution=3)
