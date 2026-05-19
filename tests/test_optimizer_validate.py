"""Tests for Optimizer.validate() — Issue #73."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from saealib.acquisition.ei import ExpectedImprovement
from saealib.acquisition.mean import MeanPrediction
from saealib.comparators import SingleObjectiveComparator
from saealib.optimizer import Optimizer
from saealib.problem import Problem
from saealib.strategies.base import OptimizationStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.surrogate.manager import GlobalSurrogateManager
from saealib.surrogate.rbf import RBFsurrogate, gaussian_kernel

DIM = 2
N_OBJ = 1


def _make_problem(n_obj: int = N_OBJ) -> Problem:
    return Problem(
        func=lambda _: np.zeros(n_obj),
        dim=DIM,
        n_obj=n_obj,
        weight=np.array([-1.0] * n_obj),
        lb=[-5.0] * DIM,
        ub=[5.0] * DIM,
        comparator=SingleObjectiveComparator(),
    )


def _stub_strategy(requires_surrogate: bool = False) -> OptimizationStrategy:
    m = MagicMock(spec=OptimizationStrategy)
    m.requires_surrogate = requires_surrogate
    return m


def _fully_configured() -> Optimizer:
    opt = Optimizer(_make_problem())
    opt.set_algorithm(MagicMock())
    opt.set_strategy(_stub_strategy(requires_surrogate=False))
    opt.initializer = MagicMock()
    opt.set_termination(MagicMock())
    return opt


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_fully_configured_returns_empty():
    assert _fully_configured().validate() == []


# ---------------------------------------------------------------------------
# Missing required components
# ---------------------------------------------------------------------------


def test_missing_algorithm():
    opt = _fully_configured()
    del opt.algorithm
    assert any("algorithm" in m for m in opt.validate())


def test_missing_strategy():
    opt = _fully_configured()
    del opt.strategy
    assert any("strategy" in m for m in opt.validate())


def test_missing_initializer():
    opt = _fully_configured()
    opt.initializer = None
    assert any("initializer" in m for m in opt.validate())


def test_missing_termination():
    opt = _fully_configured()
    del opt.termination
    assert any("termination" in m for m in opt.validate())


def test_multiple_issues_reported_together():
    opt = Optimizer(_make_problem())
    issues = opt.validate()
    for kw in ("algorithm", "strategy", "initializer", "termination"):
        assert any(kw in m for m in issues), f"expected issue about '{kw}'"


# ---------------------------------------------------------------------------
# Strategy x Surrogate
# ---------------------------------------------------------------------------


def test_strategy_requires_surrogate_missing():
    opt = _fully_configured()
    opt.set_strategy(IndividualBasedStrategy())
    assert any("surrogate_manager" in m for m in opt.validate())


def test_strategy_requires_surrogate_present():
    opt = _fully_configured()
    opt.set_strategy(IndividualBasedStrategy())
    opt.set_surrogate_manager(
        GlobalSurrogateManager(
            RBFsurrogate(kernel=gaussian_kernel, dim=DIM),
            MeanPrediction(),
        )
    )
    assert not any("surrogate_manager" in m for m in opt.validate())


# ---------------------------------------------------------------------------
# Comparator x n_obj
# ---------------------------------------------------------------------------


def test_comparator_weight_mismatch():
    opt = _fully_configured()
    opt.problem.comparator.weights = np.array([1.0, 1.0])  # len 2 != n_obj=1
    assert any("weights" in m for m in opt.validate())


# ---------------------------------------------------------------------------
# Acquisition x Surrogate uncertainty
# ---------------------------------------------------------------------------


def test_acquisition_uncertainty_mismatch():
    opt = _fully_configured()
    opt.set_surrogate_manager(
        GlobalSurrogateManager(
            RBFsurrogate(kernel=gaussian_kernel, dim=DIM),
            ExpectedImprovement(),
        )
    )
    assert any("uncertainty" in m for m in opt.validate())


def test_mean_prediction_with_rbf_ok():
    opt = _fully_configured()
    opt.set_surrogate_manager(
        GlobalSurrogateManager(
            RBFsurrogate(kernel=gaussian_kernel, dim=DIM),
            MeanPrediction(),
        )
    )
    assert opt.validate() == []


# ---------------------------------------------------------------------------
# run() / iterate() raise on misconfiguration
# ---------------------------------------------------------------------------


def test_run_raises_on_misconfiguration():
    opt = Optimizer(_make_problem())
    with pytest.raises(ValueError, match="Optimizer misconfigured"):
        opt.run()


def test_iterate_raises_on_misconfiguration():
    opt = Optimizer(_make_problem())
    with pytest.raises(ValueError, match="Optimizer misconfigured"):
        opt.iterate()
