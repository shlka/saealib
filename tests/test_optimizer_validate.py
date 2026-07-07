"""Tests for Optimizer.validate() — Issue #73."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from saealib.acquisition.ei import ExpectedImprovement
from saealib.acquisition.mean import MeanPrediction
from saealib.acquisition.pof import ProbabilityOfFeasibility, ProductOfFeasibility
from saealib.acquisition.uncertainty import MaxUncertainty
from saealib.comparators import SingleObjectiveComparator
from saealib.execution.initializer import LHSInitializer
from saealib.optimizer import Optimizer
from saealib.problem import Problem
from saealib.strategies.base import OptimizationStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.surrogate.manager import GlobalSurrogateManager
from saealib.surrogate.rbf import RBFSurrogate, gaussian_kernel
from saealib.surrogate.sklearn_surrogate import SklearnGPRSurrogate

DIM = 2
N_OBJ = 1


def _make_problem(n_obj: int = N_OBJ) -> Problem:
    return Problem(
        func=lambda _: np.zeros(n_obj),
        dim=DIM,
        n_obj=n_obj,
        direction=np.array([-1.0] * n_obj),
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
            RBFSurrogate(kernel=gaussian_kernel, dim=DIM),
            MeanPrediction(),
        )
    )
    assert not any("surrogate_manager" in m for m in opt.validate())


# ---------------------------------------------------------------------------
# Comparator x n_obj
# ---------------------------------------------------------------------------


def test_comparator_weight_mismatch():
    opt = _fully_configured()
    opt.problem.comparator.direction = np.array([1.0, 1.0])  # len 2 != n_obj=1
    assert any("direction" in m for m in opt.validate())


# ---------------------------------------------------------------------------
# Acquisition x Surrogate uncertainty
# ---------------------------------------------------------------------------


def test_acquisition_uncertainty_mismatch():
    opt = _fully_configured()
    opt.set_surrogate_manager(
        GlobalSurrogateManager(
            RBFSurrogate(kernel=gaussian_kernel, dim=DIM),
            ExpectedImprovement(),
        )
    )
    assert any("uncertainty" in m for m in opt.validate())


def test_mean_prediction_with_rbf_ok():
    opt = _fully_configured()
    opt.set_surrogate_manager(
        GlobalSurrogateManager(
            RBFSurrogate(kernel=gaussian_kernel, dim=DIM),
            MeanPrediction(),
        )
    )
    assert opt.validate() == []


# ---------------------------------------------------------------------------
# run() / iterate() raise on misconfiguration that defaults cannot resolve
# ---------------------------------------------------------------------------


def test_run_raises_on_unresolvable_misconfiguration():
    opt = _fully_configured()
    opt.problem.comparator.direction = np.array([1.0, 1.0])  # len 2 != n_obj=1
    with pytest.raises(ValueError, match="Optimizer misconfigured"):
        opt.run()


def test_iterate_raises_on_unresolvable_misconfiguration():
    opt = _fully_configured()
    opt.problem.comparator.direction = np.array([1.0, 1.0])  # len 2 != n_obj=1
    with pytest.raises(ValueError, match="Optimizer misconfigured"):
        opt.iterate()


# ---------------------------------------------------------------------------
# run() / iterate() auto-resolve unset components instead of raising
# ---------------------------------------------------------------------------


def test_run_succeeds_with_no_components_set():
    opt = Optimizer(_make_problem())
    ctx = opt.run()
    assert ctx.fe > 0


def test_iterate_succeeds_with_no_components_set():
    opt = Optimizer(_make_problem())
    ctx = next(opt.iterate())
    assert ctx is not None


# ---------------------------------------------------------------------------
# Acquisition direction auto-injection (Issue #198)
# ---------------------------------------------------------------------------


def test_acquisition_direction_length_mismatch():
    opt = _fully_configured()
    opt.set_surrogate_manager(
        GlobalSurrogateManager(
            RBFSurrogate(kernel=gaussian_kernel, dim=DIM),
            ExpectedImprovement(direction=np.array([1.0, 1.0])),  # len 2 != n_obj=1
        )
    )
    assert any("direction" in m for m in opt.validate())


def test_inject_acquisition_directions_sets_from_problem():
    problem = _make_problem(n_obj=2)
    opt = Optimizer(problem)
    opt.set_algorithm(MagicMock())
    opt.set_strategy(_stub_strategy())
    opt.initializer = MagicMock()
    opt.set_termination(MagicMock())
    acq = MeanPrediction()
    opt.set_surrogate_manager(
        GlobalSurrogateManager(RBFSurrogate(kernel=gaussian_kernel, dim=2), acq)
    )
    opt._inject_acquisition_directions()
    np.testing.assert_array_equal(acq.direction, problem.direction)


def test_inject_acquisition_directions_idempotent():
    opt = _fully_configured()
    acq = MeanPrediction()
    opt.set_surrogate_manager(
        GlobalSurrogateManager(RBFSurrogate(kernel=gaussian_kernel, dim=DIM), acq)
    )
    opt._inject_acquisition_directions()
    first = acq.direction
    opt._inject_acquisition_directions()
    assert acq.direction is first


def test_inject_acquisition_directions_preserves_explicit_direction():
    opt = _fully_configured()
    explicit = np.array([1.0])
    acq = MeanPrediction(direction=explicit)
    opt.set_surrogate_manager(
        GlobalSurrogateManager(RBFSurrogate(kernel=gaussian_kernel, dim=DIM), acq)
    )
    opt._inject_acquisition_directions()
    assert acq.direction is explicit


def test_inject_acquisition_directions_skips_direction_insensitive():
    """PoF/ProductOfFeasibility/MaxUncertainty opt out and never get a direction."""
    opt = _fully_configured()
    for acq in (
        ProbabilityOfFeasibility(),
        ProductOfFeasibility(),
        MaxUncertainty(),
    ):
        opt.set_surrogate_manager(
            GlobalSurrogateManager(RBFSurrogate(kernel=gaussian_kernel, dim=DIM), acq)
        )
        opt._inject_acquisition_directions()
        # These acquisitions never declare a `direction` field at all; opting
        # out via direction_sensitive=False means injection must not add one.
        assert getattr(acq, "direction", None) is None


def test_iterate_injects_acquisition_direction_end_to_end():
    """End-to-end: iterate() auto-injects problem.direction into an unset
    acquisition."""
    dim = 2
    problem = Problem(
        func=lambda x: np.array([-np.sum(x**2)]),
        dim=dim,
        n_obj=1,
        direction=np.array([1.0]),  # maximize
        lb=[-5.0] * dim,
        ub=[5.0] * dim,
    )
    acq = ExpectedImprovement()
    opt = (
        Optimizer(problem, seed=0)
        .set_initializer(LHSInitializer(n_init_archive=10, n_init_population=8, seed=0))
        .set_strategy(IndividualBasedStrategy(evaluation_ratio=0.5))
        .set_surrogate_manager(GlobalSurrogateManager(SklearnGPRSurrogate(), acq))
    )
    gen = opt.iterate()
    next(gen)

    np.testing.assert_array_equal(acq.direction, problem.direction)

    # Idempotent: re-injecting does not change an already-set direction.
    opt._inject_acquisition_directions()
    np.testing.assert_array_equal(acq.direction, problem.direction)
