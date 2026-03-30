"""
Tests for the acquisition module.

Tests cover:
- MeanPrediction: no weights (first objective), with weights (scalarization), shape
- MaxUncertainty: no weights (mean std), with weights, requires uncertainty
- ExpectedImprovement: basic EI formula, xi parameter, requires uncertainty
- LowerConfidenceBound: negated LCB, kappa parameter, requires uncertainty
- ProbabilityOfFeasibility: P(g<=0), requires uncertainty
- AcquisitionFunction: abstract base class cannot be instantiated
"""

import numpy as np
import pytest
from scipy.stats import norm

from saealib.acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    LowerConfidenceBound,
    MaxUncertainty,
    MeanPrediction,
    ProbabilityOfFeasibility,
)
from saealib.surrogate.prediction import SurrogatePrediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pred(mean, std=None):
    """Build a SurrogatePrediction from plain arrays."""
    m = np.asarray(mean, dtype=float)
    s = np.asarray(std, dtype=float) if std is not None else None
    return SurrogatePrediction(mean=m, std=s)


# ===========================================================================
# AcquisitionFunction (abstract base class) Tests
# ===========================================================================
class TestAcquisitionFunctionABC:
    """Tests for the AcquisitionFunction abstract base class."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            AcquisitionFunction()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_score(self) -> None:
        class IncompleteAF(AcquisitionFunction):
            pass

        with pytest.raises(TypeError):
            IncompleteAF()  # type: ignore[abstract]


# ===========================================================================
# MeanPrediction Tests
# ===========================================================================
class TestMeanPrediction:
    """Tests for MeanPrediction acquisition function."""

    def test_no_weights_returns_first_objective(self) -> None:
        """Without weights, returns prediction.mean[:, 0]."""
        pred = _pred(mean=[[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])
        scores = MeanPrediction().score(pred, reference=None)
        np.testing.assert_array_equal(scores, [1.0, 2.0, 3.0])

    def test_with_weights_returns_dot_product(self) -> None:
        """With weights, returns mean @ weights."""
        pred = _pred(mean=[[1.0, 2.0], [3.0, 4.0]])
        weights = np.array([-1.0, -1.0])
        scores = MeanPrediction(weights=weights).score(pred, reference=None)
        np.testing.assert_array_almost_equal(scores, [-3.0, -7.0])

    def test_output_shape(self) -> None:
        pred = _pred(mean=np.random.rand(5, 2))
        scores = MeanPrediction().score(pred, reference=None)
        assert scores.shape == (5,)

    def test_output_shape_with_weights(self) -> None:
        pred = _pred(mean=np.random.rand(7, 3))
        scores = MeanPrediction(weights=np.array([-1.0, -1.0, -1.0])).score(
            pred, reference=None
        )
        assert scores.shape == (7,)

    def test_reference_not_used(self) -> None:
        """reference parameter is accepted but ignored."""
        pred = _pred(mean=[[2.0]])
        s1 = MeanPrediction().score(pred, reference=None)
        s2 = MeanPrediction().score(pred, reference=np.array([99.0]))
        np.testing.assert_array_equal(s1, s2)

    def test_single_sample(self) -> None:
        pred = _pred(mean=[[4.0]])
        scores = MeanPrediction().score(pred, reference=None)
        assert scores.shape == (1,)
        assert scores[0] == pytest.approx(4.0)

    def test_single_objective_with_weight(self) -> None:
        pred = _pred(mean=[[3.0]])
        scores = MeanPrediction(weights=np.array([-1.0])).score(pred, reference=None)
        assert scores[0] == pytest.approx(-3.0)


# ===========================================================================
# MaxUncertainty Tests
# ===========================================================================
class TestMaxUncertainty:
    """Tests for MaxUncertainty acquisition function."""

    def test_no_weights_returns_mean_std(self) -> None:
        """Without weights, returns std.mean(axis=1)."""
        pred = _pred(
            mean=[[0.0, 0.0], [0.0, 0.0]],
            std=[[1.0, 3.0], [2.0, 4.0]],
        )
        scores = MaxUncertainty().score(pred, reference=None)
        np.testing.assert_array_almost_equal(scores, [2.0, 3.0])

    def test_with_weights(self) -> None:
        pred = _pred(
            mean=[[0.0, 0.0]],
            std=[[1.0, 2.0]],
        )
        scores = MaxUncertainty(weights=np.array([1.0, 0.0])).score(pred, reference=None)
        assert scores[0] == pytest.approx(1.0)

    def test_requires_uncertainty(self) -> None:
        pred = _pred(mean=[[1.0, 2.0]])
        with pytest.raises(TypeError, match="uncertainty"):
            MaxUncertainty().score(pred, reference=None)

    def test_output_shape(self) -> None:
        pred = _pred(
            mean=np.zeros((6, 2)),
            std=np.ones((6, 2)),
        )
        scores = MaxUncertainty().score(pred, reference=None)
        assert scores.shape == (6,)

    def test_single_objective(self) -> None:
        pred = _pred(mean=[[0.0]], std=[[0.5]])
        scores = MaxUncertainty().score(pred, reference=None)
        assert scores[0] == pytest.approx(0.5)


# ===========================================================================
# ExpectedImprovement Tests
# ===========================================================================
class TestExpectedImprovement:
    """Tests for ExpectedImprovement acquisition function."""

    def test_requires_uncertainty(self) -> None:
        pred = _pred(mean=[[1.0]])
        with pytest.raises(TypeError, match="uncertainty"):
            ExpectedImprovement().score(pred, reference=1.0)

    def test_output_shape(self) -> None:
        pred = _pred(mean=np.zeros((5, 1)), std=np.ones((5, 1)))
        scores = ExpectedImprovement().score(pred, reference=0.0)
        assert scores.shape == (5,)

    def test_ei_nonnegative(self) -> None:
        """EI scores are always >= 0."""
        rng = np.random.default_rng(0)
        mean = rng.standard_normal((20, 1))
        std = np.abs(rng.standard_normal((20, 1))) + 0.1
        pred = _pred(mean=mean, std=std)
        scores = ExpectedImprovement().score(pred, reference=0.0)
        assert np.all(scores >= 0.0)

    def test_ei_higher_for_better_candidate(self) -> None:
        """Candidate with lower mean (closer to current best from below) scores higher."""
        # reference best = 2.0; candidate with mean=1.5 should score higher than mean=3.0
        pred = _pred(
            mean=[[3.0], [1.5]],
            std=[[0.5], [0.5]],
        )
        scores = ExpectedImprovement(xi=0.0).score(pred, reference=2.0)
        assert scores[1] > scores[0]

    def test_ei_zero_for_worse_candidate(self) -> None:
        """EI is clipped to 0 when mu >> f_best."""
        pred = _pred(mean=[[100.0]], std=[[0.001]])
        scores = ExpectedImprovement(xi=0.0).score(pred, reference=1.0)
        assert scores[0] == pytest.approx(0.0, abs=1e-6)

    def test_ei_formula_manual(self) -> None:
        """Verify EI matches the analytical formula."""
        mu, sigma, f_best, xi = 1.0, 0.5, 2.0, 0.01
        z = (f_best - mu - xi) / sigma
        expected = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        pred = _pred(mean=[[mu]], std=[[sigma]])
        scores = ExpectedImprovement(xi=xi).score(pred, reference=f_best)
        assert scores[0] == pytest.approx(max(expected, 0.0), rel=1e-5)

    def test_ei_obj_idx(self) -> None:
        """obj_idx selects which objective to compute EI for."""
        pred = _pred(
            mean=[[10.0, 1.0]],
            std=[[0.5, 0.5]],
        )
        score_obj0 = ExpectedImprovement(obj_idx=0).score(pred, reference=np.array([2.0, 2.0]))
        score_obj1 = ExpectedImprovement(obj_idx=1).score(pred, reference=np.array([2.0, 2.0]))
        # obj1: mu=1.0 < f_best=2.0, positive EI; obj0: mu=10.0 >> f_best=2.0, ~0
        assert score_obj1[0] > score_obj0[0]

    def test_ei_xi_increases_exploration(self) -> None:
        """Higher xi shifts Z down, generally reducing score near f_best."""
        pred = _pred(mean=[[1.9]], std=[[0.1]])
        score_low_xi = ExpectedImprovement(xi=0.0).score(pred, reference=2.0)
        score_high_xi = ExpectedImprovement(xi=1.0).score(pred, reference=2.0)
        # xi=1.0 makes z=(2.0-1.9-1.0)/0.1 negative → lower EI
        assert score_low_xi[0] >= score_high_xi[0]


# ===========================================================================
# LowerConfidenceBound Tests
# ===========================================================================
class TestLowerConfidenceBound:
    """Tests for LowerConfidenceBound acquisition function."""

    def test_requires_uncertainty(self) -> None:
        pred = _pred(mean=[[1.0]])
        with pytest.raises(TypeError, match="uncertainty"):
            LowerConfidenceBound().score(pred, reference=None)

    def test_output_shape(self) -> None:
        pred = _pred(mean=np.zeros((4, 1)), std=np.ones((4, 1)))
        scores = LowerConfidenceBound().score(pred, reference=None)
        assert scores.shape == (4,)

    def test_negated_lcb_formula(self) -> None:
        """score = -(mu - kappa * sigma) = -mu + kappa * sigma."""
        mu, sigma, kappa = 2.0, 0.5, 3.0
        pred = _pred(mean=[[mu]], std=[[sigma]])
        scores = LowerConfidenceBound(kappa=kappa).score(pred, reference=None)
        expected = -(mu - kappa * sigma)
        assert scores[0] == pytest.approx(expected)

    def test_higher_score_for_lower_mean(self) -> None:
        """Candidate with lower predicted mean gets a higher score."""
        pred = _pred(mean=[[1.0], [3.0]], std=[[0.1], [0.1]])
        scores = LowerConfidenceBound(kappa=0.0).score(pred, reference=None)
        # kappa=0 → score = -mu; lower mu → higher score
        assert scores[0] > scores[1]

    def test_higher_score_for_higher_uncertainty(self) -> None:
        """Candidate with higher std gets a higher score (exploration)."""
        pred = _pred(mean=[[1.0], [1.0]], std=[[0.1], [1.0]])
        scores = LowerConfidenceBound(kappa=2.0).score(pred, reference=None)
        assert scores[1] > scores[0]

    def test_kappa_zero_equals_negative_mean(self) -> None:
        pred = _pred(mean=[[2.5], [3.5]], std=[[0.5], [0.5]])
        scores = LowerConfidenceBound(kappa=0.0).score(pred, reference=None)
        np.testing.assert_array_almost_equal(scores, [-2.5, -3.5])

    def test_obj_idx(self) -> None:
        """obj_idx selects which objective to compute LCB for."""
        pred = _pred(
            mean=[[1.0, 5.0]],
            std=[[0.1, 0.1]],
        )
        s0 = LowerConfidenceBound(kappa=0.0, obj_idx=0).score(pred, reference=None)
        s1 = LowerConfidenceBound(kappa=0.0, obj_idx=1).score(pred, reference=None)
        assert s0[0] == pytest.approx(-1.0)
        assert s1[0] == pytest.approx(-5.0)


# ===========================================================================
# ProbabilityOfFeasibility Tests
# ===========================================================================
class TestProbabilityOfFeasibility:
    """Tests for ProbabilityOfFeasibility acquisition function."""

    def test_requires_uncertainty(self) -> None:
        pred = _pred(mean=[[0.5]])
        with pytest.raises(TypeError, match="uncertainty"):
            ProbabilityOfFeasibility().score(pred, reference=None)

    def test_output_shape(self) -> None:
        pred = _pred(mean=np.zeros((5, 1)), std=np.ones((5, 1)))
        scores = ProbabilityOfFeasibility().score(pred, reference=None)
        assert scores.shape == (5,)

    def test_scores_in_0_1(self) -> None:
        """PoF scores are always in [0, 1]."""
        rng = np.random.default_rng(1)
        pred = _pred(
            mean=rng.standard_normal((20, 1)),
            std=np.abs(rng.standard_normal((20, 1))) + 0.01,
        )
        scores = ProbabilityOfFeasibility().score(pred, reference=None)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_pof_formula(self) -> None:
        """PoF = Phi(-mu / sigma)."""
        mu, sigma = -1.0, 0.5
        expected = norm.cdf((0.0 - mu) / sigma)
        pred = _pred(mean=[[mu]], std=[[sigma]])
        scores = ProbabilityOfFeasibility().score(pred, reference=None)
        assert scores[0] == pytest.approx(expected, rel=1e-5)

    def test_feasible_candidate_scores_near_one(self) -> None:
        """If mu << 0 (clearly feasible), PoF is near 1."""
        pred = _pred(mean=[[-10.0]], std=[[0.1]])
        scores = ProbabilityOfFeasibility().score(pred, reference=None)
        assert scores[0] > 0.99

    def test_infeasible_candidate_scores_near_zero(self) -> None:
        """If mu >> 0 (clearly infeasible), PoF is near 0."""
        pred = _pred(mean=[[10.0]], std=[[0.1]])
        scores = ProbabilityOfFeasibility().score(pred, reference=None)
        assert scores[0] < 0.01

    def test_obj_idx(self) -> None:
        """obj_idx selects which objective (constraint) to evaluate."""
        pred = _pred(
            mean=[[-5.0, 5.0]],
            std=[[0.1, 0.1]],
        )
        s0 = ProbabilityOfFeasibility(obj_idx=0).score(pred, reference=None)
        s1 = ProbabilityOfFeasibility(obj_idx=1).score(pred, reference=None)
        assert s0[0] > 0.99   # mu=-5: clearly feasible
        assert s1[0] < 0.01   # mu=+5: clearly infeasible
