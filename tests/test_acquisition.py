"""
Tests for the acquisition module.

Tests cover:
- MeanPrediction: no weights (first objective), with weights (scalarization), shape
- MaxUncertainty: no weights (mean std), with weights, requires uncertainty
- ExpectedImprovement: basic EI formula, xi parameter, requires uncertainty
- LowerConfidenceBound: negated LCB, kappa parameter, requires uncertainty
- ProbabilityOfFeasibility: P(g<=0), requires uncertainty
- CORSDistance: distance-constrained mean prediction, beta_i cycling (Issue #212)
- AcquisitionFunction: abstract base class cannot be instantiated
- direction-aware minimize-space conversion for EI/LCB (Issue #198)
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
    ProductOfFeasibility,
)
from saealib.acquisition.mean import CORSDistance
from saealib.exceptions import ValidationError
from saealib.population import Archive, PopulationAttribute
from saealib.surrogate.prediction import SurrogatePrediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pred(value, std=None):
    """Build a SurrogatePrediction from plain arrays."""
    m = np.asarray(value, dtype=float)
    s = np.asarray(std, dtype=float) if std is not None else None
    return SurrogatePrediction.objective(value=m, std=s)


def _archive(rows):
    """Build a single-objective Archive from objective-value rows."""
    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(1,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
    ]
    arc = Archive(attrs, init_capacity=len(rows) + 5)
    for i, f_row in enumerate(rows):
        arc.add(x=np.array([float(i)]), f=np.asarray(f_row, dtype=float))
    return arc


def _pred_x(value, x):
    """Build a SurrogatePrediction with candidate coordinates in the x field."""
    m = np.asarray(value, dtype=float)
    return SurrogatePrediction.objective(value=m, x=np.asarray(x, dtype=float))


def _archive_x(xs):
    """Build a 1-feature Archive containing exactly the given x coordinates."""
    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(1,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
    ]
    arc = Archive(attrs, init_capacity=len(xs) + 5)
    for x_val in xs:
        arc.add(x=np.array([float(x_val)]), f=np.array([0.0]))
    return arc


# ===========================================================================
# AcquisitionFunction (abstract base class) Tests
# ===========================================================================
class TestAcquisitionFunctionABC:
    """Tests for the AcquisitionFunction abstract base class."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            AcquisitionFunction()  # type: ignore[abstract]  # intentional: testing abstract instantiation raises TypeError

    def test_concrete_subclass_must_implement_score(self) -> None:
        class IncompleteAF(AcquisitionFunction):
            pass

        with pytest.raises(TypeError):
            IncompleteAF()  # type: ignore[abstract]  # intentional: testing abstract instantiation raises TypeError


# ===========================================================================
# MeanPrediction Tests
# ===========================================================================
class TestMeanPrediction:
    """Tests for MeanPrediction acquisition function."""

    def test_no_weights_returns_first_objective(self) -> None:
        """Without weights, returns prediction.mean[:, 0]."""
        pred = _pred(value=[[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])
        scores = MeanPrediction().score(pred, reference=None)
        np.testing.assert_array_equal(scores, [1.0, 2.0, 3.0])

    def test_with_weights_returns_dot_product(self) -> None:
        """With weights, returns mean @ weights."""
        pred = _pred(value=[[1.0, 2.0], [3.0, 4.0]])
        weights = np.array([-1.0, -1.0])
        scores = MeanPrediction(weights=weights).score(pred, reference=None)
        np.testing.assert_array_almost_equal(scores, [-3.0, -7.0])

    def test_output_shape(self) -> None:
        rng = np.random.default_rng(0)
        pred = _pred(value=rng.random((5, 2)))
        scores = MeanPrediction().score(pred, reference=None)
        assert scores.shape == (5,)

    def test_output_shape_with_weights(self) -> None:
        rng = np.random.default_rng(0)
        pred = _pred(value=rng.random((7, 3)))
        scores = MeanPrediction(weights=np.array([-1.0, -1.0, -1.0])).score(
            pred, reference=None
        )
        assert scores.shape == (7,)

    def test_reference_not_used(self) -> None:
        """reference parameter is accepted but ignored."""
        pred = _pred(value=[[2.0]])
        s1 = MeanPrediction().score(pred, reference=None)
        s2 = MeanPrediction().score(pred, reference=np.array([99.0]))
        np.testing.assert_array_equal(s1, s2)

    def test_single_sample(self) -> None:
        pred = _pred(value=[[4.0]])
        scores = MeanPrediction().score(pred, reference=None)
        assert scores.shape == (1,)
        assert scores[0] == pytest.approx(4.0)

    def test_single_objective_with_weight(self) -> None:
        pred = _pred(value=[[3.0]])
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
            value=[[0.0, 0.0], [0.0, 0.0]],
            std=[[1.0, 3.0], [2.0, 4.0]],
        )
        scores = MaxUncertainty().score(pred, reference=None)
        np.testing.assert_array_almost_equal(scores, [2.0, 3.0])

    def test_with_weights(self) -> None:
        pred = _pred(
            value=[[0.0, 0.0]],
            std=[[1.0, 2.0]],
        )
        scores = MaxUncertainty(weights=np.array([1.0, 0.0])).score(
            pred, reference=None
        )
        assert scores[0] == pytest.approx(1.0)

    def test_requires_uncertainty(self) -> None:
        pred = _pred(value=[[1.0, 2.0]])
        with pytest.raises(TypeError, match="uncertainty"):
            MaxUncertainty().score(pred, reference=None)

    def test_output_shape(self) -> None:
        pred = _pred(
            value=np.zeros((6, 2)),
            std=np.ones((6, 2)),
        )
        scores = MaxUncertainty().score(pred, reference=None)
        assert scores.shape == (6,)

    def test_single_objective(self) -> None:
        pred = _pred(value=[[0.0]], std=[[0.5]])
        scores = MaxUncertainty().score(pred, reference=None)
        assert scores[0] == pytest.approx(0.5)


# ===========================================================================
# ExpectedImprovement Tests
# ===========================================================================
class TestExpectedImprovement:
    """Tests for ExpectedImprovement acquisition function."""

    def test_requires_uncertainty(self) -> None:
        pred = _pred(value=[[1.0]])
        with pytest.raises(TypeError, match="uncertainty"):
            ExpectedImprovement().score(pred, reference=1.0)

    def test_output_shape(self) -> None:
        pred = _pred(value=np.zeros((5, 1)), std=np.ones((5, 1)))
        scores = ExpectedImprovement().score(pred, reference=0.0)
        assert scores.shape == (5,)

    def test_ei_nonnegative(self) -> None:
        """EI scores are always >= 0."""
        rng = np.random.default_rng(0)
        mean = rng.standard_normal((20, 1))
        std = np.abs(rng.standard_normal((20, 1))) + 0.1
        pred = _pred(value=mean, std=std)
        scores = ExpectedImprovement().score(pred, reference=0.0)
        assert np.all(scores >= 0.0)

    def test_ei_higher_for_better_candidate(self) -> None:
        """Candidate with lower mean scores higher EI."""
        # reference best = 2.0; mean=1.5 should score higher than mean=3.0
        pred = _pred(
            value=[[3.0], [1.5]],
            std=[[0.5], [0.5]],
        )
        scores = ExpectedImprovement(xi=0.0).score(pred, reference=2.0)
        assert scores[1] > scores[0]

    def test_ei_zero_for_worse_candidate(self) -> None:
        """EI is clipped to 0 when mu >> f_best."""
        pred = _pred(value=[[100.0]], std=[[0.001]])
        scores = ExpectedImprovement(xi=0.0).score(pred, reference=1.0)
        assert scores[0] == pytest.approx(0.0, abs=1e-6)

    def test_ei_formula_manual(self) -> None:
        """Verify EI matches the analytical formula."""
        mu, sigma, f_best, xi = 1.0, 0.5, 2.0, 0.01
        z = (f_best - mu - xi) / sigma
        expected = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        pred = _pred(value=[[mu]], std=[[sigma]])
        scores = ExpectedImprovement(xi=xi).score(pred, reference=f_best)
        assert scores[0] == pytest.approx(max(expected, 0.0), rel=1e-5)

    def test_ei_obj_idx(self) -> None:
        """obj_idx selects which objective to compute EI for."""
        pred = _pred(
            value=[[10.0, 1.0]],
            std=[[0.5, 0.5]],
        )
        score_obj0 = ExpectedImprovement(obj_idx=0).score(
            pred, reference=np.array([2.0, 2.0])
        )
        score_obj1 = ExpectedImprovement(obj_idx=1).score(
            pred, reference=np.array([2.0, 2.0])
        )
        # obj1: mu=1.0 < f_best=2.0, positive EI; obj0: mu=10.0 >> f_best=2.0, ~0
        assert score_obj1[0] > score_obj0[0]

    def test_ei_xi_increases_exploration(self) -> None:
        """Higher xi shifts Z down, generally reducing score near f_best."""
        pred = _pred(value=[[1.9]], std=[[0.1]])
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
        pred = _pred(value=[[1.0]])
        with pytest.raises(TypeError, match="uncertainty"):
            LowerConfidenceBound().score(pred, reference=None)

    def test_output_shape(self) -> None:
        pred = _pred(value=np.zeros((4, 1)), std=np.ones((4, 1)))
        scores = LowerConfidenceBound().score(pred, reference=None)
        assert scores.shape == (4,)

    def test_negated_lcb_formula(self) -> None:
        """score = -(mu - kappa * sigma) = -mu + kappa * sigma."""
        mu, sigma, kappa = 2.0, 0.5, 3.0
        pred = _pred(value=[[mu]], std=[[sigma]])
        scores = LowerConfidenceBound(kappa=kappa).score(pred, reference=None)
        expected = -(mu - kappa * sigma)
        assert scores[0] == pytest.approx(expected)

    def test_higher_score_for_lower_mean(self) -> None:
        """Candidate with lower predicted mean gets a higher score."""
        pred = _pred(value=[[1.0], [3.0]], std=[[0.1], [0.1]])
        scores = LowerConfidenceBound(kappa=0.0).score(pred, reference=None)
        # kappa=0 → score = -mu; lower mu → higher score
        assert scores[0] > scores[1]

    def test_higher_score_for_higher_uncertainty(self) -> None:
        """Candidate with higher std gets a higher score (exploration)."""
        pred = _pred(value=[[1.0], [1.0]], std=[[0.1], [1.0]])
        scores = LowerConfidenceBound(kappa=2.0).score(pred, reference=None)
        assert scores[1] > scores[0]

    def test_kappa_zero_equals_negative_mean(self) -> None:
        pred = _pred(value=[[2.5], [3.5]], std=[[0.5], [0.5]])
        scores = LowerConfidenceBound(kappa=0.0).score(pred, reference=None)
        np.testing.assert_array_almost_equal(scores, [-2.5, -3.5])

    def test_obj_idx(self) -> None:
        """obj_idx selects which objective to compute LCB for."""
        pred = _pred(
            value=[[1.0, 5.0]],
            std=[[0.1, 0.1]],
        )
        s0 = LowerConfidenceBound(kappa=0.0, obj_idx=0).score(pred, reference=None)
        s1 = LowerConfidenceBound(kappa=0.0, obj_idx=1).score(pred, reference=None)
        assert s0[0] == pytest.approx(-1.0)
        assert s1[0] == pytest.approx(-5.0)


# ===========================================================================
# CORSDistance Tests (Issue #212)
# ===========================================================================
class TestCORSDistance:
    """Tests for the CORS distance-constrained acquisition function."""

    def test_compute_reference_returns_archive_x(self) -> None:
        arc = _archive_x([0.0, 5.0, 10.0])
        af = CORSDistance(delta=10.0)
        ref = af.compute_reference(arc)
        np.testing.assert_array_equal(np.sort(ref.ravel()), [0.0, 5.0, 10.0])

    def test_far_candidate_scores_by_predicted_mean(self) -> None:
        """A candidate far from every evaluated point is unaffected by the constraint."""  # noqa: E501
        reference = _archive_x([0.0, 1.0, 2.0]).x
        pred = _pred_x(value=[[5.0]], x=[[100.0]])
        scores = CORSDistance(delta=1.0).score(pred, reference=reference)
        assert scores[0] == pytest.approx(5.0)

    def test_close_candidate_gets_worst_score(self) -> None:
        """A candidate violating beta_1 * delta gets -inf, never the predicted mean."""
        reference = _archive_x([0.0, 5.0, 10.0]).x
        # First score() call uses beta_1 = 0.95 (default SP1); threshold = 9.5.
        pred = _pred_x(value=[[5.0]], x=[[0.05]])
        scores = CORSDistance(delta=10.0).score(pred, reference=reference)
        assert scores[0] == -np.inf

    def test_beta_cycles_across_calls(self) -> None:
        """beta_i cycles through search_pattern, advancing once per score() call."""
        reference = _archive_x([0.0]).x
        pred = _pred_x(value=[[5.0]], x=[[0.05]])
        af = CORSDistance(delta=10.0, search_pattern=(1.0, 0.0))

        # Call 1: beta_1 = 1.0 -> threshold = 10.0 -> dist 0.05 violates.
        assert af.score(pred, reference=reference)[0] == -np.inf
        # Call 2: beta_2 = 0.0 -> threshold = 0.0 -> Eq. (1) trivially satisfied.
        assert af.score(pred, reference=reference)[0] == pytest.approx(5.0)
        # Call 3: wraps back to beta_1 = 1.0 -> violates again.
        assert af.score(pred, reference=reference)[0] == -np.inf

    def test_beta_zero_never_excludes(self) -> None:
        """A search_pattern of all zeros never enforces the distance constraint."""
        reference = _archive_x([0.0]).x
        pred = _pred_x(value=[[5.0]], x=[[0.0]])
        af = CORSDistance(delta=10.0, search_pattern=(0.0,))
        for _ in range(3):
            assert af.score(pred, reference=reference)[0] == pytest.approx(5.0)

    def test_empty_archive_no_constraint(self) -> None:
        """With no previously evaluated points, the constraint is vacuously satisfied."""  # noqa: E501
        reference = _archive_x([]).x
        pred = _pred_x(value=[[5.0]], x=[[0.0]])
        scores = CORSDistance(delta=10.0).score(pred, reference=reference)
        assert scores[0] == pytest.approx(5.0)

    def test_missing_x_raises(self) -> None:
        reference = _archive_x([0.0]).x
        pred = _pred(value=[[5.0]])
        with pytest.raises(ValueError, match="requires prediction"):
            CORSDistance(delta=10.0).score(pred, reference=reference)

    def test_x_row_mismatch_raises(self) -> None:
        with pytest.raises(ValidationError, match="shape"):
            _pred_x(value=[[5.0], [6.0]], x=[[0.0], [1.0], [2.0]])

    def test_direction_scalarizes_base_score(self) -> None:
        """The unconstrained base score respects direction, like MeanPrediction."""
        reference = _archive_x([0.0, 1.0]).x
        pred = _pred_x(value=[[3.0]], x=[[100.0]])
        scores = CORSDistance(delta=1.0, direction=np.array([-1.0])).score(
            pred, reference=reference
        )
        assert scores[0] == pytest.approx(-3.0)


# ===========================================================================
# ProbabilityOfFeasibility Tests
# ===========================================================================
class TestProbabilityOfFeasibility:
    """Tests for ProbabilityOfFeasibility acquisition function."""

    def test_requires_uncertainty(self) -> None:
        pred = _pred(value=[[0.5]])
        with pytest.raises(TypeError, match="uncertainty"):
            ProbabilityOfFeasibility().score(pred, reference=None)

    def test_output_shape(self) -> None:
        pred = _pred(value=np.zeros((5, 1)), std=np.ones((5, 1)))
        scores = ProbabilityOfFeasibility().score(pred, reference=None)
        assert scores.shape == (5,)

    def test_scores_in_0_1(self) -> None:
        """PoF scores are always in [0, 1]."""
        rng = np.random.default_rng(1)
        pred = _pred(
            value=rng.standard_normal((20, 1)),
            std=np.abs(rng.standard_normal((20, 1))) + 0.01,
        )
        scores = ProbabilityOfFeasibility().score(pred, reference=None)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_pof_formula(self) -> None:
        """PoF = Phi(-mu / sigma)."""
        mu, sigma = -1.0, 0.5
        expected = norm.cdf((0.0 - mu) / sigma)
        pred = _pred(value=[[mu]], std=[[sigma]])
        scores = ProbabilityOfFeasibility().score(pred, reference=None)
        assert scores[0] == pytest.approx(expected, rel=1e-5)

    def test_feasible_candidate_scores_near_one(self) -> None:
        """If mu << 0 (clearly feasible), PoF is near 1."""
        pred = _pred(value=[[-10.0]], std=[[0.1]])
        scores = ProbabilityOfFeasibility().score(pred, reference=None)
        assert scores[0] > 0.99

    def test_infeasible_candidate_scores_near_zero(self) -> None:
        """If mu >> 0 (clearly infeasible), PoF is near 0."""
        pred = _pred(value=[[10.0]], std=[[0.1]])
        scores = ProbabilityOfFeasibility().score(pred, reference=None)
        assert scores[0] < 0.01

    def test_obj_idx(self) -> None:
        """obj_idx selects which objective (constraint) to evaluate."""
        pred = _pred(
            value=[[-5.0, 5.0]],
            std=[[0.1, 0.1]],
        )
        s0 = ProbabilityOfFeasibility(obj_idx=0).score(pred, reference=None)
        s1 = ProbabilityOfFeasibility(obj_idx=1).score(pred, reference=None)
        assert s0[0] > 0.99  # mu=-5: clearly feasible
        assert s1[0] < 0.01  # mu=+5: clearly infeasible


# ===========================================================================
# ProductOfFeasibility Tests
# ===========================================================================
class TestProductOfFeasibility:
    """Tests for ProductOfFeasibility acquisition function."""

    def test_requires_uncertainty(self) -> None:
        pred = _pred(value=[[0.5, 0.5]])
        with pytest.raises(TypeError, match="uncertainty"):
            ProductOfFeasibility().score(pred, reference=None)

    def test_output_shape(self) -> None:
        pred = _pred(value=np.zeros((5, 2)), std=np.ones((5, 2)))
        scores = ProductOfFeasibility().score(pred, reference=None)
        assert scores.shape == (5,)

    def test_scores_in_0_1(self) -> None:
        rng = np.random.default_rng(2)
        pred = _pred(
            value=rng.standard_normal((20, 3)),
            std=np.abs(rng.standard_normal((20, 3))) + 0.01,
        )
        scores = ProductOfFeasibility().score(pred, reference=None)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_single_constraint_matches_pof(self) -> None:
        """With one constraint column, equals ProbabilityOfFeasibility(obj_idx=0)."""
        rng = np.random.default_rng(3)
        mu = rng.standard_normal((10, 1))
        sigma = np.abs(rng.standard_normal((10, 1))) + 0.01
        pred = _pred(value=mu, std=sigma)
        pof_scores = ProbabilityOfFeasibility(obj_idx=0).score(pred)
        product_scores = ProductOfFeasibility().score(pred)
        np.testing.assert_allclose(product_scores, pof_scores, rtol=1e-6)

    def test_all_feasible_scores_near_one(self) -> None:
        """If all mu << 0, joint PoF is near 1."""
        pred = _pred(value=[[-10.0, -10.0]], std=[[0.1, 0.1]])
        scores = ProductOfFeasibility().score(pred)
        assert scores[0] > 0.99

    def test_one_infeasible_constraint_pulls_score_to_zero(self) -> None:
        """Even if one constraint has mu >> 0, joint PoF is near 0."""
        pred = _pred(value=[[-10.0, 10.0]], std=[[0.1, 0.1]])
        scores = ProductOfFeasibility().score(pred)
        assert scores[0] < 0.01

    def test_product_formula(self) -> None:
        """PoF_joint = Phi(-mu1/s1) * Phi(-mu2/s2)."""
        mu1, s1, mu2, s2 = -1.0, 0.5, 0.5, 1.0
        expected = norm.cdf(-mu1 / s1) * norm.cdf(-mu2 / s2)
        pred = _pred(value=[[mu1, mu2]], std=[[s1, s2]])
        scores = ProductOfFeasibility().score(pred)
        assert scores[0] == pytest.approx(expected, rel=1e-5)


# ===========================================================================
# Direction-aware minimize-space conversion (Issue #198)
# ===========================================================================
class TestDirectionSensitivity:
    """
    EI/LCB internally convert to minimize-space via ``direction`` before
    running their (minimize-only) formulas. Scoring (mu, sigma) under
    ``direction=+1`` (maximize) must mirror scoring (-mu, sigma) under
    ``direction=-1`` (minimize) exactly, since ``direction=-1`` is a no-op
    conversion (sign=+1.0), anchoring the maximize side to already-verified
    minimize behavior.
    """

    def test_ei_maximize_mirrors_minimize_negated(self) -> None:
        archive_rows = [[2.0], [0.0], [-1.0]]
        mu = np.array([[1.0], [3.0], [-2.0], [0.5]])
        sigma = np.full((4, 1), 0.5)

        af_max = ExpectedImprovement(xi=0.0, direction=np.array([1.0]))
        af_min = ExpectedImprovement(xi=0.0, direction=np.array([-1.0]))

        ref_max = af_max.compute_reference(_archive(archive_rows))
        ref_min = af_min.compute_reference(_archive([[-v[0]] for v in archive_rows]))

        scores_max = af_max.score(_pred(mu, sigma), ref_max)
        scores_min = af_min.score(_pred(-mu, sigma), ref_min)

        np.testing.assert_allclose(scores_max, scores_min)
        assert np.argsort(scores_max).tolist() == np.argsort(scores_min).tolist()

    def test_lcb_maximize_mirrors_minimize_negated(self) -> None:
        mu = np.array([[1.0], [3.0], [-2.0], [0.5]])
        sigma = np.full((4, 1), 0.3)

        af_max = LowerConfidenceBound(kappa=1.5, direction=np.array([1.0]))
        af_min = LowerConfidenceBound(kappa=1.5, direction=np.array([-1.0]))

        scores_max = af_max.score(_pred(mu, sigma), reference=None)
        scores_min = af_min.score(_pred(-mu, sigma), reference=None)

        np.testing.assert_allclose(scores_max, scores_min)
        assert np.argsort(scores_max).tolist() == np.argsort(scores_min).tolist()

    def test_ei_direction_none_matches_direction_minus_one(self) -> None:
        """Default (direction=None) is a no-op, same as an explicit -1."""
        pred = _pred(value=[[1.5]], std=[[0.5]])
        s_default = ExpectedImprovement(xi=0.0).score(pred, reference=2.0)
        s_explicit = ExpectedImprovement(xi=0.0, direction=np.array([-1.0])).score(
            pred, reference=2.0
        )
        np.testing.assert_allclose(s_default, s_explicit)


class TestDirectionSensitiveOptOut:
    """PoF/ProductOfFeasibility/MaxUncertainty opt out via direction_sensitive=False."""

    def test_probability_of_feasibility_opts_out(self) -> None:
        assert ProbabilityOfFeasibility.direction_sensitive is False

    def test_product_of_feasibility_opts_out(self) -> None:
        assert ProductOfFeasibility.direction_sensitive is False

    def test_max_uncertainty_opts_out(self) -> None:
        assert MaxUncertainty.direction_sensitive is False

    def test_ei_lcb_parego_smsego_ehvi_are_direction_sensitive(self) -> None:
        assert ExpectedImprovement.direction_sensitive is True
        assert LowerConfidenceBound.direction_sensitive is True
