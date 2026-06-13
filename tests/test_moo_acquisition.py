"""
Tests for MOO acquisition functions: ParEGOAcquisition, SMSEGOAcquisition,
EHVIAcquisition.
"""

import numpy as np
import pytest

from saealib.acquisition import (
    EHVIAcquisition,
    ParEGOAcquisition,
    SMSEGOAcquisition,
)
from saealib.population import Archive, PopulationAttribute
from saealib.surrogate.prediction import SurrogatePrediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pred(value, std=None):
    m = np.asarray(value, dtype=float)
    s = np.asarray(std, dtype=float) if std is not None else None
    return SurrogatePrediction(value=m, std=s)


def _archive(*rows):
    """Build an Archive from objective-value rows (2-objective assumed)."""
    n_obj = len(rows[0])
    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(1,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(n_obj,)),
    ]
    arc = Archive(attrs, init_capacity=len(rows) + 5)
    for i, f_row in enumerate(rows):
        arc.add(x=np.array([float(i)]), f=np.asarray(f_row, dtype=float))
    return arc


# ===========================================================================
# ParEGOAcquisition
# ===========================================================================
class TestParEGOAcquisition:
    def test_requires_uncertainty(self) -> None:
        arc = _archive([1.0, 2.0])
        af = ParEGOAcquisition(rng=np.random.default_rng(0))
        ref = af.compute_reference(arc)
        pred = _pred(value=[[0.5, 1.5]])
        with pytest.raises(TypeError, match="uncertainty"):
            af.score(pred, ref)

    def test_output_shape(self) -> None:
        arc = _archive([1.0, 2.0], [2.0, 1.0])
        af = ParEGOAcquisition(rng=np.random.default_rng(0))
        ref = af.compute_reference(arc)
        pred = _pred(value=np.zeros((5, 2)), std=np.ones((5, 2)))
        scores = af.score(pred, ref)
        assert scores.shape == (5,)

    def test_nonnegative(self) -> None:
        rng = np.random.default_rng(1)
        arc = _archive([1.0, 3.0], [3.0, 1.0])
        af = ParEGOAcquisition(rng=rng)
        ref = af.compute_reference(arc)
        pred = _pred(
            value=rng.standard_normal((20, 2)),
            std=np.abs(rng.standard_normal((20, 2))) + 0.1,
        )
        scores = af.score(pred, ref)
        assert np.all(scores >= 0.0)

    def test_better_candidate_scores_higher(self) -> None:
        """Candidate near ideal point scores higher than one far away."""
        arc = _archive([1.0, 3.0], [3.0, 1.0])
        rng = np.random.default_rng(42)
        af = ParEGOAcquisition(alpha=0.05, rng=rng)
        ref = af.compute_reference(arc)
        # Candidate A: near ideal = (1,1)
        # Candidate B: far from ideal
        pred = _pred(
            value=[[1.1, 1.1], [5.0, 5.0]],
            std=[[0.1, 0.1], [0.1, 0.1]],
        )
        scores = af.score(pred, ref)
        assert scores[0] >= scores[1]

    def test_compute_reference_returns_tuple(self) -> None:
        arc = _archive([1.0, 2.0], [2.0, 1.0])
        af = ParEGOAcquisition(rng=np.random.default_rng(0))
        ref = af.compute_reference(arc)
        z_star, weights, f_best = ref
        assert z_star.shape == (2,)
        assert weights.shape == (2,)
        assert weights.sum() == pytest.approx(1.0, rel=1e-6)
        assert np.all(weights >= 0)
        assert isinstance(f_best, float)

    def test_ideal_point_is_component_min(self) -> None:
        arc = _archive([1.0, 5.0], [3.0, 2.0])
        af = ParEGOAcquisition(rng=np.random.default_rng(0))
        z_star, _, _ = af.compute_reference(arc)
        np.testing.assert_array_equal(z_star, [1.0, 2.0])

    def test_alpha_zero_is_pure_tchebycheff(self) -> None:
        """alpha=0 reduces to pure Tchebycheff; aug term is zero."""
        arc = _archive([1.0, 3.0], [3.0, 1.0])
        rng = np.random.default_rng(7)
        af = ParEGOAcquisition(alpha=0.0, rng=rng)
        ref = af.compute_reference(arc)
        # Should still produce valid EI scores
        pred = _pred(value=[[0.5, 0.5]], std=[[0.1, 0.1]])
        scores = af.score(pred, ref)
        assert scores.shape == (1,)
        assert scores[0] >= 0.0


# ===========================================================================
# SMSEGOAcquisition
# ===========================================================================
class TestSMSEGOAcquisition:
    def test_requires_uncertainty(self) -> None:
        arc = _archive([1.0, 2.0], [2.0, 1.0])
        af = SMSEGOAcquisition()
        ref = af.compute_reference(arc)
        pred = _pred(value=[[0.5, 0.5]])
        with pytest.raises(TypeError, match="uncertainty"):
            af.score(pred, ref)

    def test_output_shape(self) -> None:
        arc = _archive([1.0, 2.0], [2.0, 1.0])
        af = SMSEGOAcquisition(reference_point=[3.0, 3.0])
        ref = af.compute_reference(arc)
        pred = _pred(value=np.zeros((6, 2)), std=np.ones((6, 2)) * 0.1)
        scores = af.score(pred, ref)
        assert scores.shape == (6,)

    def test_nonnegative(self) -> None:
        arc = _archive([1.0, 3.0], [2.0, 1.5], [3.0, 1.0])
        af = SMSEGOAcquisition(reference_point=[5.0, 5.0])
        ref = af.compute_reference(arc)
        pred = _pred(
            value=np.random.default_rng(0).standard_normal((10, 2)) + 2,
            std=np.ones((10, 2)) * 0.5,
        )
        scores = af.score(pred, ref)
        assert np.all(scores >= 0.0)

    def test_dominated_candidate_scores_zero(self) -> None:
        """A candidate dominated by the Pareto front has HVI=0."""
        arc = _archive([1.0, 1.0])
        af = SMSEGOAcquisition(reference_point=[3.0, 3.0])
        ref = af.compute_reference(arc)
        # LCB well above the Pareto point → dominated
        pred = _pred(value=[[2.0, 2.0]], std=[[0.01, 0.01]])
        scores = af.score(pred, ref)
        assert scores[0] == pytest.approx(0.0, abs=1e-9)

    def test_improving_candidate_positive_hvi(self) -> None:
        """A non-dominated candidate with low LCB has positive HVI."""
        arc = _archive([2.0, 2.0])
        af = SMSEGOAcquisition(reference_point=[4.0, 4.0])
        ref = af.compute_reference(arc)
        # LCB(x) = [0.5, 0.5] — non-dominated, within ref box
        pred = _pred(value=[[0.6, 0.6]], std=[[0.1, 0.1]])
        # lam=1 → lcb = [0.5, 0.5]
        scores = af.score(pred, ref)
        assert scores[0] > 0.0

    def test_outside_ref_box_scores_zero(self) -> None:
        """A candidate with LCB outside the reference box scores zero."""
        arc = _archive([1.0, 1.0])
        af = SMSEGOAcquisition(reference_point=[2.0, 2.0])
        ref = af.compute_reference(arc)
        # LCB = mu - sigma = [3.0-0.1, 1.0-0.1] = [2.9, 0.9] — f1 > ref[0]
        pred = _pred(value=[[3.0, 1.0]], std=[[0.1, 0.1]])
        scores = af.score(pred, ref)
        assert scores[0] == pytest.approx(0.0, abs=1e-9)

    def test_auto_reference_point(self) -> None:
        """Auto-computed reference point should be strictly > nadir."""
        arc = _archive([1.0, 3.0], [3.0, 1.0])
        af = SMSEGOAcquisition()
        _, ref, _ = af.compute_reference(arc)
        assert np.all(ref > arc.f.max(axis=0))

    def test_lam_parameter(self) -> None:
        """Higher lam → more pessimistic LCB → generally lower or equal HVI."""
        arc = _archive([2.0, 2.0])
        ref_pt = np.array([5.0, 5.0])
        # mu=[1.5, 1.5], std=[0.5, 0.5]: lam=1→lcb=[1.0,1.0], lam=2→lcb=[0.5,0.5]
        pred = _pred(value=[[1.5, 1.5]], std=[[0.5, 0.5]])
        af1 = SMSEGOAcquisition(lam=1.0, reference_point=ref_pt)
        af2 = SMSEGOAcquisition(lam=2.0, reference_point=ref_pt)
        ref1 = af1.compute_reference(arc)
        ref2 = af2.compute_reference(arc)
        s1 = af1.score(pred, ref1)
        s2 = af2.score(pred, ref2)
        # lam=2 gives smaller lcb → larger improvement region
        assert s2[0] >= s1[0]


# ===========================================================================
# EHVIAcquisition
# ===========================================================================
class TestEHVIAcquisition:
    def test_requires_uncertainty(self) -> None:
        arc = _archive([1.0, 2.0], [2.0, 1.0])
        af = EHVIAcquisition(rng=np.random.default_rng(0))
        ref = af.compute_reference(arc)
        pred = _pred(value=[[0.5, 0.5]])
        with pytest.raises(TypeError, match="uncertainty"):
            af.score(pred, ref)

    def test_output_shape(self) -> None:
        arc = _archive([1.0, 2.0], [2.0, 1.0])
        af = EHVIAcquisition(
            n_samples=32, reference_point=[3.0, 3.0],
            rng=np.random.default_rng(0),
        )
        ref = af.compute_reference(arc)
        pred = _pred(value=np.zeros((5, 2)), std=np.ones((5, 2)) * 0.1)
        scores = af.score(pred, ref)
        assert scores.shape == (5,)

    def test_nonnegative(self) -> None:
        arc = _archive([1.0, 3.0], [3.0, 1.0])
        af = EHVIAcquisition(
            n_samples=32, reference_point=[5.0, 5.0],
            rng=np.random.default_rng(0),
        )
        ref = af.compute_reference(arc)
        rng = np.random.default_rng(1)
        pred = _pred(
            value=rng.standard_normal((8, 2)) + 2.0,
            std=np.abs(rng.standard_normal((8, 2))) + 0.1,
        )
        scores = af.score(pred, ref)
        assert np.all(scores >= 0.0)

    def test_dominated_region_scores_near_zero(self) -> None:
        """Candidate well dominated by Pareto front has near-zero EHVI."""
        arc = _archive([1.0, 1.0])
        af = EHVIAcquisition(
            n_samples=64, reference_point=[3.0, 3.0],
            rng=np.random.default_rng(0),
        )
        ref = af.compute_reference(arc)
        # mu=[2.5, 2.5], tiny std → nearly all samples are dominated
        pred = _pred(value=[[2.5, 2.5]], std=[[0.001, 0.001]])
        scores = af.score(pred, ref)
        assert scores[0] < 0.01

    def test_improving_candidate_positive_ehvi(self) -> None:
        """Candidate that is likely to improve HV has positive EHVI."""
        arc = _archive([2.0, 2.0])
        af = EHVIAcquisition(
            n_samples=128, reference_point=[4.0, 4.0],
            rng=np.random.default_rng(0),
        )
        ref = af.compute_reference(arc)
        # mu=[0.5, 0.5], small std → most samples non-dominated
        pred = _pred(value=[[0.5, 0.5]], std=[[0.1, 0.1]])
        scores = af.score(pred, ref)
        assert scores[0] > 0.0

    def test_better_candidate_scores_higher(self) -> None:
        """Candidate closer to ideal point has higher EHVI."""
        arc = _archive([2.0, 2.0])
        af = EHVIAcquisition(
            n_samples=256, reference_point=[4.0, 4.0],
            rng=np.random.default_rng(0),
        )
        ref = af.compute_reference(arc)
        pred = _pred(
            value=[[0.5, 0.5], [1.5, 1.5]],
            std=[[0.05, 0.05], [0.05, 0.05]],
        )
        scores = af.score(pred, ref)
        assert scores[0] > scores[1]

    def test_auto_reference_point(self) -> None:
        arc = _archive([1.0, 3.0], [3.0, 1.0])
        af = EHVIAcquisition(n_samples=16, rng=np.random.default_rng(0))
        _, ref, _ = af.compute_reference(arc)
        assert np.all(ref > arc.f.max(axis=0))

    def test_n_samples_parameter(self) -> None:
        """n_samples affects variance but both runs give valid shapes."""
        arc = _archive([1.0, 2.0], [2.0, 1.0])
        pred = _pred(value=[[0.5, 0.5]], std=[[0.1, 0.1]])
        ref_pt = [3.0, 3.0]
        for n in [16, 128]:
            af = EHVIAcquisition(
                n_samples=n, reference_point=ref_pt,
                rng=np.random.default_rng(0),
            )
            ref = af.compute_reference(arc)
            scores = af.score(pred, ref)
            assert scores.shape == (1,)
            assert scores[0] >= 0.0
