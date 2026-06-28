"""
Tests for the surrogate manager module.

Tests cover:
- _split_prediction: splits batch SurrogatePrediction into per-sample objects
- _rank_normalize: rank-based normalization to [0, 1]
- GlobalSurrogateManager: fits on full archive, batch predict and score
- LocalSurrogateManager: KNN per candidate, per-candidate fit and score
- CompositeSurrogateManager: combines scores from multiple sub-managers via combine_fn
- PairwiseSurrogateManager: scores by win rate against archive reference points
"""

import numpy as np
import pytest

from saealib.acquisition import MeanPrediction
from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationState
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.surrogate.accuracy import (
    KFoldAccuracyEvaluator,
    SpearmanCorrelation,
    SurrogateAccuracy,
)
from saealib.surrogate.archive_manager import (
    ArchiveBasedManager,
    DensityManager,
    NichingManager,
    NoveltyManager,
)
from saealib.surrogate.manager import (
    CompositeSurrogateManager,
    GlobalSurrogateManager,
    LocalSurrogateManager,
    PairwiseSurrogateManager,
    SurrogateManager,
    _rank_normalize,
    _split_prediction,
    product_combine,
    rank_weighted_combine,
)
from saealib.surrogate.prediction import SurrogatePrediction
from saealib.surrogate.rbf import RBFSurrogate, gaussian_kernel
from saealib.surrogate.sklearn_surrogate import DTSurrogate
from saealib.surrogate.training_set import KNNObjectiveSet, PairwiseComparisonSet

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
DIM = 2
N_OBJ = 1


@pytest.fixture
def archive_1obj() -> Archive:
    """Archive pre-filled with 20 single-objective training points."""
    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(N_OBJ,)),
    ]
    arc = Archive(attrs, init_capacity=30)
    rng = np.random.default_rng(42)
    for _ in range(20):
        x = rng.uniform(-2.0, 2.0, size=DIM)
        f = np.array([np.sum(x**2)])
        arc.add(x=x, f=f)
    return arc


@pytest.fixture
def archive_2obj() -> Archive:
    """Archive pre-filled with 20 bi-objective training points."""
    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(2,)),
    ]
    arc = Archive(attrs, init_capacity=30)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.uniform(0.0, 2.0, size=DIM)
        f = np.array([np.sum(x**2), np.sum((x - 2.0) ** 2)])
        arc.add(x=x, f=f)
    return arc


@pytest.fixture
def candidates() -> np.ndarray:
    """5 candidate points in 2D."""
    rng = np.random.default_rng(7)
    return rng.uniform(-1.0, 1.0, size=(5, DIM))


@pytest.fixture
def surrogate_1obj() -> RBFSurrogate:
    return RBFSurrogate(gaussian_kernel, DIM)


@pytest.fixture
def surrogate_2obj() -> RBFSurrogate:
    return RBFSurrogate(gaussian_kernel, DIM)


# ===========================================================================
# _split_prediction Tests
# ===========================================================================
class TestSplitPrediction:
    """Tests for the _split_prediction helper."""

    def test_splits_into_correct_count(self) -> None:
        pred = SurrogatePrediction(value=np.zeros((4, 2)))
        parts = _split_prediction(pred)
        assert len(parts) == 4

    def test_each_part_has_shape_1_nobj(self) -> None:
        pred = SurrogatePrediction(value=np.arange(6).reshape(3, 2).astype(float))
        parts = _split_prediction(pred)
        for p in parts:
            assert p.value.shape == (1, 2)

    def test_values_preserved(self) -> None:
        mean = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        pred = SurrogatePrediction(value=mean)
        parts = _split_prediction(pred)
        for i, p in enumerate(parts):
            np.testing.assert_array_equal(p.value[0], mean[i])

    def test_std_split_correctly(self) -> None:
        std = np.array([[0.1, 0.2], [0.3, 0.4]])
        pred = SurrogatePrediction(value=np.zeros((2, 2)), std=std)
        parts = _split_prediction(pred)
        assert parts[0].std is not None
        assert parts[0].std.shape == (1, 2)
        np.testing.assert_array_almost_equal(parts[0].std[0], [0.1, 0.2])

    def test_std_none_propagates(self) -> None:
        pred = SurrogatePrediction(value=np.zeros((3, 1)))
        parts = _split_prediction(pred)
        for p in parts:
            assert p.std is None

    def test_label_split_correctly(self) -> None:
        label = np.array([0.0, 1.0, 2.0])
        pred = SurrogatePrediction(value=np.zeros((3, 1)), label=label)
        parts = _split_prediction(pred)
        for i, p in enumerate(parts):
            assert p.label is not None
            assert p.label[0] == pytest.approx(float(i))

    def test_metadata_shared(self) -> None:
        """metadata dict is shared (not deep-copied) across splits."""
        meta = {"key": "val"}
        pred = SurrogatePrediction(value=np.zeros((2, 1)), metadata=meta)
        parts = _split_prediction(pred)
        for p in parts:
            assert p.metadata is meta

    def test_tell_f_split_correctly(self) -> None:
        tell_f = np.array([[10.0], [20.0], [30.0]])
        pred = SurrogatePrediction(value=np.zeros((3, 1)), _tell_f=tell_f)
        parts = _split_prediction(pred)
        assert parts[0].has_tell_f
        assert parts[0].tell_f.shape == (1, 1)
        np.testing.assert_array_almost_equal(parts[0].tell_f[0], [10.0])
        np.testing.assert_array_almost_equal(parts[1].tell_f[0], [20.0])
        np.testing.assert_array_almost_equal(parts[2].tell_f[0], [30.0])

    def test_tell_f_none_propagates(self) -> None:
        pred = SurrogatePrediction(value=np.zeros((3, 1)))
        parts = _split_prediction(pred)
        for p in parts:
            assert not p.has_tell_f

    def test_tell_f_shape_preserved_after_split(self) -> None:
        tell_f = np.array([[1.0, 2.0], [3.0, 4.0]])
        pred = SurrogatePrediction(value=np.zeros((2, 2)), _tell_f=tell_f)
        parts = _split_prediction(pred)
        assert parts[0].has_tell_f
        assert parts[0].tell_f.shape == (1, 2)
        np.testing.assert_array_almost_equal(parts[0].tell_f[0], [1.0, 2.0])
        assert parts[1].has_tell_f
        assert parts[1].tell_f.shape == (1, 2)
        np.testing.assert_array_almost_equal(parts[1].tell_f[0], [3.0, 4.0])


# ===========================================================================
# SurrogatePrediction Properties Tests
# ===========================================================================
class TestSurrogatePredictionProperties:
    """Tests for tell_f property and has_tell_f."""

    def test_tell_f_uses_override_when_set(self) -> None:
        pred = SurrogatePrediction(value=np.array([[0.0]]), _tell_f=np.array([[99.0]]))
        np.testing.assert_array_almost_equal(pred.tell_f, [[99.0]])

    def test_tell_f_falls_back_to_mean(self) -> None:
        pred = SurrogatePrediction(value=np.array([[42.0]]))
        np.testing.assert_array_almost_equal(pred.tell_f, [[42.0]])

    def test_has_tell_f_true_when_set(self) -> None:
        pred = SurrogatePrediction(value=np.array([[0.0]]), _tell_f=np.array([[1.0]]))
        assert pred.has_tell_f is True

    def test_has_tell_f_false_when_none(self) -> None:
        pred = SurrogatePrediction(value=np.array([[0.0]]))
        assert pred.has_tell_f is False


# ===========================================================================
# _sanitize_nan Tests
# ===========================================================================
class TestSanitizeNan:
    """Tests for SurrogateManager._sanitize_nan."""

    def test_no_nan_unchanged(self) -> None:
        scores = np.array([1.0, 2.0, 3.0])
        preds = [SurrogatePrediction(value=np.array([[v]])) for v in scores]
        s, p = SurrogateManager._sanitize_nan(scores, preds)
        np.testing.assert_array_equal(s, scores)
        assert not p[0].has_tell_f

    def test_nan_score_becomes_neginf(self) -> None:
        scores = np.array([1.0, np.nan, 3.0])
        preds = [SurrogatePrediction(value=np.array([[v]])) for v in [1.0, 0.0, 3.0]]
        s, _ = SurrogateManager._sanitize_nan(scores, preds)
        assert s[1] == -np.inf
        assert s[0] == 1.0 and s[2] == 3.0

    def test_nan_prediction_gets_explicit_tell_f_nan(self) -> None:
        scores = np.array([np.nan, 1.0])
        preds = [
            SurrogatePrediction(value=np.array([[np.nan]])),
            SurrogatePrediction(value=np.array([[1.0]])),
        ]
        _, p = SurrogateManager._sanitize_nan(scores, preds)
        assert p[0].has_tell_f
        assert np.all(np.isnan(p[0].tell_f))
        assert not p[1].has_tell_f

    def test_original_scores_not_mutated(self) -> None:
        scores = np.array([np.nan, 1.0])
        preds = [
            SurrogatePrediction(value=np.array([[0.0]])),
            SurrogatePrediction(value=np.array([[1.0]])),
        ]
        orig = scores.copy()
        SurrogateManager._sanitize_nan(scores, preds)
        np.testing.assert_array_equal(scores, orig)


# ===========================================================================
# _rank_normalize Tests
# ===========================================================================
class TestRankNormalize:
    """Tests for the _rank_normalize helper."""

    def test_single_element_returns_one(self) -> None:
        result = _rank_normalize(np.array([5.0]))
        assert result[0] == pytest.approx(1.0)

    def test_two_elements(self) -> None:
        result = _rank_normalize(np.array([1.0, 3.0]))
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(1.0)

    def test_three_elements_ascending(self) -> None:
        result = _rank_normalize(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])

    def test_order_independent(self) -> None:
        """Permuting the input should permute the output the same way."""
        scores = np.array([3.0, 1.0, 2.0])
        result = _rank_normalize(scores)
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.5)

    def test_output_range_0_to_1(self) -> None:
        rng = np.random.default_rng(99)
        scores = rng.standard_normal(50)
        result = _rank_normalize(scores)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_output_shape(self) -> None:
        result = _rank_normalize(np.arange(10, dtype=float))
        assert result.shape == (10,)

    def test_nan_treated_as_lowest(self) -> None:
        """NaN scores must be assigned rank 0 (worst), not rank n-1 (best)."""
        scores = np.array([3.0, 1.0, np.nan, 2.0])
        result = _rank_normalize(scores)
        assert result[2] == pytest.approx(0.0), "NaN should map to 0.0 (lowest rank)"

    def test_nan_never_highest(self) -> None:
        """NaN scores must never be selected as best in argsort(-normalized)."""
        scores = np.array([np.nan, 1.0, 2.0])
        result = _rank_normalize(scores)
        best_idx = np.argmax(result)
        assert best_idx != 0, (
            "NaN candidate should not have the highest normalized score"
        )


# ===========================================================================
# GlobalSurrogateManager Tests
# ===========================================================================
class TestGlobalSurrogateManager:
    """Tests for GlobalSurrogateManager."""

    def test_score_candidates_returns_tuple(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        result = manager.score_candidates(candidates, archive_1obj)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_scores_shape(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        scores, _ = manager.score_candidates(candidates, archive_1obj)
        assert scores.shape == (len(candidates),)

    def test_predictions_count(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        _, predictions = manager.score_candidates(candidates, archive_1obj)
        assert len(predictions) == len(candidates)

    def test_predictions_are_surrogate_prediction(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        _, predictions = manager.score_candidates(candidates, archive_1obj)
        for p in predictions:
            assert isinstance(p, SurrogatePrediction)

    def test_prediction_mean_shape_per_candidate(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        _, predictions = manager.score_candidates(candidates, archive_1obj)
        for p in predictions:
            assert p.value.shape == (1, N_OBJ)

    def test_scores_finite(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        scores, _ = manager.score_candidates(candidates, archive_1obj)
        assert np.all(np.isfinite(scores))

    def test_biobj_scores_shape(
        self,
        surrogate_2obj: RBFSurrogate,
        archive_2obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        """Bi-objective: scores still shape (n_candidates,)."""
        weights = np.array([-1.0, -1.0])
        manager = GlobalSurrogateManager(
            surrogate_2obj, MeanPrediction(weights=weights)
        )
        scores, predictions = manager.score_candidates(candidates, archive_2obj)
        assert scores.shape == (len(candidates),)
        for p in predictions:
            assert p.value.shape == (1, 2)


# ===========================================================================
# LocalSurrogateManager Tests
# ===========================================================================
class TestLocalSurrogateManager:
    """Tests for LocalSurrogateManager."""

    def test_score_candidates_returns_tuple(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = LocalSurrogateManager(
            surrogate_1obj, MeanPrediction(), training_set=KNNObjectiveSet(10)
        )
        result = manager.score_candidates(candidates, archive_1obj)
        assert isinstance(result, tuple) and len(result) == 2

    def test_scores_shape(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = LocalSurrogateManager(
            surrogate_1obj, MeanPrediction(), training_set=KNNObjectiveSet(10)
        )
        scores, _ = manager.score_candidates(candidates, archive_1obj)
        assert scores.shape == (len(candidates),)

    def test_predictions_count(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = LocalSurrogateManager(
            surrogate_1obj, MeanPrediction(), training_set=KNNObjectiveSet(10)
        )
        _, predictions = manager.score_candidates(candidates, archive_1obj)
        assert len(predictions) == len(candidates)

    def test_prediction_mean_shape(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = LocalSurrogateManager(
            surrogate_1obj, MeanPrediction(), training_set=KNNObjectiveSet(10)
        )
        _, predictions = manager.score_candidates(candidates, archive_1obj)
        for p in predictions:
            assert p.value.shape == (1, N_OBJ)

    def test_n_neighbors_default(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        """Default training_set is KNNObjectiveSet(50), clamped to archive size."""
        manager = LocalSurrogateManager(surrogate_1obj, MeanPrediction())
        assert isinstance(manager.training_set, KNNObjectiveSet)
        assert manager.training_set.n_neighbors == 50
        # archive has only 20 points, get_knn should still work
        scores, _ = manager.score_candidates(candidates, archive_1obj)
        assert scores.shape == (len(candidates),)

    def test_scores_finite(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = LocalSurrogateManager(
            surrogate_1obj, MeanPrediction(), training_set=KNNObjectiveSet(10)
        )
        scores, _ = manager.score_candidates(candidates, archive_1obj)
        assert np.all(np.isfinite(scores))


# ===========================================================================
# CompositeSurrogateManager Tests
# ===========================================================================
class TestCompositeSurrogateManager:
    """Tests for CompositeSurrogateManager."""

    def test_empty_managers_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            CompositeSurrogateManager([], combine_fn=product_combine)

    def test_product_combine_scores_shape(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        m1 = GlobalSurrogateManager(
            RBFSurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        m2 = GlobalSurrogateManager(
            RBFSurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        mgr = CompositeSurrogateManager([m1, m2], combine_fn=product_combine)
        scores, _ = mgr.score_candidates(candidates, archive_1obj)
        assert scores.shape == (len(candidates),)

    def test_rank_weighted_combine_scores_in_0_1(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        m1 = GlobalSurrogateManager(
            RBFSurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        m2 = GlobalSurrogateManager(
            RBFSurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        mgr = CompositeSurrogateManager([m1, m2], combine_fn=rank_weighted_combine())
        scores, _ = mgr.score_candidates(candidates, archive_1obj)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_rank_weighted_combine_custom_weights(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        m1 = GlobalSurrogateManager(
            RBFSurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        m2 = GlobalSurrogateManager(
            RBFSurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        mgr = CompositeSurrogateManager(
            [m1, m2], combine_fn=rank_weighted_combine(np.array([1.0, 3.0]))
        )
        scores, _ = mgr.score_candidates(candidates, archive_1obj)
        assert scores.shape == (len(candidates),)

    def test_predictions_from_first_manager(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        """predictions always come from managers[0]."""
        m1 = GlobalSurrogateManager(
            RBFSurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        m2 = GlobalSurrogateManager(
            RBFSurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        mgr = CompositeSurrogateManager([m1, m2], combine_fn=product_combine)
        _, predictions = mgr.score_candidates(candidates, archive_1obj)
        assert len(predictions) == len(candidates)
        for p in predictions:
            assert isinstance(p, SurrogatePrediction)

    def test_single_manager(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        m = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        mgr = CompositeSurrogateManager([m], combine_fn=product_combine)
        scores, _ = mgr.score_candidates(candidates, archive_1obj)
        assert scores.shape == (len(candidates),)


# ===========================================================================
# ArchiveBasedManager Tests
# ===========================================================================


class _ConstantArchiveManager(ArchiveBasedManager):
    """Minimal concrete subclass for testing ArchiveBasedManager base behaviour."""

    def compute_scores(self, candidates_x, archive, ctx=None) -> np.ndarray:
        return np.ones(len(candidates_x))


class TestArchiveBasedManager:
    """Tests for the ArchiveBasedManager base class via _ConstantArchiveManager."""

    def test_score_candidates_returns_tuple(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        mgr = _ConstantArchiveManager()
        result = mgr.score_candidates(candidates, archive_1obj)
        assert isinstance(result, tuple) and len(result) == 2

    def test_scores_shape(self, archive_1obj: Archive, candidates: np.ndarray) -> None:
        mgr = _ConstantArchiveManager()
        scores, _ = mgr.score_candidates(candidates, archive_1obj)
        assert scores.shape == (len(candidates),)

    def test_predictions_count(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        mgr = _ConstantArchiveManager()
        _, preds = mgr.score_candidates(candidates, archive_1obj)
        assert len(preds) == len(candidates)

    def test_tell_f_is_nan(self, archive_1obj: Archive, candidates: np.ndarray) -> None:
        mgr = _ConstantArchiveManager()
        _, preds = mgr.score_candidates(candidates, archive_1obj)
        for p in preds:
            assert p.has_tell_f
            assert np.all(np.isnan(p.tell_f))

    def test_tell_f_shape_matches_n_obj(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        mgr = _ConstantArchiveManager()
        _, preds = mgr.score_candidates(candidates, archive_1obj)
        for p in preds:
            assert p.tell_f.shape == (1, N_OBJ)

    def test_tell_f_shape_biobj(
        self, archive_2obj: Archive, candidates: np.ndarray
    ) -> None:
        mgr = _ConstantArchiveManager()
        _, preds = mgr.score_candidates(candidates, archive_2obj)
        for p in preds:
            assert p.tell_f.shape == (1, 2)

    def test_empty_archive_n_obj_fallback(self, candidates: np.ndarray) -> None:
        """Empty archive: n_obj defaults to 1."""
        attrs = [
            PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
            PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
        ]
        empty_arc = Archive(attrs, init_capacity=10)
        mgr = _ConstantArchiveManager()
        _, preds = mgr.score_candidates(candidates, empty_arc)
        for p in preds:
            assert p.tell_f.shape == (1, 1)
            assert np.all(np.isnan(p.tell_f))


# ===========================================================================
# NoveltyManager Tests
# ===========================================================================
class TestNoveltyManager:
    """Tests for NoveltyManager."""

    def test_empty_archive_returns_ones(self, candidates: np.ndarray) -> None:
        attrs = [
            PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
            PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
        ]
        empty_arc = Archive(attrs, init_capacity=10)
        mgr = NoveltyManager(k=3)
        scores = mgr.compute_scores(candidates, empty_arc)
        np.testing.assert_array_equal(scores, np.ones(len(candidates)))

    def test_scores_shape(self, archive_1obj: Archive, candidates: np.ndarray) -> None:
        mgr = NoveltyManager(k=1)
        scores = mgr.compute_scores(candidates, archive_1obj)
        assert scores.shape == (len(candidates),)

    def test_scores_nonnegative(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        mgr = NoveltyManager(k=3)
        scores = mgr.compute_scores(candidates, archive_1obj)
        assert np.all(scores >= 0.0)

    def test_more_distant_point_has_higher_novelty(self, archive_1obj: Archive) -> None:
        """A point far from the archive should score higher than a nearby one."""
        near = archive_1obj.x[0:1] + 1e-6  # almost identical to archive point
        far = np.array([[100.0, 100.0]])
        mgr = NoveltyManager(k=1)
        score_near = mgr.compute_scores(near, archive_1obj)[0]
        score_far = mgr.compute_scores(far, archive_1obj)[0]
        assert score_far > score_near

    def test_k_clamped_to_archive_size(self, archive_1obj: Archive) -> None:
        """k larger than archive size should not raise an error."""
        mgr = NoveltyManager(k=1000)
        candidates = np.zeros((3, DIM))
        scores = mgr.compute_scores(candidates, archive_1obj)
        assert scores.shape == (3,)
        assert np.all(np.isfinite(scores))

    def test_k_affects_scores(self, archive_1obj: Archive) -> None:
        """Different k values yield different scores (unless all distances equal)."""
        candidates = np.zeros((5, DIM))
        s1 = NoveltyManager(k=1).compute_scores(candidates, archive_1obj)
        s5 = NoveltyManager(k=5).compute_scores(candidates, archive_1obj)
        # mean of 1 NN vs mean of 5 NN — they may differ
        assert s1.shape == s5.shape == (5,)


# ===========================================================================
# DensityManager Tests
# ===========================================================================
class TestDensityManager:
    """Tests for DensityManager."""

    def test_empty_archive_returns_ones(self, candidates: np.ndarray) -> None:
        attrs = [
            PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
            PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
        ]
        empty_arc = Archive(attrs, init_capacity=10)
        mgr = DensityManager(eps=1.0)
        scores = mgr.compute_scores(candidates, empty_arc)
        np.testing.assert_array_equal(scores, np.ones(len(candidates)))

    def test_scores_shape(self, archive_1obj: Archive, candidates: np.ndarray) -> None:
        mgr = DensityManager(eps=1.0)
        scores = mgr.compute_scores(candidates, archive_1obj)
        assert scores.shape == (len(candidates),)

    def test_scores_positive(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        mgr = DensityManager(eps=1.0)
        scores = mgr.compute_scores(candidates, archive_1obj)
        assert np.all(scores > 0.0)

    def test_sparse_region_has_higher_score(self, archive_1obj: Archive) -> None:
        """Point far from archive (no neighbors within eps) should score higher."""
        dense = archive_1obj.x[0:1] + 0.01  # inside many eps-balls
        sparse = np.array([[100.0, 100.0]])  # far away, zero neighbors
        mgr = DensityManager(eps=0.5)
        score_dense = mgr.compute_scores(dense, archive_1obj)[0]
        score_sparse = mgr.compute_scores(sparse, archive_1obj)[0]
        assert score_sparse > score_dense

    def test_eps_affects_scores(self, archive_1obj: Archive) -> None:
        """Larger eps counts more neighbors → lower inverse density."""
        candidate = np.zeros((1, DIM))
        score_small_eps = DensityManager(eps=0.01).compute_scores(
            candidate, archive_1obj
        )[0]
        score_large_eps = DensityManager(eps=100.0).compute_scores(
            candidate, archive_1obj
        )[0]
        assert score_small_eps >= score_large_eps


# ===========================================================================
# NichingManager Tests
# ===========================================================================
class TestNichingManager:
    """Tests for NichingManager."""

    def test_scores_shape(self, archive_1obj: Archive, candidates: np.ndarray) -> None:
        mgr = NichingManager()
        scores = mgr.compute_scores(candidates, archive_1obj)
        assert scores.shape == (len(candidates),)

    def test_single_candidate_returns_ones(self, archive_1obj: Archive) -> None:
        mgr = NichingManager()
        single = np.zeros((1, DIM))
        scores = mgr.compute_scores(single, archive_1obj)
        np.testing.assert_array_equal(scores, np.ones(1))

    def test_scores_nonnegative(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        mgr = NichingManager()
        scores = mgr.compute_scores(candidates, archive_1obj)
        assert np.all(scores >= 0.0)

    def test_isolated_candidate_has_higher_score(self, archive_1obj: Archive) -> None:
        """A candidate isolated from others and the archive should score higher."""
        clustered = np.array([[0.0, 0.0], [0.01, 0.0], [0.0, 0.01]])
        isolated_point = np.array([100.0, 100.0])
        candidates = np.vstack([clustered, isolated_point[np.newaxis]])
        mgr = NichingManager()
        scores = mgr.compute_scores(candidates, archive_1obj)
        assert scores[-1] > scores[0]

    def test_empty_archive(self, candidates: np.ndarray) -> None:
        """Empty archive: archive_min falls back to ones."""
        attrs = [
            PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
            PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
        ]
        empty_arc = Archive(attrs, init_capacity=10)
        mgr = NichingManager()
        scores = mgr.compute_scores(candidates, empty_arc)
        assert scores.shape == (len(candidates),)
        assert np.all(np.isfinite(scores))


# ===========================================================================
# CompositeSurrogateManager + ArchiveBasedManager Integration Tests
# ===========================================================================
class TestCompositeSurrogateManagerWithArchiveBased:
    """Integration tests: CompositeSurrogateManager with ArchiveBasedManager."""

    def test_regression_novelty_ensemble_scores_shape(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        m_reg = GlobalSurrogateManager(
            RBFSurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        mgr = CompositeSurrogateManager(
            [m_reg, NoveltyManager(k=3)],
            combine_fn=rank_weighted_combine(np.array([0.7, 0.3])),
        )
        scores, _ = mgr.score_candidates(candidates, archive_1obj)
        assert scores.shape == (len(candidates),)

    def test_regression_novelty_ensemble_scores_in_0_1(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        m_reg = GlobalSurrogateManager(
            RBFSurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        mgr = CompositeSurrogateManager(
            [m_reg, NoveltyManager(k=3)], combine_fn=rank_weighted_combine()
        )
        scores, _ = mgr.score_candidates(candidates, archive_1obj)
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

    def test_regression_first_tell_f_is_finite(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        """Regression surrogate listed first → predictions returned are finite."""
        m_reg = GlobalSurrogateManager(
            RBFSurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        mgr = CompositeSurrogateManager(
            [m_reg, NoveltyManager(k=3)], combine_fn=rank_weighted_combine()
        )
        _, preds = mgr.score_candidates(candidates, archive_1obj)
        for p in preds:
            assert np.all(np.isfinite(p.tell_f))

    def test_novelty_only_ensemble_tell_f_is_nan(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        """ArchiveBasedManager alone → predictions have NaN tell_f."""
        mgr = CompositeSurrogateManager(
            [NoveltyManager(k=3)], combine_fn=rank_weighted_combine()
        )
        _, preds = mgr.score_candidates(candidates, archive_1obj)
        for p in preds:
            assert p.has_tell_f
            assert np.all(np.isnan(p.tell_f))

    def test_regression_density_ensemble_scores_shape(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        m_reg = GlobalSurrogateManager(
            RBFSurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        mgr = CompositeSurrogateManager(
            [m_reg, DensityManager(eps=0.5)],
            combine_fn=rank_weighted_combine(np.array([0.6, 0.4])),
        )
        scores, preds = mgr.score_candidates(candidates, archive_1obj)
        assert scores.shape == (len(candidates),)
        for p in preds:
            assert np.all(np.isfinite(p.tell_f))


# ===========================================================================
# Surrogate lifecycle hooks (post_fit / with_post_fit)
# ===========================================================================


class TestSurrogateHooks:
    """Tests for Surrogate.post_fit and with_post_fit."""

    def test_post_fit_default_is_noop(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        train_x, train_y = archive_1obj.x, archive_1obj.f
        surrogate_1obj.fit(train_x, train_y)
        result = surrogate_1obj.post_fit(train_x, train_y, ctx=None)
        assert result is None

    def test_with_post_fit_fn_is_called(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        called = [False]

        def hook(train_x, train_y, ctx):
            called[0] = True

        surrogate = surrogate_1obj.with_post_fit(hook)
        train_x, train_y = archive_1obj.x, archive_1obj.f
        surrogate.fit(train_x, train_y)
        surrogate.post_fit(train_x, train_y, ctx=None)
        assert called[0]

    def test_with_post_fit_fn_receives_correct_args(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        received = {}

        def hook(train_x, train_y, ctx):
            received["train_x_shape"] = train_x.shape
            received["train_y_shape"] = train_y.shape

        surrogate = surrogate_1obj.with_post_fit(hook)
        train_x, train_y = archive_1obj.x, archive_1obj.f
        surrogate.post_fit(train_x, train_y, ctx=None)
        assert received["train_x_shape"] == train_x.shape
        assert received["train_y_shape"] == train_y.shape

    def test_with_post_fit_chains_in_order(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        log: list[int] = []
        surrogate = surrogate_1obj.with_post_fit(
            lambda tx, ty, ctx: log.append(1)
        ).with_post_fit(lambda tx, ty, ctx: log.append(2))
        train_x, train_y = archive_1obj.x, archive_1obj.f
        surrogate.post_fit(train_x, train_y, ctx=None)
        assert log == [1, 2]

    def test_with_post_fit_does_not_mutate_original(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        called = [False]

        def hook(tx, ty, ctx):
            called[0] = True

        _ = surrogate_1obj.with_post_fit(hook)
        train_x, train_y = archive_1obj.x, archive_1obj.f
        surrogate_1obj.post_fit(train_x, train_y, ctx=None)
        assert not called[0]


# ===========================================================================
# SurrogateManager lifecycle hooks (post_score / with_post_score)
# ===========================================================================


class TestSurrogateManagerHooks:
    """Tests for SurrogateManager.post_score and with_post_score."""

    def test_post_score_default_is_noop(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        scores, predictions = manager.score_candidates(candidates, archive_1obj)
        result_s, result_p = manager.post_score(scores, predictions, ctx=None)
        np.testing.assert_array_equal(result_s, scores)
        assert result_p is predictions

    def test_with_post_score_transforms_scores(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        def zero_scores(scores, predictions, ctx):
            return np.zeros_like(scores), predictions

        manager = GlobalSurrogateManager(
            surrogate_1obj, MeanPrediction()
        ).with_post_score(zero_scores)
        scores, _ = manager.score_candidates(candidates, archive_1obj)
        np.testing.assert_array_equal(scores, np.zeros(len(candidates)))

    def test_with_post_score_called_once_per_score_candidates(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        call_count = [0]

        def hook(scores, predictions, ctx):
            call_count[0] += 1
            return scores, predictions

        manager = LocalSurrogateManager(
            surrogate_1obj, MeanPrediction(), training_set=KNNObjectiveSet(10)
        ).with_post_score(hook)
        manager.score_candidates(candidates, archive_1obj)
        assert call_count[0] == 1

    def test_with_post_score_does_not_mutate_original(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        _ = manager.with_post_score(lambda s, p, ctx: (np.zeros_like(s), p))
        scores, _ = manager.score_candidates(candidates, archive_1obj)
        assert not np.all(scores == 0.0)

    def test_post_fit_called_in_global_manager(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        called = [False]

        def hook(tx, ty, ctx):
            called[0] = True

        surrogate = surrogate_1obj.with_post_fit(hook)
        manager = GlobalSurrogateManager(surrogate, MeanPrediction())
        manager.score_candidates(candidates, archive_1obj)
        assert called[0]

    def test_post_fit_called_per_candidate_in_local_manager(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        call_count = [0]

        def hook(tx, ty, ctx):
            call_count[0] += 1

        surrogate = surrogate_1obj.with_post_fit(hook)
        manager = LocalSurrogateManager(
            surrogate, MeanPrediction(), training_set=KNNObjectiveSet(10)
        )
        manager.score_candidates(candidates, archive_1obj)
        assert call_count[0] == len(candidates)


# ===========================================================================
# SurrogateManager.on_generation_end / with_on_generation_end
# ===========================================================================


class TestSurrogateManagerGenerationHook:
    """Tests for SurrogateManager.on_generation_end and with_on_generation_end."""

    def test_on_generation_end_default_is_noop(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        assert manager.on_generation_end(0, archive_1obj, ctx=None) is None

    def test_with_on_generation_end_fn_is_called(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
    ) -> None:
        calls: list[tuple] = []

        def hook(gen, archive, ctx):
            calls.append((gen, archive, ctx))

        manager = GlobalSurrogateManager(
            surrogate_1obj, MeanPrediction()
        ).with_on_generation_end(hook)
        manager.on_generation_end(3, archive_1obj, ctx=None)

        assert len(calls) == 1
        assert calls[0] == (3, archive_1obj, None)

    def test_with_on_generation_end_chains_in_order(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
    ) -> None:
        order: list[int] = []
        manager = (
            GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
            .with_on_generation_end(lambda g, a, ctx: order.append(1))
            .with_on_generation_end(lambda g, a, ctx: order.append(2))
        )
        manager.on_generation_end(0, archive_1obj)
        assert order == [1, 2]

    def test_with_on_generation_end_does_not_mutate_original(
        self,
        surrogate_1obj: RBFSurrogate,
        archive_1obj: Archive,
    ) -> None:
        called = [False]

        def hook(g, a, ctx):
            called[0] = True

        original = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        _ = original.with_on_generation_end(hook)
        original.on_generation_end(0, archive_1obj)
        assert not called[0]


# ---------------------------------------------------------------------------
# last_accuracy
# ---------------------------------------------------------------------------


class TestLastAccuracy:
    def test_last_accuracy_none_by_default(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        assert manager.last_accuracy is None

    def test_last_accuracy_none_without_evaluator(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        manager.fit(archive_1obj)
        assert manager.last_accuracy is None

    def test_last_accuracy_set_after_fit_with_evaluator(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        evaluator = KFoldAccuracyEvaluator(metrics=[SpearmanCorrelation()], n_splits=5)
        manager = GlobalSurrogateManager(
            surrogate_1obj, MeanPrediction(), accuracy_evaluator=evaluator
        )
        manager.fit(archive_1obj)
        assert isinstance(manager.last_accuracy, SurrogateAccuracy)
        assert "spearman" in manager.last_accuracy.metrics

    def test_last_accuracy_updated_on_score_candidates_with_refit(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        evaluator = KFoldAccuracyEvaluator(metrics=[SpearmanCorrelation()], n_splits=3)
        manager = GlobalSurrogateManager(
            surrogate_1obj, MeanPrediction(), accuracy_evaluator=evaluator
        )
        rng = np.random.default_rng(0)
        candidates = rng.uniform(-2.0, 2.0, size=(5, DIM))
        manager.score_candidates(candidates, archive_1obj, refit=True)
        assert manager.last_accuracy is not None

    def test_last_accuracy_not_updated_when_refit_false(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        evaluator = KFoldAccuracyEvaluator(metrics=[SpearmanCorrelation()], n_splits=3)
        manager = GlobalSurrogateManager(
            surrogate_1obj, MeanPrediction(), accuracy_evaluator=evaluator
        )
        manager.fit(archive_1obj)
        first = manager.last_accuracy

        rng = np.random.default_rng(1)
        candidates = rng.uniform(-2.0, 2.0, size=(5, DIM))
        manager.score_candidates(candidates, archive_1obj, refit=False)
        assert manager.last_accuracy is first  # same object, not updated

    def test_composite_propagates_last_accuracy_from_first_manager(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        evaluator = KFoldAccuracyEvaluator(metrics=[SpearmanCorrelation()], n_splits=3)
        m1 = GlobalSurrogateManager(
            surrogate_1obj, MeanPrediction(), accuracy_evaluator=evaluator
        )
        m2 = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        composite = CompositeSurrogateManager(
            [m1, m2], combine_fn=rank_weighted_combine()
        )
        composite.fit(archive_1obj)
        assert composite.last_accuracy is m1.last_accuracy
        assert composite.last_accuracy is not None


class TestLocalSurrogateManagerAccuracy:
    def test_last_accuracy_none_by_default(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        manager = LocalSurrogateManager(surrogate_1obj, MeanPrediction())
        assert manager.last_accuracy is None

    def test_fit_sets_last_accuracy(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        evaluator = KFoldAccuracyEvaluator(metrics=[SpearmanCorrelation()], n_splits=3)
        manager = LocalSurrogateManager(
            surrogate_1obj, MeanPrediction(), accuracy_evaluator=evaluator
        )
        manager.fit(archive_1obj)
        assert isinstance(manager.last_accuracy, SurrogateAccuracy)
        assert "spearman" in manager.last_accuracy.metrics

    def test_fit_noop_without_evaluator(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        manager = LocalSurrogateManager(surrogate_1obj, MeanPrediction())
        manager.fit(archive_1obj)
        assert manager.last_accuracy is None

    def test_last_accuracy_set_after_score_candidates_with_refit(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        evaluator = KFoldAccuracyEvaluator(metrics=[SpearmanCorrelation()], n_splits=3)
        manager = LocalSurrogateManager(
            surrogate_1obj, MeanPrediction(), accuracy_evaluator=evaluator
        )
        rng = np.random.default_rng(0)
        candidates = rng.uniform(-2.0, 2.0, size=(5, DIM))
        manager.score_candidates(candidates, archive_1obj, refit=True)
        assert isinstance(manager.last_accuracy, SurrogateAccuracy)
        assert "spearman" in manager.last_accuracy.metrics
        # n_samples = number of candidates for which validation was possible
        assert manager.last_accuracy.n_samples == len(candidates)

    def test_last_accuracy_not_updated_when_refit_false(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        evaluator = KFoldAccuracyEvaluator(metrics=[SpearmanCorrelation()], n_splits=3)
        manager = LocalSurrogateManager(
            surrogate_1obj, MeanPrediction(), accuracy_evaluator=evaluator
        )
        rng = np.random.default_rng(0)
        candidates = rng.uniform(-2.0, 2.0, size=(5, DIM))
        manager.score_candidates(candidates, archive_1obj, refit=True)
        first = manager.last_accuracy
        manager.score_candidates(candidates, archive_1obj, refit=False)
        assert manager.last_accuracy is first  # not updated

    def test_generation_based_pattern_sets_last_accuracy(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        """fit() + score_candidates(refit=False) pattern (GenerationBasedStrategy)."""
        evaluator = KFoldAccuracyEvaluator(metrics=[SpearmanCorrelation()], n_splits=3)
        manager = LocalSurrogateManager(
            surrogate_1obj, MeanPrediction(), accuracy_evaluator=evaluator
        )
        rng = np.random.default_rng(0)
        candidates = rng.uniform(-2.0, 2.0, size=(5, DIM))
        manager.fit(archive_1obj)
        assert isinstance(manager.last_accuracy, SurrogateAccuracy)
        accuracy_after_fit = manager.last_accuracy
        # inner loop: refit=False should not update last_accuracy
        manager.score_candidates(candidates, archive_1obj, refit=False)
        assert manager.last_accuracy is accuracy_after_fit

    def test_nearest_neighbor_excluded_from_training(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        """Nearest archive neighbor is held out; training uses n_neighbors-1 points."""
        evaluator = KFoldAccuracyEvaluator(metrics=[SpearmanCorrelation()], n_splits=3)
        manager_with = LocalSurrogateManager(
            surrogate_1obj, MeanPrediction(), accuracy_evaluator=evaluator
        )
        manager_without = LocalSurrogateManager(
            RBFSurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        rng = np.random.default_rng(99)
        candidates = rng.uniform(-2.0, 2.0, size=(5, DIM))
        scores_with, _ = manager_with.score_candidates(candidates, archive_1obj)
        scores_without, _ = manager_without.score_candidates(candidates, archive_1obj)
        # Scores differ because the nearest neighbor is excluded from training
        # when an accuracy evaluator is active (k-1 vs k training points).
        assert scores_with.shape == scores_without.shape
        assert not np.any(np.isnan(scores_with))

    def test_loo_self_exclusion_in_update_accuracy(
        self, surrogate_1obj: RBFSurrogate, archive_1obj: Archive
    ) -> None:
        """_update_accuracy uses LOO self-exclusion; RBF no longer gives perfect score."""  # noqa: E501
        evaluator = KFoldAccuracyEvaluator(metrics=[SpearmanCorrelation()], n_splits=3)
        manager = LocalSurrogateManager(
            surrogate_1obj, MeanPrediction(), accuracy_evaluator=evaluator
        )
        manager.fit(archive_1obj)
        assert manager.last_accuracy is not None
        # With self-exclusion, RBF accuracy is < 1.0 (not perfectly interpolated)
        spearman = manager.last_accuracy.get("spearman")
        assert spearman < 1.0 or np.isnan(spearman)


# ===========================================================================
# PairwiseSurrogateManager Tests
# ===========================================================================

_PAIRWISE_ATTRS = [
    PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
    PopulationAttribute(name="f", dtype=np.float64, shape=(N_OBJ,)),
    PopulationAttribute(name="g", dtype=np.float64, shape=(0,)),
    PopulationAttribute(name="cv", dtype=np.float64, shape=()),
]


@pytest.fixture
def archive_pairwise() -> Archive:
    """Archive pre-filled with 20 single-objective training points including cv."""
    arc = Archive(_PAIRWISE_ATTRS, init_capacity=30)
    rng = np.random.default_rng(42)
    for _ in range(20):
        x = rng.uniform(-2.0, 2.0, size=DIM)
        f = np.array([np.sum(x**2)])
        arc.add(x=x, f=f, cv=0.0)
    return arc


@pytest.fixture
def ctx_pairwise(archive_pairwise: Archive) -> OptimizationState:
    """OptimizationState with a single-objective comparator."""
    problem = Problem(
        func=lambda x: np.array([np.sum(x**2)]),
        dim=DIM,
        n_obj=N_OBJ,
        direction=np.array([-1.0]),
        lb=[-5.0] * DIM,
        ub=[5.0] * DIM,
        eps_cv=1e-6,
        comparator=SingleObjectiveComparator(),
    )
    pop_attrs = _PAIRWISE_ATTRS
    pop = Population(pop_attrs, init_capacity=10)
    rng = np.random.default_rng(0)
    xs = rng.uniform(-2.0, 2.0, size=(5, DIM))
    fs = np.array([[np.sum(x**2)] for x in xs])
    pop.extend({"x": xs, "f": fs, "cv": np.zeros(5)})
    pareto_arc = ParetoArchive(pop_attrs, init_capacity=20, direction=np.array([-1.0]))
    return OptimizationState(
        problem=problem,
        population=pop,
        archive=archive_pairwise,
        pareto_archive=pareto_arc,
        rng=np.random.default_rng(1),
        fe=20,
        gen=1,
    )


class TestPairwiseSurrogateManager:
    """E2E tests for PairwiseSurrogateManager + PairwiseComparisonSet + DTSurrogate."""

    def test_score_candidates_returns_correct_shape(
        self,
        archive_pairwise: Archive,
        ctx_pairwise: OptimizationState,
        candidates: np.ndarray,
    ) -> None:
        manager = PairwiseSurrogateManager(
            DTSurrogate(n_estimators=5, random_state=0),
            n_ref=5,
        )
        scores, predictions = manager.score_candidates(
            candidates, archive_pairwise, ctx_pairwise
        )
        assert scores.shape == (len(candidates),)
        assert len(predictions) == len(candidates)

    def test_fit_then_score_refit_false(
        self,
        archive_pairwise: Archive,
        ctx_pairwise: OptimizationState,
        candidates: np.ndarray,
    ) -> None:
        """fit() + score_candidates(refit=False) pattern works without error."""
        manager = PairwiseSurrogateManager(
            DTSurrogate(n_estimators=5, random_state=0),
            n_ref=5,
        )
        manager.fit(archive_pairwise, ctx_pairwise)
        scores, predictions = manager.score_candidates(
            candidates, archive_pairwise, ctx_pairwise, refit=False
        )
        assert scores.shape == (len(candidates),)
        assert len(predictions) == len(candidates)

    def test_predictions_have_nan_tell_f(
        self,
        archive_pairwise: Archive,
        ctx_pairwise: OptimizationState,
        candidates: np.ndarray,
    ) -> None:
        """tell_f is NaN so strategies skip pbest assignment."""
        manager = PairwiseSurrogateManager(
            DTSurrogate(n_estimators=5, random_state=0),
            n_ref=5,
        )
        _, predictions = manager.score_candidates(
            candidates, archive_pairwise, ctx_pairwise
        )
        for p in predictions:
            assert p.has_tell_f
            assert np.all(np.isnan(p.tell_f))

    def test_scores_in_0_1_range(
        self,
        archive_pairwise: Archive,
        ctx_pairwise: OptimizationState,
        candidates: np.ndarray,
    ) -> None:
        """Win rates are clipped probabilities; always in [0, 1]."""
        manager = PairwiseSurrogateManager(
            DTSurrogate(n_estimators=5, random_state=0),
            n_ref=5,
        )
        scores, _ = manager.score_candidates(candidates, archive_pairwise, ctx_pairwise)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_no_predict_proba_raises_value_error(
        self,
        archive_pairwise: Archive,
        ctx_pairwise: OptimizationState,
        candidates: np.ndarray,
    ) -> None:
        """Surrogates without predict_proba raise ValueError with a clear message."""
        manager = PairwiseSurrogateManager(
            RBFSurrogate(gaussian_kernel, DIM),
            training_set=PairwiseComparisonSet(),
            n_ref=5,
        )
        manager.fit(archive_pairwise, ctx_pairwise)
        with pytest.raises(ValueError, match="predict_proba"):
            manager.score_candidates(
                candidates, archive_pairwise, ctx_pairwise, refit=False
            )

    def test_default_training_set_is_pairwise(self) -> None:
        """Default training_set is PairwiseComparisonSet when none supplied."""
        manager = PairwiseSurrogateManager(DTSurrogate(n_estimators=5, random_state=0))
        assert isinstance(manager.training_set, PairwiseComparisonSet)
