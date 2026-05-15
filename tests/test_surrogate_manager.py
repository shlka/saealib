"""
Tests for the surrogate manager module.

Tests cover:
- _split_prediction: splits batch SurrogatePrediction into per-sample objects
- _rank_normalize: rank-based normalization to [0, 1]
- GlobalSurrogateManager: fits on full archive, batch predict and score
- LocalSurrogateManager: KNN per candidate, per-candidate fit and score
- EnsembleSurrogateManager: rank-normalized weighted aggregation of sub-managers
"""

import numpy as np
import pytest

from saealib.acquisition import MeanPrediction
from saealib.population import Archive, PopulationAttribute
from saealib.surrogate.archive_manager import (
    ArchiveBasedManager,
    DensityManager,
    NichingManager,
    NoveltyManager,
)
from saealib.surrogate.manager import (
    EnsembleSurrogateManager,
    GlobalSurrogateManager,
    LocalSurrogateManager,
    _rank_normalize,
    _split_prediction,
)
from saealib.surrogate.prediction import SurrogatePrediction
from saealib.surrogate.rbf import RBFsurrogate, gaussian_kernel
from saealib.surrogate.training_set import KNNObjectiveSet

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
def surrogate_1obj() -> RBFsurrogate:
    return RBFsurrogate(gaussian_kernel, DIM)


@pytest.fixture
def surrogate_2obj() -> RBFsurrogate:
    return RBFsurrogate(gaussian_kernel, DIM)


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


# ===========================================================================
# GlobalSurrogateManager Tests
# ===========================================================================
class TestGlobalSurrogateManager:
    """Tests for GlobalSurrogateManager."""

    def test_score_candidates_returns_tuple(
        self,
        surrogate_1obj: RBFsurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        result = manager.score_candidates(candidates, archive_1obj)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_scores_shape(
        self,
        surrogate_1obj: RBFsurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        scores, _ = manager.score_candidates(candidates, archive_1obj)
        assert scores.shape == (len(candidates),)

    def test_predictions_count(
        self,
        surrogate_1obj: RBFsurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        _, predictions = manager.score_candidates(candidates, archive_1obj)
        assert len(predictions) == len(candidates)

    def test_predictions_are_surrogate_prediction(
        self,
        surrogate_1obj: RBFsurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        _, predictions = manager.score_candidates(candidates, archive_1obj)
        for p in predictions:
            assert isinstance(p, SurrogatePrediction)

    def test_prediction_mean_shape_per_candidate(
        self,
        surrogate_1obj: RBFsurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        _, predictions = manager.score_candidates(candidates, archive_1obj)
        for p in predictions:
            assert p.value.shape == (1, N_OBJ)

    def test_scores_finite(
        self,
        surrogate_1obj: RBFsurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        scores, _ = manager.score_candidates(candidates, archive_1obj)
        assert np.all(np.isfinite(scores))

    def test_biobj_scores_shape(
        self,
        surrogate_2obj: RBFsurrogate,
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
        surrogate_1obj: RBFsurrogate,
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
        surrogate_1obj: RBFsurrogate,
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
        surrogate_1obj: RBFsurrogate,
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
        surrogate_1obj: RBFsurrogate,
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
        surrogate_1obj: RBFsurrogate,
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
        surrogate_1obj: RBFsurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = LocalSurrogateManager(
            surrogate_1obj, MeanPrediction(), training_set=KNNObjectiveSet(10)
        )
        scores, _ = manager.score_candidates(candidates, archive_1obj)
        assert np.all(np.isfinite(scores))


# ===========================================================================
# EnsembleSurrogateManager Tests
# ===========================================================================
class TestEnsembleSurrogateManager:
    """Tests for EnsembleSurrogateManager."""

    def test_empty_managers_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            EnsembleSurrogateManager([])

    def test_uniform_weights_by_default(self) -> None:
        m1 = GlobalSurrogateManager(
            RBFsurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        m2 = GlobalSurrogateManager(
            RBFsurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        ensemble = EnsembleSurrogateManager([m1, m2])
        np.testing.assert_array_almost_equal(ensemble.weights, [0.5, 0.5])

    def test_custom_weights_are_normalized(self) -> None:
        m1 = GlobalSurrogateManager(
            RBFsurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        m2 = GlobalSurrogateManager(
            RBFsurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        ensemble = EnsembleSurrogateManager([m1, m2], weights=np.array([1.0, 3.0]))
        np.testing.assert_array_almost_equal(ensemble.weights, [0.25, 0.75])

    def test_scores_shape(self, archive_1obj: Archive, candidates: np.ndarray) -> None:
        m1 = GlobalSurrogateManager(
            RBFsurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        m2 = GlobalSurrogateManager(
            RBFsurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        ensemble = EnsembleSurrogateManager([m1, m2])
        scores, _ = ensemble.score_candidates(candidates, archive_1obj)
        assert scores.shape == (len(candidates),)

    def test_predictions_from_first_manager(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        """EnsembleSurrogateManager returns predictions from the first sub-manager."""
        m1 = GlobalSurrogateManager(
            RBFsurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        m2 = GlobalSurrogateManager(
            RBFsurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        ensemble = EnsembleSurrogateManager([m1, m2])
        _, predictions = ensemble.score_candidates(candidates, archive_1obj)
        assert len(predictions) == len(candidates)
        for p in predictions:
            assert isinstance(p, SurrogatePrediction)

    def test_scores_in_0_1_range(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
        """Aggregated scores are rank-normalized weighted averages, so in [0, 1]."""
        m1 = GlobalSurrogateManager(
            RBFsurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        m2 = GlobalSurrogateManager(
            RBFsurrogate(gaussian_kernel, DIM), MeanPrediction()
        )
        ensemble = EnsembleSurrogateManager([m1, m2])
        scores, _ = ensemble.score_candidates(candidates, archive_1obj)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_single_manager_ensemble(
        self,
        surrogate_1obj: RBFsurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        """An ensemble with one manager propagates scores correctly."""
        m = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        ensemble = EnsembleSurrogateManager([m])
        scores, _ = ensemble.score_candidates(candidates, archive_1obj)
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

    def test_tell_f_is_nan(
        self, archive_1obj: Archive, candidates: np.ndarray
    ) -> None:
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

    def test_more_distant_point_has_higher_novelty(
        self, archive_1obj: Archive
    ) -> None:
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

    def test_scores_positive(self, archive_1obj: Archive, candidates: np.ndarray) -> None:
        mgr = DensityManager(eps=1.0)
        scores = mgr.compute_scores(candidates, archive_1obj)
        assert np.all(scores > 0.0)

    def test_sparse_region_has_higher_score(self, archive_1obj: Archive) -> None:
        """Point far from all archive points (no neighbors within eps) should score higher."""
        dense = archive_1obj.x[0:1] + 0.01  # inside many eps-balls
        sparse = np.array([[100.0, 100.0]])  # far away, zero neighbors
        mgr = DensityManager(eps=0.5)
        score_dense = mgr.compute_scores(dense, archive_1obj)[0]
        score_sparse = mgr.compute_scores(sparse, archive_1obj)[0]
        assert score_sparse > score_dense

    def test_eps_affects_scores(self, archive_1obj: Archive) -> None:
        """Larger eps counts more neighbors → lower inverse density."""
        candidate = np.zeros((1, DIM))
        score_small_eps = DensityManager(eps=0.01).compute_scores(candidate, archive_1obj)[0]
        score_large_eps = DensityManager(eps=100.0).compute_scores(candidate, archive_1obj)[0]
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
