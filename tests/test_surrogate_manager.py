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
from saealib.surrogate.manager import (
    EnsembleSurrogateManager,
    GlobalSurrogateManager,
    LocalSurrogateManager,
    _rank_normalize,
    _split_prediction,
)
from saealib.surrogate.prediction import SurrogatePrediction
from saealib.surrogate.rbf import RBFsurrogate, gaussian_kernel

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
        pred = SurrogatePrediction(mean=np.zeros((4, 2)))
        parts = _split_prediction(pred)
        assert len(parts) == 4

    def test_each_part_has_shape_1_nobj(self) -> None:
        pred = SurrogatePrediction(mean=np.arange(6).reshape(3, 2).astype(float))
        parts = _split_prediction(pred)
        for p in parts:
            assert p.mean.shape == (1, 2)

    def test_values_preserved(self) -> None:
        mean = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        pred = SurrogatePrediction(mean=mean)
        parts = _split_prediction(pred)
        for i, p in enumerate(parts):
            np.testing.assert_array_equal(p.mean[0], mean[i])

    def test_std_split_correctly(self) -> None:
        std = np.array([[0.1, 0.2], [0.3, 0.4]])
        pred = SurrogatePrediction(mean=np.zeros((2, 2)), std=std)
        parts = _split_prediction(pred)
        assert parts[0].std is not None
        assert parts[0].std.shape == (1, 2)
        np.testing.assert_array_almost_equal(parts[0].std[0], [0.1, 0.2])

    def test_std_none_propagates(self) -> None:
        pred = SurrogatePrediction(mean=np.zeros((3, 1)))
        parts = _split_prediction(pred)
        for p in parts:
            assert p.std is None

    def test_label_split_correctly(self) -> None:
        label = np.array([0.0, 1.0, 2.0])
        pred = SurrogatePrediction(mean=np.zeros((3, 1)), label=label)
        parts = _split_prediction(pred)
        for i, p in enumerate(parts):
            assert p.label is not None
            assert p.label[0] == pytest.approx(float(i))

    def test_metadata_shared(self) -> None:
        """metadata dict is shared (not deep-copied) across splits."""
        meta = {"key": "val"}
        pred = SurrogatePrediction(mean=np.zeros((2, 1)), metadata=meta)
        parts = _split_prediction(pred)
        for p in parts:
            assert p.metadata is meta


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
        result = manager.score_candidates(
            candidates, archive_1obj, reference=np.array([0.0])
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_scores_shape(
        self,
        surrogate_1obj: RBFsurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        scores, _ = manager.score_candidates(
            candidates, archive_1obj, reference=np.array([0.0])
        )
        assert scores.shape == (len(candidates),)

    def test_predictions_count(
        self,
        surrogate_1obj: RBFsurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        _, predictions = manager.score_candidates(
            candidates, archive_1obj, reference=np.array([0.0])
        )
        assert len(predictions) == len(candidates)

    def test_predictions_are_surrogate_prediction(
        self,
        surrogate_1obj: RBFsurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        _, predictions = manager.score_candidates(
            candidates, archive_1obj, reference=np.array([0.0])
        )
        for p in predictions:
            assert isinstance(p, SurrogatePrediction)

    def test_prediction_mean_shape_per_candidate(
        self,
        surrogate_1obj: RBFsurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        _, predictions = manager.score_candidates(
            candidates, archive_1obj, reference=np.array([0.0])
        )
        for p in predictions:
            assert p.mean.shape == (1, N_OBJ)

    def test_scores_finite(
        self,
        surrogate_1obj: RBFsurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = GlobalSurrogateManager(surrogate_1obj, MeanPrediction())
        scores, _ = manager.score_candidates(
            candidates, archive_1obj, reference=np.array([0.0])
        )
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
        scores, predictions = manager.score_candidates(
            candidates, archive_2obj, reference=np.zeros(2)
        )
        assert scores.shape == (len(candidates),)
        for p in predictions:
            assert p.mean.shape == (1, 2)


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
            surrogate_1obj, MeanPrediction(), n_neighbors=10
        )
        result = manager.score_candidates(
            candidates, archive_1obj, reference=np.array([0.0])
        )
        assert isinstance(result, tuple) and len(result) == 2

    def test_scores_shape(
        self,
        surrogate_1obj: RBFsurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = LocalSurrogateManager(
            surrogate_1obj, MeanPrediction(), n_neighbors=10
        )
        scores, _ = manager.score_candidates(
            candidates, archive_1obj, reference=np.array([0.0])
        )
        assert scores.shape == (len(candidates),)

    def test_predictions_count(
        self,
        surrogate_1obj: RBFsurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = LocalSurrogateManager(
            surrogate_1obj, MeanPrediction(), n_neighbors=10
        )
        _, predictions = manager.score_candidates(
            candidates, archive_1obj, reference=np.array([0.0])
        )
        assert len(predictions) == len(candidates)

    def test_prediction_mean_shape(
        self,
        surrogate_1obj: RBFsurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = LocalSurrogateManager(
            surrogate_1obj, MeanPrediction(), n_neighbors=10
        )
        _, predictions = manager.score_candidates(
            candidates, archive_1obj, reference=np.array([0.0])
        )
        for p in predictions:
            assert p.mean.shape == (1, N_OBJ)

    def test_n_neighbors_default(
        self,
        surrogate_1obj: RBFsurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        """Default n_neighbors=50 is clamped to archive size."""
        manager = LocalSurrogateManager(surrogate_1obj, MeanPrediction())
        assert manager.n_neighbors == 50
        # archive has only 20 points, get_knn should still work
        scores, _ = manager.score_candidates(
            candidates, archive_1obj, reference=np.array([0.0])
        )
        assert scores.shape == (len(candidates),)

    def test_scores_finite(
        self,
        surrogate_1obj: RBFsurrogate,
        archive_1obj: Archive,
        candidates: np.ndarray,
    ) -> None:
        manager = LocalSurrogateManager(
            surrogate_1obj, MeanPrediction(), n_neighbors=10
        )
        scores, _ = manager.score_candidates(
            candidates, archive_1obj, reference=np.array([0.0])
        )
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
        scores, _ = ensemble.score_candidates(
            candidates, archive_1obj, reference=np.array([0.0])
        )
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
        _, predictions = ensemble.score_candidates(
            candidates, archive_1obj, reference=np.array([0.0])
        )
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
        scores, _ = ensemble.score_candidates(
            candidates, archive_1obj, reference=np.array([0.0])
        )
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
        scores, _ = ensemble.score_candidates(
            candidates, archive_1obj, reference=np.array([0.0])
        )
        assert scores.shape == (len(candidates),)
