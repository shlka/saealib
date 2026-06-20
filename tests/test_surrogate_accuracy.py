"""Tests for surrogate/accuracy.py."""

import numpy as np
import pytest

from saealib.surrogate.accuracy import (
    RMSE,
    AccuracyEvaluator,
    HeldOutAccuracyEvaluator,
    KFoldAccuracyEvaluator,
    LOOAccuracyEvaluator,
    R2Score,
    SpearmanCorrelation,
    SurrogateAccuracy,
)
from saealib.surrogate.rbf import RBFSurrogate, gaussian_kernel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 1
N_OBJ = 1


def _make_surrogate() -> RBFSurrogate:
    return RBFSurrogate(kernel=gaussian_kernel, dim=DIM)


def _sphere_data(n: int = 30, rng_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    x = rng.uniform(-2.0, 2.0, size=(n, DIM))
    y = np.sum(x**2, axis=1, keepdims=True)
    return x, y


# ---------------------------------------------------------------------------
# SurrogateAccuracyMetric
# ---------------------------------------------------------------------------


class TestSpearmanCorrelation:
    def test_perfect_positive(self):
        y = np.array([[1.0], [2.0], [3.0], [4.0]])
        assert SpearmanCorrelation().compute(y, y) == pytest.approx(1.0)

    def test_perfect_negative(self):
        y_true = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_pred = np.array([[4.0], [3.0], [2.0], [1.0]])
        assert SpearmanCorrelation().compute(y_true, y_pred) == pytest.approx(-1.0)

    def test_name(self):
        assert SpearmanCorrelation().name == "spearman"

    def test_constant_pred_returns_zero(self):
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[5.0], [5.0], [5.0]])
        result = SpearmanCorrelation().compute(y_true, y_pred)
        assert result == pytest.approx(0.0)


class TestRMSE:
    def test_zero_error(self):
        y = np.array([[1.0], [2.0], [3.0]])
        assert RMSE().compute(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([[0.0], [0.0]])
        y_pred = np.array([[3.0], [4.0]])
        # sqrt((9+16)/2) = sqrt(12.5)
        assert RMSE().compute(y_true, y_pred) == pytest.approx(np.sqrt(12.5))

    def test_name(self):
        assert RMSE().name == "rmse"


class TestR2Score:
    def test_perfect_fit(self):
        y = np.array([[1.0], [2.0], [3.0]])
        assert R2Score().compute(y, y) == pytest.approx(1.0)

    def test_constant_true(self):
        y_true = np.array([[5.0], [5.0], [5.0]])
        y_pred = np.array([[5.0], [5.0], [5.0]])
        assert R2Score().compute(y_true, y_pred) == pytest.approx(1.0)

    def test_name(self):
        assert R2Score().name == "r2"


# ---------------------------------------------------------------------------
# SurrogateAccuracy
# ---------------------------------------------------------------------------


class TestSurrogateAccuracy:
    def test_get_existing(self):
        acc = SurrogateAccuracy(metrics={"spearman": 0.8}, n_samples=20)
        assert acc.get("spearman") == pytest.approx(0.8)

    def test_get_missing_returns_default(self):
        acc = SurrogateAccuracy()
        assert np.isnan(acc.get("spearman"))

    def test_get_custom_default(self):
        acc = SurrogateAccuracy()
        assert acc.get("rmse", 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# AccuracyEvaluator is abstract
# ---------------------------------------------------------------------------


def test_accuracy_evaluator_is_abstract():
    with pytest.raises(TypeError):
        AccuracyEvaluator()  # type: ignore[abstract]  # intentional: testing abstract instantiation raises TypeError


# ---------------------------------------------------------------------------
# KFoldAccuracyEvaluator
# ---------------------------------------------------------------------------


class TestKFoldAccuracyEvaluator:
    def test_returns_accuracy_object(self):
        x, y = _sphere_data(30)
        acc = KFoldAccuracyEvaluator(n_splits=5).evaluate(_make_surrogate(), x, y)
        assert isinstance(acc, SurrogateAccuracy)
        assert acc.n_samples == 30

    def test_has_all_default_metrics(self):
        x, y = _sphere_data(30)
        acc = KFoldAccuracyEvaluator(n_splits=5).evaluate(_make_surrogate(), x, y)
        assert "spearman" in acc.metrics
        assert "rmse" in acc.metrics
        assert "r2" in acc.metrics

    def test_spearman_in_valid_range(self):
        x, y = _sphere_data(30)
        acc = KFoldAccuracyEvaluator(
            metrics=[SpearmanCorrelation()], n_splits=5
        ).evaluate(_make_surrogate(), x, y)
        assert -1.0 <= acc.get("spearman") <= 1.0

    def test_surrogate_unmodified_after_evaluate(self):
        x, y = _sphere_data(30)
        surrogate = _make_surrogate()
        surrogate.fit(x, y)
        pred_before = surrogate.predict(x[:3]).value.copy()

        KFoldAccuracyEvaluator(n_splits=5).evaluate(surrogate, x, y)

        np.testing.assert_array_equal(pred_before, surrogate.predict(x[:3]).value)

    def test_custom_metric_only(self):
        x, y = _sphere_data(30)
        acc = KFoldAccuracyEvaluator(metrics=[RMSE()], n_splits=5).evaluate(
            _make_surrogate(), x, y
        )
        assert "rmse" in acc.metrics
        assert "spearman" not in acc.metrics

    def test_n_splits_clamped_to_n_samples(self):
        x, y = _sphere_data(4)
        acc = KFoldAccuracyEvaluator(n_splits=10).evaluate(_make_surrogate(), x, y)
        assert acc.n_samples == 4

    def test_too_few_samples_returns_empty(self):
        x, y = _sphere_data(1)
        acc = KFoldAccuracyEvaluator(n_splits=5).evaluate(_make_surrogate(), x, y)
        assert acc.metrics == {}


# ---------------------------------------------------------------------------
# LOOAccuracyEvaluator
# ---------------------------------------------------------------------------


class TestLOOAccuracyEvaluator:
    def test_returns_accuracy(self):
        x, y = _sphere_data(10)
        acc = LOOAccuracyEvaluator().evaluate(_make_surrogate(), x, y)
        assert "spearman" in acc.metrics
        assert acc.n_samples == 10

    def test_surrogate_unmodified(self):
        x, y = _sphere_data(10)
        surrogate = _make_surrogate()
        surrogate.fit(x, y)
        pred_before = surrogate.predict(x[:3]).value.copy()

        LOOAccuracyEvaluator().evaluate(surrogate, x, y)

        np.testing.assert_array_equal(pred_before, surrogate.predict(x[:3]).value)


# ---------------------------------------------------------------------------
# HeldOutAccuracyEvaluator
# ---------------------------------------------------------------------------


class TestHeldOutAccuracyEvaluator:
    def test_returns_accuracy(self):
        x, y = _sphere_data(30)
        held_x, held_y = _sphere_data(5, rng_seed=99)
        surrogate = _make_surrogate()
        surrogate.fit(x, y)

        acc = HeldOutAccuracyEvaluator(
            held_x, held_y, metrics=[SpearmanCorrelation()]
        ).evaluate(surrogate, x, y)

        assert "spearman" in acc.metrics
        assert acc.n_samples == 30

    def test_perfect_surrogate(self):
        x, y = _sphere_data(20)
        surrogate = _make_surrogate()
        surrogate.fit(x, y)

        acc = HeldOutAccuracyEvaluator(x, y).evaluate(surrogate, x, y)

        assert acc.get("spearman") == pytest.approx(1.0, abs=1e-6)
        assert acc.get("rmse") == pytest.approx(0.0, abs=1e-6)
        assert acc.get("r2") == pytest.approx(1.0, abs=1e-6)

    def test_does_not_refit_surrogate(self):
        """HeldOutAccuracyEvaluator must not modify the already-fitted surrogate."""
        x, y = _sphere_data(20)
        held_x, held_y = _sphere_data(5, rng_seed=7)
        surrogate = _make_surrogate()
        surrogate.fit(x, y)
        pred_before = surrogate.predict(held_x).value.copy()

        HeldOutAccuracyEvaluator(held_x, held_y).evaluate(surrogate, x, y)

        np.testing.assert_array_equal(pred_before, surrogate.predict(held_x).value)
