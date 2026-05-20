"""
Tests for sklearn-based surrogate adapters.

Covers:
- SklearnSurrogate: fit/predict, single/multi-objective, missing sklearn
- SVMSurrogate, NNSurrogate, DTSurrogate: smoke tests
- XGBSurrogate, LGBMSurrogate: smoke tests
- Integration with GlobalSurrogateManager
"""

import sys

import numpy as np
import pytest

from saealib.acquisition import MeanPrediction
from saealib.population import Archive, PopulationAttribute
from saealib.surrogate.manager import GlobalSurrogateManager
from saealib.surrogate.prediction import SurrogatePrediction
from saealib.surrogate.sklearn_surrogate import (
    DTSurrogate,
    LGBMSurrogate,
    NNSurrogate,
    SklearnSurrogate,
    SVMSurrogate,
    XGBSurrogate,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
DIM = 2
N_SAMPLES = 20


@pytest.fixture
def train_data_1obj():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, size=(N_SAMPLES, DIM))
    y = np.sum(X**2, axis=1)
    return X, y


@pytest.fixture
def train_data_2obj():
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 2.0, size=(N_SAMPLES, DIM))
    y = np.column_stack([np.sum(X**2, axis=1), np.sum((X - 2.0) ** 2, axis=1)])
    return X, y


@pytest.fixture
def test_x():
    rng = np.random.default_rng(7)
    return rng.uniform(-1.0, 1.0, size=(5, DIM))


@pytest.fixture
def archive_1obj():
    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(1,)),
    ]
    arc = Archive(attrs, init_capacity=30)
    rng = np.random.default_rng(42)
    for _ in range(N_SAMPLES):
        x = rng.uniform(-2.0, 2.0, size=DIM)
        f = np.array([np.sum(x**2)])
        arc.add(x=x, f=f)
    return arc


@pytest.fixture
def candidates():
    rng = np.random.default_rng(7)
    return rng.uniform(-1.0, 1.0, size=(5, DIM))


# ===========================================================================
# SklearnSurrogate Tests
# ===========================================================================
class TestSklearnSurrogate:
    def test_fit_predict_1obj(self, train_data_1obj, test_x) -> None:
        from sklearn.linear_model import Ridge

        X, y = train_data_1obj
        s = SklearnSurrogate(Ridge())
        s.fit(X, y)
        pred = s.predict(test_x)
        assert isinstance(pred, SurrogatePrediction)
        assert pred.value.shape == (5, 1)

    def test_fit_predict_2obj(self, train_data_2obj, test_x) -> None:
        from sklearn.linear_model import Ridge

        X, y = train_data_2obj
        s = SklearnSurrogate(Ridge())
        s.fit(X, y)
        pred = s.predict(test_x)
        assert pred.value.shape == (5, 2)

    def test_predict_1d_input(self, train_data_1obj) -> None:
        from sklearn.linear_model import Ridge

        X, y = train_data_1obj
        s = SklearnSurrogate(Ridge())
        s.fit(X, y)
        pred = s.predict(X[0])  # 1D input
        assert pred.value.shape == (1, 1)

    def test_std_is_none(self, train_data_1obj, test_x) -> None:
        from sklearn.linear_model import Ridge

        X, y = train_data_1obj
        s = SklearnSurrogate(Ridge())
        s.fit(X, y)
        assert s.predict(test_x).std is None

    def test_refit_different_nobj(
        self, train_data_1obj, train_data_2obj, test_x
    ) -> None:
        from sklearn.linear_model import Ridge

        X1, y1 = train_data_1obj
        X2, y2 = train_data_2obj
        s = SklearnSurrogate(Ridge())
        s.fit(X1, y1)
        s.fit(X2, y2)
        pred = s.predict(test_x)
        assert pred.value.shape == (5, 2)

    def test_missing_sklearn_raises(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "sklearn", None)
        monkeypatch.setitem(sys.modules, "sklearn.base", None)
        with pytest.raises(ImportError, match="scikit-learn"):
            from sklearn.linear_model import Ridge

            SklearnSurrogate(Ridge())

    def test_estimator_is_cloned_per_objective(self, train_data_2obj) -> None:
        from sklearn.linear_model import Ridge

        X, y = train_data_2obj
        estimator = Ridge()
        s = SklearnSurrogate(estimator)
        s.fit(X, y)
        assert s._models[0] is not s._models[1]
        assert s._models[0] is not estimator


# ===========================================================================
# SVMSurrogate Tests
# ===========================================================================
class TestSVMSurrogate:
    def test_fit_predict_1obj(self, train_data_1obj, test_x) -> None:
        s = SVMSurrogate()
        X, y = train_data_1obj
        s.fit(X, y)
        pred = s.predict(test_x)
        assert pred.value.shape == (5, 1)

    def test_fit_predict_2obj(self, train_data_2obj, test_x) -> None:
        s = SVMSurrogate()
        X, y = train_data_2obj
        s.fit(X, y)
        assert s.predict(test_x).value.shape == (5, 2)

    def test_kwargs_forwarded(self) -> None:
        from sklearn.svm import SVR

        s = SVMSurrogate(kernel="linear", C=10.0)
        assert isinstance(s.estimator, SVR)
        assert s.estimator.kernel == "linear"
        assert s.estimator.C == 10.0


# ===========================================================================
# NNSurrogate Tests
# ===========================================================================
class TestNNSurrogate:
    def test_fit_predict_1obj(self, train_data_1obj, test_x) -> None:
        s = NNSurrogate(max_iter=200, random_state=0)
        X, y = train_data_1obj
        s.fit(X, y)
        pred = s.predict(test_x)
        assert pred.value.shape == (5, 1)

    def test_fit_predict_2obj(self, train_data_2obj, test_x) -> None:
        s = NNSurrogate(max_iter=200, random_state=0)
        X, y = train_data_2obj
        s.fit(X, y)
        assert s.predict(test_x).value.shape == (5, 2)


# ===========================================================================
# DTSurrogate Tests
# ===========================================================================
class TestDTSurrogate:
    def test_fit_predict_1obj(self, train_data_1obj, test_x) -> None:
        s = DTSurrogate(n_estimators=10, random_state=0)
        X, y = train_data_1obj
        s.fit(X, y)
        pred = s.predict(test_x)
        assert pred.value.shape == (5, 1)

    def test_fit_predict_2obj(self, train_data_2obj, test_x) -> None:
        s = DTSurrogate(n_estimators=10, random_state=0)
        X, y = train_data_2obj
        s.fit(X, y)
        assert s.predict(test_x).value.shape == (5, 2)


# ===========================================================================
# XGBSurrogate Tests
# ===========================================================================
class TestXGBSurrogate:
    def test_fit_predict_1obj(self, train_data_1obj, test_x) -> None:
        s = XGBSurrogate(n_estimators=10, random_state=0, verbosity=0)
        X, y = train_data_1obj
        s.fit(X, y)
        pred = s.predict(test_x)
        assert pred.value.shape == (5, 1)

    def test_fit_predict_2obj(self, train_data_2obj, test_x) -> None:
        s = XGBSurrogate(n_estimators=10, random_state=0, verbosity=0)
        X, y = train_data_2obj
        s.fit(X, y)
        assert s.predict(test_x).value.shape == (5, 2)

    def test_missing_xgboost_raises(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "xgboost", None)
        with pytest.raises(ImportError, match="xgboost"):
            XGBSurrogate()


# ===========================================================================
# LGBMSurrogate Tests
# ===========================================================================
class TestLGBMSurrogate:
    def test_fit_predict_1obj(self, train_data_1obj, test_x) -> None:
        s = LGBMSurrogate(n_estimators=10, random_state=0, verbose=-1)
        X, y = train_data_1obj
        s.fit(X, y)
        pred = s.predict(test_x)
        assert pred.value.shape == (5, 1)

    def test_fit_predict_2obj(self, train_data_2obj, test_x) -> None:
        s = LGBMSurrogate(n_estimators=10, random_state=0, verbose=-1)
        X, y = train_data_2obj
        s.fit(X, y)
        assert s.predict(test_x).value.shape == (5, 2)

    def test_missing_lightgbm_raises(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "lightgbm", None)
        with pytest.raises(ImportError, match="lightgbm"):
            LGBMSurrogate()


# ===========================================================================
# Integration: GlobalSurrogateManager
# ===========================================================================
class TestSklearnSurrogateIntegration:
    def test_global_manager_svm(self, archive_1obj, candidates) -> None:
        manager = GlobalSurrogateManager(SVMSurrogate(), MeanPrediction())
        scores, preds = manager.score_candidates(candidates, archive_1obj)
        assert scores.shape == (len(candidates),)
        assert len(preds) == len(candidates)
        for p in preds:
            assert p.value.shape == (1, 1)

    def test_global_manager_dt(self, archive_1obj, candidates) -> None:
        manager = GlobalSurrogateManager(
            DTSurrogate(n_estimators=10, random_state=0), MeanPrediction()
        )
        scores, _ = manager.score_candidates(candidates, archive_1obj)
        assert scores.shape == (len(candidates),)

    def test_global_manager_xgb(self, archive_1obj, candidates) -> None:
        manager = GlobalSurrogateManager(
            XGBSurrogate(n_estimators=10, verbosity=0), MeanPrediction()
        )
        scores, _ = manager.score_candidates(candidates, archive_1obj)
        assert scores.shape == (len(candidates),)

    def test_global_manager_lgbm(self, archive_1obj, candidates) -> None:
        manager = GlobalSurrogateManager(
            LGBMSurrogate(n_estimators=10, verbose=-1), MeanPrediction()
        )
        scores, _ = manager.score_candidates(candidates, archive_1obj)
        assert scores.shape == (len(candidates),)
