"""Scikit-learn surrogate adapter and convenience classes."""

from __future__ import annotations

import numpy as np

from saealib.surrogate.base import Surrogate
from saealib.surrogate.prediction import SurrogatePrediction


class SklearnSurrogate(Surrogate):
    """
    Surrogate adapter for scikit-learn compatible estimators.

    Wraps any estimator that implements ``fit(X, y)`` and ``predict(X)``.
    Multi-objective problems are handled by fitting one cloned estimator
    per objective, following the same pattern as ``RBFsurrogate``.

    Parameters
    ----------
    estimator : sklearn estimator
        A scikit-learn compatible regressor instance (e.g. ``SVR()``).
        The estimator is cloned once per objective on the first ``fit`` call.

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    """

    def __init__(self, estimator: object) -> None:
        try:
            import sklearn  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for SklearnSurrogate. "
                "Install it with: pip install saealib[sklearn]"
            ) from e
        self.estimator = estimator
        self._models: list | None = None
        self.n_obj: int | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """
        Fit the surrogate model.

        Parameters
        ----------
        train_x : np.ndarray
            Training input data. shape: (n_samples, n_features)
        train_y : np.ndarray
            Training output data. shape: (n_samples, n_obj) or (n_samples,).
        """
        from sklearn.base import clone

        arr = np.asarray(train_y, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n_obj = arr.shape[1]

        if self._models is None or n_obj != self.n_obj:
            self.n_obj = n_obj
            self._models = [clone(self.estimator) for _ in range(n_obj)]

        for i, model in enumerate(self._models):
            model.fit(train_x, arr[:, i])

    def predict(self, test_x: np.ndarray) -> SurrogatePrediction:
        """
        Predict using the surrogate model.

        Parameters
        ----------
        test_x : np.ndarray
            Input data. shape: (n_samples, n_features) or (n_features,)

        Returns
        -------
        SurrogatePrediction
            prediction.value shape: (n_samples, n_obj)
            prediction.std is None
        """
        test = np.asarray(test_x)
        if test.ndim == 1:
            test = test.reshape(1, -1)
        preds = [m.predict(test) for m in self._models]
        value = np.column_stack(preds)
        return SurrogatePrediction(value=value)

