"""
RBF surrogate model module.

This module defines the Radial Basis Function (RBF) surrogate model.
RBFsurrogate supports multi-objective problems by maintaining one
independent RBF model per objective (ensemble approach).
"""

import logging

import numpy as np
import scipy.spatial

from saealib.surrogate.base import Surrogate
from saealib.surrogate.prediction import SurrogatePrediction

logger = logging.getLogger(__name__)


def gaussian_kernel(x1: np.ndarray, x2: np.ndarray, sigma=2.0) -> np.ndarray:
    """
    Gaussian radial basis function kernel.

    Parameters
    ----------
    x1 : np.ndarray
        Input data 1.
    x2 : np.ndarray
        Input data 2.
    sigma : float
        Kernel width parameter.

    Returns
    -------
    np.ndarray
        Matrix of kernel evaluations between x1 and x2. shape: (len(x1), len(x2))
    """
    sq_dist = scipy.spatial.distance.cdist(x1, x2, "sqeuclidean")
    return np.exp(-sq_dist / (2 * (sigma**2)))


class _RBFModel:
    """
    Single-objective RBF interpolation model (internal use only).

    Holds all state for one objective's RBF fit. Used as a building
    block by RBFsurrogate to support multi-objective problems.
    """

    def __init__(self, kernel: callable, dim: int):
        self.kernel = kernel
        self.dim = dim
        self.train_x: np.ndarray | None = None
        self.train_y: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.kernel_matrix: np.ndarray | None = None
        self.sigma: float | None = None

    def fit(self, train_x: np.ndarray, train_y_1d: np.ndarray) -> None:
        """
        Fit the RBF model.

        Parameters
        ----------
        train_x : np.ndarray
            Training input data. shape: (n_samples, n_features)
        train_y_1d : np.ndarray
            Training output data for one objective. shape: (n_samples,)
        """
        self.train_x = np.asarray(train_x)
        self.train_y = np.asarray(train_y_1d)
        n_samples = len(train_x)
        self.sigma = np.median(scipy.spatial.distance.pdist(self.train_x))
        self.kernel_matrix = self.kernel(self.train_x, self.train_x, sigma=self.sigma)
        rcond = 1 / np.linalg.cond(self.kernel_matrix)
        if rcond < np.finfo(self.kernel_matrix.dtype).eps:
            logger.warning(f"Kernel matrix is ill-conditioned. RCOND: {rcond}")
        try:
            self.weights = np.linalg.solve(
                self.kernel_matrix, (self.train_y - np.mean(self.train_y))
            )
        except np.linalg.LinAlgError:
            logger.error(
                "Failed to solve linear system (Kernel matrix might be singular)."
            )
            self.weights = np.nan * np.ones(n_samples)

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        """
        Predict for one objective.

        Parameters
        ----------
        test_x : np.ndarray
            Input data. shape: (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predicted values. shape: (n_samples,)
        """
        test = np.asarray(test_x)
        if test.ndim == 1:
            test = test.reshape(1, -1)
        k = self.kernel(self.train_x, test, sigma=self.sigma)
        preds = k.T.dot(self.weights) + np.mean(self.train_y)
        return np.asarray(preds).flatten()


class RBFsurrogate(Surrogate):
    """
    Radial Basis Function (RBF) Interpolation surrogate model.

    Supports multi-objective problems by maintaining one independent
    _RBFModel per objective. The number of objectives is inferred from
    ``train_y`` on the first call to ``fit`` (lazy initialization).

    Attributes
    ----------
    kernel : callable
        Kernel function (e.g. gaussian_kernel).
    dim : int
        Dimensionality of the input data.
    n_obj : int or None
        Number of objectives. Set on first fit call.
    """

    def __init__(self, kernel: callable, dim: int):
        self.kernel = kernel
        self.dim = dim
        self.n_obj: int | None = None
        self._models: list[_RBFModel] | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """
        Fit the surrogate model.

        Parameters
        ----------
        train_x : np.ndarray
            Training input data. shape: (n_samples, n_features)
        train_y : np.ndarray
            Training output data. shape: (n_samples, n_obj) or (n_samples,).
            1-D input is treated as single-objective: shape (n_samples, 1).
        """
        arr = np.asarray(train_y, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)  # (n_samples,) -> (n_samples, 1)
        n_obj = arr.shape[1]

        # (Re-)initialize models when n_obj changes or on first fit
        if self._models is None or n_obj != self.n_obj:
            self.n_obj = n_obj
            self._models = [_RBFModel(self.kernel, self.dim) for _ in range(n_obj)]

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
            prediction.mean shape: (n_samples, n_obj)
            prediction.std  is None (RBF interpolation provides no uncertainty)
        """
        preds = [m.predict(test_x) for m in self._models]
        mean = np.column_stack(preds)  # (n_samples, n_obj)
        return SurrogatePrediction(mean=mean)
