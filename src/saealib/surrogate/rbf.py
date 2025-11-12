"""
RBF surrogate model module.

This module defines the Radial Basis Function (RBF) surrogate model.
"""
import logging

import numpy as np
import scipy.spatial

from saealib.surrogate.base import Surrogate


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
    # return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))
    sq_dist = scipy.spatial.distance.cdist(x1, x2, 'sqeuclidean')
    return np.exp(-sq_dist / (2 * (sigma ** 2)))


class RBFsurrogate(Surrogate):
    """
    Radial Basis Function (RBF) Interpolation surrogate model.

    Attributes
    ----------
    kernel : callable -> np.ndarray
        Kernel function to use. (e.g., .gaussian_kernel)
    dim : int
        Dimensionality of the input data.
    train_x : np.ndarray
        Training input data.
    train_y : np.ndarray
        Training output data.
    weights : np.ndarray
        Weights for the RBF model.
    kernel_matrix : np.ndarray
        Kernel matrix of the training data.
    sigma : float
        Kernel width parameter.
    """
    def __init__(self, kernel: callable, dim: int):
        """
        Initialize RBF surrogate model.

        Parameters
        ----------
        kernel : callable -> np.ndarray
            Kernel function to use.
        dim : int
            Dimensionality of the input data.
        """
        self.dim = dim
        self.train_x = None
        self.train_y = None
        self.kernel = kernel
        self.weights = None
        self.kernel_matrix = None
        self.sigma = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        self.train_x = np.asarray(train_x)
        self.train_y = np.asarray(train_y)
        n_samples = len(train_x)
        self.sigma = np.median(scipy.spatial.distance.pdist(self.train_x))
        self.kernel_matrix = self.kernel(self.train_x, self.train_x, sigma=self.sigma)
        rcond = 1 / np.linalg.cond(self.kernel_matrix)
        if rcond < np.finfo(self.kernel_matrix.dtype).eps:
            logger.warning(f"Kernel matrix is ill-conditioned. RCOND: {rcond}")
        try:
            self.weights = np.linalg.solve(self.kernel_matrix, (train_y - np.mean(train_y)))
        except np.linalg.LinAlgError:
            logger.error("Failed to solve linear system (Kernel matrix might be singular).")
            self.weights = np.zeros(n_samples)

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        test = np.asarray(test_x)
        if test.ndim == 1:
            test = test.reshape(1, -1)
        k = self.kernel(self.train_x, test, sigma=self.sigma)
        # k shape: (n_train, n_test)
        # weights shape: (n_train,)
        preds = k.T.dot(self.weights) + np.mean(self.train_y)
        return np.asarray(preds).flatten()
