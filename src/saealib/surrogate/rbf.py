import logging

import numpy as np
import scipy.spatial

from .base import Surrogate


def gaussian_kernel(x1: np.ndarray, x2: np.ndarray, sigma=2.0):
    # return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))
    sq_dist = scipy.spatial.distance.cdist(x1, x2, 'sqeuclidean')
    return np.exp(-sq_dist / (2 * (sigma ** 2)))


class RBFsurrogate(Surrogate):
    def __init__(self, kernel, dim):
        self.dim = dim
        self.train_x = None
        self.train_y = None
        self.kernel = kernel
        self.weights = None
        self.kernel_matrix = None
        self.sigma = None

    def fit(self, train_x, train_y):
        self.train_x = np.asarray(train_x)
        self.train_y = np.asarray(train_y)
        n_samples = len(train_x)
        self.sigma = np.median(scipy.spatial.distance.pdist(self.train_x))
        self.kernel_matrix = self.kernel(self.train_x, self.train_x, sigma=self.sigma)
        rcond = 1 / np.linalg.cond(self.kernel_matrix)
        if rcond < np.finfo(self.kernel_matrix.dtype).eps:
            logging.warning(f"Kernel matrix is ill-conditioned. RCOND: {rcond}")
        try:
            self.weights = np.linalg.solve(self.kernel_matrix, (train_y - np.mean(train_y)))
        except np.linalg.LinAlgError:
            logging.error("Failed to solve linear system (Kernel matrix might be singular).")
            self.weights = np.zeros(n_samples)

    def predict(self, test_x: np.ndarray):
        tx = np.asarray(test_x)
        if tx.ndim == 1:
            tx = tx.reshape(1, -1)
        k = self.kernel(self.train_x, tx, sigma=self.sigma)
        # k shape: (n_train, n_test)
        # weights shape: (n_train,)
        preds = k.T.dot(self.weights) + np.mean(self.train_y)
        return np.asarray(preds).flatten()
