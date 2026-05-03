"""Abstract base class for surrogate models."""

from abc import ABC, abstractmethod

import numpy as np

from saealib.surrogate.prediction import SurrogatePrediction


class Surrogate(ABC):
    """Base class for surrogate models."""

    @abstractmethod
    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """
        Fit the surrogate model.

        Parameters
        ----------
        train_x : np.ndarray
            Training input data. shape: (n_samples, n_features)
        train_y : np.ndarray
            Training output data. shape: (n_samples, n_obj)
            For n_obj == 1, shape (n_samples,) is also accepted.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def predict(self, test_x: np.ndarray) -> SurrogatePrediction:
        """
        Predict using the surrogate model.

        Parameters
        ----------
        test_x : np.ndarray
            Input data for prediction. shape: (n_samples, n_features)

        Returns
        -------
        SurrogatePrediction
            prediction.mean shape: (n_samples, n_obj)
            prediction.std  shape: (n_samples, n_obj) or None
        """
        pass
