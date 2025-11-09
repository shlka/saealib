from abc import ABC, abstractmethod

import numpy as np


# TODO: this class is regression surrogate, need to generalize for classification surrogate
class Surrogate(ABC):
    """
    Base class for surrogate models.
    """
    @abstractmethod
    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """
        Fit the surrogate model.
        
        Parameters
        ----------
        train_x : np.ndarray
            Training input data. shape: (n_samples, n_features)
        train_y : np.ndarray
            Training output data. shape: (n_samples, )

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def predict(self, test_x: np.ndarray) -> np.ndarray:
        """
        Predict using the surrogate model.

        Parameters
        ----------
        test_x : np.ndarray
            Input data for prediction. shape: (n_samples, n_features)
        
        Returns
        -------
        np.ndarray
            Predicted output data. shape: (n_samples, )
        """
        pass
