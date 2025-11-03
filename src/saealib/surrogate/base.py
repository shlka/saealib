import numpy as np


class Surrogate:
    """
    Base class for surrogate models.
    """
    def __init__(self):
        pass

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        pass

    def predict(self, test_x: np.ndarray) -> float:
        pass
