"""MeanPrediction acquisition function module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from saealib.acquisition.base import AcquisitionFunction
from saealib.surrogate.prediction import SurrogatePrediction

if TYPE_CHECKING:
    from saealib.population import Archive


class MeanPrediction(AcquisitionFunction):
    """
    Acquisition function based on predicted mean value (exploitation).

    For single-objective problems, returns the predicted mean directly.
    For multi-objective problems, returns a weighted scalarization of the
    predicted mean.

    A higher score indicates a more promising candidate.
    The sign convention follows the weight: use a negative weight for
    minimization (e.g., weights=np.array([-1.0])) so that lower objective
    values yield higher scores.

    Parameters
    ----------
    weights : np.ndarray or None
        Weights for scalarizing multi-objective predictions.
        shape: (n_obj,). If None, uses the first objective only.
    """

    def __init__(self, weights: np.ndarray | None = None, reference: Any = None):
        self.weights = weights
        self.reference = reference

    def compute_reference(self, archive: Archive) -> Any:
        """Return fixed reference if set, otherwise None."""
        return self.reference

    def score(
        self,
        prediction: SurrogatePrediction,
        reference: Any = None,
    ) -> np.ndarray:
        """
        Compute scores based on predicted mean.

        Parameters
        ----------
        prediction : SurrogatePrediction
            Surrogate predictions. prediction.mean shape: (n_samples, n_obj)
        reference : Any
            Not used. Accepted for interface compatibility.

        Returns
        -------
        np.ndarray
            Scores. shape: (n_samples,)
        """
        m = prediction.mean  # (n_samples, n_obj)
        if self.weights is not None:
            return m @ np.asarray(self.weights)  # (n_samples,)
        return m[:, 0]  # single-objective default
