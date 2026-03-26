"""MaxUncertainty acquisition function module."""

from __future__ import annotations

from typing import Any

import numpy as np

from saealib.acquisition.base import AcquisitionFunction
from saealib.surrogate.prediction import SurrogatePrediction


class MaxUncertainty(AcquisitionFunction):
    """
    Acquisition function that maximizes predictive uncertainty (exploration).

    Selects candidates where the surrogate model is least confident.
    Requires a surrogate that provides uncertainty estimates (std).

    For multi-objective problems, aggregates uncertainty across objectives
    using a weighted sum.

    Parameters
    ----------
    weights : np.ndarray or None
        Weights for aggregating uncertainty across objectives.
        shape: (n_obj,). If None, uses the mean across objectives.
    """

    def __init__(self, weights: np.ndarray | None = None):
        self.weights = weights

    def score(
        self,
        prediction: SurrogatePrediction,
        reference: Any = None,
    ) -> np.ndarray:
        """
        Compute scores based on predictive standard deviation.

        Parameters
        ----------
        prediction : SurrogatePrediction
            Surrogate predictions. Must have std (has_uncertainty == True).
        reference : Any
            Not used. Accepted for interface compatibility.

        Returns
        -------
        np.ndarray
            Scores. shape: (n_samples,)

        Raises
        ------
        TypeError
            If prediction does not contain uncertainty estimates.
        """
        if not prediction.has_uncertainty:
            raise TypeError(
                "MaxUncertainty requires a surrogate with uncertainty estimates "
                "(prediction.std must not be None)."
            )
        std = prediction.std  # (n_samples, n_obj)
        if self.weights is not None:
            return std @ np.asarray(self.weights)
        return std.mean(axis=1)  # mean uncertainty across objectives
