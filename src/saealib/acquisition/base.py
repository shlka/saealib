"""
Acquisition function base module.

This module defines the abstract base class for acquisition (infill criterion)
functions used in surrogate-assisted optimization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from saealib.surrogate.prediction import SurrogatePrediction


class AcquisitionFunction(ABC):
    """
    Abstract base class for acquisition functions (infill criteria).

    An acquisition function converts a SurrogatePrediction into a scalar
    score per candidate, which is used to rank candidates for true evaluation.

    The AcquisitionFunction is completely decoupled from Surrogate:
    it knows nothing about how predictions are generated.
    """

    @abstractmethod
    def score(
        self,
        prediction: SurrogatePrediction,
        reference: Any,
    ) -> np.ndarray:
        """
        Compute acquisition scores for a set of candidates.

        Parameters
        ----------
        prediction : SurrogatePrediction
            Predictions from a surrogate model.
        reference : Any
            Reference information for the acquisition function.
            - Single-objective: scalar or ndarray of shape (n_obj,)
              representing the current best objective value(s).
            - Multi-objective: Pareto front (Population), reference point,
              or any structure required by the specific acquisition function.
            Implementations that require a specific type should raise
            TypeError if the type is incompatible.

        Returns
        -------
        np.ndarray
            Acquisition scores. shape: (n_samples,)
            Higher scores indicate more promising candidates.
        """
        ...
