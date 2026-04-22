"""Probability of Feasibility (PoF) acquisition function module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import norm

from saealib.acquisition.base import AcquisitionFunction
from saealib.surrogate.prediction import SurrogatePrediction

if TYPE_CHECKING:
    from saealib.population import Archive


class ProbabilityOfFeasibility(AcquisitionFunction):
    """
    Probability of Feasibility (PoF) acquisition function.

    Estimates the probability that a candidate satisfies a constraint
    g(x) <= 0, using a surrogate model that predicts the constraint value.

    PoF(x) = Phi((0 - mu(x)) / sigma(x))

    Typically used in combination with another acquisition function
    (e.g., EI * PoF) to handle black-box constraints.

    Requires a surrogate that provides uncertainty estimates (std).

    Parameters
    ----------
    obj_idx : int
        Index of the predicted constraint to evaluate. Default: 0.
    """

    def __init__(self, obj_idx: int = 0, reference: Any = None):
        self.obj_idx = obj_idx
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
        Compute Probability of Feasibility scores.

        Parameters
        ----------
        prediction : SurrogatePrediction
            Surrogate predictions of constraint values.
            Must have std (has_uncertainty == True).
        reference : Any
            Not used. Accepted for interface compatibility.

        Returns
        -------
        np.ndarray
            PoF scores in [0, 1]. shape: (n_samples,)
            Higher scores indicate a higher probability of feasibility.

        Raises
        ------
        TypeError
            If prediction does not contain uncertainty estimates.
        """
        if not prediction.has_uncertainty:
            raise TypeError(
                "ProbabilityOfFeasibility requires a surrogate with uncertainty "
                "estimates (prediction.std must not be None)."
            )
        mu = prediction.mean[:, self.obj_idx]  # (n_samples,)
        sigma = prediction.std[:, self.obj_idx]  # (n_samples,)
        sigma = np.maximum(sigma, 1e-9)
        return norm.cdf((0.0 - mu) / sigma)  # P(g(x) <= 0)
