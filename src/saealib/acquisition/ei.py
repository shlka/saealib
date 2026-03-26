"""Expected Improvement (EI) acquisition function module."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm

from saealib.acquisition.base import AcquisitionFunction
from saealib.surrogate.prediction import SurrogatePrediction


class ExpectedImprovement(AcquisitionFunction):
    """
    Expected Improvement (EI) acquisition function.

    EI balances exploration and exploitation by computing the expected
    amount of improvement over the current best observed value.

    For single-objective minimization:
        EI(x) = E[max(f_best - f(x), 0)]
               = (f_best - mu) * Phi(Z) + sigma * phi(Z)
        where Z = (f_best - mu) / sigma

    Requires a surrogate that provides uncertainty estimates (std).

    Parameters
    ----------
    xi : float
        Exploration-exploitation trade-off parameter. Higher values
        encourage more exploration. Default: 0.01.
    obj_idx : int
        Index of the objective to optimize. Used for multi-objective
        problems where EI is applied to a single objective. Default: 0.
    """

    def __init__(self, xi: float = 0.01, obj_idx: int = 0):
        self.xi = xi
        self.obj_idx = obj_idx

    def score(
        self,
        prediction: SurrogatePrediction,
        reference: Any,
    ) -> np.ndarray:
        """
        Compute Expected Improvement scores.

        Parameters
        ----------
        prediction : SurrogatePrediction
            Surrogate predictions. Must have std (has_uncertainty == True).
        reference : Any
            Current best objective value. Scalar or ndarray of shape (n_obj,).
            The objective at index obj_idx is used.

        Returns
        -------
        np.ndarray
            EI scores. shape: (n_samples,). Higher is better.

        Raises
        ------
        TypeError
            If prediction does not contain uncertainty estimates.
        """
        if not prediction.has_uncertainty:
            raise TypeError(
                "ExpectedImprovement requires a surrogate with uncertainty "
                "estimates (prediction.std must not be None)."
            )
        mu = prediction.mean[:, self.obj_idx]    # (n_samples,)
        sigma = prediction.std[:, self.obj_idx]  # (n_samples,)

        ref = np.asarray(reference, dtype=float)
        f_best = float(ref.flat[self.obj_idx]) if ref.ndim > 0 else float(ref)

        sigma = np.maximum(sigma, 1e-9)  # avoid division by zero
        z = (f_best - mu - self.xi) / sigma
        ei = (f_best - mu - self.xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.maximum(ei, 0.0)
