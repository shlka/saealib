"""ParEGO acquisition function module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import norm

from saealib.acquisition.base import AcquisitionFunction
from saealib.surrogate.prediction import SurrogatePrediction

if TYPE_CHECKING:
    from saealib.population import Archive


class ParEGOAcquisition(AcquisitionFunction):
    """
    ParEGO acquisition function for multi-objective Bayesian optimisation.

    Scalarises objectives using the Augmented Tchebycheff function with a
    randomly sampled weight vector, then applies Expected Improvement on the
    scalarised prediction (Knowles 2006; formula from Chugh 2020 Eq. 8).

    Scalarised objective (minimised)::

        g(f; w, z*) = max_i [w_i |f_i - z_i*|] + alpha * sum_i w_i |f_i - z_i*|

    where z* is the component-wise minimum of the current archive and w is
    drawn uniformly from the (k-1)-simplex at each call to
    ``compute_reference``.

    Because saealib's ABC receives ``(n_samples, n_obj)`` predictions rather
    than per-objective GP posteriors, the scalarised GP is approximated::

        mu_s  = g(mu; w, z*)
        std_s = (w . std).sum(axis=-1)   [linear-combination approximation]

    EI is then computed on (mu_s, std_s) against the best scalarised archive
    value.

    Parameters
    ----------
    alpha : float
        Augmentation coefficient (rho in some references).
        Default: 0.05 (mlr3mbo convention).
    rng : np.random.Generator or None
        Random number generator for weight sampling.
    """

    requires_uncertainty: bool = True

    def __init__(
        self,
        alpha: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.alpha = alpha
        self._rng = rng if rng is not None else np.random.default_rng()

    def _scalarize(
        self,
        f: np.ndarray,
        weights: np.ndarray,
        z_star: np.ndarray,
    ) -> np.ndarray:
        """Augmented Tchebycheff scalarization (Chugh 2020 Eq. 8)."""
        diff = np.abs(f - z_star)  # (..., n_obj)
        weighted = weights * diff  # (..., n_obj)
        return weighted.max(axis=-1) + self.alpha * weighted.sum(axis=-1)

    def compute_reference(
        self, archive: Archive
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Sample a new weight vector and compute the ideal point.

        Parameters
        ----------
        archive : Archive
            Archive of evaluated solutions.

        Returns
        -------
        tuple
            ``(z_star, weights, f_best_scalar)`` — component-wise minimum of
            archive objective values, a new weight vector drawn uniformly from
            the (k-1)-simplex, and the minimum scalarised archive value under
            these weights.
        """
        n_obj = archive.f.shape[1]
        z_star = archive.f.min(axis=0)
        weights = self._rng.dirichlet(np.ones(n_obj))
        f_best = float(self._scalarize(archive.f, weights, z_star).min())
        return z_star, weights, f_best

    def score(
        self,
        prediction: SurrogatePrediction,
        reference: Any,
    ) -> np.ndarray:
        """
        Compute EI on the Augmented Tchebycheff scalarised prediction.

        Parameters
        ----------
        prediction : SurrogatePrediction
            Surrogate predictions.  Must have std (has_uncertainty == True).
        reference : tuple
            ``(z_star, weights, f_best_scalar)`` from ``compute_reference``.

        Returns
        -------
        np.ndarray
            EI scores.  shape: (n_samples,).  Higher is better.

        Raises
        ------
        TypeError
            If prediction does not contain uncertainty estimates.
        """
        if not prediction.has_uncertainty:
            raise TypeError(
                "ParEGOAcquisition requires uncertainty estimates "
                "(prediction.std must not be None)."
            )
        z_star, weights, f_best = reference

        mu_s = self._scalarize(prediction.value, weights, z_star)  # (n,)
        sigma_s = (weights * prediction.std).sum(axis=-1)  # (n,)
        sigma_s = np.maximum(sigma_s, 1e-9)

        z = (f_best - mu_s) / sigma_s
        ei = (f_best - mu_s) * norm.cdf(z) + sigma_s * norm.pdf(z)
        return np.maximum(ei, 0.0)
