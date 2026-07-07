"""EHVI (Expected Hypervolume Improvement) acquisition function module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from saealib.acquisition.base import AcquisitionFunction, direction_to_minimize_sign
from saealib.surrogate.prediction import SurrogatePrediction
from saealib.utils.indicators import _non_dominated, hypervolume

if TYPE_CHECKING:
    from saealib.population import Archive


class EHVIAcquisition(AcquisitionFunction):
    """
    Expected Hypervolume Improvement (EHVI) acquisition function.

    Estimates EHVI via Monte Carlo sampling (Hupkens 2015 Eq. 1; HVI
    definition from Daulton 2020 Definition 2)::

        EHVI(x) = (1/S) * sum_s HVI(y_s, P, r)
        y_s ~ N(mu(x), diag(sigma^2(x)))   [independent objectives]

    where HVI(y, P, r) = HV(P u {y}, r) - HV(P, r).

    ``archive.f``, ``reference_point``, and the predicted mean are internally
    converted to minimisation convention via ``direction_to_minimize_sign``
    (see ``direction`` below) before the formula above runs.

    Parameters
    ----------
    n_samples : int
        Number of MC samples per candidate.  Default: 256.
    reference_point : array-like or None
        Hypervolume reference point, given in raw objective space (i.e. in
        the same convention as ``problem.direction``, not necessarily
        minimisation).  If None, auto-computed from the archive nadir with
        a 10 % margin.
    rng : np.random.Generator or None
        Random number generator for MC sampling.
    direction : np.ndarray or None
        Per-objective optimization direction (+1 = maximize, -1 = minimize).
        shape: (n_obj,). ``None`` (default) means already-minimize, so
        behaviour for existing minimize-only callers is unchanged; when
        unset, it is auto-injected from ``problem.direction`` at run start.
    """

    requires_uncertainty: bool = True

    def __init__(
        self,
        n_samples: int = 256,
        reference_point: npt.ArrayLike | None = None,
        rng: np.random.Generator | None = None,
        direction: np.ndarray | None = None,
    ) -> None:
        self.n_samples = n_samples
        self.reference_point = (
            np.asarray(reference_point, dtype=float)
            if reference_point is not None
            else None
        )
        self._rng = rng if rng is not None else np.random.default_rng()
        self.direction = direction

    def compute_reference(
        self,
        archive: Archive,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Extract the Pareto front and (if needed) compute the reference point.

        Parameters
        ----------
        archive : Archive
            Archive of evaluated solutions.

        Returns
        -------
        tuple
            ``(pareto_f, ref_point, base_hv)`` — non-dominated objective
            matrix, hypervolume reference point, and current hypervolume.
        """
        s = direction_to_minimize_sign(self.direction)
        f = archive.f * s
        pareto_f = _non_dominated(f)

        if self.reference_point is not None:
            ref = self.reference_point * s
        else:
            nadir = f.max(axis=0)
            span = nadir - f.min(axis=0)
            ref = nadir + 0.1 * np.maximum(span, 1e-6)

        base_hv = hypervolume(pareto_f, ref)
        return pareto_f, ref, base_hv

    def score(
        self,
        prediction: SurrogatePrediction,
        reference: Any,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Estimate EHVI via Monte Carlo sampling.

        Parameters
        ----------
        prediction : SurrogatePrediction
            Surrogate predictions.  Must have std (has_uncertainty == True).
        reference : tuple
            ``(pareto_f, ref_point, base_hv)`` from ``compute_reference``.
        rng : np.random.Generator or None, optional
            Random number generator.  When provided (e.g. ``ctx.rng``), it is
            used instead of the instance-level ``_rng`` so all randomness flows
            through the master RNG.

        Returns
        -------
        np.ndarray
            EHVI scores.  shape: (n_samples,).  Higher is better.

        Raises
        ------
        TypeError
            If prediction does not contain uncertainty estimates.
        """
        if not prediction.has_uncertainty:
            raise TypeError(
                "EHVIAcquisition requires uncertainty estimates "
                "(prediction.std must not be None)."
            )
        pareto_f, ref_point, base_hv = reference
        assert prediction.std is not None
        s = direction_to_minimize_sign(self.direction)
        mu_conv = prediction.value * s  # (n_cand, n_obj)
        sigma = prediction.std  # (n_cand, n_obj)
        n_cand, n_obj = mu_conv.shape

        # Draw all MC samples at once: (n_samples, n_cand, n_obj)
        _rng = rng if rng is not None else self._rng
        eps = _rng.standard_normal((self.n_samples, n_cand, n_obj))
        mc_samples = mu_conv[np.newaxis] + sigma[np.newaxis] * eps

        ehvi = np.zeros(n_cand)
        for i in range(n_cand):
            hvi_sum = 0.0
            for s in range(self.n_samples):
                y = mc_samples[s, i]
                if np.any(y >= ref_point):
                    continue
                combined = (
                    y[np.newaxis]
                    if len(pareto_f) == 0
                    else np.vstack([pareto_f, y[np.newaxis]])
                )
                hvi_sum += max(hypervolume(combined, ref_point) - base_hv, 0.0)
            ehvi[i] = hvi_sum / self.n_samples

        return ehvi
