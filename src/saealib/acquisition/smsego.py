"""SMS-EGO acquisition function module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from saealib.acquisition.base import AcquisitionFunction
from saealib.surrogate.prediction import SurrogatePrediction
from saealib.utils.indicators import _non_dominated, hypervolume

if TYPE_CHECKING:
    from saealib.population import Archive


class SMSEGOAcquisition(AcquisitionFunction):
    """
    SMS-EGO acquisition function for multi-objective Bayesian optimisation.

    Scores each candidate by the hypervolume improvement (HVI) of its
    per-objective LCB prediction over the current Pareto front (Ponweiser
    2008; LCB formula from mlr3mbo ``AcqFunctionSmsEgo.R``).

    LCB per objective::

        y_hat_j(x) = mu_j(x) - lambda * sigma_j(x)

    Acquisition score::

        SMS(x) = HV(P u {y_hat(x)}, r) - HV(P, r)

    Parameters
    ----------
    lam : float
        LCB width parameter λ.  Default: 1.0 (mlr3mbo convention).
    reference_point : array-like or None
        Hypervolume reference point (minimisation convention).  If None,
        auto-computed from the archive nadir with a 10 % margin.
    """

    requires_uncertainty: bool = True

    def __init__(
        self,
        lam: float = 1.0,
        reference_point: np.ndarray | None = None,
    ) -> None:
        self.lam = lam
        self.reference_point = (
            np.asarray(reference_point, dtype=float)
            if reference_point is not None
            else None
        )

    def compute_reference(
        self, archive: Archive
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
        f = archive.f
        pareto_f = _non_dominated(f)

        if self.reference_point is not None:
            ref = self.reference_point
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
    ) -> np.ndarray:
        """
        Compute hypervolume improvement of the LCB-predicted candidates.

        Parameters
        ----------
        prediction : SurrogatePrediction
            Surrogate predictions.  Must have std (has_uncertainty == True).
        reference : tuple
            ``(pareto_f, ref_point, base_hv)`` from ``compute_reference``.

        Returns
        -------
        np.ndarray
            HVI scores.  shape: (n_samples,).  Higher is better.

        Raises
        ------
        TypeError
            If prediction does not contain uncertainty estimates.
        """
        if not prediction.has_uncertainty:
            raise TypeError(
                "SMSEGOAcquisition requires uncertainty estimates "
                "(prediction.std must not be None)."
            )
        pareto_f, ref_point, base_hv = reference
        lcb = prediction.value - self.lam * prediction.std  # (n, n_obj)
        n = lcb.shape[0]
        hvi = np.zeros(n)

        for i in range(n):
            y = lcb[i]
            if np.any(y >= ref_point):
                continue
            combined = (
                y[np.newaxis] if len(pareto_f) == 0
                else np.vstack([pareto_f, y[np.newaxis]])
            )
            hvi[i] = max(hypervolume(combined, ref_point) - base_hv, 0.0)

        return hvi
