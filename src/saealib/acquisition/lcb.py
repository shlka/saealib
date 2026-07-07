"""Lower Confidence Bound (LCB) acquisition function module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from saealib.acquisition.base import AcquisitionFunction, direction_to_minimize_sign
from saealib.surrogate.prediction import SurrogatePrediction

if TYPE_CHECKING:
    from saealib.population import Archive


class LowerConfidenceBound(AcquisitionFunction):
    """
    Lower Confidence Bound (LCB) acquisition function.

    LCB trades off exploitation (low predicted mean) and exploration
    (high predicted uncertainty):

        LCB(x) = mu(x) - kappa * sigma(x)

    For minimization, lower LCB values are better. This class returns
    the negated LCB so that higher scores indicate more promising candidates,
    consistent with the convention used by other acquisition functions.

    Requires a surrogate that provides uncertainty estimates (std).

    Parameters
    ----------
    kappa : float
        Exploration-exploitation trade-off parameter. Higher values
        encourage more exploration. Default: 2.0.
    obj_idx : int
        Index of the objective to optimize. Used for multi-objective
        problems where LCB is applied to a single objective. Default: 0.
    direction : np.ndarray or None
        Per-objective optimization direction (+1 = maximize, -1 = minimize).
        shape: (n_obj,). The predicted mean is converted to minimize-space
        via ``direction_to_minimize_sign`` before the (minimize-only) LCB
        formula above runs. ``None`` (default) means already-minimize; when
        unset, it is auto-injected from ``problem.direction`` at run start.
    """

    requires_uncertainty: bool = True

    def __init__(
        self,
        kappa: float = 2.0,
        obj_idx: int = 0,
        reference: Any = None,
        direction: np.ndarray | None = None,
    ):
        self.kappa = kappa
        self.obj_idx = obj_idx
        self.reference = reference
        self.direction = direction

    def compute_reference(
        self, archive: Archive, rng: np.random.Generator | None = None
    ) -> Any:
        """Return fixed reference if set, otherwise None."""
        return self.reference

    def score(
        self,
        prediction: SurrogatePrediction,
        reference: Any = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Compute negated LCB scores (higher is better).

        Parameters
        ----------
        prediction : SurrogatePrediction
            Surrogate predictions. Must have std (has_uncertainty == True).
        reference : Any
            Not used. Accepted for interface compatibility.

        Returns
        -------
        np.ndarray
            Negated LCB scores. shape: (n_samples,)

        Raises
        ------
        TypeError
            If prediction does not contain uncertainty estimates.
        """
        if not prediction.has_uncertainty:
            raise TypeError(
                "LowerConfidenceBound requires a surrogate with uncertainty "
                "estimates (prediction.std must not be None)."
            )
        assert prediction.std is not None
        s = direction_to_minimize_sign(self.direction)
        s_idx = (
            s[self.obj_idx]  # type: ignore  # ty narrows bare np.ndarray isinstance to object dtype; s is float ndarray at runtime
            if isinstance(s, np.ndarray)
            else s
        )
        mu = prediction.value[:, self.obj_idx] * s_idx  # (n_samples,)
        sigma = prediction.std[:, self.obj_idx]  # (n_samples,)
        return -(mu - self.kappa * sigma)  # negate: higher = better
