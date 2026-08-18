"""Lower Confidence Bound (LCB) acquisition function module."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from saealib.acquisition.base import PointwiseAcquisition, direction_to_minimize_sign
from saealib.acquisition.kernels import lower_confidence_bound_kernel
from saealib.surrogate.prediction import SurrogatePrediction

if TYPE_CHECKING:
    from saealib.population import Archive


class LowerConfidenceBound(PointwiseAcquisition):
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
        Exploration-exploitation trade-off parameter used when
        ``beta_schedule`` is ``None``. Higher values encourage more
        exploration. Default: 2.0.
    obj_idx : int
        Index of the objective to optimize. Used for multi-objective
        problems where LCB is applied to a single objective. Default: 0.
    beta_schedule : callable or None
        Optional ``beta_t`` schedule ``schedule(t) -> float``, in the
        notation of the cited GP-UCB regret bound (``kappa`` corresponds to
        ``sqrt(beta_t)``). When set, ``score()`` counts its own calls as the
        round index ``t`` (1 on the first call, incrementing by 1 per call)
        and uses ``kappa_t = sqrt(schedule(t))`` in place of the fixed
        ``kappa``. See :func:`gp_ucb_beta_schedule` for the schedule from the
        cited finite-domain regret bound. ``None`` (default) keeps ``kappa``
        fixed.
    direction : np.ndarray or None
        Per-objective optimization direction (+1 = maximize, -1 = minimize).
        shape: (n_obj,). The predicted mean is converted to minimize-space
        via ``direction_to_minimize_sign`` before the (minimize-only) LCB
        formula above runs. ``None`` (default) means already-minimize; when
        unset, it is auto-injected from ``problem.direction`` at run start.

    References
    ----------
    :cite:`srinivas2012gpucb`: Srinivas, N., Krause, A., Kakade, S. M., &
    Seeger, M. W. (2012). Information-theoretic regret bounds for Gaussian
    process optimization in the bandit setting. *IEEE Transactions on
    Information Theory*, 58(5), 3250-3265.
    """

    requires_uncertainty: bool = True

    def __init__(
        self,
        kappa: float = 2.0,
        obj_idx: int = 0,
        beta_schedule: Callable[[int], float] | None = None,
        direction: np.ndarray | None = None,
    ):
        self.kappa = kappa
        self.obj_idx = obj_idx
        self.beta_schedule = beta_schedule
        self.direction = direction
        self._t = 0

    def compute_reference(
        self, archive: Archive, rng: np.random.Generator | None = None
    ) -> Any:
        """LowerConfidenceBound uses no reference value; always returns None."""
        return None

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
        self._t += 1
        kappa = (
            math.sqrt(self.beta_schedule(self._t)) if self.beta_schedule else self.kappa
        )
        s = direction_to_minimize_sign(self.direction)
        s_idx = s[self.obj_idx] if isinstance(s, np.ndarray) else s
        mu = prediction.value[:, self.obj_idx] * s_idx  # (n_samples,)
        sigma = prediction.std[:, self.obj_idx]  # (n_samples,)
        return -lower_confidence_bound_kernel(mu, sigma, kappa)


def gp_ucb_beta_schedule(
    domain_size: int, delta: float = 0.1
) -> Callable[[int], float]:
    """
    Return the finite-domain GP-UCB ``beta_t`` schedule for use with ``beta_schedule``.

    ``beta_t = 2 log(domain_size * t^2 * pi^2 / (6 * delta))``, the schedule
    the cited regret bound uses for a finite search domain of size
    ``domain_size`` :cite:`srinivas2012gpucb`. For a continuous domain,
    ``domain_size`` is a discretization-count proxy supplied by the caller;
    this function does not derive one.

    Parameters
    ----------
    domain_size : int
        Size of the (possibly discretized) search domain (``|D|`` in the
        cited bound).
    delta : float
        Confidence parameter (``0 < delta < 1``). Default: 0.1.

    Returns
    -------
    callable
        ``schedule(t) -> float``, increasing in the round index ``t``
        (``t`` starting at 1).
    """

    def _schedule(t: int) -> float:
        return 2.0 * math.log(domain_size * t**2 * math.pi**2 / (6.0 * delta))

    return _schedule
