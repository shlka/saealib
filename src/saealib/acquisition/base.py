"""
Acquisition function base module.

This module defines the abstract base class for acquisition (infill criterion)
functions used in surrogate-assisted optimization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from saealib.population import Archive
    from saealib.surrogate.prediction import SurrogatePrediction


def direction_to_minimize_sign(direction: np.ndarray | None) -> np.ndarray | float:
    """Return the multiplicative sign converting objectives to minimize-space.

    Every direction-sensitive acquisition function's formulas are written
    assuming minimization. Multiplying a raw-objective-space quantity
    (``archive.f``, ``prediction.value``, a user-supplied ``reference``) by
    this sign converts it into minimize-space before the formula runs.
    Uncertainty magnitudes (``prediction.std``, ``sigma``) must never be
    multiplied by this sign.

    Parameters
    ----------
    direction : np.ndarray or None
        Per-objective optimization direction (+1 = maximize, -1 = minimize).
        shape: (n_obj,). ``None`` means already-minimize.

    Returns
    -------
    np.ndarray or float
        ``-direction`` if *direction* is given, else the scalar ``1.0``.
        Both broadcast correctly against ``(n_obj,)`` or ``(n, n_obj)`` arrays.
    """
    return -direction if direction is not None else 1.0


class AcquisitionFunction(ABC):
    """
    Abstract base class for acquisition functions (infill criteria).

    An acquisition function converts a SurrogatePrediction into a scalar
    score per candidate, which is used to rank candidates for true evaluation.

    The AcquisitionFunction is completely decoupled from Surrogate:
    it knows nothing about how predictions are generated.
    """

    # Optimizer.validate() cross-checks this with surrogate.provides_uncertainty.
    requires_uncertainty: bool = False

    # Optimizer._inject_acquisition_directions() only auto-injects
    # problem.direction into acquisition functions that opt in via this flag.
    direction_sensitive: bool = True

    @abstractmethod
    def compute_reference(
        self,
        archive: Archive,
        rng: np.random.Generator | None = None,
    ) -> Any:
        """
        Compute the reference value required by this acquisition function.

        Called by SurrogateManager before scoring. Each acquisition function
        derives its appropriate reference from the archive. If ``self.reference``
        is set (injected externally at construction or later), implementations
        should return it instead of computing from the archive, allowing users
        to supply domain knowledge or a fixed reference point.

        Parameters
        ----------
        archive : Archive
            Archive of evaluated solutions.
        rng : np.random.Generator or None, optional
            Random number generator from ``ctx.rng``.  Implementations that
            require randomness should prefer this over a stored ``_rng`` so
            that all randomness flows through the single master RNG.

        Returns
        -------
        Any
            Reference value passed to ``score``. Return ``None`` if this
            acquisition function does not use a reference.
        """
        ...

    @abstractmethod
    def score(
        self,
        prediction: SurrogatePrediction,
        reference: Any,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Compute acquisition scores for a set of candidates.

        Parameters
        ----------
        prediction : SurrogatePrediction
            Predictions from a surrogate model.
        reference : Any
            Reference value produced by ``compute_reference``.
        rng : np.random.Generator or None, optional
            Random number generator from ``ctx.rng``.

        Returns
        -------
        np.ndarray
            Acquisition scores. shape: (n_samples,)
            Higher scores indicate more promising candidates.
        """
        ...
