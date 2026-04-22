"""
Acquisition function base module.

This module defines the abstract base class for acquisition (infill criterion)
functions used in surrogate-assisted optimization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from saealib.surrogate.prediction import SurrogatePrediction

if TYPE_CHECKING:
    from saealib.population import Archive


class AcquisitionFunction(ABC):
    """
    Abstract base class for acquisition functions (infill criteria).

    An acquisition function converts a SurrogatePrediction into a scalar
    score per candidate, which is used to rank candidates for true evaluation.

    The AcquisitionFunction is completely decoupled from Surrogate:
    it knows nothing about how predictions are generated.
    """

    @abstractmethod
    def compute_reference(self, archive: Archive) -> Any:
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
    ) -> np.ndarray:
        """
        Compute acquisition scores for a set of candidates.

        Parameters
        ----------
        prediction : SurrogatePrediction
            Predictions from a surrogate model.
        reference : Any
            Reference value produced by ``compute_reference``.

        Returns
        -------
        np.ndarray
            Acquisition scores. shape: (n_samples,)
            Higher scores indicate more promising candidates.
        """
        ...
