"""Abstract base class for surrogate models."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from saealib.surrogate.prediction import SurrogatePrediction

if TYPE_CHECKING:
    from saealib.context import OptimizationContext


class Surrogate(ABC):
    """Base class for surrogate models."""

    # GP implementation will override this with True.
    provides_uncertainty: bool = False

    @abstractmethod
    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """
        Fit the surrogate model.

        Parameters
        ----------
        train_x : np.ndarray
            Training input data. shape: (n_samples, n_features)
        train_y : np.ndarray
            Training output data. shape: (n_samples, n_obj)
            For n_obj == 1, shape (n_samples,) is also accepted.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def predict(self, test_x: np.ndarray) -> SurrogatePrediction:
        """
        Predict using the surrogate model.

        Parameters
        ----------
        test_x : np.ndarray
            Input data for prediction. shape: (n_samples, n_features)

        Returns
        -------
        SurrogatePrediction
            prediction.value shape: (n_samples, n_obj)
            prediction.std  shape: (n_samples, n_obj) or None
        """
        pass

    def post_fit(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        ctx: OptimizationContext | None = None,
    ) -> None:
        """Post-fit lifecycle hook; override to inject custom processing.

        Parameters
        ----------
        train_x : np.ndarray
            Training input data used for this fit. shape: (n_samples, n_features)
        train_y : np.ndarray
            Training output data used for this fit. shape: (n_samples, n_obj)
        ctx : OptimizationContext or None, optional
            Current optimization context.
        """

    def with_post_fit(
        self,
        fn: Callable[[np.ndarray, np.ndarray, OptimizationContext | None], None],
    ) -> Surrogate:
        """Return a copy of this surrogate with ``fn`` appended to the post-fit hook.

        Parameters
        ----------
        fn : callable
            ``fn(train_x, train_y, ctx) -> None``

        Returns
        -------
        Surrogate
            Shallow copy with the hook registered.
        """
        new = copy.copy(self)
        prev = self.post_fit

        def _hook(train_x: np.ndarray, train_y: np.ndarray, ctx=None) -> None:
            prev(train_x, train_y, ctx)
            fn(train_x, train_y, ctx)

        new.post_fit = _hook
        return new
