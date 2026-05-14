"""SurrogatePrediction: unified return type for all surrogate model predictions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SurrogatePrediction:
    """
    Unified return type for surrogate model predictions.

    Attributes
    ----------
    mean : np.ndarray
        Predicted mean values. shape: (n_samples, n_obj)
    std : np.ndarray or None
        Predicted standard deviations (uncertainty).
        shape: (n_samples, n_obj). None if the surrogate does not
        provide uncertainty estimates (e.g., RBF interpolation).
    label : np.ndarray or None
        Predicted class labels. shape: (n_samples,).
        None unless the surrogate is a classification model.
    tell_f : np.ndarray
        Values to assign to offspring.f before calling algorithm.tell().
        Falls back to ``mean`` when no override is set (``_tell_f is None``).
        Pass ``_tell_f=np.full(..., np.nan)`` to prevent pbest corruption
        when the surrogate returns non-objective values (e.g., class
        probabilities, novelty scores).
    metadata : dict
        Implementation-specific additional information
        (e.g., SHAP values, gradient estimates).
    """

    mean: np.ndarray
    std: np.ndarray | None = None
    label: np.ndarray | None = None
    _tell_f: np.ndarray | None = field(default=None)
    metadata: dict = field(default_factory=dict)
    # Values that are conventionally used should be implemented
    # as attributes rather than metadata.

    @property
    def has_uncertainty(self) -> bool:
        """Return True if uncertainty estimates are available."""
        return self.std is not None

    @property
    def has_label(self) -> bool:
        """Return True if classification labels are available."""
        return self.label is not None

    @property
    def tell_f(self) -> np.ndarray:
        """Return the override array if set, otherwise fall back to mean."""
        return self._tell_f if self._tell_f is not None else self.mean

    @property
    def has_tell_f(self) -> bool:
        """Return True if a dedicated tell_f override is set."""
        return self._tell_f is not None
