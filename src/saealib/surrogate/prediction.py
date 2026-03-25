"""
SurrogatePrediction module.

This module defines the SurrogatePrediction dataclass,
which is the unified return type for all surrogate model predictions.
"""

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
    metadata : dict
        Implementation-specific additional information
        (e.g., SHAP values, gradient estimates).
    """

    mean: np.ndarray
    std: np.ndarray | None = None
    label: np.ndarray | None = None
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
