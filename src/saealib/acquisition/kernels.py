"""Pure numerical kernels used by built-in acquisitions.

The functions in this module accept normalized NumPy arrays and do not access
framework objects, RNG state, or mutate their inputs.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def expected_improvement_kernel(
    mean: np.ndarray, std: np.ndarray, reference: float, xi: float
) -> np.ndarray:
    """Return expected improvement for minimize-space means and stds."""
    mean = np.asarray(mean, dtype=np.float64)
    std = np.maximum(np.asarray(std, dtype=np.float64), 1e-9)
    z = (reference - mean - xi) / std
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return np.maximum((reference - mean - xi) * ndtr(z) + std * pdf, 0.0)


def probability_of_feasibility_kernel(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Return ``P(value <= 0)`` elementwise for Gaussian predictions."""
    mean = np.asarray(mean, dtype=np.float64)
    std = np.maximum(np.asarray(std, dtype=np.float64), 1e-9)
    return ndtr(-mean / std)


def product_of_feasibility_kernel(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Return the independent-constraint feasibility product per row."""
    return np.prod(probability_of_feasibility_kernel(mean, std), axis=1)


def lower_confidence_bound_kernel(
    mean: np.ndarray, std: np.ndarray, beta: float
) -> np.ndarray:
    """Return the lower confidence bound in minimize space."""
    return np.asarray(mean, dtype=np.float64) - beta * np.asarray(std, dtype=np.float64)
