"""
RBF surrogate model module.

This module defines the Radial Basis Function (RBF) surrogate model.
RBFSurrogate supports multi-objective problems by maintaining one
independent RBF model per objective (ensemble approach).
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.spatial

from saealib.exceptions import ValidationError
from saealib.registry import register
from saealib.surrogate.base import RegressionSurrogate
from saealib.surrogate.prediction import SurrogatePrediction

logger = logging.getLogger(__name__)


def gaussian_kernel(x1: np.ndarray, x2: np.ndarray, length_scale=2.0) -> np.ndarray:
    """
    Gaussian radial basis function kernel.

    Uses the ``exp(-r^2 / (2 * length_scale^2))`` parameterization;
    equivalent to the papers' ``exp(-gamma * r^2)`` form with
    ``gamma = 1 / (2 * length_scale^2)``. Based on
    :cite:`gutmann2001rbf,regis2005cors`.

    Parameters
    ----------
    x1 : np.ndarray
        Input data 1.
    x2 : np.ndarray
        Input data 2.
    length_scale : float
        Kernel width parameter, in the same distance units as the input
        data.

    Returns
    -------
    np.ndarray
        Matrix of kernel evaluations between x1 and x2. shape: (len(x1), len(x2))
    """
    sq_dist = scipy.spatial.distance.cdist(x1, x2, "sqeuclidean")
    return np.exp(-sq_dist / (2 * (length_scale**2)))


def thin_plate_spline_kernel(
    x1: np.ndarray, x2: np.ndarray, length_scale: float | None = None
) -> np.ndarray:
    """Thin-plate spline radial basis function kernel.

    Parameters
    ----------
    x1 : np.ndarray
        Input data 1.
    x2 : np.ndarray
        Input data 2.
    length_scale : float or None
        Unused length-scale parameter accepted for kernel compatibility.

    Returns
    -------
    np.ndarray
        Matrix of kernel evaluations between x1 and x2. shape: (len(x1), len(x2))
    """
    r = scipy.spatial.distance.cdist(x1, x2, "euclidean")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = r**2 * np.log(r)
    return np.nan_to_num(out, nan=0.0)


def linear_kernel(
    x1: np.ndarray, x2: np.ndarray, length_scale: float | None = None
) -> np.ndarray:
    """Linear radial basis function kernel.

    This conditionally positive definite kernel has order 0 and requires a
    constant polynomial term for a well-posed system. Based on
    :cite:`gutmann2001rbf`.

    Parameters
    ----------
    x1 : np.ndarray
        Input data 1.
    x2 : np.ndarray
        Input data 2.
    length_scale : float or None
        Unused length-scale parameter accepted for kernel compatibility.

    Returns
    -------
    np.ndarray
        Matrix of kernel evaluations between x1 and x2. shape: (len(x1), len(x2))
    """
    return scipy.spatial.distance.cdist(x1, x2, "euclidean")


def cubic_kernel(
    x1: np.ndarray, x2: np.ndarray, length_scale: float | None = None
) -> np.ndarray:
    """Cubic radial basis function kernel.

    This conditionally positive definite kernel has order 1 and requires a
    linear polynomial term for a well-posed system. Based on
    :cite:`gutmann2001rbf`.

    Parameters
    ----------
    x1 : np.ndarray
        Input data 1.
    x2 : np.ndarray
        Input data 2.
    length_scale : float or None
        Unused length-scale parameter accepted for kernel compatibility.

    Returns
    -------
    np.ndarray
        Matrix of kernel evaluations between x1 and x2. shape: (len(x1), len(x2))
    """
    return scipy.spatial.distance.cdist(x1, x2, "euclidean") ** 3


def multiquadric_kernel(
    x1: np.ndarray, x2: np.ndarray, length_scale: float = 2.0
) -> np.ndarray:
    """Multiquadric radial basis function kernel.

    This conditionally positive definite kernel has order 0 and requires a
    constant polynomial term for a well-posed system. Based on
    :cite:`gutmann2001rbf`.

    Parameters
    ----------
    x1 : np.ndarray
        Input data 1.
    x2 : np.ndarray
        Input data 2.
    length_scale : float
        Shape parameter (denoted ``gamma`` in the paper), in the same
        distance units as the input data.

    Returns
    -------
    np.ndarray
        Matrix of kernel evaluations between x1 and x2. shape: (len(x1), len(x2))
    """
    sq_dist = scipy.spatial.distance.cdist(x1, x2, "sqeuclidean")
    return np.sqrt(sq_dist + length_scale**2)


def matern_kernel(
    x1: np.ndarray, x2: np.ndarray, length_scale: float = 2.0, nu: float = 1.5
) -> np.ndarray:
    """Matérn radial basis function kernel.

    The supported cases are strictly positive definite and do not require a
    polynomial term. Based on :cite:`rasmussen2006gpml`.

    Parameters
    ----------
    x1 : np.ndarray
        Input data 1.
    x2 : np.ndarray
        Input data 2.
    length_scale : float
        Length-scale parameter (denoted ``ell`` in the paper).
    nu : float
        Smoothness parameter. Must be 0.5, 1.5, or 2.5.

    Returns
    -------
    np.ndarray
        Matrix of kernel evaluations between x1 and x2. shape: (len(x1), len(x2))

    Raises
    ------
    ValidationError
        If ``nu`` is not 0.5, 1.5, or 2.5.
    """
    if nu not in (0.5, 1.5, 2.5):
        raise ValidationError("nu must be 0.5, 1.5, or 2.5")
    r = scipy.spatial.distance.cdist(x1, x2, "euclidean")
    if nu == 0.5:
        return np.exp(-r / length_scale)
    if nu == 1.5:
        scaled = np.sqrt(3) * r / length_scale
        return (1 + scaled) * np.exp(-scaled)
    scaled = np.sqrt(5) * r / length_scale
    return (1 + scaled + scaled**2 / 3) * np.exp(-scaled)


class _RBFModel:
    """
    Single-objective RBF interpolation model (internal use only).

    Holds all state for one objective's RBF fit. Used as a building
    block by RBFSurrogate to support multi-objective problems.
    """

    def __init__(
        self,
        kernel: Callable[..., Any],
        dim: int,
        polynomial_degree: int,
        solver: str,
        alpha: float,
    ) -> None:
        self.kernel = kernel
        self.dim = dim
        self.polynomial_degree = polynomial_degree
        self.solver = solver
        self.alpha = alpha
        self.train_x: np.ndarray | None = None
        self.train_y: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.poly_coeffs: np.ndarray | None = None
        self.kernel_matrix: np.ndarray | None = None
        self.length_scale: np.floating[Any] | float | None = None
        self._last_ill_conditioned = False
        self._last_singular = False

    def fit(self, train_x: np.ndarray, train_y_1d: np.ndarray) -> None:
        """
        Fit the RBF model.

        Parameters
        ----------
        train_x : np.ndarray
            Training input data. shape: (n_samples, n_features)
        train_y_1d : np.ndarray
            Training output data for one objective. shape: (n_samples,)
        """
        self.train_x = np.asarray(train_x)
        self.train_y = np.asarray(train_y_1d)
        n_samples = len(self.train_x)
        self.length_scale = np.median(scipy.spatial.distance.pdist(self.train_x))
        phi = self.kernel(self.train_x, self.train_x, length_scale=self.length_scale)

        if self.polynomial_degree == -1:
            target = self.train_y - np.mean(self.train_y)
            n_poly = 0
            a = np.array(phi, copy=True)
        else:
            target = self.train_y
            if self.polynomial_degree == 0:
                polynomial_basis = np.ones((n_samples, 1))
            else:
                polynomial_basis = np.hstack([np.ones((n_samples, 1)), self.train_x])
            n_poly = polynomial_basis.shape[1]
            a = np.block(
                [
                    [phi, polynomial_basis],
                    [polynomial_basis.T, np.zeros((n_poly, n_poly))],
                ]
            )

        rhs = np.concatenate((target, np.zeros(n_poly))) if n_poly else target

        if self.solver == "tikhonov":
            a[:n_samples, :n_samples] += self.alpha * np.eye(n_samples)

        self.kernel_matrix = a
        rcond = 1 / np.linalg.cond(a)
        ill_conditioned = rcond < np.finfo(a.dtype).eps
        if ill_conditioned:
            logger.debug(f"Kernel matrix is ill-conditioned. RCOND: {rcond}")
        self._last_ill_conditioned = ill_conditioned

        try:
            if self.solver == "lstsq":
                solution = np.linalg.lstsq(a, rhs, rcond=None)[0]
            else:
                solution = np.linalg.solve(a, rhs)
        except np.linalg.LinAlgError:
            logger.debug(f"Kernel matrix solve failed (singular). RCOND: {rcond}")
            if not self._last_singular:
                logger.warning(
                    "Failed to solve linear system (Kernel matrix might be singular)."
                )
            self._last_singular = True
            solution = np.nan * np.ones(a.shape[0])
        else:
            if self._last_singular:
                logger.debug("Kernel matrix solve recovered from singularity.")
            self._last_singular = False

        self.weights = solution[:n_samples]
        self.poly_coeffs = solution[n_samples:] if n_poly else None

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        """
        Predict for one objective.

        Parameters
        ----------
        test_x : np.ndarray
            Input data. shape: (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predicted values. shape: (n_samples,)
        """
        test = np.asarray(test_x)
        if test.ndim == 1:
            test = test.reshape(1, -1)
        assert self.train_x is not None
        assert self.weights is not None and self.train_y is not None
        k = self.kernel(self.train_x, test, length_scale=self.length_scale)
        if self.polynomial_degree == -1:
            preds = k.T.dot(self.weights) + np.mean(self.train_y)
        else:
            assert self.poly_coeffs is not None
            if self.polynomial_degree == 0:
                polynomial_test = np.ones((len(test), 1))
            else:
                polynomial_test = np.hstack([np.ones((len(test), 1)), test])
            preds = k.T.dot(self.weights) + polynomial_test.dot(self.poly_coeffs)
        return np.asarray(preds).flatten()


@register()
class RBFSurrogate(RegressionSurrogate):
    """Radial Basis Function interpolation surrogate model.

    Supports multi-objective problems with one independent model per
    objective. The number of objectives is inferred from ``train_y`` on the
    first call to ``fit``.

    Attributes
    ----------
    kernel : callable
        Kernel function (e.g. gaussian_kernel).
    dim : int
        Dimensionality of the input data.
    n_obj : int or None
        Number of objectives. Set on first fit call.
    polynomial_degree : int
        Degree of the optional polynomial term, or ``-1`` to disable it.
    solver : str
        Linear system solver used during fitting.
    alpha : float
        Tikhonov regularization strength.

    References
    ----------
    Origin of RBF interpolation: Hardy (1971).

    :cite:`gutmann2001rbf`: Gutmann, H.-M. (2001). A radial basis function
    method for global optimization. *Journal of Global Optimization*,
    19(3), 201-227.

    :cite:`regis2005cors`: Regis, R. G., & Shoemaker, C. A. (2005).
    Constrained global optimization of expensive black box functions using
    radial basis functions. *Journal of Global Optimization*, 31(1),
    153-171.

    :cite:`rasmussen2006gpml`: Rasmussen, C. E., & Williams, C. K. I. (2006).
    Gaussian processes for machine learning. MIT Press.
    """

    def __init__(
        self,
        kernel: Callable[..., Any],
        dim: int,
        polynomial_degree: int = -1,
        solver: str = "solve",
        alpha: float = 1e-8,
    ) -> None:
        if polynomial_degree not in (-1, 0, 1):
            raise ValidationError("polynomial_degree must be -1, 0, or 1")
        if solver not in ("solve", "lstsq", "tikhonov"):
            raise ValidationError("solver must be 'solve', 'lstsq', or 'tikhonov'")
        if alpha <= 0:
            raise ValidationError("alpha must be greater than 0")

        self.kernel = kernel
        self.dim = dim
        self.polynomial_degree = polynomial_degree
        self.solver = solver
        self.alpha = alpha
        self.n_obj: int | None = None
        self._models: list[_RBFModel] | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """
        Fit the surrogate model.

        Parameters
        ----------
        train_x : np.ndarray
            Training input data. shape: (n_samples, n_features)
        train_y : np.ndarray
            Training output data. shape: (n_samples, n_obj) or (n_samples,).
            1-D input is treated as single-objective: shape (n_samples, 1).
        """
        arr = np.asarray(train_y, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)  # (n_samples,) -> (n_samples, 1)
        n_obj = arr.shape[1]

        # (Re-)initialize models when n_obj changes or on first fit
        if self._models is None or n_obj != self.n_obj:
            self.n_obj = n_obj
            self._models = [
                _RBFModel(
                    self.kernel,
                    self.dim,
                    self.polynomial_degree,
                    self.solver,
                    self.alpha,
                )
                for _ in range(n_obj)
            ]

        for i, model in enumerate(self._models):
            model.fit(train_x, arr[:, i])

    def predict(self, test_x: np.ndarray) -> SurrogatePrediction:
        """
        Predict using the surrogate model.

        Parameters
        ----------
        test_x : np.ndarray
            Input data. shape: (n_samples, n_features) or (n_features,)

        Returns
        -------
        SurrogatePrediction
            prediction.value shape: (n_samples, n_obj)
            prediction.std  is None (RBF interpolation provides no uncertainty)
            prediction.x holds the (n_samples, n_features) query points
            passed to this call (candidates when called from a prediction
            stage, holdout/archive points when called for
            accuracy evaluation), needed by acquisition functions that have
            no other channel to the points being scored (e.g. CORSDistance).
        """
        assert self._models is not None
        test = np.atleast_2d(np.asarray(test_x, dtype=float))
        preds = [m.predict(test) for m in self._models]
        value = np.column_stack(preds)  # (n_samples, n_obj)
        return SurrogatePrediction.objective(value=value, x=test)
