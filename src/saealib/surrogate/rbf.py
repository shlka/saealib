"""
RBF surrogate model module.

This module defines the Radial Basis Function (RBF) surrogate model.
RBFSurrogate supports multi-objective problems by maintaining one
independent RBF model per objective (ensemble approach).
"""

import logging

import numpy as np

from saealib.exceptions import ValidationError
from saealib.registry import register
from saealib.surrogate.base import RegressionSurrogate
from saealib.surrogate.prediction import SurrogatePrediction
from saealib.surrogate.rbf_kernels import RBFKernel

logger = logging.getLogger(__name__)


def _resolve_polynomial_degree(kernel: RBFKernel, polynomial_degree: int | None) -> int:
    if polynomial_degree is None:
        return (
            -1 if kernel.min_polynomial_degree is None else kernel.min_polynomial_degree
        )
    if polynomial_degree not in (-1, 0, 1):
        raise ValidationError("polynomial_degree must be -1, 0, or 1")
    if (
        kernel.min_polynomial_degree is not None
        and polynomial_degree < kernel.min_polynomial_degree
    ):
        raise ValidationError(
            f"{type(kernel).__name__} requires polynomial_degree >= "
            f"{kernel.min_polynomial_degree}, got {polynomial_degree}"
        )
    return polynomial_degree


def _validate_solver(value: str) -> None:
    if value not in ("solve", "lstsq", "tikhonov"):
        raise ValidationError("solver must be 'solve', 'lstsq', or 'tikhonov'")


def _validate_alpha(value: float) -> None:
    if value <= 0:
        raise ValidationError("alpha must be greater than 0")


class _RBFModel:
    """
    Single-objective RBF interpolation model (internal use only).

    Holds all state for one objective's RBF fit. Used as a building
    block by RBFSurrogate to support multi-objective problems.
    """

    def __init__(
        self,
        kernel: RBFKernel,
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
        phi = self.kernel.pairwise(self.train_x, self.train_x)

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
                rank = np.linalg.matrix_rank(polynomial_basis)
                if rank < polynomial_basis.shape[1]:
                    raise ValidationError(
                        "polynomial_degree=1 requires the augmented basis [1, X] to "
                        "have full column rank; got too few training points or "
                        "affinely degenerate training data"
                    )
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
            # Deliberate recoverable surrogate failure: expose NaN predictions
            # so downstream acquisition/feedback code can handle this candidate.
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
        k = self.kernel.pairwise(self.train_x, test)
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
    objective. The number of objectives is inferred from ``train_y`` on
    the first call to ``fit``.

    Attributes
    ----------
    kernel : RBFKernel
        Kernel object (e.g. ``GaussianKernel()``). Reassigning this
        property invalidates any fitted state; call ``fit()`` again before
        the next ``predict()``.
    dim : int
        Dimensionality of the input data.
    n_obj : int or None
        Number of objectives. Set on first fit call.
    polynomial_degree : int
        Resolved degree of the optional polynomial term (``-1`` disables
        it). When the constructor argument is ``None``, this is resolved
        from ``kernel.min_polynomial_degree``.
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
        kernel: RBFKernel,
        dim: int,
        polynomial_degree: int | None = None,
        solver: str = "solve",
        alpha: float = 1e-8,
    ) -> None:
        _validate_solver(solver)
        _validate_alpha(alpha)
        self._kernel = kernel
        self._polynomial_degree_arg = polynomial_degree
        self._polynomial_degree = _resolve_polynomial_degree(kernel, polynomial_degree)
        self.dim = dim
        self._solver = solver
        self._alpha = alpha
        self.n_obj: int | None = None
        self._models: list[_RBFModel] | None = None

    def _invalidate_fit(self) -> None:
        self._models = None
        self.n_obj = None

    @property
    def kernel(self) -> RBFKernel:
        """Return the configured RBF kernel."""
        return self._kernel

    @kernel.setter
    def kernel(self, value: RBFKernel) -> None:
        polynomial_degree = _resolve_polynomial_degree(
            value, self._polynomial_degree_arg
        )
        self._kernel = value
        self._polynomial_degree = polynomial_degree
        self._invalidate_fit()

    @property
    def polynomial_degree(self) -> int:
        """Return the resolved polynomial degree."""
        return self._polynomial_degree

    @polynomial_degree.setter
    def polynomial_degree(self, value: int | None) -> None:
        polynomial_degree = _resolve_polynomial_degree(self._kernel, value)
        self._polynomial_degree_arg = value
        self._polynomial_degree = polynomial_degree
        self._invalidate_fit()

    @property
    def solver(self) -> str:
        """Return the configured linear system solver."""
        return self._solver

    @solver.setter
    def solver(self, value: str) -> None:
        _validate_solver(value)
        self._solver = value
        self._invalidate_fit()

    @property
    def alpha(self) -> float:
        """Return the configured Tikhonov regularization strength."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        _validate_alpha(value)
        self._alpha = value
        self._invalidate_fit()

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
        try:
            train_x_arr = np.asarray(train_x, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(
                "train_x must be convertible to a finite 2-D array"
            ) from exc
        if train_x_arr.ndim != 2:
            raise ValidationError("train_x must be a 2-D array")
        if train_x_arr.shape[0] == 0:
            raise ValidationError("train_x must contain at least one sample")
        if not np.all(np.isfinite(train_x_arr)):
            raise ValidationError("train_x must contain only finite values")

        try:
            arr = np.asarray(train_y, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(
                "train_y must be convertible to a finite 1-D or 2-D array"
            ) from exc
        if arr.ndim not in (1, 2):
            raise ValidationError("train_y must be a 1-D or 2-D array")
        if not np.all(np.isfinite(arr)):
            raise ValidationError("train_y must contain only finite values")
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)  # (n_samples,) -> (n_samples, 1)
        if arr.shape[0] != train_x_arr.shape[0]:
            raise ValidationError(
                "train_x and train_y must contain the same number of samples"
            )
        n_obj = arr.shape[1]

        try:
            resolved_kernel = self._kernel.resolve(train_x_arr)
            new_models = [
                _RBFModel(
                    resolved_kernel,
                    self.dim,
                    self.polynomial_degree,
                    self.solver,
                    self.alpha,
                )
                for _ in range(n_obj)
            ]
            for i, model in enumerate(new_models):
                model.fit(train_x_arr, arr[:, i])
        except Exception:
            self._invalidate_fit()
            raise

        self._models = new_models
        self.n_obj = n_obj

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

        Raises
        ------
        RuntimeError
            If the surrogate has not been fitted, or ``kernel`` was
            reassigned since the last ``fit()`` call.
        """
        if self._models is None:
            raise RuntimeError(
                "RBFSurrogate must be fitted before predict() can be called, "
                "or re-fitted after changing kernel configuration."
            )
        try:
            test = np.asarray(test_x, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(
                "test_x must be convertible to a finite 1-D or 2-D array"
            ) from exc
        if test.ndim not in (1, 2):
            raise ValidationError("test_x must be a 1-D or 2-D array")
        test = np.atleast_2d(test)
        if not np.all(np.isfinite(test)):
            raise ValidationError("test_x must contain only finite values")
        if test.shape[1] != self.dim:
            raise ValidationError(
                f"test_x must have {self.dim} features, got {test.shape[1]}"
            )
        preds = [m.predict(test) for m in self._models]
        value = np.column_stack(preds)  # (n_samples, n_obj)
        return SurrogatePrediction.objective(value=value, x=test)
