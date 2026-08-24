"""RBF kernel classes.

Defines the ``RBFKernel`` extension point and the built-in kernel
implementations used by :class:`~saealib.surrogate.rbf.RBFSurrogate`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np
import scipy.spatial
from typing_extensions import Self

from saealib.exceptions import ValidationError
from saealib.registry import register


def _validate_length_scale(length_scale: float | None) -> None:
    if length_scale is None:
        return
    if not np.isfinite(length_scale) or length_scale <= 0:
        raise ValidationError("length_scale must be finite and greater than 0")


def _resolve_length_scale(train_x: np.ndarray) -> float:
    distances = scipy.spatial.distance.pdist(np.asarray(train_x))
    scale = float(np.median(distances)) if len(distances) else float("nan")
    if not np.isfinite(scale) or scale <= 0:
        raise ValidationError(
            "Unable to resolve a positive finite length_scale from training "
            "data. Provide an explicit length_scale, or check for "
            "duplicate/degenerate training points."
        )
    return scale


class RBFKernel(ABC):
    """Abstract base for RBF kernel implementations.

    A subclass implements the radial profile ``phi(r)``; kernel-specific
    parameters (e.g. ``length_scale``, ``nu``) are held as object state
    rather than passed through a shared call signature.
    """

    min_polynomial_degree: ClassVar[int | None] = None
    auto_polynomial_degree: ClassVar[int | None] = None

    def __post_init__(self) -> None:
        self.validate()

    @abstractmethod
    def evaluate(self, r: np.ndarray) -> np.ndarray:
        """Evaluate ``phi(r)`` for a radial-distance matrix."""
        ...

    def validate(self) -> None:
        """Validate static kernel configuration."""
        return None

    def resolve(self, train_x: np.ndarray) -> Self:
        """Return a kernel whose data-dependent parameters are resolved.

        Calls :meth:`validate` before returning, so a custom subclass that
        overrides ``validate()`` but not ``resolve()`` still gets its
        configuration checked at resolve time.
        """
        self.validate()
        return self

    def pairwise(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Evaluate the kernel between every row of ``x1`` and ``x2``."""
        r = scipy.spatial.distance.cdist(x1, x2, "euclidean")
        return self.evaluate(r)


@register()
@dataclass(frozen=True, kw_only=True)
class GaussianKernel(RBFKernel):
    """Gaussian radial basis function kernel.

    Uses the ``exp(-0.5 * (r / length_scale)^2)`` parameterization;
    equivalent to the papers' ``exp(-gamma * r^2)`` form with
    ``gamma = 1 / (2 * length_scale^2)``. Based on
    :cite:`gutmann2001rbf,regis2005cors`.

    Parameters
    ----------
    length_scale : float or None
        Kernel width parameter, in the same distance units as the input
        data. ``None`` resolves to the median pairwise training-point
        distance at fit time.
    """

    length_scale: float | None = None
    min_polynomial_degree: ClassVar[int | None] = None
    auto_polynomial_degree: ClassVar[int | None] = 0

    def validate(self) -> None:
        """Validate ``length_scale``."""
        _validate_length_scale(self.length_scale)

    def resolve(self, train_x: np.ndarray) -> Self:
        """Resolve ``length_scale`` from training data if unset."""
        if self.length_scale is not None:
            return self
        return replace(self, length_scale=_resolve_length_scale(train_x))

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        """Evaluate the Gaussian kernel ``exp(-0.5 * (r / length_scale)**2)``."""
        if self.length_scale is None:
            raise ValidationError(
                "GaussianKernel.length_scale is unresolved (None); call "
                "resolve(train_x) first, or construct with an explicit length_scale."
            )
        q = r / self.length_scale
        return np.exp(-0.5 * q**2)


@register()
@dataclass(frozen=True, kw_only=True)
class ThinPlateSplineKernel(RBFKernel):
    """Thin-plate spline radial basis function kernel.

    This conditionally positive definite kernel has order 1 and requires a
    linear polynomial term for a well-posed system.
    """

    min_polynomial_degree: ClassVar[int | None] = 1
    auto_polynomial_degree: ClassVar[int | None] = 1

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        """Evaluate the thin-plate spline kernel for a radial-distance matrix."""
        with np.errstate(divide="ignore", invalid="ignore"):
            out = r**2 * np.log(r)
        return np.nan_to_num(out, nan=0.0)


@register()
@dataclass(frozen=True, kw_only=True)
class LinearKernel(RBFKernel):
    """Linear radial basis function kernel.

    This conditionally positive definite kernel has order 0 and requires a
    constant polynomial term for a well-posed system. Based on
    :cite:`gutmann2001rbf`.
    """

    min_polynomial_degree: ClassVar[int | None] = 0
    auto_polynomial_degree: ClassVar[int | None] = 0

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        """Evaluate the linear kernel for a radial-distance matrix."""
        return r


@register()
@dataclass(frozen=True, kw_only=True)
class CubicKernel(RBFKernel):
    """Cubic radial basis function kernel.

    This conditionally positive definite kernel has order 1 and requires a
    linear polynomial term for a well-posed system. Based on
    :cite:`gutmann2001rbf`.
    """

    min_polynomial_degree: ClassVar[int | None] = 1
    auto_polynomial_degree: ClassVar[int | None] = 1

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        """Evaluate the cubic kernel for a radial-distance matrix."""
        return r**3


@register()
@dataclass(frozen=True, kw_only=True)
class MultiquadricKernel(RBFKernel):
    """Multiquadric radial basis function kernel.

    This conditionally positive definite kernel has order 0 and requires a
    constant polynomial term for a well-posed system. Based on
    :cite:`gutmann2001rbf`.

    Parameters
    ----------
    length_scale : float or None
        Shape parameter (denoted ``gamma`` in the paper), in the same
        distance units as the input data. ``None`` resolves to the median
        pairwise training-point distance at fit time.
    """

    length_scale: float | None = None
    min_polynomial_degree: ClassVar[int | None] = 0
    auto_polynomial_degree: ClassVar[int | None] = 0

    def validate(self) -> None:
        """Validate ``length_scale``."""
        _validate_length_scale(self.length_scale)

    def resolve(self, train_x: np.ndarray) -> Self:
        """Resolve ``length_scale`` from training data if unset."""
        if self.length_scale is not None:
            return self
        return replace(self, length_scale=_resolve_length_scale(train_x))

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        """Evaluate the multiquadric kernel ``sqrt(r**2 + length_scale**2)``."""
        if self.length_scale is None:
            raise ValidationError(
                "MultiquadricKernel.length_scale is unresolved (None); call "
                "resolve(train_x) first, or construct with an explicit length_scale."
            )
        return np.sqrt(r**2 + self.length_scale**2)


@register()
@dataclass(frozen=True, kw_only=True)
class MaternKernel(RBFKernel):
    """Matérn radial basis function kernel.

    Strictly positive definite; does not require a polynomial term. Based
    on :cite:`rasmussen2006gpml`.

    Parameters
    ----------
    length_scale : float or None
        Length-scale parameter (denoted ``ell`` in the paper). ``None``
        resolves to the median pairwise training-point distance at fit
        time.
    nu : float
        Smoothness parameter. Must be 0.5, 1.5, or 2.5.
    """

    length_scale: float | None = None
    nu: float = 1.5
    min_polynomial_degree: ClassVar[int | None] = None
    auto_polynomial_degree: ClassVar[int | None] = 0

    def validate(self) -> None:
        """Validate ``length_scale`` and ``nu``."""
        _validate_length_scale(self.length_scale)
        if self.nu not in (0.5, 1.5, 2.5):
            raise ValidationError("nu must be 0.5, 1.5, or 2.5")

    def resolve(self, train_x: np.ndarray) -> Self:
        """Resolve ``length_scale`` from training data if unset."""
        if self.length_scale is not None:
            return self
        return replace(self, length_scale=_resolve_length_scale(train_x))

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        """Evaluate the Matérn kernel's nu-dependent radial profile."""
        if self.length_scale is None:
            raise ValidationError(
                "MaternKernel.length_scale is unresolved (None); call "
                "resolve(train_x) first, or construct with an explicit length_scale."
            )
        length_scale = self.length_scale
        if self.nu == 0.5:
            return np.exp(-r / length_scale)
        if self.nu == 1.5:
            scaled = np.sqrt(3) * r / length_scale
            return (1 + scaled) * np.exp(-scaled)
        scaled = np.sqrt(5) * r / length_scale
        return (1 + scaled + scaled**2 / 3) * np.exp(-scaled)
