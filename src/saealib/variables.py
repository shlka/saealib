"""Variable type definitions for mixed-variable optimization."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from saealib.exceptions import ValidationError

__all__ = ["CategoricalVariable", "ContinuousVariable", "IntegerVariable", "Variable"]


class Variable(ABC):
    """Abstract base for a single design variable.

    Subclasses define the domain and how to project values back onto it.
    All bounds are expressed in the encoded float representation used by
    :class:`~saealib.Population` arrays.
    """

    @property
    @abstractmethod
    def lb(self) -> float:
        """Lower bound in encoded (float) space."""

    @property
    @abstractmethod
    def ub(self) -> float:
        """Upper bound in encoded (float) space."""

    @abstractmethod
    def repair(self, x: npt.ArrayLike) -> np.ndarray:
        """Project *x* onto the valid domain.

        Parameters
        ----------
        x : np.ndarray
            Values for this dimension.  Shape ``()`` (scalar) or ``(n,)`` (batch).

        Returns
        -------
        np.ndarray
            Repaired values, same shape as *x*.
        """


class ContinuousVariable(Variable):
    """Continuous real-valued variable on ``[lb, ub]``.

    Parameters
    ----------
    lb, ub : float
        Lower and upper bounds (inclusive).
    """

    def __init__(self, lb: float, ub: float) -> None:
        if lb > ub:
            raise ValidationError(f"lb must be <= ub, got lb={lb}, ub={ub}")
        self._lb = float(lb)
        self._ub = float(ub)

    @property
    def lb(self) -> float:
        """Lower bound."""
        return self._lb

    @property
    def ub(self) -> float:
        """Upper bound."""
        return self._ub

    def repair(self, x: npt.ArrayLike) -> np.ndarray:
        """Clip *x* to ``[lb, ub]``.

        Parameters
        ----------
        x : np.ndarray
            Values for this dimension.

        Returns
        -------
        np.ndarray
            Clipped values.
        """
        return np.clip(x, self._lb, self._ub)


class IntegerVariable(Variable):
    """Integer-valued variable on ``[lb, ub]`` (both inclusive).

    Stored as ``float64`` in population arrays; rounded by :meth:`repair`.

    Parameters
    ----------
    lb, ub : int
        Lower and upper bounds (inclusive).
    """

    def __init__(self, lb: int, ub: int) -> None:
        if int(lb) > int(ub):
            raise ValidationError(f"lb must be <= ub, got lb={lb}, ub={ub}")
        self._lb = int(lb)
        self._ub = int(ub)

    @property
    def lb(self) -> float:
        """Lower bound as float."""
        return float(self._lb)

    @property
    def ub(self) -> float:
        """Upper bound as float."""
        return float(self._ub)

    def repair(self, x: npt.ArrayLike) -> np.ndarray:
        """Round *x* to the nearest integer and clip to ``[lb, ub]``.

        Parameters
        ----------
        x : np.ndarray
            Values for this dimension.

        Returns
        -------
        np.ndarray
            Rounded and clipped values.
        """
        return np.clip(np.round(x), self._lb, self._ub)


class CategoricalVariable(Variable):
    """Categorical variable; stored internally as integer index.

    Values are encoded as indices in ``[0, n_categories - 1]`` and stored as
    ``float64`` in population arrays.  Use :meth:`encode` / :meth:`decode` to
    convert between category values and their integer indices.

    Parameters
    ----------
    categories : list
        Ordered list of category values.  Must be non-empty.

    Examples
    --------
    >>> v = CategoricalVariable(["red", "green", "blue"])
    >>> v.encode("green")
    1
    >>> v.decode(1)
    'green'
    """

    def __init__(self, categories: list) -> None:
        if len(categories) == 0:
            raise ValidationError("categories must not be empty")
        self._categories = list(categories)
        self._n = len(self._categories)

    @property
    def lb(self) -> float:
        """Lower bound (always ``0.0``)."""
        return 0.0

    @property
    def ub(self) -> float:
        """Upper bound (``n_categories - 1``)."""
        return float(self._n - 1)

    @property
    def n_categories(self) -> int:
        """Number of categories."""
        return self._n

    @property
    def categories(self) -> list:
        """Ordered list of category values (copy)."""
        return list(self._categories)

    def repair(self, x: npt.ArrayLike) -> np.ndarray:
        """Round *x* and clip to a valid category index.

        Parameters
        ----------
        x : np.ndarray
            Values for this dimension.

        Returns
        -------
        np.ndarray
            Rounded and clipped values in ``[0, n_categories - 1]``.
        """
        return np.clip(np.round(x), 0, self._n - 1)

    def encode(self, category) -> int:
        """Return the integer index for *category*.

        Parameters
        ----------
        category :
            A value present in :attr:`categories`.

        Returns
        -------
        int
        """
        try:
            return self._categories.index(category)
        except ValueError:
            raise ValidationError(
                f"{category!r} is not in categories {self._categories}"
            )

    def decode(self, index: int | float):
        """Return the category value for *index*.

        Parameters
        ----------
        index : int or float
            Index into :attr:`categories`; rounded to nearest integer.

        Returns
        -------
        category value
        """
        idx = round(float(index))
        if not (0 <= idx < self._n):
            raise ValidationError(f"index {idx} out of range [0, {self._n - 1}]")
        return self._categories[idx]
