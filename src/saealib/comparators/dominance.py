from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Dominator(ABC):
    """
    Abstract base for dominance predicates.

    A ``Dominator`` encapsulates the definition of dominance between
    objective vectors independently of the sorting algorithm.  Two concrete
    operations are required and MUST agree with each other:

    - ``dominance_matrix`` — batched NxN boolean matrix (primary operation).
    - ``dominates`` — scalar pairwise predicate, derived from the matrix to
      guarantee consistency.

    Parameters are the same for both methods:

    Parameters
    ----------
    f : np.ndarray
        Objective matrix. shape: (n, n_obj)  [``dominance_matrix``]
    fa, fb : np.ndarray
        Objective vectors. shape: (n_obj,)  [``dominates``]
    direction : np.ndarray or None
        Per-objective optimization direction: +1 = maximize, -1 = minimize.
        ``None`` defaults to minimization for all objectives.
    """

    @abstractmethod
    def dominance_matrix(
        self,
        f: np.ndarray,
        direction: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute the NxN boolean dominance matrix.

        ``D[i, j]`` is True if row i dominates row j.

        Parameters
        ----------
        f : np.ndarray
            Objective matrix. shape: (n, n_obj).  Must contain no NaN values
            (callers are responsible for pre-filtering).
        direction : np.ndarray or None
            Per-objective direction. ``None`` → minimize all.

        Returns
        -------
        np.ndarray
            Boolean matrix of shape (n, n).
        """

    def dominates(
        self,
        fa: np.ndarray,
        fb: np.ndarray,
        direction: np.ndarray | None = None,
    ) -> bool:
        """
        Return True if fa dominates fb.

        Derived from ``dominance_matrix`` on a 2-row stack to guarantee
        agreement with the batched path.  NaN values in fa always return
        False (consistent with ``dominance_matrix`` assuming finite input).

        Parameters
        ----------
        fa, fb : np.ndarray
            Objective vectors. shape: (n_obj,)
        direction : np.ndarray or None
            Per-objective direction. ``None`` → minimize all.

        Returns
        -------
        bool
        """
        fa = np.asarray(fa, dtype=float)
        fb = np.asarray(fb, dtype=float)
        if np.any(np.isnan(fa)):
            return False
        # Replace NaN in fb with +inf so it appears infinitely bad (dominated).
        fb_safe = np.where(np.isnan(fb), np.inf, fb)
        stacked = np.stack([fa, fb_safe])
        return bool(self.dominance_matrix(stacked, direction)[0, 1])


class ParetoDominator(Dominator):
    """
    Pareto dominance predicate.

    ``fa`` dominates ``fb`` iff ``fa`` is at most as large as ``fb`` in all
    objectives and strictly smaller in at least one (after applying the
    direction transform).

    This is the default dominance relation used throughout the library.
    """

    def dominance_matrix(
        self,
        f: np.ndarray,
        direction: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute the NxN Pareto-dominance matrix.

        Applies the ``f * (-direction)`` transform so that smaller is always
        better, then accumulates per-objective ``leq_all`` / ``less_any``
        comparisons.  Peak memory is O(N²); no (N, N, M) tensor is created.

        Parameters
        ----------
        f : np.ndarray
            Objective matrix. shape: (n, n_obj).  Must contain no NaN values.
        direction : np.ndarray or None
            Per-objective direction. ``None`` → minimize all.

        Returns
        -------
        np.ndarray
            Boolean matrix of shape (n, n).  ``D[i, j]`` = True iff row i
            Pareto-dominates row j.
        """
        g = np.asarray(f, dtype=float)
        if direction is not None:
            g = g * (-direction)
        k, m = g.shape
        leq_all = np.ones((k, k), dtype=bool)
        less_any = np.zeros((k, k), dtype=bool)
        for obj in range(m):
            col = g[:, obj]
            leq_all &= col[:, None] <= col[None, :]
            less_any |= col[:, None] < col[None, :]
        return leq_all & less_any


class EpsilonDominator(Dominator):
    """
    ε-dominance predicate (Laumanns et al., 2002).

    Quantizes each objective into ε-boxes and applies ordinary Pareto
    dominance on the resulting integer box coordinates.  Two solutions that
    fall in the same ε-box are mutually non-dominating; a solution in a
    strictly better box (lower box index in every minimization objective,
    higher in every maximization objective) dominates the other.

    Two quantization modes are supported:

    - **additive** (default): box index = ``floor(f_i / eps_i)``.
      As ``eps → 0`` the boxes shrink to individual points and the relation
      recovers ordinary Pareto dominance.

    - **multiplicative**: box index = ``floor(log f_i / log(1 + eps_i))``.
      Requires strictly positive objective values (``f > 0``).

    Parameters
    ----------
    eps : float or np.ndarray
        Box width per objective.  A scalar applies the same width to every
        objective; an array of shape ``(n_obj,)`` sets per-objective widths.
        All values must be strictly positive (> 0).
    mode : {"additive", "multiplicative"}
        Quantization mode.  Default is ``"additive"``.

    Raises
    ------
    ValueError
        If any element of ``eps`` is not strictly positive (≤ 0).
    ValueError
        If ``mode`` is not one of ``"additive"`` or ``"multiplicative"``.

    Notes
    -----
    Additive box rule::

        b_i = floor(f_i / eps_i)

    Multiplicative box rule (requires f_i > 0 for all i)::

        b_i = floor(log(f_i) / log(1 + eps_i))

    Because ``log(1 + eps_i) = log1p(eps_i)`` is used internally, the
    multiplicative rule is numerically stable for small ``eps``.

    The dominance-matrix computation delegates entirely to an internal
    :class:`ParetoDominator` instance operating on the quantized box
    coordinates.  This guarantees that :meth:`dominates` (inherited from
    :class:`Dominator`) and :meth:`dominance_matrix` always agree.

    References
    ----------
    :cite:`laumanns2002epsilon`: Laumanns, M., Thiele, L., Deb, K., &
    Zitzler, E. (2002). Combining Convergence and Diversity in Evolutionary
    Multiobjective Optimization. *Evolutionary Computation*, 10(3), 263-282.
    """

    def __init__(
        self,
        eps: float | np.ndarray,
        mode: str = "additive",
    ) -> None:
        eps_arr = np.atleast_1d(np.asarray(eps, dtype=float))
        if np.any(eps_arr <= 0):
            raise ValueError(
                f"All eps values must be strictly positive (> 0); got {eps_arr}"
            )
        if mode not in ("additive", "multiplicative"):
            raise ValueError(
                f"mode must be 'additive' or 'multiplicative'; got {mode!r}"
            )
        self._eps = eps_arr
        self._mode = mode
        # Internal delegate: Pareto dominance on box coordinates.
        self._pareto = ParetoDominator()

    @property
    def eps(self) -> np.ndarray:
        """Box widths per objective."""
        return self._eps

    @property
    def mode(self) -> str:
        """Quantization mode ('additive' or 'multiplicative')."""
        return self._mode

    def _quantize(self, f: np.ndarray) -> np.ndarray:
        """
        Map objective matrix *f* to integer ε-box coordinates.

        Parameters
        ----------
        f : np.ndarray
            Objective matrix, shape (n, n_obj).

        Returns
        -------
        np.ndarray
            Integer box-index matrix, shape (n, n_obj), dtype float64
            (floor of real-valued box coordinates; kept as float so that
            ParetoDominator can operate on them without type issues).

        Raises
        ------
        ValueError
            In multiplicative mode, if any element of *f* is ≤ 0.
        """
        if self._mode == "additive":
            return np.floor(f / self._eps)
        # multiplicative mode
        if np.any(f <= 0):
            raise ValueError(
                "Multiplicative ε-dominance requires all objective values to be "
                "strictly positive (f > 0).  Found non-positive values in f."
            )
        return np.floor(np.log(f) / np.log1p(self._eps))

    def dominance_matrix(
        self,
        f: np.ndarray,
        direction: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute the NxN ε-dominance matrix.

        Quantizes *f* into ε-box coordinates and delegates to
        :class:`ParetoDominator` on those coordinates so that direction
        handling and the accumulation logic are not duplicated.

        ``D[i, j]`` is True if row i ε-dominates row j.

        Parameters
        ----------
        f : np.ndarray
            Objective matrix. shape: (n, n_obj).  Must contain no NaN values
            (callers are responsible for pre-filtering).
        direction : np.ndarray or None
            Per-objective direction. ``None`` → minimize all.

        Returns
        -------
        np.ndarray
            Boolean matrix of shape (n, n).

        Raises
        ------
        ValueError
            In multiplicative mode, if any element of *f* is ≤ 0.
        """
        b = self._quantize(np.asarray(f, dtype=float))
        return self._pareto.dominance_matrix(b, direction)


# Module-level singleton used by legacy wrappers (_pareto_dominates,
# _dominance_matrix) so callers that import them directly stay correct.
_PARETO_DOMINATOR: ParetoDominator = ParetoDominator()


def _pareto_dominates(
    fa: np.ndarray, fb: np.ndarray, direction: np.ndarray | None = None
) -> bool:
    """
    Return True if fa Pareto-dominates fb.

    NaN values in fa are treated as non-dominating (returns False).

    .. deprecated::
        Thin wrapper kept for backward compatibility.  Use
        ``ParetoDominator().dominates(fa, fb, direction)`` instead.

    Parameters
    ----------
    fa, fb : np.ndarray
        Objective vectors to compare.
    direction : np.ndarray or None
        Per-objective optimization direction: +1 = maximize, -1 = minimize.
        None defaults to minimization for all objectives.
    """
    return _PARETO_DOMINATOR.dominates(fa, fb, direction)


def _dominance_matrix(g: np.ndarray) -> np.ndarray:
    """
    Compute the NxN boolean dominance matrix from a direction-transformed matrix.

    ``D[i, j]`` is True if row i Pareto-dominates row j.  The input ``g``
    must already have the direction transform applied (smaller = better for
    every objective) and must contain no NaN values.

    .. deprecated::
        Thin wrapper kept for backward compatibility.  Use
        ``ParetoDominator().dominance_matrix(g)`` instead.

    Parameters
    ----------
    g : np.ndarray
        Objective matrix after direction transform. shape: (k, n_obj).
        Must contain no NaN values.

    Returns
    -------
    np.ndarray
        Boolean matrix of shape (k, k).
    """
    # g is already direction-transformed; pass direction=None so no second
    # transform is applied inside ParetoDominator.
    return _PARETO_DOMINATOR.dominance_matrix(g)
