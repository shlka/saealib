"""Duplicate elimination operator for offspring filtering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DuplicateElimination:
    """Filter offspring that duplicate any member of the current population.

    Parameters
    ----------
    atol : float
        Absolute tolerance passed to ``numpy.isclose``.
        Matches the ``atol`` semantics of
        :class:`~saealib.population.archive.ArchiveMixin`.
    rtol : float
        Relative tolerance passed to ``numpy.isclose``.
        Matches the ``rtol`` semantics of
        :class:`~saealib.population.archive.ArchiveMixin`.
    max_retries : int
        Maximum number of regeneration attempts per
        :meth:`~saealib.algorithms.ga.GA.ask` call.
    """

    atol: float = 1e-16
    rtol: float = 0.0
    max_retries: int = 100

    def find_duplicates(self, off_x: np.ndarray, pop_x: np.ndarray) -> np.ndarray:
        """Return a boolean mask of offspring that duplicate the population.

        Parameters
        ----------
        off_x : numpy.ndarray of shape (N, dim)
            Offspring decision vectors.
        pop_x : numpy.ndarray of shape (M, dim)
            Current population decision vectors.

        Returns
        -------
        numpy.ndarray of shape (N,), dtype bool
            ``True`` where ``off_x[i]`` is element-wise close to any row of
            ``pop_x``, using the same ``atol`` / ``rtol`` as
            :meth:`~saealib.population.archive.ArchiveMixin._find_idx`.
        """
        close = np.all(
            np.isclose(pop_x[None], off_x[:, None], atol=self.atol, rtol=self.rtol),
            axis=2,
        )
        return np.any(close, axis=1)
