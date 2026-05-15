"""
Archive-based surrogate managers: score candidates directly from the archive
without training a surrogate model.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import cdist

from saealib.surrogate.manager import SurrogateManager
from saealib.surrogate.prediction import SurrogatePrediction

if TYPE_CHECKING:
    from saealib.context import OptimizationContext
    from saealib.optimizer import ComponentProvider
    from saealib.population import Archive


class ArchiveBasedManager(SurrogateManager):
    """
    SurrogateManager that scores candidates directly from the archive
    without training a surrogate model.

    Subclasses implement compute_scores(). score_candidates() calls
    compute_scores() and wraps each result in a SurrogatePrediction with
    tell_f=NaN so that pbest is not corrupted when non-objective scores
    would otherwise be written to offspring.f.
    """

    @abstractmethod
    def compute_scores(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
        ctx: OptimizationContext | None = None,
    ) -> np.ndarray:
        """
        Compute and return scores (higher = more promising).

        Parameters
        ----------
        candidates_x : shape (n_candidates, dim)
        archive : archive of evaluated solutions
        ctx : optimization context (None if not needed)

        Returns
        -------
        scores : shape (n_candidates,)
        """
        ...

    def score_candidates(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
        provider: ComponentProvider | None = None,
        ctx: OptimizationContext | None = None,
    ) -> tuple[np.ndarray, list[SurrogatePrediction]]:
        scores = self.compute_scores(candidates_x, archive, ctx)
        n_obj = archive.f.shape[1] if len(archive) > 0 else 1
        nan_f = np.full((1, n_obj), np.nan)
        predictions = [
            SurrogatePrediction(value=nan_f.copy(), _tell_f=nan_f.copy())
            for _ in range(len(candidates_x))
        ]
        return scores, predictions


class NoveltyManager(ArchiveBasedManager):
    """
    Score = mean k-nearest-neighbor distance to archive. Larger = more novel.

    Parameters
    ----------
    k : int
        Number of nearest neighbors to average. Clamped to archive size.
    """

    def __init__(self, k: int = 1):
        self.k = k

    def compute_scores(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
        ctx: OptimizationContext | None = None,
    ) -> np.ndarray:
        if len(archive) == 0:
            return np.ones(len(candidates_x))
        dists = cdist(candidates_x, archive.x)
        k = min(self.k, dists.shape[1])
        return np.sort(dists, axis=1)[:, :k].mean(axis=1)


class DensityManager(ArchiveBasedManager):
    """
    Score = inverse ε-NN density. Prefer candidates in sparse regions.

    Parameters
    ----------
    eps : float
        Radius for counting archive neighbors.
    """

    def __init__(self, eps: float = 1.0):
        self.eps = eps

    def compute_scores(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
        ctx: OptimizationContext | None = None,
    ) -> np.ndarray:
        if len(archive) == 0:
            return np.ones(len(candidates_x))
        dists = cdist(candidates_x, archive.x)
        counts = (dists < self.eps).sum(axis=1)
        density = counts / len(archive)
        return 1.0 / (density + 1e-9)


class NichingManager(ArchiveBasedManager):
    """
    Score = min distance to other candidates + min distance to archive.
    Promotes diversity among candidates.

    Note: scores all candidates jointly. At small population sizes, score
    variance among candidates will be low — this is expected behavior.
    """

    def compute_scores(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
        ctx: OptimizationContext | None = None,
    ) -> np.ndarray:
        n = len(candidates_x)
        if n == 1:
            return np.ones(1)
        intra = cdist(candidates_x, candidates_x)
        np.fill_diagonal(intra, np.inf)
        archive_min = (
            cdist(candidates_x, archive.x).min(axis=1)
            if len(archive) > 0
            else np.ones(n)
        )
        return intra.min(axis=1) + archive_min
