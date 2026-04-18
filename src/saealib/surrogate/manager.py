"""
Surrogate manager module.

This module defines SurrogateManager and its implementations, which
orchestrate the fit/predict/score pipeline for surrogate-assisted selection.

Responsibility split:
  Surrogate        -- fits a model and predicts SurrogatePrediction
  AcquisitionFunction -- converts SurrogatePrediction to scalar scores
  SurrogateManager -- coordinates the two above; exposes score_candidates()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from saealib.acquisition.base import AcquisitionFunction
from saealib.callback import PostSurrogateFitEvent
from saealib.surrogate.base import Surrogate
from saealib.surrogate.prediction import SurrogatePrediction

if TYPE_CHECKING:
    from saealib.context import OptimizationContext
    from saealib.optimizer import ComponentProvider
    from saealib.population import Archive


class SurrogateManager(ABC):
    """
    Abstract base class for surrogate managers.

    A SurrogateManager coordinates the fit/predict/score pipeline and
    returns both scalar acquisition scores and the underlying predictions
    so that callers (e.g. IndividualBasedStrategy) can assign predicted
    objective values to offspring.
    """

    @abstractmethod
    def score_candidates(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
        reference: Any,
        provider: ComponentProvider | None = None,
        ctx: OptimizationContext | None = None,
    ) -> tuple[np.ndarray, list[SurrogatePrediction]]:
        """
        Score candidate solutions using the surrogate model.

        Parameters
        ----------
        candidates_x : np.ndarray
            Candidate design variable matrix. shape: (n_candidates, dim)
        archive : Archive
            Archive of evaluated solutions used for surrogate training.
        reference : Any
            Reference information for the acquisition function.
            Typically the current best objective value(s) for single-objective,
            or a Pareto front / reference point for multi-objective.
        provider : ComponentProvider or None, optional
            Component provider used to dispatch ``PostSurrogateFitEvent``
            after each surrogate fit. If ``None``, no event is dispatched.
        ctx : OptimizationContext or None, optional
            Current optimization context. Required when ``provider`` is given.

        Returns
        -------
        scores : np.ndarray
            Acquisition scores. shape: (n_candidates,). Higher is better.
        predictions : list[SurrogatePrediction]
            Surrogate predictions for each candidate.
            predictions[i].mean shape: (1, n_obj)
        """
        ...


class GlobalSurrogateManager(SurrogateManager):
    """
    Surrogate manager that fits once on the full archive.

    Fits the surrogate on all archived solutions, then predicts and scores
    all candidates in a single batch. Suitable when global approximation
    quality is sufficient.

    Parameters
    ----------
    surrogate : Surrogate
        Surrogate model instance.
    acquisition : AcquisitionFunction
        Acquisition function used to score predictions.
    """

    def __init__(self, surrogate: Surrogate, acquisition: AcquisitionFunction):
        self.surrogate = surrogate
        self.acquisition = acquisition

    def score_candidates(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
        reference: Any,
        provider: ComponentProvider | None = None,
        ctx: OptimizationContext | None = None,
    ) -> tuple[np.ndarray, list[SurrogatePrediction]]:
        """Fit on full archive, predict and score all candidates at once."""
        train_x = archive.x
        train_f = archive.f  # (n_archive, n_obj)
        self.surrogate.fit(train_x, train_f)
        if provider is not None and ctx is not None:
            provider.dispatch(
                PostSurrogateFitEvent(
                    ctx=ctx,
                    provider=provider,
                    surrogate=self.surrogate,
                    train_x=train_x,
                    train_f=train_f,
                )
            )

        prediction = self.surrogate.predict(candidates_x)  # mean: (n_candidates, n_obj)
        scores = self.acquisition.score(prediction, reference)

        # Split batch prediction into per-candidate SurrogatePrediction objects
        predictions = _split_prediction(prediction)
        return scores, predictions


class LocalSurrogateManager(SurrogateManager):
    """
    Surrogate manager that fits a local model per candidate (pre-selection).

    For each candidate, retrieves the k nearest neighbors from the archive,
    fits the surrogate on that local neighborhood, and predicts the
    candidate's objective value. This corresponds to the individual-based
    local modeling strategy (Guo et al., 2018).

    NOTE: The same surrogate instance is reused across candidates (re-fit
    each iteration). This is not thread-safe. For parallel use, provide a
    surrogate factory instead (future work).

    Parameters
    ----------
    surrogate : Surrogate
        Surrogate model instance (re-fit per candidate).
    acquisition : AcquisitionFunction
        Acquisition function used to score predictions.
    n_neighbors : int
        Number of nearest neighbors for local model training.
    """

    def __init__(
        self,
        surrogate: Surrogate,
        acquisition: AcquisitionFunction,
        n_neighbors: int = 50,
    ):
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.n_neighbors = n_neighbors

    def score_candidates(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
        reference: Any,
        provider: ComponentProvider | None = None,
        ctx: OptimizationContext | None = None,
    ) -> tuple[np.ndarray, list[SurrogatePrediction]]:
        """Fit a local model per candidate and score each individually."""
        predictions: list[SurrogatePrediction] = []
        _dispatch = provider is not None and ctx is not None

        for x in candidates_x:
            train_idx, _ = archive.get_knn(x, self.n_neighbors)
            train_x = archive.x[train_idx]
            train_f = archive.f[train_idx]  # (n_neighbors, n_obj)
            self.surrogate.fit(train_x, train_f)
            if _dispatch:
                provider.dispatch(
                    PostSurrogateFitEvent(
                        ctx=ctx,
                        provider=provider,
                        surrogate=self.surrogate,
                        train_x=train_x,
                        train_f=train_f,
                    )
                )
            pred = self.surrogate.predict(x)  # mean: (1, n_obj)
            predictions.append(pred)

        scores = np.array(
            [self.acquisition.score(p, reference)[0] for p in predictions]
        )
        return scores, predictions


class EnsembleSurrogateManager(SurrogateManager):
    """
    Surrogate manager that aggregates scores from multiple sub-managers.

    Each sub-manager produces scores, which are rank-normalized to [0, 1]
    before being aggregated via a weighted average. Rank normalization
    ensures scores from managers with incompatible scales (e.g., EI vs.
    raw mean) are made comparable before combining.

    Parameters
    ----------
    managers : list[SurrogateManager]
        Sub-managers to aggregate.
    weights : np.ndarray or None
        Weights for the weighted average. shape: (len(managers),).
        If None, uniform weights are used.
    """

    def __init__(
        self,
        managers: list[SurrogateManager],
        weights: np.ndarray | None = None,
    ):
        if not managers:
            raise ValueError("EnsembleSurrogateManager requires at least one manager.")
        self.managers = managers
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            self.weights = w / w.sum()
        else:
            self.weights = np.full(len(managers), 1.0 / len(managers))

    def score_candidates(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
        reference: Any,
        provider: ComponentProvider | None = None,
        ctx: OptimizationContext | None = None,
    ) -> tuple[np.ndarray, list[SurrogatePrediction]]:
        """
        Aggregate rank-normalized scores from all sub-managers.

        Returns the predictions from the first sub-manager as representative.
        """
        all_scores: list[np.ndarray] = []
        first_predictions: list[SurrogatePrediction] | None = None

        for i, manager in enumerate(self.managers):
            scores, preds = manager.score_candidates(
                candidates_x, archive, reference, provider, ctx
            )
            all_scores.append(_rank_normalize(scores))
            if i == 0:
                first_predictions = preds

        combined = np.stack(all_scores, axis=0)  # (n_managers, n_candidates)
        aggregated = (self.weights[:, None] * combined).sum(axis=0)  # (n_candidates,)
        return aggregated, first_predictions  # type: ignore[return-value]


def _split_prediction(prediction: SurrogatePrediction) -> list[SurrogatePrediction]:
    """Split a batch SurrogatePrediction into per-sample SurrogatePrediction objects."""
    n = prediction.mean.shape[0]
    result = []
    for i in range(n):
        std_i = prediction.std[i : i + 1] if prediction.std is not None else None
        label_i = prediction.label[i : i + 1] if prediction.label is not None else None
        result.append(
            SurrogatePrediction(
                mean=prediction.mean[i : i + 1],
                std=std_i,
                label=label_i,
                metadata=prediction.metadata,
            )
        )
    return result


def _rank_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Normalize scores to [0, 1] via rank transform.

    Rank 0 (lowest score) -> 0.0, rank n-1 (highest score) -> 1.0.
    Ties receive the same normalized rank.
    """
    n = len(scores)
    if n == 1:
        return np.ones(1)
    order = np.argsort(scores)
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    return ranks / (n - 1)
