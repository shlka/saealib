"""
Surrogate manager: coordinates the fit/predict/score pipeline.

Responsibility split:
  Surrogate           -- fits a model and predicts SurrogatePrediction
  AcquisitionFunction -- converts SurrogatePrediction to scalar scores
  SurrogateManager    -- coordinates the two above; exposes score_candidates()

Classes
-------
GlobalSurrogateManager
    Fits once on the full archive; batch predict and score.
LocalSurrogateManager
    Fits a local KNN model per candidate.
CompositeSurrogateManager
    Combines scores from multiple sub-managers via an injectable combine_fn.

Combine functions
-----------------
product_combine
    Element-wise product of score arrays (e.g. EI x PoF).
rank_weighted_combine
    Returns a combine function that rank-normalises then takes a weighted average.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from saealib.surrogate.base import Surrogate
from saealib.surrogate.prediction import SurrogatePrediction
from saealib.surrogate.training_set import (
    ArchiveObjectiveSet,
    KNNObjectiveSet,
    TrainingSet,
)

if TYPE_CHECKING:
    from saealib.acquisition.base import AcquisitionFunction
    from saealib.context import OptimizationContext
    from saealib.population import Archive


class SurrogateManager(ABC):
    """
    Abstract base class for surrogate managers.

    A SurrogateManager coordinates the fit/predict/score pipeline and
    returns both scalar acquisition scores and the underlying predictions
    so that callers (e.g. IndividualBasedStrategy) can assign predicted
    objective values to offspring.
    """

    @staticmethod
    def _sanitize_nan(
        scores: np.ndarray,
        predictions: list[SurrogatePrediction],
    ) -> tuple[np.ndarray, list[SurrogatePrediction]]:
        """Replace NaN scores with -inf and mark failed predictions explicitly.

        Any candidate whose acquisition score is NaN (surrogate fit failed,
        external library returned NaN, etc.) is given score=-inf so it is
        never selected by the strategy's pre-selection step.  The
        corresponding prediction's _tell_f is set to an explicit NaN array,
        following the same convention as ArchiveBasedManager, so strategies
        can detect and handle invalid predictions without comparing NaN
        objective values.
        """
        nan_mask = np.isnan(scores)
        if not nan_mask.any():
            return scores, predictions
        scores = scores.copy()
        scores[nan_mask] = -np.inf
        for i in np.where(nan_mask)[0]:
            p = predictions[i]
            nan_f = np.full_like(p.value, np.nan)
            predictions[i] = SurrogatePrediction(
                value=p.value,
                std=p.std,
                label=p.label,
                _tell_f=nan_f,
                metadata=p.metadata,
            )
        return scores, predictions

    @abstractmethod
    def score_candidates(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
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
        ctx : OptimizationContext or None, optional
            Current optimization context. Passed to ``TrainingSet.build``
            for strategies that require comparator or population access.

        Returns
        -------
        scores : np.ndarray
            Acquisition scores. shape: (n_candidates,). Higher is better.
        predictions : list[SurrogatePrediction]
            Surrogate predictions for each candidate.
            predictions[i].value shape: (1, n_obj)
        """
        ...

    def post_score(
        self,
        scores: np.ndarray,
        predictions: list[SurrogatePrediction],
        ctx: OptimizationContext | None = None,
    ) -> tuple[np.ndarray, list[SurrogatePrediction]]:
        """Post-score lifecycle hook; override to inject custom processing.

        Parameters
        ----------
        scores : np.ndarray
            Acquisition scores. shape: (n_candidates,)
        predictions : list[SurrogatePrediction]
            Surrogate predictions for each candidate.
        ctx : OptimizationContext or None, optional
            Current optimization context.

        Returns
        -------
        tuple[np.ndarray, list[SurrogatePrediction]]
            Processed scores and predictions.
        """
        return scores, predictions

    def with_post_score(
        self,
        fn: Callable[
            [np.ndarray, list[SurrogatePrediction], OptimizationContext | None],
            tuple[np.ndarray, list[SurrogatePrediction]],
        ],
    ) -> SurrogateManager:
        """Return a copy of this manager with ``fn`` appended to the hook.

        Parameters
        ----------
        fn : callable
            ``fn(scores, predictions, ctx) -> (scores, predictions)``

        Returns
        -------
        SurrogateManager
            Shallow copy with the hook registered.
        """
        new = copy.copy(self)
        prev = self.post_score
        new.post_score = lambda scores, predictions, ctx=None: fn(
            *prev(scores, predictions, ctx), ctx
        )
        return new


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
    training_set : TrainingSet or None
        Strategy object for building training data. Defaults to
        ``ArchiveObjectiveSet()``, which preserves the previous behaviour.
    """

    def __init__(
        self,
        surrogate: Surrogate,
        acquisition: AcquisitionFunction,
        training_set: TrainingSet | None = None,
    ):
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.training_set: TrainingSet = (
            training_set if training_set is not None else ArchiveObjectiveSet()
        )

    def score_candidates(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
        ctx: OptimizationContext | None = None,
    ) -> tuple[np.ndarray, list[SurrogatePrediction]]:
        """Fit on full archive, predict and score all candidates at once."""
        population = ctx.population if ctx is not None else None
        data = self.training_set.build(archive, population, ctx, candidate_x=None)
        self.surrogate.fit(data.train_x, data.train_y)
        self.surrogate.post_fit(data.train_x, data.train_y, ctx)

        reference = self.acquisition.compute_reference(archive)
        prediction = self.surrogate.predict(candidates_x)  # mean: (n_candidates, n_obj)
        scores = self.acquisition.score(prediction, reference)

        # Split batch prediction into per-candidate SurrogatePrediction objects
        predictions = _split_prediction(prediction)
        scores, predictions = self._sanitize_nan(scores, predictions)
        return self.post_score(scores, predictions, ctx)


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
    training_set : TrainingSet or None
        Strategy object for building training data per candidate. Defaults to
        ``KNNObjectiveSet(n_neighbors=50)``, which preserves the previous
        behaviour.
    """

    def __init__(
        self,
        surrogate: Surrogate,
        acquisition: AcquisitionFunction,
        training_set: TrainingSet | None = None,
    ):
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.training_set: TrainingSet = (
            training_set
            if training_set is not None
            else KNNObjectiveSet(n_neighbors=50)
        )

    def score_candidates(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
        ctx: OptimizationContext | None = None,
    ) -> tuple[np.ndarray, list[SurrogatePrediction]]:
        """Fit a local model per candidate and score each individually."""
        predictions: list[SurrogatePrediction] = []
        reference = self.acquisition.compute_reference(archive)
        population = ctx.population if ctx is not None else None

        for x in candidates_x:
            data = self.training_set.build(archive, population, ctx, candidate_x=x)
            self.surrogate.fit(data.train_x, data.train_y)
            self.surrogate.post_fit(data.train_x, data.train_y, ctx)
            pred = self.surrogate.predict(x)  # mean: (1, n_obj)
            predictions.append(pred)

        scores = np.array(
            [self.acquisition.score(p, reference)[0] for p in predictions]
        )
        scores, predictions = self._sanitize_nan(scores, predictions)
        return self.post_score(scores, predictions, ctx)


def product_combine(scores: list[np.ndarray]) -> np.ndarray:
    """Combine scores by element-wise product.

    Parameters
    ----------
    scores : list[np.ndarray]
        Score arrays, each shape ``(n_candidates,)``.

    Returns
    -------
    np.ndarray
        Element-wise product. shape: ``(n_candidates,)``.
    """
    return np.prod(np.stack(scores, axis=0), axis=0)


def rank_weighted_combine(
    weights: np.ndarray | None = None,
) -> Callable[[list[np.ndarray]], np.ndarray]:
    """Return a combine function that rank-normalises then takes a weighted average.

    Parameters
    ----------
    weights : np.ndarray or None
        Weights for each manager. If None, uniform weights are used.
        Need not sum to 1; they are normalised internally.

    Returns
    -------
    callable
        A function ``(list[np.ndarray]) -> np.ndarray`` suitable for
        ``CompositeSurrogateManager``.
    """
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
    else:
        w = None  # resolved lazily to uniform at call time

    def _combine(score_list: list[np.ndarray]) -> np.ndarray:
        normalized = [_rank_normalize(s) for s in score_list]
        _w = w if w is not None else np.full(len(normalized), 1.0 / len(normalized))
        return (np.stack(normalized, axis=0) * _w[:, None]).sum(axis=0)

    return _combine


class CompositeSurrogateManager(SurrogateManager):
    """Surrogate manager that combines scores from multiple sub-managers.

    Each sub-manager's ``score_candidates`` is called independently.
    The resulting score arrays are combined by ``combine_fn``.
    Predictions (for ``tell_f`` assignment) are taken from ``managers[0]``.

    Parameters
    ----------
    managers : list[SurrogateManager]
        Sub-managers to combine. Must be non-empty.
        ``managers[0]`` provides the ``SurrogatePrediction`` objects returned
        by ``score_candidates``; its predicted objective values flow into
        ``assign_tell_f`` in the strategies.
    combine_fn : callable(list[np.ndarray]) -> np.ndarray
        Accepts a list of score arrays (each shape ``(n_candidates,)``) and
        returns a single combined score array of the same shape.
        Use :func:`product_combine` for element-wise product (e.g. EI x PoF)
        or :func:`rank_weighted_combine` for rank-normalised weighted average.

    Examples
    --------
    EI x PoF (objective x feasibility product):

    >>> manager = CompositeSurrogateManager(
    ...     [ei_manager, pof_manager],
    ...     combine_fn=product_combine,
    ... )

    Rank-normalised ensemble (equivalent to former EnsembleSurrogateManager):

    >>> manager = CompositeSurrogateManager(
    ...     [m1, m2],
    ...     combine_fn=rank_weighted_combine(weights=[0.3, 0.7]),
    ... )
    """

    def __init__(
        self,
        managers: list[SurrogateManager],
        combine_fn: Callable[[list[np.ndarray]], np.ndarray],
    ):
        if not managers:
            raise ValueError("CompositeSurrogateManager requires at least one manager.")
        self.managers = managers
        self.combine_fn = combine_fn

    def score_candidates(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
        ctx: OptimizationContext | None = None,
    ) -> tuple[np.ndarray, list[SurrogatePrediction]]:
        """Combine scores from all sub-managers using ``combine_fn``.

        Returns the predictions from ``managers[0]`` as representative.
        """
        all_scores: list[np.ndarray] = []
        first_predictions: list[SurrogatePrediction] | None = None

        for i, manager in enumerate(self.managers):
            scores, preds = manager.score_candidates(candidates_x, archive, ctx)
            all_scores.append(scores)
            if i == 0:
                first_predictions = preds

        combined = self.combine_fn(all_scores)
        scores, predictions = self._sanitize_nan(combined, first_predictions)  # type: ignore[arg-type]
        return self.post_score(scores, predictions, ctx)


def _split_prediction(prediction: SurrogatePrediction) -> list[SurrogatePrediction]:
    """Split a batch SurrogatePrediction into per-sample SurrogatePrediction objects."""
    n = prediction.value.shape[0]
    result = []
    for i in range(n):
        std_i = prediction.std[i : i + 1] if prediction.std is not None else None
        label_i = prediction.label[i : i + 1] if prediction.label is not None else None
        tell_f_i = (
            prediction._tell_f[i : i + 1] if prediction._tell_f is not None else None
        )
        result.append(
            SurrogatePrediction(
                value=prediction.value[i : i + 1],
                std=std_i,
                label=label_i,
                _tell_f=tell_f_i,
                metadata=prediction.metadata,
            )
        )
    return result


def _rank_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Normalize scores to [0, 1] via rank transform.

    Rank 0 (lowest score) -> 0.0, rank n-1 (highest score) -> 1.0.
    NaN scores are treated as the lowest rank (0.0) so that candidates
    with failed surrogate predictions are never selected.
    """
    n = len(scores)
    if n == 1:
        return np.ones(1)
    safe = np.where(np.isnan(scores), -np.inf, scores)
    order = np.argsort(safe)
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    return ranks / (n - 1)
