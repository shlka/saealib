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
PairwiseSurrogateManager
    Fits a pairwise classifier; scores by win rate against archive references.

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

from saealib.surrogate.accuracy import AccuracyEvaluator, SurrogateAccuracy
from saealib.surrogate.base import Surrogate
from saealib.surrogate.prediction import SurrogatePrediction
from saealib.surrogate.training_set import (
    ArchiveObjectiveSet,
    KNNObjectiveSet,
    PairwiseComparisonSet,
    TrainingSet,
)

if TYPE_CHECKING:
    from saealib.acquisition.base import AcquisitionFunction
    from saealib.context import OptimizationState
    from saealib.population import Archive


class SurrogateManager(ABC):
    """
    Abstract base class for surrogate managers.

    A SurrogateManager coordinates the fit/predict/score pipeline and
    returns both scalar acquisition scores and the underlying predictions
    so that callers (e.g. IndividualBasedStrategy) can assign predicted
    objective values to offspring.

    Attributes
    ----------
    last_accuracy : SurrogateAccuracy or None
        Accuracy metrics computed after the most recent :meth:`fit` call.
        ``None`` until the first fit or when no evaluator is configured.
    """

    last_accuracy: SurrogateAccuracy | None = None

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

    def fit(
        self,
        archive: Archive,
        ctx: OptimizationState | None = None,
    ) -> None:
        """
        Pre-fit the surrogate on the archive.

        Call once before a sequence of ``score_candidates(..., refit=False)``
        calls when the archive does not change between calls (e.g. the
        surrogate-only inner loop of ``GenerationBasedStrategy``).
        The default implementation is a no-op; override in managers that
        maintain a fitted surrogate model.

        Parameters
        ----------
        archive : Archive
            Archive of evaluated solutions used for surrogate training.
        ctx : OptimizationState or None, optional
            Current optimization context.
        """

    @abstractmethod
    def score_candidates(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
        ctx: OptimizationState | None = None,
        *,
        refit: bool = True,
    ) -> tuple[np.ndarray, list[SurrogatePrediction]]:
        """
        Score candidate solutions using the surrogate model.

        Parameters
        ----------
        candidates_x : np.ndarray
            Candidate design variable matrix. shape: (n_candidates, dim)
        archive : Archive
            Archive of evaluated solutions used for surrogate training.
        ctx : OptimizationState or None, optional
            Current optimization context. Passed to ``TrainingSet.build``
            for strategies that require comparator or population access.
        refit : bool, optional
            If ``True`` (default), fit the surrogate before scoring.
            Pass ``False`` after an explicit :meth:`fit` call to skip
            redundant re-fitting when the archive has not changed.

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
        ctx: OptimizationState | None = None,
    ) -> tuple[np.ndarray, list[SurrogatePrediction]]:
        """Post-score lifecycle hook; override to inject custom processing.

        Parameters
        ----------
        scores : np.ndarray
            Acquisition scores. shape: (n_candidates,)
        predictions : list[SurrogatePrediction]
            Surrogate predictions for each candidate.
        ctx : OptimizationState or None, optional
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
            [np.ndarray, list[SurrogatePrediction], OptimizationState | None],
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
        new.post_score = lambda scores, predictions, ctx=None: fn(  # type: ignore  # lambda hook; slot type stricter than inferred lambda signature
            *prev(scores, predictions, ctx), ctx
        )
        return new

    def on_generation_end(
        self,
        gen: int,
        archive: Archive,
        ctx: OptimizationState | None = None,
    ) -> None:
        """End-of-generation hook; override to update internal state."""

    def with_on_generation_end(
        self,
        fn: Callable[[int, Archive, OptimizationState | None], None],
    ) -> SurrogateManager:
        """Return a copy of this manager with ``fn`` appended to the hook.

        Parameters
        ----------
        fn : callable
            ``fn(gen, archive, ctx) -> None``

        Returns
        -------
        SurrogateManager
            Shallow copy with the hook registered.
        """
        new = copy.copy(self)
        prev = self.on_generation_end

        def _chained(
            gen: int,
            archive: Archive,
            ctx: OptimizationState | None = None,
        ) -> None:
            prev(gen, archive, ctx)
            fn(gen, archive, ctx)

        new.on_generation_end = _chained  # type: ignore  # chained callable; hook slot type narrower than Callable
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
    accuracy_evaluator : AccuracyEvaluator or None
        If provided, :meth:`fit` computes accuracy metrics after each fit
        and stores them in :attr:`last_accuracy`.
    """

    def __init__(
        self,
        surrogate: Surrogate,
        acquisition: AcquisitionFunction,
        training_set: TrainingSet | None = None,
        accuracy_evaluator: AccuracyEvaluator | None = None,
    ):
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.training_set: TrainingSet = (
            training_set if training_set is not None else ArchiveObjectiveSet()
        )
        self.accuracy_evaluator = accuracy_evaluator

    def fit(
        self,
        archive: Archive,
        ctx: OptimizationState | None = None,
    ) -> None:
        """Fit the surrogate on the full archive.

        If an ``accuracy_evaluator`` was supplied at construction, accuracy
        metrics are computed immediately after fitting and stored in
        :attr:`last_accuracy`.
        """
        population = ctx.population if ctx is not None else None
        data = self.training_set.build(archive, population, ctx, candidate_x=None)
        self.surrogate.fit(data.train_x, data.train_y)
        self.surrogate.post_fit(data.train_x, data.train_y, ctx)
        if self.accuracy_evaluator is not None:
            self.last_accuracy = self.accuracy_evaluator.evaluate(
                self.surrogate, data.train_x, data.train_y
            )

    def score_candidates(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
        ctx: OptimizationState | None = None,
        *,
        refit: bool = True,
    ) -> tuple[np.ndarray, list[SurrogatePrediction]]:
        """Fit on full archive, predict and score all candidates at once."""
        if refit:
            self.fit(archive, ctx)

        rng = ctx.rng if ctx is not None else None
        reference = self.acquisition.compute_reference(archive, rng=rng)
        prediction = self.surrogate.predict(candidates_x)  # mean: (n_candidates, n_obj)
        scores = self.acquisition.score(prediction, reference, rng=rng)

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
    accuracy_evaluator : AccuracyEvaluator or None
        If provided, :meth:`score_candidates` computes accuracy metrics when
        ``refit=True`` and stores them in :attr:`last_accuracy`.  Accuracy is
        estimated by fitting a local model for each archive point via the same
        ``training_set`` and comparing the prediction against the true value.
    """

    def __init__(
        self,
        surrogate: Surrogate,
        acquisition: AcquisitionFunction,
        training_set: TrainingSet | None = None,
        accuracy_evaluator: AccuracyEvaluator | None = None,
    ):
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.training_set: TrainingSet = (
            training_set
            if training_set is not None
            else KNNObjectiveSet(n_neighbors=50)
        )
        self.accuracy_evaluator = accuracy_evaluator

    def fit(
        self,
        archive: Archive,
        ctx: OptimizationState | None = None,
    ) -> None:
        """Compute accuracy metrics from the archive (no global model is fitted).

        For ``LocalSurrogateManager``, there is no persistent global surrogate
        to pre-fit; local models are always built per candidate inside
        :meth:`score_candidates`.  However, calling :meth:`fit` explicitly
        (as ``GenerationBasedStrategy`` does before its inner loop) still
        triggers accuracy evaluation so that :attr:`last_accuracy` is available
        regardless of which strategy is used.
        """
        if self.accuracy_evaluator is not None:
            population = ctx.population if ctx is not None else None
            self._update_accuracy(archive, population, ctx)

    def score_candidates(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
        ctx: OptimizationState | None = None,
        *,
        refit: bool = True,
    ) -> tuple[np.ndarray, list[SurrogatePrediction]]:
        """Fit a local model per candidate and score each individually.

        When ``refit=True`` and an ``accuracy_evaluator`` is configured,
        accuracy is estimated inline using the nearest-neighbor holdout method
        (Hanawa et al., 2025): for each candidate, the closest archive neighbor
        is reserved as a validation point and excluded from training.  Metrics
        are averaged over all candidates and stored in :attr:`last_accuracy`.

        Because the holdout point is excluded from training, the effective
        training-set size is ``n_neighbors - 1`` when an accuracy evaluator is
        active.  Without an accuracy evaluator the full ``n_neighbors`` points
        are always used.
        """
        predictions: list[SurrogatePrediction] = []
        rng = ctx.rng if ctx is not None else None
        reference = self.acquisition.compute_reference(archive, rng=rng)
        population = ctx.population if ctx is not None else None
        compute_accuracy = refit and self.accuracy_evaluator is not None
        y_true_list: list[np.ndarray] = []
        y_pred_list: list[np.ndarray] = []

        for x in candidates_x:
            data = self.training_set.build(archive, population, ctx, candidate_x=x)

            if compute_accuracy and len(data.train_x) >= 2:
                # Hold out the nearest neighbor (index 0 when KNN-sorted) as
                # validation to get an unbiased estimate of local model accuracy.
                val_x = data.train_x[0:1]
                val_y = np.atleast_2d(data.train_y[0:1].T).T
                train_x = data.train_x[1:]
                train_y = data.train_y[1:]
            else:
                val_x = val_y = None
                train_x = data.train_x
                train_y = data.train_y

            self.surrogate.fit(train_x, train_y)
            self.surrogate.post_fit(train_x, train_y, ctx)
            pred = self.surrogate.predict(x)  # mean: (1, n_obj)
            predictions.append(pred)

            if val_x is not None and val_y is not None:
                try:
                    val_pred = self.surrogate.predict(val_x)
                    y_true_list.append(val_y[0])
                    y_pred_list.append(val_pred.value[0])
                except Exception:
                    pass

        scores = np.array(
            [self.acquisition.score(p, reference, rng=rng)[0] for p in predictions]
        )
        scores, predictions = self._sanitize_nan(scores, predictions)

        if compute_accuracy:
            if y_true_list:
                y_true = np.stack(y_true_list)
                y_pred = np.stack(y_pred_list)
                metrics = self.accuracy_evaluator._compute_metrics(y_true, y_pred)
                self.last_accuracy = SurrogateAccuracy(
                    metrics=metrics, n_samples=len(y_true_list)
                )
            else:
                self.last_accuracy = SurrogateAccuracy(n_samples=0)

        return self.post_score(scores, predictions, ctx)

    def _update_accuracy(
        self,
        archive: Archive,
        population: object,
        ctx: OptimizationState | None,
    ) -> None:
        """Compute accuracy via LOO with self-exclusion on archive points.

        Called by :meth:`fit` for the ``GenerationBasedStrategy`` pattern.
        For each archive point ``x_i``, the local model is built from the
        ``n_neighbors`` nearest neighbors of ``x_i`` **excluding ``x_i``
        itself**, and then predicts ``x_i``.  This avoids the self-inclusion
        bias that occurs when an interpolating surrogate (e.g. RBF) is fitted
        on data that includes the query point.
        """
        archive_x = archive.x
        archive_y = np.atleast_2d(archive.f.T).T
        n = len(archive_x)

        if n < 2:
            self.last_accuracy = SurrogateAccuracy(n_samples=n)
            return

        n_neighbors = min(getattr(self.training_set, "n_neighbors", n), n - 1)
        y_true_list: list[np.ndarray] = []
        y_pred_list: list[np.ndarray] = []

        for i in range(n):
            # Exclude x_i from its own neighborhood (LOO self-exclusion)
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            loo_x = archive_x[mask]
            loo_y = archive_y[mask]

            dists = np.sum((loo_x - archive_x[i]) ** 2, axis=1)
            k = min(n_neighbors, len(loo_x))
            idx = np.argsort(dists)[:k]
            train_x = loo_x[idx]
            train_y = loo_y[idx]

            surrogate_copy = copy.deepcopy(self.surrogate)
            try:
                surrogate_copy.fit(train_x, train_y)
                pred = surrogate_copy.predict(archive_x[i : i + 1])
                y_true_list.append(archive_y[i])
                y_pred_list.append(pred.value[0])
            except Exception:
                continue

        if not y_true_list:
            self.last_accuracy = SurrogateAccuracy(n_samples=n)
            return

        y_true = np.stack(y_true_list)
        y_pred = np.stack(y_pred_list)
        metrics = self.accuracy_evaluator._compute_metrics(y_true, y_pred)  # type: ignore  # caller guarantees non-None; ty doesn't narrow across method boundary
        self.last_accuracy = SurrogateAccuracy(metrics=metrics, n_samples=n)


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

    def fit(
        self,
        archive: Archive,
        ctx: OptimizationState | None = None,
    ) -> None:
        """Pre-fit all sub-managers.

        :attr:`last_accuracy` is propagated from ``managers[0]`` so callers
        can read a representative accuracy value from this composite manager.
        """
        for manager in self.managers:
            manager.fit(archive, ctx)
        self.last_accuracy = self.managers[0].last_accuracy

    def score_candidates(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
        ctx: OptimizationState | None = None,
        *,
        refit: bool = True,
    ) -> tuple[np.ndarray, list[SurrogatePrediction]]:
        """Combine scores from all sub-managers using ``combine_fn``.

        Returns the predictions from ``managers[0]`` as representative.
        """
        all_scores: list[np.ndarray] = []
        first_predictions: list[SurrogatePrediction] | None = None

        for i, manager in enumerate(self.managers):
            scores, preds = manager.score_candidates(
                candidates_x, archive, ctx, refit=refit
            )
            all_scores.append(scores)
            if i == 0:
                first_predictions = preds

        combined = self.combine_fn(all_scores)
        scores, predictions = self._sanitize_nan(combined, first_predictions)  # type: ignore  # _sanitize_nan return typed as Any; tuple unpacking safe at runtime
        return self.post_score(scores, predictions, ctx)


class PairwiseSurrogateManager(SurrogateManager):
    """Surrogate manager for pairwise comparison classifiers.

    Scores candidates by pairing each with reference points sampled from
    the archive and averaging the predicted win probability over all pairs.

    The surrogate must expose ``predict_proba(test_x) -> np.ndarray`` returning
    win probabilities with shape ``(n_samples,)``.  If the surrogate lacks this
    method, :meth:`score_candidates` raises ``ValueError``.

    Parameters
    ----------
    surrogate : Surrogate
        Pairwise classifier surrogate.  Must support ``predict_proba()``.
        ``DTSurrogate`` and other ``SklearnSurrogate`` subclasses provide
        ``predict_proba()`` out of the box.
    training_set : TrainingSet or None
        Training data builder.  Defaults to ``PairwiseComparisonSet()``.
        ``ctx`` is required when using ``PairwiseComparisonSet``.
    n_ref : int
        Number of archive points sampled as reference per candidate.
        When the archive has fewer than ``n_ref`` points all are used.
    """

    def __init__(
        self,
        surrogate: Surrogate,
        training_set: TrainingSet | None = None,
        n_ref: int = 10,
    ):
        self.surrogate = surrogate
        self.training_set: TrainingSet = (
            training_set if training_set is not None else PairwiseComparisonSet()
        )
        self.n_ref = n_ref

    def fit(
        self,
        archive: Archive,
        ctx: OptimizationState | None = None,
    ) -> None:
        """Fit the surrogate on pairwise comparison training data.

        Parameters
        ----------
        archive : Archive
            Archive of evaluated solutions.
        ctx : OptimizationState or None
            Required when the training set is ``PairwiseComparisonSet``
            (provides the comparator and rng).
        """
        population = ctx.population if ctx is not None else None
        data = self.training_set.build(archive, population, ctx, candidate_x=None)
        self.surrogate.fit(data.train_x, data.train_y)
        self.surrogate.post_fit(data.train_x, data.train_y, ctx)

    def score_candidates(
        self,
        candidates_x: np.ndarray,
        archive: Archive,
        ctx: OptimizationState | None = None,
        *,
        refit: bool = True,
    ) -> tuple[np.ndarray, list[SurrogatePrediction]]:
        """Score candidates by mean win rate against archive reference points.

        Parameters
        ----------
        candidates_x : np.ndarray
            Candidate design variable matrix. shape: (n_candidates, dim)
        archive : Archive
            Archive of evaluated solutions.
        ctx : OptimizationState or None
            Optimization context.  Required when ``refit=True`` and the
            training set is ``PairwiseComparisonSet``.
        refit : bool
            If ``True`` (default), fit the surrogate before scoring.

        Returns
        -------
        scores : np.ndarray
            Win rate scores. shape: (n_candidates,). Higher is better.
        predictions : list[SurrogatePrediction]
            One ``SurrogatePrediction`` per candidate.  ``value`` holds the
            scalar win rate (shape ``(1,)``); ``_tell_f`` is NaN so that
            strategies skip pbest assignment.

        Raises
        ------
        ValueError
            If the surrogate does not expose ``predict_proba()``.
        """
        if refit:
            self.fit(archive, ctx)

        rng = ctx.rng if ctx is not None else np.random.default_rng()

        archive_x = archive.x
        n_archive = len(archive_x)
        n_ref = min(self.n_ref, n_archive)
        if n_ref < n_archive:
            ref_idx = rng.choice(n_archive, size=n_ref, replace=False)
            ref_x = archive_x[ref_idx]
        else:
            ref_x = archive_x

        scores = np.empty(len(candidates_x))
        predictions: list[SurrogatePrediction] = []

        for i, x_c in enumerate(candidates_x):
            pairs = np.stack([np.concatenate([x_c, x_r]) for x_r in ref_x])
            try:
                win_probs = self.surrogate.predict_proba(pairs)
            except AttributeError as exc:
                raise ValueError(
                    f"PairwiseSurrogateManager requires a surrogate with "
                    f"predict_proba(). {type(self.surrogate).__name__} does "
                    f"not support it."
                ) from exc
            win_rate = float(np.mean(win_probs))
            scores[i] = win_rate
            predictions.append(
                SurrogatePrediction(
                    value=np.full((1,), win_rate),
                    _tell_f=np.full((1,), np.nan),
                )
            )

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
