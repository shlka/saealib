"""
Surrogate accuracy metrics and evaluators.

Classes
-------
SurrogateAccuracyMetric
    Abstract base for accuracy metric computation.
SpearmanCorrelation
    Rank correlation between predicted and true values (primary metric).
RMSE
    Root mean squared error of predictions.
R2Score
    Coefficient of determination (R²).
SurrogateAccuracy
    Container for computed accuracy metrics.
AccuracyEvaluator
    Abstract base for surrogate accuracy evaluators.
KFoldAccuracyEvaluator
    Evaluates accuracy via k-fold cross-validation.
LOOAccuracyEvaluator
    Evaluates accuracy via leave-one-out cross-validation.
HeldOutAccuracyEvaluator
    Evaluates accuracy against pre-specified held-out data.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import spearmanr

if TYPE_CHECKING:
    from saealib.surrogate.base import Surrogate


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class SurrogateAccuracyMetric(ABC):
    """Abstract base for a scalar accuracy metric."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Metric identifier used as key in :attr:`SurrogateAccuracy.metrics`."""
        ...

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the metric.

        Parameters
        ----------
        y_true : np.ndarray
            True objective values. shape: (n_samples, n_obj)
        y_pred : np.ndarray
            Predicted objective values. Same shape as ``y_true``.

        Returns
        -------
        float
            Scalar metric value. Interpretation (higher/lower = better)
            depends on the concrete metric.
        """
        ...


class SpearmanCorrelation(SurrogateAccuracyMetric):
    """Spearman rank correlation averaged over objectives.

    Measures whether the surrogate preserves the ranking of candidates,
    which is the primary criterion in EA contexts (Yu et al., 2019).
    Range: [-1, 1]. Higher is better.
    """

    @property
    def name(self) -> str:
        """Return ``"spearman"``."""
        return "spearman"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Spearman rank correlation, averaged over objectives."""
        y_true = np.atleast_2d(y_true.T).T
        y_pred = np.atleast_2d(y_pred.T).T
        rhos = []
        for j in range(y_true.shape[1]):
            result = spearmanr(y_true[:, j], y_pred[:, j])
            rho = float(result.statistic)
            rhos.append(0.0 if np.isnan(rho) else rho)
        return float(np.mean(rhos))


class RMSE(SurrogateAccuracyMetric):
    """Root mean squared error.

    Range: [0, ∞). Lower is better.
    """

    @property
    def name(self) -> str:
        """Return ``"rmse"``."""
        return "rmse"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute RMSE between true and predicted values."""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


class R2Score(SurrogateAccuracyMetric):
    """Coefficient of determination (R²).

    Range: (-∞, 1]. Higher is better. 1.0 = perfect fit.
    """

    @property
    def name(self) -> str:
        """Return ``"r2"``."""
        return "r2"

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² between true and predicted values."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Accuracy container
# ---------------------------------------------------------------------------


@dataclass
class SurrogateAccuracy:
    """Container for surrogate accuracy metrics computed after a fit.

    Attributes
    ----------
    metrics : dict[str, float]
        Mapping from metric name (e.g. ``"spearman"``) to scalar value.
    n_samples : int
        Number of training samples used for evaluation.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0

    def get(self, name: str, default: float = float("nan")) -> float:
        """Return metric value by name, or *default* if not present."""
        return self.metrics.get(name, default)


_DEFAULT_METRICS: list[SurrogateAccuracyMetric] = [
    SpearmanCorrelation(),
    RMSE(),
    R2Score(),
]


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


class AccuracyEvaluator(ABC):
    """Abstract base for surrogate accuracy evaluators.

    Subclasses implement different evaluation strategies (k-fold CV, LOO,
    held-out). All share the same :meth:`evaluate` interface so they can be
    injected into :class:`~saealib.surrogate.manager.GlobalSurrogateManager`
    interchangeably.

    Parameters
    ----------
    metrics : list[SurrogateAccuracyMetric] or None
        Metrics to compute. Defaults to
        :class:`SpearmanCorrelation`, :class:`RMSE`, :class:`R2Score`.
    """

    def __init__(
        self,
        metrics: list[SurrogateAccuracyMetric] | None = None,
    ):
        self.metrics: list[SurrogateAccuracyMetric] = (
            metrics if metrics is not None else list(_DEFAULT_METRICS)
        )

    @abstractmethod
    def evaluate(
        self,
        surrogate: Surrogate,
        train_x: np.ndarray,
        train_y: np.ndarray,
    ) -> SurrogateAccuracy:
        """Compute accuracy metrics for *surrogate* on the given training data.

        Parameters
        ----------
        surrogate : Surrogate
            Surrogate instance to evaluate.
        train_x : np.ndarray
            Training inputs. shape: (n_samples, n_features)
        train_y : np.ndarray
            Training targets. shape: (n_samples, n_obj)

        Returns
        -------
        SurrogateAccuracy
        """
        ...

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        """Apply all metrics to a single (y_true, y_pred) pair."""
        return {m.name: m.compute(y_true, y_pred) for m in self.metrics}


class KFoldAccuracyEvaluator(AccuracyEvaluator):
    """Evaluates surrogate accuracy via k-fold cross-validation.

    The training data is split into *n_splits* folds. For each fold, the
    surrogate is deep-copied and fitted on the remaining folds, then tested
    on the held-out fold. Metrics are averaged across folds.

    Parameters
    ----------
    metrics : list[SurrogateAccuracyMetric] or None
        Metrics to compute.
    n_splits : int
        Number of folds. Clamped to ``n_samples`` if larger.
    """

    def __init__(
        self,
        metrics: list[SurrogateAccuracyMetric] | None = None,
        n_splits: int = 5,
    ):
        super().__init__(metrics)
        self.n_splits = n_splits

    def evaluate(
        self,
        surrogate: Surrogate,
        train_x: np.ndarray,
        train_y: np.ndarray,
    ) -> SurrogateAccuracy:
        """Evaluate via k-fold CV."""
        return _cv_loop(surrogate, train_x, train_y, self.metrics, self.n_splits)


class LOOAccuracyEvaluator(AccuracyEvaluator):
    """Evaluates surrogate accuracy via leave-one-out cross-validation.

    Equivalent to :class:`KFoldAccuracyEvaluator` with ``n_splits=n_samples``.
    Each sample is used as a validation set exactly once.

    Parameters
    ----------
    metrics : list[SurrogateAccuracyMetric] or None
        Metrics to compute.
    """

    def evaluate(
        self,
        surrogate: Surrogate,
        train_x: np.ndarray,
        train_y: np.ndarray,
    ) -> SurrogateAccuracy:
        """Evaluate via leave-one-out CV."""
        n = len(train_x)
        return _cv_loop(surrogate, train_x, train_y, self.metrics, n_splits=n)


class HeldOutAccuracyEvaluator(AccuracyEvaluator):
    """Evaluates accuracy against pre-specified held-out data.

    The surrogate is assumed to be already fitted; it is *not* re-fitted
    inside :meth:`evaluate`. Useful for comparing surrogate predictions
    against the most recent true evaluations.

    Parameters
    ----------
    held_x : np.ndarray
        Held-out inputs. shape: (n_held, n_features)
    held_y : np.ndarray
        Held-out true objective values. shape: (n_held, n_obj)
    metrics : list[SurrogateAccuracyMetric] or None
        Metrics to compute.
    """

    def __init__(
        self,
        held_x: np.ndarray,
        held_y: np.ndarray,
        metrics: list[SurrogateAccuracyMetric] | None = None,
    ):
        super().__init__(metrics)
        self.held_x = np.asarray(held_x)
        self.held_y = np.atleast_2d(np.asarray(held_y).T).T

    def evaluate(
        self,
        surrogate: Surrogate,
        train_x: np.ndarray,
        train_y: np.ndarray,
    ) -> SurrogateAccuracy:
        """Predict on held-out data and compute metrics."""
        pred = surrogate.predict(self.held_x)
        result = self._compute_metrics(self.held_y, pred.value)
        return SurrogateAccuracy(metrics=result, n_samples=len(train_x))


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _cv_loop(
    surrogate: Surrogate,
    train_x: np.ndarray,
    train_y: np.ndarray,
    metrics: list[SurrogateAccuracyMetric],
    n_splits: int,
) -> SurrogateAccuracy:
    """Run cross-validation and return averaged metrics.

    Shared by :class:`KFoldAccuracyEvaluator` and :class:`LOOAccuracyEvaluator`.
    The surrogate is deep-copied per fold and left unmodified.
    """
    n = len(train_x)
    if n < 2:
        return SurrogateAccuracy(n_samples=n)

    train_y = np.atleast_2d(train_y.T).T
    n_splits = min(n_splits, n)
    indices = np.arange(n)
    fold_size = n // n_splits
    accumulator: dict[str, list[float]] = {m.name: [] for m in metrics}

    for fold in range(n_splits):
        start = fold * fold_size
        end = start + fold_size if fold < n_splits - 1 else n
        val_idx = indices[start:end]
        train_idx = np.setdiff1d(indices, val_idx)
        if len(train_idx) < 2:
            continue

        surrogate_copy = copy.deepcopy(surrogate)
        try:
            surrogate_copy.fit(train_x[train_idx], train_y[train_idx])
            pred = surrogate_copy.predict(train_x[val_idx])
            for m in metrics:
                accumulator[m.name].append(m.compute(train_y[val_idx], pred.value))
        except Exception:
            continue

    averaged = {
        name: float(np.mean(vals)) if vals else float("nan")
        for name, vals in accumulator.items()
    }
    return SurrogateAccuracy(metrics=averaged, n_samples=n)
