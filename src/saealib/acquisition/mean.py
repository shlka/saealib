"""MeanPrediction acquisition function module."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial import distance

from saealib.acquisition.base import PointwiseAcquisition
from saealib.exceptions import ValidationError
from saealib.registry import register
from saealib.surrogate.prediction import SurrogatePrediction

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.population import Archive


@register()
class MeanPrediction(PointwiseAcquisition):
    """
    Acquisition function based on predicted mean value (exploitation).

    For single-objective problems, returns the predicted mean directly.
    For multi-objective problems, returns a weighted scalarization of the
    predicted mean.

    A higher score indicates a more promising candidate.
    The sign convention follows the weight: use a negative weight for
    minimization (e.g., weights=np.array([-1.0])) so that lower objective
    values yield higher scores.

    Parameters
    ----------
    weights : np.ndarray or None
        Weights for magnitude-aware scalarization of multi-objective predictions.
        shape: (n_obj,). If None and direction is also None, uses the first
        objective only.
    direction : np.ndarray or None
        Per-objective optimization directions (+1 = maximize, -1 = minimize).
        Used for direction-only scalarization when magnitude does not matter.
        Takes precedence over weights when both are provided. When unset,
        it is auto-injected from ``problem.direction`` at run start.
    """

    def __init__(
        self,
        weights: np.ndarray | None = None,
        reference: Any = None,
        direction: np.ndarray | None = None,
    ):
        self.weights = weights
        self.direction = direction
        self.reference = reference

    def compute_reference(
        self, archive: Archive, rng: np.random.Generator | None = None
    ) -> Any:
        """Return fixed reference if set, otherwise None."""
        return self.reference

    def score(
        self,
        prediction: SurrogatePrediction,
        reference: Any = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Compute scores based on predicted mean.

        Parameters
        ----------
        prediction : SurrogatePrediction
            Surrogate predictions. prediction.value shape: (n_samples, n_obj)
        reference : Any
            Not used. Accepted for interface compatibility.

        Returns
        -------
        np.ndarray
            Scores. shape: (n_samples,)
        """
        m = prediction.value  # (n_samples, n_obj)
        if self.direction is not None:
            return m @ np.asarray(self.direction)  # (n_samples,)
        if self.weights is not None:
            return m @ np.asarray(self.weights)  # (n_samples,)
        return m[:, 0]  # single-objective default


@dataclass(frozen=True)
class _CORSReference:
    """Prepared archive reference and beta for one CORS decision."""

    evaluated_x: np.ndarray
    beta: float


@register()
class CORSDistance(PointwiseAcquisition):
    """
    CORS distance-constrained predicted-mean acquisition function.

    Implements the auxiliary problem (CORS-AP) of the Constrained
    Optimization using Response Surfaces (CORS) method: the predicted-mean
    score used to rank candidates for the next costly evaluation is
    overridden to the worst possible value (``-np.inf``) for any candidate
    that violates the distance constraint::

        ||x - x_j|| >= beta_i * Delta_i,  j = 1, ..., k + i - 1

    i.e. whose minimum distance to every previously evaluated point falls
    below the current iteration's threshold ``beta_i * Delta``. ``beta_i``
    cycles through ``search_pattern``: the parameters are set by performing
    cycles of ``N + 1`` iterations, with ``beta_i = beta_{i + N + 1}`` for
    all ``i >= 1`` and ``1 >= beta_1 >= ... >= beta_{N+1} = 0`` in the cited work.
    ``prepare()`` resolves one entry of ``search_pattern`` for each decision
    from ``ctx.decision_count`` using a zero-based index, so decision zero
    uses ``search_pattern[0]``. This differs from
    :class:`~saealib.acquisition.lcb.LowerConfidenceBound`, whose scheduled
    parameter uses the same field as ``t = ctx.decision_count + 1`` (one-based).
    A real context is therefore required when ``evaluate()`` prepares a CORS
    reference; direct ``score()`` calls must provide that prepared reference.

    Regis & Shoemaker (2005) define ``Delta_i``  as the maximin
    distance of any point in the feasible domain D from the previously
    evaluated points -- a quantity that requires knowledge of D's bounds,
    which ``AcquisitionFunction`` has no access to (see
    :class:`saealib.acquisition.base.AcquisitionFunction`). This
    implementation therefore requires ``delta`` as a constructor parameter
    rather than approximating it from ``archive.x``'s range: the latter
    would shrink monotonically as more points are evaluated and bias the
    schedule toward premature local search, which the paper's own
    approximate ``Delta_i`` (computed from cover points spanning the full
    domain D) does not do. Callers should supply the diagonal
    length of the design space's box bounds (e.g.
    ``np.linalg.norm(ub - lb)``) or a maximin estimate obtained via the
    paper's own cover-points approximation.

    Because neither ``prediction.value`` nor the archive-derived
    ``reference`` carries candidate coordinates, this class reads the
    candidate design vectors from ``prediction.x``; the surrogate's
    ``predict()`` must populate that field.

    Parameters
    ----------
    delta : float
        Distance scale for the current threshold ``beta_i * delta``
        ("Delta_i" in the cited work). See class docstring for why this
        must be supplied directly rather than derived from the archive.
    search_pattern : Sequence[float]
        The beta cycling sequence ``<beta_1, ..., beta_{N+1}=0>``
        in the cited work. Defaults to the paper's own SP1 =
        ``<0.95, 0.25, 0.05, 0.03, 0>``; reported as the
        stronger of the two search patterns the paper tested on the
        Dixon-Szego benchmark suite.
    weights : np.ndarray or None
        Weights for magnitude-aware scalarization of multi-objective
        predicted means. shape: (n_obj,). See :class:`MeanPrediction`.
    direction : np.ndarray or None
        Per-objective optimization direction (+1 = maximize, -1 =
        minimize). See :class:`MeanPrediction`. When unset, it is
        auto-injected from ``problem.direction`` at run start.

    References
    ----------
    :cite:`regis2005cors`: Regis, R. G., & Shoemaker, C. A. (2005).
    Constrained global optimization of expensive black box functions using
    radial basis functions. *Journal of Global Optimization*, 31(1),
    153-171.
    """

    def __init__(
        self,
        delta: float,
        search_pattern: Sequence[float] = (0.95, 0.25, 0.05, 0.03, 0.0),
        weights: np.ndarray | None = None,
        direction: np.ndarray | None = None,
    ):
        self.delta = delta
        self.search_pattern = tuple(search_pattern)
        self.weights = weights
        self.direction = direction

    def prepare(self, archive: Archive, ctx: OptimizationState | None = None) -> Any:
        """
        Prepare the evaluated points and beta for one decision.

        CORS uses ``ctx.decision_count`` directly as a zero-based index into
        ``search_pattern``. In contrast, the scheduled parameter in
        :class:`~saealib.acquisition.lcb.LowerConfidenceBound` uses
        ``ctx.decision_count + 1`` as its one-based round index. Keeping the
        offset distinction here makes the CORS phase start at SP1's first
        entry when ``decision_count == 0``.

        Returns
        -------
        _CORSReference
            The archive's evaluated design vectors together with the beta
            selected for this decision.

        Raises
        ------
        ValidationError
            If ``ctx`` is ``None``. A decision count is required to resolve
            the beta; call ``evaluate()`` with a real context rather than
            calling ``score()`` directly without a prepared reference.
        """
        if ctx is None:
            raise ValidationError(
                "CORSDistance.search_pattern requires ctx.decision_count; "
                "call evaluate() with a real OptimizationState, not score() "
                "directly with no ctx."
            )
        beta = self.search_pattern[ctx.decision_count % len(self.search_pattern)]
        return _CORSReference(
            evaluated_x=self.compute_reference(archive, rng=ctx.rng),
            beta=beta,
        )

    def compute_reference(
        self, archive: Archive, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Return the archive's evaluated design vectors for the distance constraint."""
        return archive.x

    def score(
        self,
        prediction: SurrogatePrediction,
        reference: Any,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Compute predicted-mean scores, excluding distance-constraint violators.

        Parameters
        ----------
        prediction : SurrogatePrediction
            Surrogate predictions. prediction.value shape: (n_samples, n_obj).
            prediction.x must hold the candidate design vectors,
            shape: (n_samples, n_features), aligned row-for-row with
            prediction.value.
        reference : Any
            Evaluated design vectors returned by ``compute_reference``.
            shape: (n_evaluated, n_features).
            The reference must be the ``_CORSReference`` returned by
            :meth:`prepare`, not the raw array returned by
            :meth:`compute_reference`.

        Returns
        -------
        np.ndarray
            Scores. shape: (n_samples,). Candidates whose minimum distance
            to every evaluated point in ``reference`` is below the current
            ``beta_i * delta`` threshold receive ``-np.inf``.

        Raises
        ------
        ValueError
            If ``prediction.x`` is missing or its row count does not
            match ``prediction.value``.
        ValidationError
            If ``reference`` was not prepared with :meth:`prepare`.
        """
        if not isinstance(reference, _CORSReference):
            raise ValidationError(
                "CORSDistance.score() requires a prepare()-resolved reference; "
                "call evaluate() with a real ctx, not score() directly."
            )
        m = prediction.value  # (n_samples, n_obj)
        if self.direction is not None:
            base = m @ np.asarray(self.direction)  # (n_samples,)
        elif self.weights is not None:
            base = m @ np.asarray(self.weights)  # (n_samples,)
        else:
            base = m[:, 0]  # single-objective default
        scores = np.array(base, dtype=float, copy=True)  # never mutate prediction.value

        threshold = reference.beta * self.delta

        evaluated_x = (
            np.asarray(reference.evaluated_x, dtype=float)
            if reference.evaluated_x is not None
            else np.empty((0, 0))
        )
        if threshold <= 0 or evaluated_x.shape[0] == 0:
            # beta_i == 0 makes the distance constraint trivially satisfied by
            # every point; an empty archive likewise leaves no constraint to check.
            return scores

        candidate_x = self._candidate_x(prediction)
        min_dist = distance.cdist(candidate_x, evaluated_x).min(axis=1)
        scores[min_dist < threshold] = -np.inf
        return scores

    @staticmethod
    def _candidate_x(prediction: SurrogatePrediction) -> np.ndarray:
        x = prediction.x
        if x is None:
            raise ValueError(
                "CORSDistance requires prediction.x (the candidate design "
                "vectors) to evaluate the distance constraint; the "
                "surrogate's predict() must populate it."
            )
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if x.shape[0] != prediction.value.shape[0]:
            raise ValueError(
                "prediction.x must have one row per candidate in "
                f"prediction.value (got {x.shape[0]} rows vs "
                f"{prediction.value.shape[0]})."
            )
        return x
