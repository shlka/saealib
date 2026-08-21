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

    ``requires_sequential_decisions`` is lightweight metadata for compiler and
    runtime diagnostics: one true-evaluated candidate per decision is the
    source-faithful configuration, while multi-candidate plans remain
    supported as an explicit batch extension. ``completed_decision_count == 0`` matches
    the first CORS iteration when this acquisition is used continuously from
    the beginning of a run. If a component is replaced and ``CORSDistance`` is
    introduced later, its CORS phase starts at the current
    ``completed_decision_count``
    rather than at zero.

    Implements the auxiliary problem (CORS-AP) of the Constrained
    Optimization using Response Surfaces (CORS) method: the predicted-mean
    score used to rank candidates for the next costly evaluation is
    overridden to the worst possible value (``-np.inf``) for any candidate
    that violates the distance constraint::

        ||x - x_j|| >= beta_i * Delta_i,  j = 1, ..., k + i - 1

    i.e. whose minimum distance to every previously evaluated point falls
    below the current iteration's threshold ``beta_i * Delta_i``. ``beta_i``
    cycles through ``search_pattern``: the parameters are set by performing
    cycles of ``N + 1`` iterations, with ``beta_i = beta_{i + N + 1}`` for
    all ``i >= 1`` and ``1 >= beta_1 >= ... >= beta_{N+1} = 0`` in the cited work.
    saealib accepts any non-empty finite sequence of beta values in ``[0, 1]``;
    the paper's ordering and terminal-zero conditions are not required.
    ``prepare()`` resolves one entry of ``search_pattern`` for each decision
    from ``ctx.completed_decision_count`` using a zero-based index, so decision zero
    uses ``search_pattern[0]``. This differs from
    :class:`~saealib.acquisition.lcb.LowerConfidenceBound`, whose scheduled
    parameter uses ``ctx.decision_count + 1`` as a one-based round number.
    CORS and LCB therefore use different runtime counters.
    A real context is therefore required when ``evaluate()`` prepares a CORS
    reference; direct ``score()`` calls must provide that prepared reference.

    Regis & Shoemaker (2005) define ``Delta_i`` as the maximin distance
    over the feasible domain D from the previously evaluated points::

        Delta_i = max_{x_tilde in D} min_j ||x_tilde - x_j||

    The exact domain-wide maximum is not available to an acquisition function,
    so when ``delta`` is ``None`` this implementation approximates it with
    the candidate pool in ``prediction.x`` as the cover points:
    ``max_c min_j ||candidate_c - evaluated_j||``. This approximation can
    underestimate ``Delta_i`` when a later candidate pool is concentrated in
    a promising region and therefore covers less of D than the full domain.
    Since ``beta <= 1`` guarantees that a maximin candidate in a non-empty
    pool is executable (equality is allowed), the approximation cannot silently
    leave no executable candidate and force an arbitrary choice.
    Passing a numeric ``delta`` opts into the legacy fixed distance scale.

    Because neither ``prediction.value`` nor the archive-derived
    ``reference`` carries candidate coordinates, this class reads the
    candidate design vectors from ``prediction.x``; the surrogate's
    ``predict()`` must populate that field.

    Parameters
    ----------
    delta : float or None, optional
        Fixed distance scale for the current threshold ``beta_i * delta``.
        When ``None`` (the default), compute the paper's ``Delta_i`` at each
        score call from the candidate pool as an approximation. A numeric
        value preserves the legacy fixed-scale behavior.
    search_pattern : Sequence[float]
        The beta cycling sequence. Any non-empty sequence of finite values in
        ``[0, 1]`` is accepted. Defaults to the paper's own SP1 =
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

    requires_sequential_decisions: bool = True

    def __init__(
        self,
        delta: float | None = None,
        search_pattern: Sequence[float] = (0.95, 0.25, 0.05, 0.03, 0.0),
        weights: np.ndarray | None = None,
        direction: np.ndarray | None = None,
    ):
        if delta is not None:
            try:
                valid_delta = bool(np.isfinite(delta)) and bool(delta > 0)
            except (TypeError, ValueError):
                valid_delta = False
            if not valid_delta:
                raise ValidationError(
                    "CORSDistance.delta must be finite and greater than 0"
                )
        self.delta = delta
        try:
            pattern = tuple(float(beta) for beta in search_pattern)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "CORSDistance.search_pattern must be a sequence of finite beta values"
            ) from exc
        if not pattern:
            raise ValidationError("CORSDistance.search_pattern must not be empty")
        if any(not np.isfinite(beta) or not 0.0 <= beta <= 1.0 for beta in pattern):
            raise ValidationError(
                "CORSDistance.search_pattern beta values must be finite and "
                "between 0 and 1"
            )
        self.search_pattern = pattern
        self.weights = weights
        self.direction = direction

    def prepare(self, archive: Archive, ctx: OptimizationState | None = None) -> Any:
        """
        Prepare the evaluated points and beta for one decision.

        CORS uses ``ctx.completed_decision_count`` directly as a zero-based index into
        ``search_pattern``. In contrast, the scheduled parameter in
        :class:`~saealib.acquisition.lcb.LowerConfidenceBound` uses
        ``ctx.decision_count + 1`` as its one-based round index. Keeping the
        offset distinction here makes the CORS phase start at SP1's first
        entry when ``completed_decision_count == 0``.

        Returns
        -------
        _CORSReference
            The archive's evaluated design vectors together with the beta
            selected for this decision.

        Raises
        ------
        ValidationError
            If ``ctx`` is ``None`` or does not expose the completed decision
            counter. A completed decision count is required to resolve the
            beta; call ``prepare(archive, ctx)`` with a real context.
        """
        if ctx is None or not hasattr(ctx, "completed_decision_count"):
            raise ValidationError(
                "CORSDistance requires a decision context with "
                "completed_decision_count; call prepare(archive, ctx) with a "
                "real OptimizationState."
            )
        beta = self.search_pattern[
            ctx.completed_decision_count % len(self.search_pattern)
        ]
        return _CORSReference(
            evaluated_x=archive.x,
            beta=beta,
        )

    def compute_reference(
        self, archive: Archive, rng: np.random.Generator | None = None
    ) -> _CORSReference:
        """Reject context-free reference construction.

        CORS beta is a decision-scoped schedule, so a reference constructed
        without an :class:`~saealib.context.OptimizationState` could silently
        use the wrong phase. Call :meth:`prepare` instead.
        """
        raise ValidationError(
            "CORS requires a decision context; call prepare(archive, ctx) with "
            "a real OptimizationState."
        )

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
            The ``_CORSReference`` returned by :meth:`prepare`.

        Returns
        -------
        np.ndarray
            Scores. shape: (n_samples,). Candidates whose minimum distance
            to every evaluated point in ``reference`` is below the current
            ``beta_i * Delta_i`` threshold receive ``-np.inf``. With
            ``delta=None``, ``Delta_i`` is the candidate-pool maximin
            distance; with an explicit ``delta``, it is that fixed value.

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

        evaluated_x = (
            np.asarray(reference.evaluated_x, dtype=float)
            if reference.evaluated_x is not None
            else np.empty((0, 0))
        )
        if reference.beta <= 0 or evaluated_x.shape[0] == 0:
            # beta_i == 0 makes the distance constraint trivially satisfied by
            # every point; an empty archive likewise leaves no constraint to check.
            return scores

        candidate_x = self._candidate_x(prediction)
        if candidate_x.shape[0] == 0:
            return scores

        min_dist = distance.cdist(candidate_x, evaluated_x).min(axis=1)
        delta_i = min_dist.max() if self.delta is None else self.delta
        threshold = reference.beta * delta_i
        if threshold <= 0:
            return scores
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
