"""Base class for optimization strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from saealib.context import OptimizationState
from saealib.optimizer import ComponentProvider

if TYPE_CHECKING:
    from saealib.population import Individual
    from saealib.surrogate.prediction import SurrogatePrediction


def assign_tell_f(
    individual: Individual,
    pred: SurrogatePrediction,
    ctx: OptimizationState,
) -> None:
    """Assign predicted objective to individual, replacing NaN with worst population f.

    When a surrogate fails (score sanitized to -inf by SurrogateManager), the
    prediction's tell_f is an explicit NaN array.  Assigning NaN to individual.f
    would corrupt the population sort.  Instead, we substitute the worst f value
    currently in the population so that the individual is naturally eliminated by
    the survivor selection without special NaN handling in comparators.
    """
    f = pred.tell_f[0]
    if np.any(np.isnan(f)):
        order = ctx.problem.comparator.sort_population(ctx.population)
        f = ctx.population.get_array("f")[order[-1]].copy()
    individual.f = f


class OptimizationStrategy(ABC):
    """Base class for optimization strategies.

    Built-in strategies compose their generation logic from ``Pipeline``
    stages and rebuild that pipeline on every :meth:`step` call, so
    reassigning components on the provider (e.g. ``provider.algorithm``,
    ``provider.surrogate_manager``) or mutating a strategy's own parameters
    mid-run takes effect from the next call onward. This is a convention
    followed by each built-in strategy via a ``_build_pipeline`` method
    (not part of this ABC) rather than an enforced contract; a subclass
    wanting to customize stage composition should override
    ``_build_pipeline`` following the same pattern.
    """

    # Optimizer.validate() checks this to ensure surrogate_manager is configured.
    requires_surrogate: bool = False

    @abstractmethod
    def step(
        self, ctx: OptimizationState, provider: ComponentProvider
    ) -> OptimizationState | None:
        """
        Perform one generation step: generate, score, evaluate, and update.

        Parameters
        ----------
        ctx : OptimizationState
            Current optimization context.
        provider : ComponentProvider
            Component provider.

        Returns
        -------
        OptimizationState or None
            Updated state when the strategy uses the functional Pipeline API.
            ``None`` for strategies that mutate *ctx* in-place (legacy style).
            Callers must handle both: ``ctx = result if result is not None else ctx``.
        """
        pass
