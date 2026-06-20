"""Base class for optimization strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from saealib.context import OptimizationContext
from saealib.optimizer import ComponentProvider

if TYPE_CHECKING:
    from saealib.population import Individual
    from saealib.surrogate.prediction import SurrogatePrediction


def assign_tell_f(
    individual: Individual,
    pred: SurrogatePrediction,
    ctx: OptimizationContext,
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
    """Base class for optimization strategies."""

    # Optimizer.validate() checks this to ensure surrogate_manager is configured.
    requires_surrogate: bool = False

    @abstractmethod
    def step(self, ctx: OptimizationContext, provider: ComponentProvider) -> None:
        """
        Perform one generation step: generate, score, evaluate, and update.

        Parameters
        ----------
        ctx : OptimizationContext
            Current optimization context.
        provider : ComponentProvider
            Component provider.
        """
        pass
