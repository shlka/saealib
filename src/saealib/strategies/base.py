"""Base class for optimization strategies."""

from abc import ABC, abstractmethod

from saealib.context import OptimizationContext
from saealib.optimizer import ComponentProvider


class OptimizationStrategy(ABC):
    """Base class for optimization strategies."""

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
