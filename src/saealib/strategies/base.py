from abc import ABC, abstractmethod

from saealib.context import OptimizationContext
from saealib.optimizer import ComponentProvider


class OptimizationStrategy(ABC):
    """Base class for optimization strategies."""

    @abstractmethod
    def step(self, ctx: OptimizationContext, provider: ComponentProvider) -> None:
        """
        Perform one iteration of optimization processing.

        Parameters
        ----------
        ctx : OptimizationContext
            A dataclass object that holds internal information about the Optimizer.
        provider : ComponentProvider
            Objects of the class in which the component is exposed (ex. Optimizer).
        """
        pass
