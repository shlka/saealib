"""Base class for optimization strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from saealib.context import OptimizationState
from saealib.core.contracts import ComponentContract, PortContract, StateContract
from saealib.core.state import PENDING_EVALUATIONS
from saealib.optimizer import ComponentProvider
from saealib.policies.evaluation import EvaluateAll, EvaluationPlanner
from saealib.policies.feedback import FeedbackBuilder, MixedFeedback


class OptimizationStrategy(ABC):
    """Base class for optimization strategies.

    Strategies compose their generation logic from a pipeline built for each
    step.  ``build_pipeline`` is the public extension contract, so provider
    replacements and strategy parameter changes are observed on the next
    execution.
    """

    # Optimizer.validate() checks this to ensure surrogate_manager is configured.
    requires_surrogate: bool = False

    evaluation_planner: EvaluationPlanner = EvaluateAll()

    feedback_builder: FeedbackBuilder = MixedFeedback()

    def contract(self) -> ComponentContract:
        """Return the strategy contract."""
        return ComponentContract(
            ports={"strategy": PortContract()},
            state=StateContract(
                reads=(PENDING_EVALUATIONS,),
                writes=(PENDING_EVALUATIONS,),
            ),
        )

    @abstractmethod
    def build_pipeline(self, provider: ComponentProvider):
        """Build the next pipeline from the current component provider."""
        ...

    def step(
        self, ctx: OptimizationState, provider: ComponentProvider
    ) -> OptimizationState:
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
        OptimizationState
            Updated state returned by the pipeline.
        """
        return self.build_pipeline(provider).execute(ctx)
