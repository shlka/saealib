"""Base class for optimization strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from saealib.context import OptimizationState
from saealib.optimizer import ComponentProvider
from saealib.policies.evaluation import EvaluateAll, EvaluationPolicy
from saealib.policies.feedback import FeedbackPolicy, MixedFeedback


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

    evaluation_policy: EvaluationPolicy = EvaluateAll()
    feedback_policy: FeedbackPolicy = MixedFeedback()

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
