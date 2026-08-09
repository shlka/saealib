"""Base class for optimization strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from saealib.context import OptimizationState
from saealib.core.compiler.graph import ComponentGraph
from saealib.core.contracts import ComponentContract, PortContract, StateContract
from saealib.core.graph_builder import (
    StageNodeAdapter,
    build_decomposed_component_graph,
)
from saealib.core.state import PENDING_EVALUATIONS
from saealib.optimizer import ComponentProvider
from saealib.pipeline import Pipeline
from saealib.policies.evaluation import EvaluateAll, EvaluationPlanner
from saealib.policies.feedback import FeedbackBuilder, MixedFeedback


def build_runtime_neutral_graph(
    strategy: OptimizationStrategy, provider: ComponentProvider
) -> ComponentGraph:
    """Build the strategy's one canonical graph topology.

    Runtime selection changes execution policy, never graph construction.
    Built-in strategies supply one canonical Stage sequence; ``build_pipeline``
    is reconstructed from this graph only for compatibility callers.
    """
    build_stages = getattr(strategy, "_build_stages", None)
    if callable(build_stages):
        pipeline = Pipeline(list(build_stages(provider)))
    else:
        pipeline = strategy.build_pipeline(provider)
    return build_decomposed_component_graph(pipeline)


def build_pipeline_from_graph(graph: ComponentGraph) -> Pipeline:
    """Recover the legacy Stage facade from a canonical strategy graph."""
    stages = [
        node.component.stage
        for node in graph.nodes
        if isinstance(node.component, StageNodeAdapter)
    ]
    return Pipeline(stages)


class OptimizationStrategy(ABC):
    """Base class for optimization strategies.

    Strategies compose their generation logic from one canonical pipeline.
    ``build_pipeline`` remains the public compatibility facade.
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
            ),
        )

    @abstractmethod
    def build_pipeline(self, provider: ComponentProvider):
        """Build the next pipeline from the current component provider."""
        ...

    def build_graph(self, provider: ComponentProvider) -> ComponentGraph:
        """Build the canonical graph, retaining the legacy pipeline hook."""
        return build_runtime_neutral_graph(self, provider)

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
        from saealib.execution.runtime import execute_strategy_step

        return execute_strategy_step(self, ctx, provider)
