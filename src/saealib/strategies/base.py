"""Base class for optimization strategies."""

from __future__ import annotations

from saealib.context import OptimizationState
from saealib.core.compiler.graph import ComponentGraph
from saealib.core.contracts import ComponentContract, PortContract, StateContract
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
    Built-in strategies materialize their graph directly at this boundary;
    ``build_pipeline`` is reconstructed from this graph only for compatibility
    callers.
    """
    if type(strategy).build_graph is not OptimizationStrategy.build_graph:
        return strategy.build_graph(provider)
    raise TypeError("strategy must implement build_graph")


def build_pipeline_from_graph(graph: ComponentGraph) -> Pipeline:
    """Recover the Stage facade from a canonical strategy graph."""
    from saealib.core.graph_builder import StageNodeAdapter

    stages = [
        node.component.stage
        for node in graph.nodes
        if isinstance(node.component, StageNodeAdapter)
    ]
    return Pipeline(stages)


class OptimizationStrategy:
    """Base class for optimization strategies.

    Strategies expose one canonical :class:`ComponentGraph` topology.
    ``build_pipeline`` remains the public compatibility facade recovered from
    that graph.
    """

    # Optimizer.validate() checks this to ensure surrogate_manager is configured.
    requires_surrogate: bool = False

    # Async runtimes may refill a strategy while an earlier proposal is still
    # waiting for complete feedback.  Most strategies need a generation
    # boundary first; steady-state strategies explicitly opt into overlap.
    supports_async_refill: bool = False

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

    def build_pipeline(self, provider: ComponentProvider) -> Pipeline:
        """Recover the retained Pipeline facade from the canonical graph."""
        return build_pipeline_from_graph(self.build_graph(provider))

    def build_graph(self, provider: ComponentProvider) -> ComponentGraph:
        """Build the strategy's canonical graph.

        Built-in strategies provide node/adapter specs to the canonical graph
        builder; graph-only extensions override this method and need not
        implement a Pipeline builder.
        """
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
