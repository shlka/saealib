"""Synchronous execution of a compiled sequential plan."""

from __future__ import annotations

from saealib.context import OptimizationState
from saealib.core.compiler.compiler import ExecutablePlan
from saealib.core.runtime import (
    NodeResult,
    NodeStatus,
    RuntimeSession,
    RuntimeStep,
    SequentialPlan,
)
from saealib.core.state.patch import StatePatch
from saealib.exceptions import ValidationError

__all__ = ["PipelineRuntime"]


class PipelineRuntime:
    """Run each top-level legacy Stage once per synchronous step."""

    capabilities = frozenset()

    def initialize(
        self, plan: ExecutablePlan, state: OptimizationState
    ) -> RuntimeSession:
        """Validate runtime capabilities and create a state-threading session."""
        if not isinstance(plan, ExecutablePlan):
            raise ValidationError(
                "PipelineRuntime.initialize requires an ExecutablePlan"
            )
        if not isinstance(state, OptimizationState):
            raise ValidationError(
                "PipelineRuntime.initialize requires an OptimizationState"
            )
        sequential = SequentialPlan.from_executable_plan(plan)
        if not sequential.accepts(self.capabilities):
            missing = sequential.required_runtime_capabilities - self.capabilities
            names = ", ".join(sorted(missing))
            raise ValidationError(
                f"PipelineRuntime lacks required capabilities: {names}"
            )
        return RuntimeSession(plan=sequential, state=state)

    def advance(self, session: RuntimeSession) -> RuntimeStep:
        """Execute one complete ordered pipeline and expose its boundary."""
        if not isinstance(session, RuntimeSession):
            raise ValidationError("PipelineRuntime.advance requires a RuntimeSession")
        if not isinstance(session.plan, SequentialPlan):
            raise ValidationError(
                "PipelineRuntime.advance requires a SequentialPlan session"
            )

        state = session.state
        results: list[NodeResult] = []
        executed_node_ids: list[str] = []
        for node in session.plan.nodes:
            execute = getattr(node.component, "execute", None)
            if not callable(execute):
                raise ValidationError(
                    f"SequentialPlan node {node.component_id!r} is not executable"
                )
            next_state = execute(state)
            if not isinstance(next_state, OptimizationState):
                raise ValidationError(
                    f"Stage node {node.component_id!r} did not return an "
                    "OptimizationState"
                )
            state = next_state
            executed_node_ids.append(node.component_id)
            results.append(
                NodeResult(
                    patch=StatePatch(writes={}),
                    status=NodeStatus.COMPLETED,
                )
            )

        next_session = RuntimeSession(
            plan=session.plan,
            state=state,
            finished=False,
            observable=True,
            step_index=session.step_index + 1,
        )
        return RuntimeStep(
            state=state,
            node_results=tuple(results),
            executed_node_ids=tuple(executed_node_ids),
            observable=True,
            finished=False,
            session=next_session,
        )
