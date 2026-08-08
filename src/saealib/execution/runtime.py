"""Runtime implementations and the legacy optimizer environment bridge."""

from __future__ import annotations

import time
from typing import Any, Protocol

from saealib.callback import (
    Event,
    GenerationEndEvent,
    GenerationStartEvent,
    RunEndEvent,
    RunStartEvent,
)
from saealib.comparators import NSGA3Comparator
from saealib.context import OptimizationState
from saealib.core.compiler.compiler import CompileContext, Compiler, ExecutablePlan
from saealib.core.runtime import (
    ExecutionRuntime,
    NodeResult,
    NodeStatus,
    RuntimeSession,
    RuntimeStep,
    SequentialPlan,
)
from saealib.core.state.patch import StatePatch
from saealib.exceptions import EvaluationFatalError, ValidationError

__all__ = ["PipelineRuntime", "create_runtime", "resolve_plan"]


class RuntimeEnvironment(Protocol):
    """Minimal provider boundary needed by a lifecycle runtime."""

    def execute(
        self, plan: SequentialPlan, state: OptimizationState
    ) -> OptimizationState: ...

    def process_pending(
        self, state: OptimizationState, generation_open: bool
    ) -> tuple[OptimizationState, bool, bool, bool]: ...

    def is_terminated(self, state: OptimizationState) -> bool: ...

    def dispatch(self, event: Event) -> None: ...

    def finish_generation(self, state: OptimizationState) -> None: ...

    def fatal(self, state: OptimizationState) -> None: ...


def _execute_sequential_plan(
    plan: SequentialPlan, state: OptimizationState
) -> OptimizationState:
    """Thread state through the compiled StageNodeAdapter sequence."""
    for node in plan.execution_nodes:
        execute = getattr(node.component, "execute", None)
        if not callable(execute):
            raise ValidationError(
                f"SequentialPlan node {node.component_id!r} is not executable"
            )
        state = execute(state)
        if not isinstance(state, OptimizationState):
            raise ValidationError(
                f"Stage node {node.component_id!r} did not return an OptimizationState"
            )
    return state


class _OptimizerEnvironment:
    """Temporary adapter for the pre-L4 optimizer/scheduler workflow.

    This is intentionally outside Runner.  L4 can replace this adapter with an
    async environment without changing the runtime protocol or facade.
    """

    def __init__(self, optimizer: Any, plan: SequentialPlan) -> None:
        self.optimizer = optimizer
        self.plan = plan
        self._execution_fingerprint = self._fingerprint()
        self.capabilities = frozenset(
            {"partial_feedback"}
            if getattr(
                getattr(optimizer, "algorithm", None), "allow_partial_tell", False
            )
            else set()
        )

    def execute(
        self, plan: SequentialPlan, state: OptimizationState
    ) -> OptimizationState:
        if getattr(self.optimizer, "async_evaluation_scheduler", None) is None:
            current_fingerprint = self._fingerprint()
            if current_fingerprint != self._execution_fingerprint:
                # Runtime extension seam: provider changes refresh execution
                # stages without recompiling the immutable plan or re-entering
                # strategy.step(). The unchanged path executes the plan.
                pipeline = self.optimizer.strategy.build_pipeline(self.optimizer)
                self._execution_fingerprint = current_fingerprint
                return pipeline.execute(state)
            return _execute_sequential_plan(plan, state)
        # Temporary L4 bridge: async still uses strategy.step() because its
        # scheduler polling/reattach semantics are not yet a PipelineRuntime.
        result = self.optimizer.strategy.step(state, self.optimizer)
        return state if result is None else result

    def _fingerprint(self) -> tuple[object, ...]:
        strategy = self.optimizer.strategy
        strategy_values = tuple(
            (name, repr(value))
            for name, value in vars(strategy).items()
            if name != "pipeline"
        )
        provider_values = tuple(
            (name, id(getattr(self.optimizer, name, None)))
            for name in (
                "surrogate_manager",
                "acquisition",
                "feedback_builder",
                "evaluation_planner",
                "evaluator",
            )
        )
        return strategy_values + provider_values

    def process_pending(
        self, state: OptimizationState, generation_open: bool
    ) -> tuple[OptimizationState, bool, bool, bool]:
        scheduler = getattr(self.optimizer, "async_evaluation_scheduler", None)
        if scheduler is None:
            state = self.execute(self.plan, state)
            if state.pending_evaluations:
                return state, generation_open, False, True
        else:
            if set(state.pending_evaluations) != set(state.evaluation_handles):
                state = scheduler.reattach(state)
            before = state
            state = scheduler.poll(state, wait=False)
            if state.pending_evaluations:
                if self.is_terminated(state):
                    if state is before:
                        time.sleep(0.001)
                    return state, generation_open, False, True
                state = self.execute(self.plan, state)
                if state.pending_evaluations:
                    if state is before:
                        time.sleep(0.001)
                    return state, generation_open, False, True

        if generation_open:
            self.finish_generation(state)
            return state, False, True, scheduler is None
        return state, generation_open, False, scheduler is None

    def is_terminated(self, state: OptimizationState) -> bool:
        return self.optimizer.termination.is_terminated(state)

    def dispatch(self, event: Event) -> None:
        self.optimizer.dispatch(event)

    def finish_generation(self, state: OptimizationState) -> None:
        handler = state.problem.handler
        handler.on_generation_end(state.gen, state.population)
        threshold = state.problem.handler.feasibility_threshold
        state.comparator.eps_cv = threshold
        state.pareto_archive.eps_cv = threshold
        manager = getattr(self.optimizer, "surrogate_manager", None)
        if manager is not None:
            manager.on_generation_end(state.gen, state.archive, state)
        self.dispatch(GenerationEndEvent(ctx=state))

    def fatal(self, state: OptimizationState) -> None:
        if state.async_fatal:
            raise EvaluationFatalError(
                str(state.async_fatal.get("reason", "async fatal")), state
            )


class PipelineRuntime:
    """Drive synchronous lifecycle and expose one observable step per generation."""

    capabilities = frozenset()

    def __init__(self, environment: RuntimeEnvironment | None = None) -> None:
        self.environment = environment

    def initialize(
        self, plan: ExecutablePlan, state: OptimizationState
    ) -> RuntimeSession:
        """Create a lifecycle session and dispatch the run-start boundary."""
        if not isinstance(plan, ExecutablePlan):
            raise ValidationError(
                "PipelineRuntime.initialize requires an ExecutablePlan"
            )
        if not isinstance(state, OptimizationState):
            raise ValidationError(
                "PipelineRuntime.initialize requires an OptimizationState"
            )
        sequential = SequentialPlan.from_executable_plan(plan)
        capabilities = getattr(self.environment, "capabilities", self.capabilities)
        if not sequential.accepts(capabilities):
            missing = sequential.required_runtime_capabilities - capabilities
            names = ", ".join(sorted(missing))
            raise ValidationError(
                f"PipelineRuntime lacks required capabilities: {names}"
            )
        if self.environment is None:
            return RuntimeSession(
                plan=sequential, state=state, observable=True, generation_open=False
            )
        self._prepare_state(state)
        self.environment.dispatch(RunStartEvent(ctx=state))
        return RuntimeSession(
            plan=sequential,
            state=state,
            observable=True,
            generation_open=bool(state.pending_evaluations),
        )

    def advance(self, session: RuntimeSession) -> RuntimeStep:
        """Advance lifecycle state by one runtime-owned step."""
        if not isinstance(session, RuntimeSession):
            raise ValidationError("PipelineRuntime.advance requires a RuntimeSession")
        if not isinstance(session.plan, SequentialPlan):
            raise ValidationError(
                "PipelineRuntime.advance requires a SequentialPlan session"
            )
        if self.environment is None:
            state = self._execute_plan(session.plan, session.state)
            next_session = RuntimeSession(
                plan=session.plan,
                state=state,
                step_index=session.step_index + 1,
                observable=True,
            )
            return RuntimeStep(
                state=state,
                node_results=tuple(
                    NodeResult(patch=StatePatch(writes={}), status=NodeStatus.COMPLETED)
                    for _ in session.plan.execution_nodes
                ),
                executed_node_ids=tuple(
                    node.component_id for node in session.plan.execution_nodes
                ),
                observable=True,
                session=next_session,
            )

        env = self.environment
        env.fatal(session.state)
        state = session.state
        generation_open = session.generation_open
        if state.pending_evaluations:
            state, generation_open, generation_finished, continue_loop = (
                env.process_pending(state, generation_open)
            )
            if generation_finished:
                return self._step(session, state, generation_open, observable=True)
            if continue_loop:
                return self._step(session, state, generation_open, observable=False)
        if env.is_terminated(state):
            env.dispatch(RunEndEvent(ctx=state))
            return self._step(session, state, generation_open, finished=True)
        env.dispatch(GenerationStartEvent(ctx=state))
        generation_open = True
        state = env.execute(session.plan, state)
        if not state.pending_evaluations:
            env.finish_generation(state)
            generation_open = False
            return self._step(session, state, generation_open, observable=True)
        return self._step(session, state, generation_open, observable=False)

    def _step(
        self,
        session: RuntimeSession,
        state: OptimizationState,
        generation_open: bool,
        *,
        observable: bool = False,
        finished: bool = False,
    ) -> RuntimeStep:
        next_session = RuntimeSession(
            plan=session.plan,
            state=state,
            finished=finished,
            observable=observable,
            step_index=session.step_index + 1,
            generation_open=generation_open,
        )
        return RuntimeStep(
            state=state,
            observable=observable,
            finished=finished,
            session=next_session,
        )

    @staticmethod
    def _execute_plan(
        plan: SequentialPlan, state: OptimizationState
    ) -> OptimizationState:
        return _execute_sequential_plan(plan, state)

    @staticmethod
    def _prepare_state(state: OptimizationState) -> None:
        threshold = state.problem.handler.feasibility_threshold
        state.comparator.eps_cv = threshold
        state.pareto_archive.eps_cv = threshold
        if (
            isinstance(state.comparator, NSGA3Comparator)
            and state.comparator._rng is None
        ):
            state.comparator.rng = state.rng.spawn(1)[0]


def create_runtime(optimizer: object) -> ExecutionRuntime:
    """Create the default runtime without exposing a concrete type to Runner."""
    plan = resolve_plan(optimizer)
    return PipelineRuntime(
        _OptimizerEnvironment(optimizer, SequentialPlan.from_executable_plan(plan))
    )


def resolve_plan(optimizer: object) -> ExecutablePlan:
    """Return the optimizer plan, compiling only for legacy bare providers."""
    plan = getattr(optimizer, "executable_plan", None)
    if isinstance(plan, ExecutablePlan):
        return plan
    strategy = getattr(optimizer, "strategy", None)
    build_graph = getattr(strategy, "build_graph", None)
    if not callable(build_graph):
        raise ValidationError("default runtime requires an executable plan")
    graph = build_graph(optimizer)
    problem = getattr(optimizer, "problem", None)
    context = None
    if problem is not None:
        context = CompileContext(
            space=problem.space,
            problem=problem,
            offered_runtime_capabilities=frozenset(),
        )
    return Compiler().compile(graph, context)
