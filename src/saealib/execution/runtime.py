"""Runtime implementations and the legacy optimizer environment bridge."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol, cast

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
from saealib.core.contracts.execution import RuntimeCapability
from saealib.core.contracts.vocabulary import validate_name
from saealib.core.graph_builder import build_decomposed_component_graph
from saealib.core.runtime import (
    ExecutionRuntime,
    NodeResult,
    NodeStatus,
    RuntimeSession,
    RuntimeStep,
    SequentialPlan,
    validate_plan_contracts,
)
from saealib.core.state import OPTIMIZATION_STATE_INITIAL_KEYS
from saealib.core.state.patch import StatePatch
from saealib.exceptions import ConfigurationError, EvaluationFatalError, ValidationError
from saealib.stages import AsyncEvaluationSubmitStage

__all__ = [
    "AsyncPipelineRuntime",
    "PipelineRuntime",
    "RuntimeFactory",
    "RuntimeRegistration",
    "RuntimeRegistry",
    "create_runtime",
    "default_runtime_registry",
    "execute_strategy_step",
    "resolve_plan",
]


class RuntimeEnvironment(Protocol):
    """Minimal provider boundary needed by a lifecycle runtime."""

    def execute(
        self, plan: SequentialPlan, state: OptimizationState
    ) -> OptimizationState: ...

    def is_terminated(self, state: OptimizationState) -> bool: ...

    def dispatch(self, event: Event) -> None: ...

    def finish_generation(self, state: OptimizationState) -> None: ...

    def fatal(self, state: OptimizationState) -> None: ...


class AsyncRuntimeEnvironment(RuntimeEnvironment, Protocol):
    """Provider seam used by :class:`AsyncPipelineRuntime`."""

    def execute_async(
        self, plan: SequentialPlan, state: OptimizationState
    ) -> OptimizationState: ...

    def reattach(self, state: OptimizationState) -> OptimizationState: ...

    def poll(self, state: OptimizationState) -> OptimizationState: ...

    def can_refill(self, state: OptimizationState) -> bool: ...


def _execute_sequential_plan(
    plan: SequentialPlan, state: OptimizationState
) -> OptimizationState:
    """Thread state through the compiled StageNodeAdapter sequence."""
    for node, execute in zip(plan.execution_nodes, plan._execute_targets):
        state = execute(state)
        if not isinstance(state, OptimizationState):
            raise ValidationError(
                f"Stage node {node.component_id!r} did not return an OptimizationState"
            )
    return state


def _algorithm_runtime_capabilities(
    optimizer: object,
) -> frozenset[RuntimeCapability]:
    """Return capabilities explicitly required by the configured algorithm."""
    algorithm = getattr(optimizer, "algorithm", None)
    contract_method = getattr(algorithm, "contract", None)
    contract = contract_method() if callable(contract_method) else None
    execution = getattr(contract, "execution", None)
    required = getattr(execution, "required_runtime_capabilities", ())
    if "partial_feedback" in required:
        return frozenset({"partial_feedback"})
    return frozenset()


class _OptimizerEnvironment:
    """Runtime environment adapter for optimizer-owned services.

    This is intentionally outside Runner.  L4 can replace this adapter with an
    async environment without changing the runtime protocol or facade.
    """

    def __init__(self, optimizer: Any, plan: SequentialPlan) -> None:
        self.optimizer = optimizer
        self.plan = plan
        self._execution_fingerprint = self._fingerprint()
        self.capabilities = _algorithm_runtime_capabilities(optimizer)
        scheduler = getattr(optimizer, "async_evaluation_scheduler", None)
        if scheduler is not None and any(
            getattr(insertion, "adapter_name", None) == "feedback_accumulator"
            for insertion in plan.plan.inserted_adapters
        ):
            # The Phase 7 graph contains the accumulator delivery boundary;
            # this seam only enables its stateful async runtime service.
            scheduler.enable_feedback_accumulator()

    def execute(
        self, plan: SequentialPlan, state: OptimizationState
    ) -> OptimizationState:
        self._refresh_plan_if_needed()
        return _execute_sequential_plan(self.plan, state)

    def _fingerprint(self) -> tuple[object, ...]:
        """Capture provider and strategy inputs that shape the executable graph."""
        strategy = getattr(self.optimizer, "strategy", None)
        strategy_values = (
            tuple(
                (name, repr(value))
                for name, value in vars(strategy).items()
                if name != "pipeline"
            )
            if strategy is not None
            else ()
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

    def _refresh_plan_if_needed(self) -> None:
        """Recompile through the runtime when graph-shaping inputs changed."""
        current_fingerprint = self._fingerprint()
        if current_fingerprint == self._execution_fingerprint:
            return
        strategy = getattr(self.optimizer, "strategy", None)
        build_graph = getattr(strategy, "build_graph", None)
        problem = getattr(self.optimizer, "problem", None)
        if not callable(build_graph) or problem is None:
            raise ValidationError(
                "runtime plan refresh requires strategy.build_graph and problem"
            )
        graph = build_graph(self.optimizer)
        executable = Compiler().compile(
            graph,
            CompileContext(
                space=problem.space,
                problem=problem,
                offered_runtime_capabilities=default_runtime_registry.offered_capabilities(
                    self.optimizer
                ),
                initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
            ),
        )
        validate_plan_contracts(executable)
        self.plan = SequentialPlan.from_executable_plan(executable)
        if hasattr(self.optimizer, "_executable_plan"):
            self.optimizer._executable_plan = executable
        self._execution_fingerprint = current_fingerprint

    def execute_async(
        self, plan: SequentialPlan, state: OptimizationState
    ) -> OptimizationState:
        """Execute the canonical graph through the async runtime dispatcher.

        The graph contains the synchronous evaluation contract for both
        modes.  Async runtime owns submission and scheduler feedback delivery;
        the graph's synchronous submit/collect/apply/feedback/tell tail is not
        executed, preventing a second tell for the same proposal.
        """
        scheduler = getattr(self.optimizer, "async_evaluation_scheduler", None)
        if scheduler is None:
            raise ValidationError("Async runtime requires an evaluation scheduler")
        self._refresh_plan_if_needed()
        nodes = self.plan.execution_nodes
        plan_index = next(
            (
                index
                for index, node in enumerate(nodes)
                if getattr(getattr(node.component, "stage", None), "name", None)
                == "evaluation_plan"
            ),
            None,
        )
        if plan_index is None:
            # Keep the runtime seam useful for small custom plans that do not
            # contain the optimization evaluation protocol.
            return _execute_sequential_plan(self.plan, state)

        current = state
        plan_state = current.evaluation_plan_state
        plan_is_terminal = (
            current.evaluation_plan is not None
            and plan_state is not None
            and all(
                int(item.request_id)
                in set(plan_state.completed) | set(plan_state.acknowledged)
                for item in current.evaluation_plan.requests
            )
        )
        if current.evaluation_plan is None or plan_is_terminal:
            for execute in self.plan._execute_targets[:plan_index]:
                current = execute(current)

        plan_stage = getattr(nodes[plan_index].component, "stage", None)
        planner = getattr(plan_stage, "_planner", None)
        if planner is None:
            raise ValidationError("evaluation_plan node has no planner")
        strategy = self.optimizer.strategy
        builder = getattr(self.optimizer, "feedback_builder", None)
        if builder is None:
            builder = getattr(strategy, "feedback_builder", None)
        cbmanager = getattr(self.optimizer, "cbmanager", None)
        async_submit = AsyncEvaluationSubmitStage(
            scheduler,
            planner,
            builder,
            getattr(self.optimizer, "algorithm", None),
            cbmanager,
        )
        return async_submit.execute(current)

    def reattach(self, state: OptimizationState) -> OptimizationState:
        scheduler = getattr(self.optimizer, "async_evaluation_scheduler", None)
        if scheduler is None:
            raise ValidationError("Async runtime requires an evaluation scheduler")
        return scheduler.reattach(state)

    def poll(self, state: OptimizationState) -> OptimizationState:
        scheduler = getattr(self.optimizer, "async_evaluation_scheduler", None)
        if scheduler is None:
            raise ValidationError("Async runtime requires an evaluation scheduler")
        return scheduler.poll(state, wait=False)

    def can_refill(self, state: OptimizationState) -> bool:
        scheduler = getattr(self.optimizer, "async_evaluation_scheduler", None)
        if scheduler is None:
            raise ValidationError("Async runtime requires an evaluation scheduler")
        return len(state.pending_evaluations) < scheduler.max_pending

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


def execute_strategy_step(
    strategy: object, state: OptimizationState, provider: object
) -> OptimizationState:
    """Compatibility step facade with async lifecycle owned by Runtime.

    This supports callers that still invoke ``strategy.step`` directly.  The
    strategy contributes only its canonical pipeline; scheduler polling,
    capacity checks, and async submission remain runtime responsibilities.
    """
    scheduler = getattr(provider, "async_evaluation_scheduler", None)
    build_pipeline = cast(Any, getattr(strategy, "build_pipeline", None))
    if not callable(build_pipeline):
        raise ValidationError("strategy requires a callable build_pipeline")
    if scheduler is None:
        pipeline = build_pipeline(provider)
        setattr(strategy, "pipeline", pipeline)
        return pipeline.execute(state)
    if state.pending_evaluations:
        state = scheduler.poll(state, wait=False)
        if (
            not state.pending_evaluations
            or len(state.pending_evaluations) >= scheduler.max_pending
        ):
            return state
    pipeline = build_pipeline(provider)
    graph = build_decomposed_component_graph(pipeline)
    plan = Compiler().compile(
        graph,
        CompileContext(
            space=state.problem.space,
            problem=state.problem,
            offered_runtime_capabilities=frozenset({"partial_feedback"}),
            initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
        ),
    )
    optimizer = SimpleNamespace(
        strategy=strategy,
        problem=state.problem,
        algorithm=getattr(provider, "algorithm", None),
        evaluator=getattr(provider, "evaluator", None),
        feedback_builder=getattr(provider, "feedback_builder", None),
        cbmanager=getattr(provider, "cbmanager", None),
        async_evaluation_scheduler=scheduler,
    )
    setattr(strategy, "pipeline", pipeline)
    environment = _OptimizerEnvironment(
        optimizer, SequentialPlan.from_executable_plan(plan)
    )
    return environment.execute_async(environment.plan, state)


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
        validate_plan_contracts(plan)
        services: dict[str, object] = {}
        for node in plan.graph.nodes:
            required = {
                requirement.name
                for port in node.contract.ports.values()
                for spec in (port.inputs + port.outputs)
                for requirement in spec.required_services
            }
            if not required <= set(node.resolved_services):
                missing = ", ".join(sorted(required - set(node.resolved_services)))
                raise ValidationError(
                    f"compiled plan node {node.component_id!r} is missing resolved "
                    f"services: {missing}"
                )
            for name, service in node.resolved_services.items():
                previous = services.get(name)
                if previous is not None and previous is not service:
                    raise ValidationError(
                        f"compiled plan resolves service {name!r} to "
                        "conflicting objects"
                    )
                services[name] = service
        state.bind_compiled_services(services)
        sequential = SequentialPlan.from_executable_plan(plan)
        capabilities = self.capabilities | frozenset(
            getattr(self.environment, "capabilities", ())
        )
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
            state = env.execute(session.plan, state)
            if state.pending_evaluations:
                return self._step(session, state, generation_open)
            if generation_open:
                env.finish_generation(state)
                return self._step(session, state, False, observable=True)
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


class AsyncPipelineRuntime(PipelineRuntime):
    """Drive pending asynchronous evaluation lifecycle through a scheduler."""

    capabilities = frozenset({"partial_feedback"})

    def advance(self, session: RuntimeSession) -> RuntimeStep:
        """Poll, refill, drain, and expose asynchronous generation boundaries."""
        if not isinstance(session, RuntimeSession):
            raise ValidationError(
                "AsyncPipelineRuntime.advance requires a RuntimeSession"
            )
        if not isinstance(session.plan, SequentialPlan):
            raise ValidationError(
                "AsyncPipelineRuntime.advance requires a SequentialPlan session"
            )
        if self.environment is None:
            raise ValidationError("AsyncPipelineRuntime requires a runtime environment")
        env = cast(AsyncRuntimeEnvironment, self.environment)
        env.fatal(session.state)
        state = session.state
        generation_open = session.generation_open

        if state.pending_evaluations:
            if set(state.pending_evaluations) != set(state.evaluation_handles):
                state = env.reattach(state)
            before = state
            state = env.poll(state)
            if state.pending_evaluations:
                if env.is_terminated(state):
                    if state is before:
                        time.sleep(0.001)
                    return self._step(session, state, generation_open)
                if not env.can_refill(state):
                    return self._step(session, state, generation_open)
                state = env.execute_async(session.plan, state)
                if state.pending_evaluations:
                    if state is before:
                        time.sleep(0.001)
                    return self._step(session, state, generation_open)

            if generation_open:
                env.finish_generation(state)
                return self._step(session, state, False, observable=True)

        if env.is_terminated(state):
            env.dispatch(RunEndEvent(ctx=state))
            return self._step(session, state, generation_open, finished=True)
        env.dispatch(GenerationStartEvent(ctx=state))
        state = env.execute_async(session.plan, state)
        generation_open = True
        if not state.pending_evaluations:
            env.finish_generation(state)
            generation_open = False
            return self._step(session, state, generation_open, observable=True)
        return self._step(session, state, generation_open)


class RuntimeFactory(Protocol):
    """Factory protocol for runtimes selected by a provider registry."""

    def __call__(self, optimizer: object, plan: ExecutablePlan) -> ExecutionRuntime:
        """Create a runtime for an already resolved plan."""
        ...


@dataclass(frozen=True, kw_only=True)
class RuntimeRegistration:
    """One explicit runtime selection rule."""

    name: str
    matches: Callable[[object], bool]
    factory: RuntimeFactory
    offered_runtime_capabilities: frozenset[RuntimeCapability] = frozenset()
    capability_provider: Callable[[object], Iterable[RuntimeCapability]] | None = None

    def __post_init__(self) -> None:
        """Normalize the provider's effective runtime offer."""
        if self.capability_provider is not None and not callable(
            self.capability_provider
        ):
            raise ValidationError("capability_provider must be callable")
        capabilities = frozenset(self.offered_runtime_capabilities)
        for capability in capabilities:
            validate_name(capability)
        object.__setattr__(
            self,
            "offered_runtime_capabilities",
            capabilities,
        )

    def offered_capabilities(self, optimizer: object) -> frozenset[RuntimeCapability]:
        """Resolve this provider's static or optimizer-dependent offer."""
        values = (
            self.capability_provider(optimizer)
            if self.capability_provider is not None
            else self.offered_runtime_capabilities
        )
        capabilities = frozenset(values)
        for capability in capabilities:
            validate_name(capability)
        return capabilities


class RuntimeRegistry:
    """Runtime selection errors on conflict; compiler rules do not select a runtime."""

    def __init__(self, registrations: tuple[RuntimeRegistration, ...] = ()) -> None:
        self._registrations: dict[str, RuntimeRegistration] = {}
        for registration in registrations:
            self.register(registration)

    def register(self, registration: RuntimeRegistration) -> None:
        """Add a uniquely named runtime provider."""
        if registration.name in self._registrations:
            raise ValidationError(
                f"runtime registration {registration.name!r} already exists"
            )
        if not callable(registration.matches) or not callable(registration.factory):
            raise ValidationError(
                "runtime registration requires callable matches and factory"
            )
        self._registrations[registration.name] = registration

    def unregister(self, name: str) -> None:
        """Remove the runtime provider registered under ``name``."""
        if name not in self._registrations:
            raise ValidationError(f"runtime registration {name!r} is not registered")
        del self._registrations[name]

    def replace(self, name: str, registration: RuntimeRegistration) -> None:
        """Replace the uniquely named runtime provider under ``name``."""
        if name not in self._registrations:
            raise ValidationError(f"runtime registration {name!r} is not registered")
        if not callable(registration.matches) or not callable(registration.factory):
            raise ValidationError(
                "runtime registration requires callable matches and factory"
            )
        if registration.name != name and registration.name in self._registrations:
            raise ValidationError(
                f"runtime registration {registration.name!r} already exists"
            )
        del self._registrations[name]
        self._registrations[registration.name] = registration

    def _resolve(self, optimizer: object) -> RuntimeRegistration:
        matches = tuple(
            registration
            for registration in self._registrations.values()
            if registration.matches(optimizer)
        )
        registered_names = ", ".join(repr(name) for name in self._registrations)
        if not matches:
            names = registered_names or "(none)"
            raise ConfigurationError(
                "no registered runtime matches the optimizer; "
                f"registered runtime names: {names}"
            )
        if len(matches) > 1:
            names = ", ".join(repr(registration.name) for registration in matches)
            raise ConfigurationError(
                "multiple registered runtimes match the optimizer; "
                f"conflicting registration names: {names}"
            )
        return matches[0]

    def create(self, optimizer: object, plan: ExecutablePlan) -> ExecutionRuntime:
        """Select and instantiate the unique matching provider."""
        return self._resolve(optimizer).factory(optimizer, plan)

    def offered_capabilities(self, optimizer: object) -> frozenset[RuntimeCapability]:
        """Return the effective offer from the selected runtime provider."""
        return self._resolve(optimizer).offered_capabilities(optimizer)

    def registrations(self) -> tuple[RuntimeRegistration, ...]:
        """Return registered providers in registration order."""
        return tuple(self._registrations.values())


def _sync_runtime_factory(optimizer: object, plan: ExecutablePlan) -> ExecutionRuntime:
    return PipelineRuntime(
        _OptimizerEnvironment(optimizer, SequentialPlan.from_executable_plan(plan))
    )


def _async_runtime_factory(optimizer: object, plan: ExecutablePlan) -> ExecutionRuntime:
    return AsyncPipelineRuntime(
        _OptimizerEnvironment(optimizer, SequentialPlan.from_executable_plan(plan))
    )


def _sync_runtime_capability_provider(
    optimizer: object,
) -> frozenset[RuntimeCapability]:
    """Offer partial feedback only when the algorithm contract requires it."""
    return _algorithm_runtime_capabilities(optimizer)


default_runtime_registry = RuntimeRegistry(
    (
        RuntimeRegistration(
            name="async",
            matches=lambda optimizer: getattr(
                optimizer, "async_evaluation_scheduler", None
            )
            is not None,
            factory=_async_runtime_factory,
            offered_runtime_capabilities=AsyncPipelineRuntime.capabilities,
        ),
        RuntimeRegistration(
            name="sync",
            matches=lambda optimizer: getattr(
                optimizer, "async_evaluation_scheduler", None
            )
            is None,
            factory=_sync_runtime_factory,
            offered_runtime_capabilities=PipelineRuntime.capabilities,
            capability_provider=_sync_runtime_capability_provider,
        ),
    )
)


def create_runtime(optimizer: object) -> ExecutionRuntime:
    """Create a runtime through the replaceable default provider registry."""
    plan = resolve_plan(optimizer)
    return default_runtime_registry.create(optimizer, plan)


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
            offered_runtime_capabilities=default_runtime_registry.offered_capabilities(
                optimizer
            ),
            initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
        )
    return Compiler().compile(graph, context)
