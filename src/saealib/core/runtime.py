"""Immutable vocabulary at the compiled-plan/runtime boundary.

This module deliberately contains no execution loop.  It defines the values
which a runtime consumes and returns; concrete runtimes are added by later
phase units.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from saealib.callback.events import Event
from saealib.context import OptimizationState
from saealib.core.compiler.compiler import ExecutablePlan
from saealib.core.compiler.graph import ComponentNode
from saealib.core.contracts.execution import RuntimeCapability
from saealib.core.state.patch import StatePatch
from saealib.exceptions import StalePlanError, ValidationError

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "ExecutionRuntime",
    "IssueCandidateIds",
    "NodeResult",
    "NodeStatus",
    "RequestCheckpoint",
    "RequestRecompile",
    "RequestTermination",
    "RuntimeCommand",
    "RuntimeSession",
    "RuntimeStep",
    "SequentialPlan",
]


class NodeStatus(str, Enum):
    """The execution state observed by a plan node."""

    COMPLETED = "completed"
    BLOCKED = "blocked"
    RUNNING = "running"
    FAILED = "failed"
    RECOMPILE_REQUIRED = "recompile_required"


@dataclass(frozen=True, kw_only=True)
class RuntimeCommand:
    """A refusal-capable request from a node to its execution runtime."""

    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate the optional human-readable reason."""
        if self.reason is not None and not isinstance(self.reason, str):
            raise ValidationError("RuntimeCommand reason must be a string or None")


@dataclass(frozen=True, kw_only=True)
class RequestRecompile(RuntimeCommand):
    """Request that a subsequent runtime step use a rebuilt plan."""


@dataclass(frozen=True, kw_only=True)
class IssueCandidateIds(RuntimeCommand):
    """Request allocation of a number of candidate identifiers."""

    count: int

    def __post_init__(self) -> None:
        """Validate that the request asks for a positive number of IDs."""
        super().__post_init__()
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise ValidationError("IssueCandidateIds count must be an integer")
        if self.count < 1:
            raise ValidationError("IssueCandidateIds count must be positive")


@dataclass(frozen=True, kw_only=True)
class RequestTermination(RuntimeCommand):
    """Request that the runtime terminate execution."""


@dataclass(frozen=True, kw_only=True)
class RequestCheckpoint(RuntimeCommand):
    """Request that the runtime create a checkpoint."""


@dataclass(frozen=True, kw_only=True)
class NodeResult:
    """The immutable result envelope returned by one plan node."""

    patch: StatePatch
    events: tuple[Event, ...] = ()
    commands: tuple[RuntimeCommand, ...] = ()
    status: NodeStatus = NodeStatus.COMPLETED

    def __post_init__(self) -> None:
        """Normalize collections and enforce node result invariants."""
        if not isinstance(self.patch, StatePatch):
            raise ValidationError("NodeResult patch must be a StatePatch")
        events = tuple(self.events)
        commands = tuple(self.commands)
        if any(not isinstance(event, Event) for event in events):
            raise ValidationError("NodeResult events must contain Event values")
        if any(not isinstance(command, RuntimeCommand) for command in commands):
            raise ValidationError(
                "NodeResult commands must contain RuntimeCommand values"
            )
        if not isinstance(self.status, NodeStatus):
            raise ValidationError("NodeResult status must be a NodeStatus")
        if self.status is NodeStatus.RECOMPILE_REQUIRED and any(
            isinstance(command, RequestRecompile) for command in commands
        ):
            raise ValidationError(
                "RECOMPILE_REQUIRED cannot be combined with RequestRecompile"
            )
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "commands", commands)


def _unique_control_order(
    node_ids: set[str], successors: dict[str, set[str]], label: str
) -> tuple[str, ...]:
    """Return a control order only when every position is unique."""
    predecessors = {node_id: set() for node_id in node_ids}
    for source in node_ids:
        for target in successors[source]:
            predecessors[target].add(source)
    available = [node_id for node_id, parents in predecessors.items() if not parents]
    if len(available) != 1:
        raise ValidationError(f"SequentialPlan {label} has no unique entry")
    ordered: list[str] = []
    remaining = predecessors
    while available:
        if len(available) != 1:
            raise ValidationError(f"SequentialPlan {label} is ambiguous")
        current = available.pop()
        ordered.append(current)
        for target in successors[current]:
            remaining[target].remove(current)
            if not remaining[target]:
                available.append(target)
    if len(ordered) != len(node_ids):
        raise ValidationError(f"SequentialPlan {label} contains a cycle")
    return tuple(ordered)


@dataclass(frozen=True, kw_only=True)
class SequentialPlan:
    """An immutable ordered view over an :class:`ExecutablePlan`.

    Node ordering is supplied by the graph adapter.  Keeping it explicit
    makes the boundary usable by the later graph adapter without making this
    vocabulary module responsible for graph execution or topological policy.
    """

    plan: ExecutablePlan
    nodes: tuple[ComponentNode, ...]
    execution_nodes: tuple[ComponentNode, ...] = ()
    _execute_targets: tuple[Callable[..., object], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate the immutable ordered view against its compiled plan."""
        if not isinstance(self.plan, ExecutablePlan):
            raise ValidationError("SequentialPlan plan must be an ExecutablePlan")
        nodes = tuple(self.nodes)
        if any(not isinstance(node, ComponentNode) for node in nodes):
            raise ValidationError(
                "SequentialPlan nodes must contain ComponentNode values"
            )
        ids = tuple(node.component_id for node in nodes)
        if len(set(ids)) != len(ids):
            raise ValidationError("SequentialPlan nodes must have unique component IDs")
        graph_ids = {node.component_id for node in self.plan.graph.nodes}
        if any(node_id not in graph_ids for node_id in ids):
            raise ValidationError(
                "SequentialPlan nodes must belong to the executable plan"
            )
        execution_nodes = tuple(self.execution_nodes) or nodes
        execution_ids = tuple(node.component_id for node in execution_nodes)
        if any(not isinstance(node, ComponentNode) for node in execution_nodes):
            raise ValidationError(
                "SequentialPlan execution_nodes must contain ComponentNode values"
            )
        if len(set(execution_ids)) != len(execution_ids):
            raise ValidationError(
                "SequentialPlan execution_nodes must have unique component IDs"
            )
        if any(node_id not in graph_ids for node_id in execution_ids):
            raise ValidationError(
                "SequentialPlan execution_nodes must belong to the executable plan"
            )
        from saealib.core.graph_builder import cached_execution_target

        execute_targets = tuple(
            cached_execution_target(node.component) for node in execution_nodes
        )
        if any(not callable(execute) for execute in execute_targets):
            invalid_node = next(
                node
                for node, execute in zip(execution_nodes, execute_targets)
                if not callable(execute)
            )
            raise ValidationError(
                f"SequentialPlan node {invalid_node.component_id!r} is not executable"
            )
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "execution_nodes", execution_nodes)
        object.__setattr__(self, "_execute_targets", execute_targets)

    @classmethod
    def from_executable_plan(cls, plan: ExecutablePlan) -> SequentialPlan:
        """Build the ordered executable view consumed by sync runtime.

        Stage adapters remain the convenience path for the standard profile.
        A graph containing only framework components is also executable: its
        ``execute(StateView)`` nodes are ordered by the graph's control edges.
        """
        if not isinstance(plan, ExecutablePlan):
            raise ValidationError("SequentialPlan requires an ExecutablePlan")

        # Import locally: graph_builder owns the legacy Stage bridge and also
        # imports compiler vocabulary used while constructing the graph.
        from saealib.core.graph_builder import StageNodeAdapter

        graph = plan.graph
        nodes_by_id = {node.component_id: node for node in graph.nodes}
        stage_ids = {
            node.component_id
            for node in graph.nodes
            if isinstance(node.component, StageNodeAdapter)
        }
        if not graph.entry_points:
            raise ValidationError("SequentialPlan requires a graph entry point")

        control_successors: dict[str, set[str]] = {
            node_id: set() for node_id in nodes_by_id
        }
        for edge in graph.control_edges:
            source = edge.source.component_id
            target = edge.target.component_id
            if source not in nodes_by_id or target not in nodes_by_id:
                raise ValidationError("SequentialPlan control edge has an unknown node")
            control_successors[source].add(target)

        reachable: set[str] = set()
        pending = [entry.component_id for entry in graph.entry_points]
        while pending:
            node_id = pending.pop()
            if node_id not in nodes_by_id:
                raise ValidationError(
                    f"SequentialPlan entry point {node_id!r} is not in the graph"
                )
            if node_id in reachable:
                continue
            reachable.add(node_id)
            pending.extend(control_successors[node_id])

        reachable_stages = stage_ids & reachable

        executable_ids = {
            node_id
            for node_id in reachable
            if callable(getattr(nodes_by_id[node_id].component, "execute", None))
        }

        def executable_successors_for(node_id: str) -> set[str]:
            """Collapse contract-only nodes between executable graph nodes."""
            result: set[str] = set()
            pending_nodes = list(control_successors[node_id])
            visited_non_executables: set[str] = set()
            while pending_nodes:
                target = pending_nodes.pop()
                if target in executable_ids:
                    result.add(target)
                    continue
                if target in visited_non_executables:
                    raise ValidationError(
                        "SequentialPlan executable control order contains a cycle"
                    )
                visited_non_executables.add(target)
                pending_nodes.extend(control_successors[target])
            return result

        if not stage_ids:
            if not executable_ids:
                return cls(plan=plan, nodes=(), execution_nodes=())
            executable_successors = {
                node_id: executable_successors_for(node_id)
                for node_id in executable_ids
            }
            ordered_ids = _unique_control_order(
                executable_ids, executable_successors, "control order"
            )
            ordered_nodes = tuple(nodes_by_id[node_id] for node_id in ordered_ids)
            return cls(
                plan=plan,
                nodes=ordered_nodes,
                execution_nodes=ordered_nodes,
            )

        if not reachable_stages:
            raise ValidationError(
                "SequentialPlan entry points do not reach a StageNodeAdapter"
            )
        if reachable_stages != stage_ids:
            missing = ", ".join(sorted(stage_ids - reachable_stages))
            raise ValidationError(
                f"SequentialPlan has unreachable stage nodes: {missing}"
            )

        def stage_successors(source: str) -> set[str]:
            """Collapse control-only compile nodes between two Stage nodes."""
            result: set[str] = set()
            pending_nodes = list(control_successors[source])
            visited_non_stages: set[str] = set()
            while pending_nodes:
                target = pending_nodes.pop()
                if target in reachable_stages:
                    result.add(target)
                    continue
                if target in visited_non_stages:
                    raise ValidationError(
                        "SequentialPlan control order contains a cycle"
                    )
                visited_non_stages.add(target)
                pending_nodes.extend(control_successors[target])
            return result

        successors = {
            node_id: stage_successors(node_id) for node_id in reachable_stages
        }
        ordered_ids = _unique_control_order(
            reachable_stages, successors, "control order"
        )

        executable_successors = {
            node_id: executable_successors_for(node_id) for node_id in executable_ids
        }
        execution_ids = _unique_control_order(
            executable_ids, executable_successors, "executable control order"
        )
        return cls(
            plan=plan,
            nodes=tuple(nodes_by_id[node_id] for node_id in ordered_ids),
            execution_nodes=tuple(nodes_by_id[node_id] for node_id in execution_ids),
        )

    @property
    def required_runtime_capabilities(self) -> frozenset[RuntimeCapability]:
        """Return the capabilities required by the compiled plan."""
        return self.plan.required_runtime_capabilities

    def accepts(self, capabilities: Iterable[RuntimeCapability]) -> bool:
        """Return whether *capabilities* satisfy the plan's requirements."""
        offered = frozenset(capabilities)
        return self.required_runtime_capabilities <= offered


@dataclass(frozen=True, kw_only=True)
class RuntimeSession:
    """Immutable orchestration snapshot; ``OptimizationState`` is the state carrier."""

    plan: ExecutablePlan | SequentialPlan
    state: OptimizationState
    finished: bool = False
    observable: bool = False
    step_index: int = 0
    generation_open: bool = False

    def __post_init__(self) -> None:
        """Validate the immutable session snapshot."""
        if not isinstance(self.plan, (ExecutablePlan, SequentialPlan)):
            raise ValidationError("RuntimeSession plan must be an executable plan")
        if not isinstance(self.state, OptimizationState):
            raise ValidationError("RuntimeSession state must be an OptimizationState")
        if (
            isinstance(self.finished, bool) is False
            or isinstance(self.observable, bool) is False
        ):
            raise ValidationError(
                "RuntimeSession finished and observable must be booleans"
            )
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int):
            raise ValidationError("RuntimeSession step_index must be an integer")
        if self.step_index < 0:
            raise ValidationError("RuntimeSession step_index must not be negative")
        if not isinstance(self.generation_open, bool):
            raise ValidationError("RuntimeSession generation_open must be a boolean")


@dataclass(frozen=True, kw_only=True)
class RuntimeStep:
    """The outcome of exactly one ``ExecutionRuntime.advance`` call."""

    state: OptimizationState
    node_results: tuple[NodeResult, ...] = ()
    executed_node_ids: tuple[str, ...] = ()
    observable: bool = False
    finished: bool = False
    refused_commands: tuple[RuntimeCommand, ...] = ()
    session: RuntimeSession | None = None

    def __post_init__(self) -> None:
        """Normalize the outcome collections and validate its boundary."""
        if not isinstance(self.state, OptimizationState):
            raise ValidationError("RuntimeStep state must be an OptimizationState")
        results = tuple(self.node_results)
        executed_node_ids = tuple(self.executed_node_ids)
        refused = tuple(self.refused_commands)
        if any(not isinstance(result, NodeResult) for result in results):
            raise ValidationError(
                "RuntimeStep node_results must contain NodeResult values"
            )
        for result in results:
            if result.status is NodeStatus.RECOMPILE_REQUIRED and any(
                isinstance(command, RequestRecompile) for command in result.commands
            ):
                raise ValidationError(
                    "RECOMPILE_REQUIRED cannot be combined with RequestRecompile"
                )
        if any(not isinstance(command, RuntimeCommand) for command in refused):
            raise ValidationError(
                "RuntimeStep refused_commands must contain RuntimeCommand values"
            )
        if any(not isinstance(node_id, str) for node_id in executed_node_ids):
            raise ValidationError("RuntimeStep executed_node_ids must contain strings")
        if not isinstance(self.observable, bool) or not isinstance(self.finished, bool):
            raise ValidationError(
                "RuntimeStep observable and finished must be booleans"
            )
        if self.session is not None and not isinstance(self.session, RuntimeSession):
            raise ValidationError(
                "RuntimeStep session must be a RuntimeSession or None"
            )
        object.__setattr__(self, "node_results", results)
        object.__setattr__(self, "executed_node_ids", executed_node_ids)
        object.__setattr__(self, "refused_commands", refused)

    @property
    def recompile_required(self) -> bool:
        """Whether this step requests the next step-boundary recompile."""
        return any(
            result.status is NodeStatus.RECOMPILE_REQUIRED
            for result in self.node_results
        )


class ExecutionRuntime(Protocol):
    """Protocol consumed by a runner; concrete execution is a later unit."""

    capabilities: frozenset[RuntimeCapability]

    def initialize(
        self, plan: ExecutablePlan, state: OptimizationState
    ) -> RuntimeSession:
        """Create a session from a compiled plan and existing state."""
        ...

    def advance(self, session: RuntimeSession) -> RuntimeStep:
        """Advance exactly one runtime step."""
        ...


def validate_plan_contracts(plan: ExecutablePlan) -> None:
    """Read each compiled node contract once and reject a stale plan."""
    snapshots = dict(plan.contract_snapshots)
    if not snapshots:
        return
    stale_node: str | None = None
    for node in plan.graph.nodes:
        contract_method = getattr(node.component, "contract", None)
        if not callable(contract_method):
            stale_node = stale_node or node.component_id
            continue
        current = contract_method()
        expected = snapshots.get(node.component_id)
        if current != expected:
            stale_node = stale_node or node.component_id
    if stale_node is not None:
        raise StalePlanError(f"stale_plan: node {stale_node!r} contract changed")
