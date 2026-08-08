"""Immutable vocabulary at the compiled-plan/runtime boundary.

This module deliberately contains no execution loop.  It defines the values
which a runtime consumes and returns; concrete runtimes are added by later
phase units.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from saealib.callback.events import Event
from saealib.context import OptimizationState
from saealib.core.compiler.compiler import ExecutablePlan
from saealib.core.compiler.graph import ComponentNode
from saealib.core.contracts.execution import RuntimeCapability
from saealib.core.state.patch import StatePatch
from saealib.exceptions import ValidationError

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


@dataclass(frozen=True, kw_only=True)
class SequentialPlan:
    """An immutable ordered view over an :class:`ExecutablePlan`.

    Node ordering is supplied by the graph adapter.  Keeping it explicit
    makes the boundary usable by the later graph adapter without making this
    vocabulary module responsible for graph execution or topological policy.
    """

    plan: ExecutablePlan
    nodes: tuple[ComponentNode, ...]

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
        object.__setattr__(self, "nodes", nodes)

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


@dataclass(frozen=True, kw_only=True)
class RuntimeStep:
    """The outcome of exactly one ``ExecutionRuntime.advance`` call."""

    state: OptimizationState
    node_results: tuple[NodeResult, ...] = ()
    observable: bool = False
    finished: bool = False
    refused_commands: tuple[RuntimeCommand, ...] = ()
    session: RuntimeSession | None = None

    def __post_init__(self) -> None:
        """Normalize the outcome collections and validate its boundary."""
        if not isinstance(self.state, OptimizationState):
            raise ValidationError("RuntimeStep state must be an OptimizationState")
        results = tuple(self.node_results)
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
        if not isinstance(self.observable, bool) or not isinstance(self.finished, bool):
            raise ValidationError(
                "RuntimeStep observable and finished must be booleans"
            )
        if self.session is not None and not isinstance(self.session, RuntimeSession):
            raise ValidationError(
                "RuntimeStep session must be a RuntimeSession or None"
            )
        object.__setattr__(self, "node_results", results)
        object.__setattr__(self, "refused_commands", refused)


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
