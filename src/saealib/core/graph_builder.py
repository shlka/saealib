"""Build component graphs from the legacy stage-based pipelines.

The graph builder is deliberately a bridge.  ``Stage`` remains the execution
object used by the existing optimizer, while ``StageNodeAdapter`` gives the
compiler vocabulary a component-shaped view of the components a stage holds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from saealib.core.compiler.graph import (
    ComponentGraph,
    ComponentNode,
    ControlEdge,
    DataEdge,
    NodeRef,
    StateBinding,
)
from saealib.core.contracts import (
    AssumptionSet,
    ComponentContract,
    ExecutionContract,
    LifecycleContract,
    PortContract,
    PortDirection,
    StateContract,
)
from saealib.core.state import StateKey
from saealib.exceptions import ValidationError
from saealib.pipeline import Pipeline, Stage

__all__ = ["StageNodeAdapter", "build_component_graph"]


@dataclass(frozen=True)
class _HeldComponent:
    """A component found on a stage and its path within that stage."""

    path: tuple[str, ...]
    installed_name: str | None
    component: object
    contract: ComponentContract


def _component_contract(value: object) -> ComponentContract | None:
    """Return a component contract without treating a Stage as a component."""
    if isinstance(value, Stage):
        return None
    method = getattr(value, "contract", None)
    if not callable(method):
        return None
    contract = method()
    if not isinstance(contract, ComponentContract):
        raise ValidationError(
            f"{type(value).__name__}.contract() must return ComponentContract"
        )
    return contract


def _path_piece(value: object) -> str:
    """Make an attribute/container path useful in diagnostics and bindings."""
    if isinstance(value, str):
        return value.lstrip("_") or "component"
    return str(value)


def _held_components(stage: Stage) -> tuple[_HeldComponent, ...]:
    """Collect contracts held by *stage*, including nested stage components.

    The walk is intentionally structural rather than based on concrete Stage
    classes.  This keeps custom stages usable and covers the nested component
    collections used by composite surrogate managers.  A Stage's own
    ``contract()`` is never consulted; only its held components contribute.
    """
    found: list[_HeldComponent] = []
    seen: set[int] = set()

    def visit(
        value: object,
        path: tuple[str, ...],
        installed_name: str | None = None,
    ) -> None:
        value_id = id(value)
        if value_id in seen:
            return
        if isinstance(value, Stage):
            seen.add(value_id)
            for index, child in enumerate(value.stages):
                visit(child, (*path, f"stage_{index}"), installed_name)
            attributes = getattr(value, "__dict__", {})
            if isinstance(attributes, Mapping):
                for name, child in attributes.items():
                    if name != "stages":
                        visit(child, (*path, _path_piece(name)), installed_name)
            return
        if isinstance(value, Mapping):
            seen.add(value_id)
            for key, child in value.items():
                visit(
                    child,
                    (*path, _path_piece(key)),
                    _path_piece(key),
                )
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            seen.add(value_id)
            for index, child in enumerate(value):
                visit(child, (*path, f"item_{index}"), installed_name)
            return

        contract = _component_contract(value)
        if contract is not None:
            found.append(
                _HeldComponent(
                    path=path,
                    installed_name=installed_name,
                    component=value,
                    contract=contract,
                )
            )

        # Components such as CompositeSurrogateManager keep named child
        # managers in ordinary attributes.  Walk those attributes as well so
        # each installed surrogate can receive its own graph binding.
        seen.add(value_id)
        attributes = getattr(value, "__dict__", {})
        if isinstance(attributes, Mapping):
            for name, child in attributes.items():
                visit(child, (*path, _path_piece(name)), installed_name)

    visit(stage, (stage.name or type(stage).__name__,))
    return tuple(found)


def _append_unique(values: list[Any], value: Any) -> None:
    """Append a value once while preserving declaration order."""
    if value not in values:
        values.append(value)


def _qualified_role(path: tuple[str, ...], role: str, index: int) -> str:
    """Namespace a colliding role without inventing a new port schema."""
    owner = _path_piece(path[-1] if path else f"component_{index}")
    return f"{owner}:{role}"


def _compose_contracts(held: Sequence[_HeldComponent]) -> ComponentContract:
    """Compose the contracts of the components held by a Stage.

    State, lifecycle, execution, and assumptions are unions.  Ports retain
    their original role when possible; a colliding role is qualified by the
    held-component path.  Held components are not declared as parts because
    they are a collection inside the legacy Stage, not named adapter attrs.
    """
    ports: dict[str, PortContract] = {}
    reads: list[StateKey[object]] = []
    writes: list[StateKey[object]] = []
    exports: list[StateKey[object]] = []
    events = []
    feedback = None
    capabilities: list[str] = []
    offered_capabilities: list[str] = []
    assumptions: dict[str, bool] = {}

    for index, item in enumerate(held, start=1):
        contract = item.contract
        for role, port in contract.ports.items():
            role_name = role
            if role_name in ports:
                if ports[role_name] == port:
                    continue
                role_name = _qualified_role(item.path, role, index)
                while role_name in ports:
                    role_name = _qualified_role(
                        (*item.path, f"held_{index}"), role, index
                    )
            ports[role_name] = port
        for key in contract.state.reads:
            _append_unique(reads, key)
        for key in contract.state.writes:
            _append_unique(writes, key)
        for key in contract.state.exports:
            _append_unique(exports, key)
        for event in contract.lifecycle.events:
            _append_unique(events, event)
        if contract.lifecycle.feedback is not None:
            if feedback is None:
                feedback = contract.lifecycle.feedback
            elif feedback != contract.lifecycle.feedback:
                # A Stage adapter has no lifecycle policy of its own.  A
                # conflicting pair cannot be represented by LifecycleContract;
                # retain the structural union and leave the later lifecycle
                # rule to inspect the underlying parts.
                feedback = None
        for capability in contract.execution.required_runtime_capabilities:
            _append_unique(capabilities, capability)
        for capability in contract.execution.offered_runtime_capabilities:
            _append_unique(offered_capabilities, capability)
        for name, value in contract.assumptions.items():
            if name not in assumptions:
                assumptions[name] = value
            elif assumptions[name] != value:
                # The conservative value keeps a composed contract from
                # claiming an assumption that one of its parts rejects.
                assumptions[name] = False

    return ComponentContract(
        ports=ports,
        lifecycle=LifecycleContract(events=tuple(events), feedback=feedback),
        state=StateContract(
            reads=tuple(reads), writes=tuple(writes), exports=tuple(exports)
        ),
        execution=ExecutionContract(
            required_runtime_capabilities=tuple(capabilities),
            offered_runtime_capabilities=tuple(offered_capabilities),
        ),
        assumptions=AssumptionSet(assumptions),
    )


def _surrogate_key_name(node_id: str, item: _HeldComponent) -> str:
    """Qualify an installed surrogate name by its graph node path."""
    installed_name = item.installed_name
    if installed_name is None:
        ignored = {
            "manager",
            "managers",
            "sm",
            "surrogate",
            "surrogate_manager",
        }
        for piece in reversed(item.path[1:]):
            if piece in ignored or piece.startswith("stage_"):
                continue
            installed_name = piece
            break
    if installed_name is None:
        return node_id
    return f"{node_id}:{installed_name}"


class StageNodeAdapter:
    """Represent a legacy ``Stage`` as a graph component.

    The adapter delegates execution to the unchanged Stage and exposes only a
    composed contract for the components held by that Stage.  It intentionally
    does not declare the Stage's direct ``OptimizationState`` accesses; those
    declarations belong to Phase 7's per-Stage contract work.
    """

    def __init__(self, stage: Stage, *, node_path: str | None = None) -> None:
        if not isinstance(stage, Stage):
            raise ValidationError("StageNodeAdapter stage must be a Stage")
        self.stage = stage
        self.node_path = node_path or stage.name or type(stage).__name__

    def execute(self, state: Any) -> Any:
        """Delegate execution to the existing Stage implementation."""
        return self.stage.execute(state)

    def contract(self) -> ComponentContract:
        """Compose the held components' contracts for this adapter."""
        return _compose_contracts(_held_components(self.stage))

    def state_bindings(self, node: NodeRef | str) -> tuple[StateBinding, ...]:
        """Return node-qualified bindings for exported surrogate state."""
        node_ref = NodeRef.from_value(node)
        bindings: list[StateBinding] = []
        seen: set[StateKey[object]] = set()
        for item in _held_components(self.stage):
            for key in item.contract.state.exports:
                if key.namespace != "surrogates":
                    continue
                qualified = StateKey[object](
                    namespace=key.namespace,
                    name=_surrogate_key_name(node_ref.component_id, item),
                    schema_version=key.schema_version,
                )
                if qualified in seen:
                    continue
                seen.add(qualified)
                bindings.append(StateBinding(node=node_ref, state_key=qualified))
        return tuple(bindings)


_DATA_PORTS: tuple[tuple[str, str, str, str, str, str], ...] = (
    ("ask", "proposer", "genomes", "surrogate_predict", "predictor", "candidates"),
    (
        "ask",
        "proposer",
        "genomes",
        "evaluation_plan",
        "evaluation_planner",
        "candidates",
    ),
    (
        "ask",
        "proposer",
        "genomes",
        "async_evaluation_submit",
        "evaluation_planner",
        "candidates",
    ),
    (
        "surrogate_predict",
        "predictor",
        "prediction",
        "acquisition",
        "acquisition",
        "prediction",
    ),
    (
        "acquisition",
        "acquisition",
        "scores",
        "evaluation_plan",
        "evaluation_planner",
        "acquisition",
    ),
    (
        "acquisition",
        "acquisition",
        "scores",
        "async_evaluation_submit",
        "evaluation_planner",
        "acquisition",
    ),
    (
        "feedback",
        "feedback_builder",
        "feedback",
        "tell",
        "feedback_consumer",
        "offspring",
    ),
)


def _has_port(
    contract: ComponentContract,
    role: str,
    name: str,
    direction: PortDirection,
) -> bool:
    """Return whether a composed role has exactly one matching port."""
    port_contract = contract.ports.get(role)
    if port_contract is None:
        return False
    matches = tuple(
        port
        for port in (*port_contract.inputs, *port_contract.outputs)
        if port.name == name and port.direction is direction
    )
    return len(matches) == 1


def _unique_node_ids(stages: Sequence[Stage]) -> tuple[str, ...]:
    """Give repeated stage names stable, valid graph identities."""
    counts: dict[str, int] = {}
    ids: list[str] = []
    for index, stage in enumerate(stages):
        base = stage.name or f"stage_{index}"
        count = counts.get(base, 0) + 1
        counts[base] = count
        ids.append(base if count == 1 else f"{base}_{count}")
    return tuple(ids)


def build_component_graph(pipeline: Pipeline) -> ComponentGraph:
    """Build a graph with the same top-level stages as *pipeline*.

    Adjacent stages always receive a control edge. Data edges are emitted only
    when both endpoints expose the declared role, name, and direction.
    """
    if not isinstance(pipeline, Pipeline):
        raise ValidationError("build_component_graph requires a Pipeline")
    stages = tuple(pipeline.stages)
    node_ids = _unique_node_ids(stages)
    adapters = tuple(
        StageNodeAdapter(stage, node_path=node_id)
        for stage, node_id in zip(stages, node_ids)
    )
    nodes = tuple(
        ComponentNode(component_id=node_id, component=adapter)
        for node_id, adapter in zip(node_ids, adapters)
    )
    control_edges = tuple(
        ControlEdge(
            source=NodeRef(component_id=node_ids[index]),
            target=NodeRef(component_id=node_ids[index + 1]),
        )
        for index in range(len(node_ids) - 1)
    )

    data_edges: list[DataEdge] = []
    for (
        source_name,
        source_role,
        source_port,
        target_name,
        target_role,
        target_port,
    ) in _DATA_PORTS:
        source_indices = [
            index for index, stage in enumerate(stages) if stage.name == source_name
        ]
        target_indices = [
            index for index, stage in enumerate(stages) if stage.name == target_name
        ]
        for source_index in source_indices:
            for target_index in target_indices:
                if source_index < target_index:
                    source_node = nodes[source_index]
                    target_node = nodes[target_index]
                    if not (
                        _has_port(
                            source_node.contract,
                            source_role,
                            source_port,
                            PortDirection.OUTPUT,
                        )
                        and _has_port(
                            target_node.contract,
                            target_role,
                            target_port,
                            PortDirection.INPUT,
                        )
                    ):
                        continue
                    data_edges.append(
                        DataEdge(
                            source=NodeRef(
                                component_id=node_ids[source_index], role=source_role
                            ),
                            target=NodeRef(
                                component_id=node_ids[target_index], role=target_role
                            ),
                            source_port=source_port,
                            target_port=target_port,
                        )
                    )

    bindings: list[StateBinding] = []
    for node_id, adapter in zip(node_ids, adapters):
        bindings.extend(adapter.state_bindings(node_id))

    return ComponentGraph(
        nodes=nodes,
        data_edges=tuple(data_edges),
        control_edges=control_edges,
        state_bindings=tuple(bindings),
        entry_points=(NodeRef(component_id=node_ids[0]),) if node_ids else (),
    )
