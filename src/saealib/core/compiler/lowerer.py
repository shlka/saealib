"""Lower structured pipeline values into an immutable :class:`StructuredGraph`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast

from saealib.core.compiler.graph import (
    ComponentNode,
    ControlEdge,
    NodeRef,
    StateBinding,
)
from saealib.core.compiler.regions import (
    BranchRegion,
    LoopRegion,
    RegionNode,
    RepeatRegion,
    SequenceRegion,
    StructuredRegion,
    compose_effects,
)
from saealib.core.compiler.structured import StructuredGraph
from saealib.core.contracts.contract import ComponentContract
from saealib.core.contracts.state import StateContract
from saealib.exceptions import ValidationError

__all__ = ["lower_pipeline", "lower_structured"]


def _pipeline_items(value: object) -> tuple[object, ...] | None:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    if type(value).__name__ == "Pipeline" or (
        hasattr(value, "stages") and not callable(getattr(value, "contract", None))
    ):
        return tuple(cast(Any, value).stages)
    return None


def _pipeline_name(value: object) -> str:
    name = getattr(value, "name", "")
    return name if isinstance(name, str) else ""


def _body_items(value: object) -> tuple[object, ...] | None:
    items = _pipeline_items(value)
    if items is not None:
        return items
    if isinstance(value, StructuredGraph):
        return None
    return (value,)


def _lower_sequence(items: tuple[object, ...], namespace: str) -> StructuredGraph:
    nodes: list[ComponentNode] = []
    edges: list[ControlEdge] = []
    region_nodes: list[RegionNode] = []
    operations: list[ComponentNode | RegionNode] = []
    effects: list[StateContract] = []
    bindings: list[StateBinding] = []
    previous: tuple[ComponentNode, ...] = ()
    for index, item in enumerate(items):
        entries: tuple[ComponentNode, ...] = ()
        exits: tuple[ComponentNode, ...] = ()
        item_name = getattr(item, "name", "")
        if isinstance(item, StructuredRegion):
            item_name = item.region_id
        local_id = (
            item_name if isinstance(item_name, str) and item_name else f"node{index}"
        )
        child_namespace = f"{namespace}.{local_id}" if namespace else local_id
        nested_items = _pipeline_items(item)
        structured_kind = getattr(item, "_structured_kind", None)
        if structured_kind == "repeat":
            structured_item = cast(Any, item)
            item = RepeatRegion(
                region_id=local_id,
                namespace=namespace,
                body=structured_item.body,
                count=structured_item.count,
            )
        elif structured_kind == "loop":
            structured_item = cast(Any, item)
            item = LoopRegion(
                region_id=local_id,
                namespace=namespace,
                body=structured_item.body,
                condition=structured_item.condition,
            )
        elif structured_kind == "branch":
            structured_item = cast(Any, item)
            item = BranchRegion(
                region_id=local_id,
                namespace=namespace,
                body=structured_item.then,
                condition=structured_item.condition,
                otherwise=structured_item.else_,
            )
        if isinstance(item, (RepeatRegion, LoopRegion, BranchRegion)):
            nested_items = None
        if nested_items is not None:
            body = _lower_sequence(nested_items, child_namespace)
            region = SequenceRegion(
                region_id=local_id, namespace=namespace, body=body, effect=body.effect
            )
            region_nodes.append(
                RegionNode(region=region, metadata={"kind": "sequence"})
            )
            operations.append(region_nodes[-1])
            nodes.extend(body.nodes)
            edges.extend(body.control_edges)
            bindings.extend(body.state_bindings)
            effects.append(body.effect)
            if body.nodes:
                entries = (body.nodes[0],)
                exits = (body.nodes[-1],)
        elif isinstance(item, StructuredRegion):
            body_items = _body_items(item.body)
            if body_items is None:
                if isinstance(item.body, StructuredGraph):
                    body = item.body
                else:
                    raise ValidationError(
                        "Structured region body must be a sequence or StructuredGraph"
                    )
            else:
                body = _lower_sequence(body_items, child_namespace)
            otherwise = getattr(item, "otherwise", None)
            otherwise_graph: StructuredGraph | None = None
            if otherwise is not None:
                otherwise_items = _body_items(otherwise)
                if otherwise_items is None:
                    if not isinstance(otherwise, StructuredGraph):
                        raise ValidationError(
                            "Structured region alternate body must be a sequence "
                            "or StructuredGraph"
                        )
                    otherwise_graph = otherwise
                else:
                    otherwise_graph = _lower_sequence(
                        otherwise_items, f"{child_namespace}.else"
                    )
            condition = getattr(item, "condition", None)
            condition_contract = (
                condition.contract() if condition is not None else StateContract()
            )
            lowered = item.with_body(
                body,
                effect=compose_effects((body.effect, item.effect, condition_contract)),
            )
            if otherwise_graph is not None:
                lowered = replace(
                    lowered,
                    effect=compose_effects((lowered.effect, otherwise_graph.effect)),
                )
                lowered = replace(lowered, otherwise=otherwise_graph)
            region_nodes.append(
                RegionNode(
                    region=lowered,
                    metadata={
                        "kind": type(item).__name__,
                        "qualified_id": lowered.qualified_id,
                    },
                )
            )
            operations.append(region_nodes[-1])
            nodes.extend(body.nodes)
            edges.extend(body.control_edges)
            bindings.extend(body.state_bindings)
            if otherwise_graph is not None:
                nodes.extend(otherwise_graph.nodes)
                edges.extend(otherwise_graph.control_edges)
                bindings.extend(otherwise_graph.state_bindings)
            effects.append(lowered.effect)
            if otherwise_graph is not None:
                effects.append(otherwise_graph.effect)
            entries = tuple(
                node
                for graph in (body, otherwise_graph)
                if graph is not None and graph.nodes
                for node in (graph.nodes[0],)
            )
            exits = tuple(
                node
                for graph in (body, otherwise_graph)
                if graph is not None and graph.nodes
                for node in (graph.nodes[-1],)
            )
        else:
            contract_method = getattr(item, "contract", None)
            if not callable(contract_method):
                raise ValidationError(
                    "Structured lowering requires components with contract(); "
                    "pipeline values must contain components with contract()"
                )
            contract = contract_method()
            if not isinstance(contract, ComponentContract):
                raise ValidationError(
                    "Component contract() must return ComponentContract"
                )
            node = ComponentNode(component_id=child_namespace, component=item)
            nodes.append(node)
            operations.append(node)
            bindings.extend(
                StateBinding(node=NodeRef(component_id=child_namespace), state_key=key)
                for key in (
                    *contract.state.reads,
                    *contract.state.writes,
                    *contract.state.exports,
                )
            )
            effects.append(contract.state)
            entries = (node,)
            exits = (node,)
        if previous and entries:
            edges.extend(
                ControlEdge(
                    source=NodeRef(component_id=source.component_id),
                    target=NodeRef(component_id=target.component_id),
                )
                for source in previous
                for target in entries
                if source.component_id != target.component_id
            )
        if entries:
            previous = exits
    entry = (NodeRef(component_id=nodes[0].component_id),) if nodes else ()
    return StructuredGraph(
        nodes=tuple(nodes),
        control_edges=tuple(edges),
        state_bindings=tuple(bindings),
        entry_points=entry,
        region_nodes=tuple(region_nodes),
        operations=tuple(operations),
        effect=compose_effects(effects),
    )


def lower_structured(value: object | Sequence[object]) -> StructuredGraph:
    """Recursively lower a Pipeline-like sequence without runtime execution."""
    items = _pipeline_items(value)
    if items is None:
        raise ValidationError("lower_structured requires a Pipeline or sequence")
    return _lower_sequence(items, _pipeline_name(value))


lower_pipeline = lower_structured
