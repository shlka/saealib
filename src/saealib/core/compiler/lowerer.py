"""Lower structured pipeline values into an immutable :class:`StructuredGraph`."""

from __future__ import annotations

from collections.abc import Sequence

from saealib.core.compiler.graph import (
    ComponentNode,
    ControlEdge,
    NodeRef,
    StateBinding,
)
from saealib.core.compiler.regions import (
    RegionNode,
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
    if hasattr(value, "stages") and (
        type(value).__name__ == "Pipeline"
        or not callable(getattr(value, "contract", None))
    ):
        return tuple(value.stages)
    return None


def _pipeline_name(value: object) -> str:
    name = getattr(value, "name", "")
    return name if isinstance(name, str) else ""


def _lower_sequence(items: tuple[object, ...], namespace: str) -> StructuredGraph:
    nodes: list[ComponentNode] = []
    edges: list[ControlEdge] = []
    region_nodes: list[RegionNode] = []
    effects: list[StateContract] = []
    bindings: list[StateBinding] = []
    previous: ComponentNode | None = None
    for index, item in enumerate(items):
        item_name = getattr(item, "name", "")
        if isinstance(item, StructuredRegion):
            item_name = item.region_id
        local_id = (
            item_name if isinstance(item_name, str) and item_name else f"node{index}"
        )
        child_namespace = f"{namespace}.{local_id}" if namespace else local_id
        nested_items = _pipeline_items(item)
        if nested_items is not None:
            body = _lower_sequence(nested_items, child_namespace)
            region = SequenceRegion(
                region_id=local_id, namespace=namespace, body=body, effect=body.effect
            )
            region_nodes.append(
                RegionNode(region=region, metadata={"kind": "sequence"})
            )
            nodes.extend(body.nodes)
            edges.extend(body.control_edges)
            effects.append(body.effect)
            current = body.nodes[-1] if body.nodes else None
        elif isinstance(item, StructuredRegion):
            body_items = _pipeline_items(item.body)
            if body_items is None:
                if isinstance(item.body, StructuredGraph):
                    body = item.body
                else:
                    raise ValidationError(
                        "Structured region body must be a sequence or StructuredGraph"
                    )
            else:
                body = _lower_sequence(body_items, child_namespace)
            condition = getattr(item, "condition", None)
            condition_contract = (
                condition.contract() if condition is not None else StateContract()
            )
            lowered = item.with_body(
                body,
                effect=compose_effects((body.effect, item.effect, condition_contract)),
            )
            region_nodes.append(
                RegionNode(region=lowered, metadata={"kind": type(item).__name__})
            )
            nodes.extend(body.nodes)
            edges.extend(body.control_edges)
            effects.append(lowered.effect)
            current = body.nodes[-1] if body.nodes else None
        else:
            contract_method = getattr(item, "contract", None)
            if not callable(contract_method):
                raise ValidationError(
                    "Structured lowering requires components with contract(); "
                    "legacy Stage values are not supported"
                )
            if hasattr(item, "execute") and type(item).__module__ == "saealib.pipeline":
                raise ValidationError(
                    "Legacy Stage values are not supported by structured lowering"
                )
            contract = contract_method()
            if not isinstance(contract, ComponentContract):
                raise ValidationError(
                    "Component contract() must return ComponentContract"
                )
            node = ComponentNode(component_id=child_namespace, component=item)
            nodes.append(node)
            bindings.extend(
                StateBinding(node=NodeRef(component_id=child_namespace), state_key=key)
                for key in (
                    *contract.state.reads,
                    *contract.state.writes,
                    *contract.state.exports,
                )
            )
            effects.append(contract.state)
            current = node
        if previous is not None and current is not None:
            edges.append(
                ControlEdge(
                    source=NodeRef(component_id=previous.component_id),
                    target=NodeRef(component_id=current.component_id),
                )
            )
        if current is not None:
            previous = current
    entry = (NodeRef(component_id=nodes[0].component_id),) if nodes else ()
    return StructuredGraph(
        nodes=tuple(nodes),
        control_edges=tuple(edges),
        state_bindings=tuple(bindings),
        entry_points=entry,
        region_nodes=tuple(region_nodes),
        effect=compose_effects(effects),
    )


def lower_structured(value: object | Sequence[object]) -> StructuredGraph:
    """Recursively lower a Pipeline-like sequence without runtime execution."""
    items = _pipeline_items(value)
    if items is None:
        raise ValidationError("lower_structured requires a Pipeline or sequence")
    return _lower_sequence(items, _pipeline_name(value))


lower_pipeline = lower_structured
