"""Structured graph representation retained alongside ordinary graph edges."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from saealib.core.compiler.graph import ComponentGraph, ComponentNode
from saealib.core.compiler.regions import RegionNode, compose_effects
from saealib.core.contracts.state import StateContract
from saealib.exceptions import ValidationError

__all__ = ["StructuredGraph"]


@dataclass(frozen=True, kw_only=True)
class StructuredGraph(ComponentGraph):
    """A frozen :class:`ComponentGraph` retaining structured control regions."""

    region_nodes: tuple[RegionNode, ...] = ()
    # Source-order operations.  Region bodies carry their own operations.
    operations: tuple[ComponentNode | RegionNode, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    effect: StateContract = field(default_factory=StateContract)

    def __post_init__(self) -> None:
        super().__post_init__()
        regions = tuple(self.region_nodes)
        if any(not isinstance(node, RegionNode) for node in regions):
            raise ValidationError(
                "StructuredGraph region_nodes must contain RegionNode values"
            )
        ids = [node.region.qualified_id for node in regions]
        if len(ids) != len(set(ids)):
            raise ValidationError("StructuredGraph region ids must be unique")
        for node in regions:
            if isinstance(node.region.body, StructuredGraph):
                node.region.body.validate()
            if hasattr(node.region, "otherwise") and isinstance(
                node.region.otherwise, StructuredGraph
            ):
                node.region.otherwise.validate()
        if not isinstance(self.effect, StateContract):
            raise ValidationError("StructuredGraph effect must be a StateContract")
        object.__setattr__(self, "region_nodes", regions)
        operations = tuple(self.operations) or (*self.nodes, *regions)
        if any(
            not isinstance(item, (ComponentNode, RegionNode)) for item in operations
        ):
            raise ValidationError(
                "StructuredGraph operations must contain ComponentNode or "
                "RegionNode values"
            )
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def regions(self) -> tuple[RegionNode, ...]:
        """Alias emphasizing that region nodes are structured graph content."""
        return self.region_nodes

    def validate(self) -> None:
        """Raise when this graph or one of its nested bodies is invalid.

        ``ComponentGraph`` deliberately keeps its structural checks diagnostic-
        based.  Structured control, however, is consumed directly by the
        structured runtime, so its operation tree must be complete and lowered
        before a plan can be created.
        """
        ids = tuple(node.component_id for node in self.nodes)
        if len(set(ids)) != len(ids):
            raise ValidationError("StructuredGraph component ids must be unique")

        regions = tuple(self.region_nodes)
        region_ids = tuple(node.region.qualified_id for node in regions)
        if len(set(region_ids)) != len(region_ids):
            raise ValidationError("StructuredGraph region ids must be unique")
        region_set = {id(region) for region in regions}
        node_ids = set(ids)
        operation_component_ids: set[str] = set()
        operation_regions: set[int] = set()
        for operation in self.operations:
            if isinstance(operation, ComponentNode):
                if operation.component_id not in node_ids:
                    raise ValidationError(
                        "StructuredGraph operation component must belong to graph"
                    )
                if operation.component_id in operation_component_ids:
                    raise ValidationError(
                        "StructuredGraph operations must not repeat components"
                    )
                operation_component_ids.add(operation.component_id)
            elif isinstance(operation, RegionNode):
                if id(operation) not in region_set:
                    raise ValidationError(
                        "StructuredGraph operation region must belong to graph"
                    )
                if id(operation) in operation_regions:
                    raise ValidationError(
                        "StructuredGraph operations must not repeat regions"
                    )
                operation_regions.add(id(operation))
            else:
                raise ValidationError(
                    "StructuredGraph operations must contain ComponentNode or "
                    "RegionNode values"
                )

        for region_node in regions:
            region = region_node.region
            body = region.body
            if not isinstance(body, StructuredGraph):
                raise ValidationError(
                    f"StructuredGraph region {region.qualified_id!r} body must "
                    "be a StructuredGraph"
                )
            body.validate()
            otherwise = getattr(region, "otherwise", None)
            if otherwise is not None:
                if not isinstance(otherwise, StructuredGraph):
                    raise ValidationError(
                        f"StructuredGraph region {region.qualified_id!r} alternate "
                        "body must be a StructuredGraph"
                    )
                otherwise.validate()

    @classmethod
    def from_graph(
        cls,
        graph: ComponentGraph,
        *,
        region_nodes: tuple[RegionNode, ...] = (),
        metadata: Mapping[str, object] | None = None,
        effect: StateContract | None = None,
    ) -> StructuredGraph:
        """Promote an ordinary graph while retaining supplied structure."""
        return cls(
            nodes=graph.nodes,
            data_edges=graph.data_edges,
            control_edges=graph.control_edges,
            state_bindings=graph.state_bindings,
            entry_points=graph.entry_points,
            region_nodes=region_nodes,
            operations=tuple(graph.nodes) + tuple(region_nodes),
            metadata={} if metadata is None else metadata,
            effect=compose_effects((effect or StateContract(),)),
        )


def _rebind_nodes(graph: StructuredGraph) -> StructuredGraph:
    nodes_by_id: dict[str, ComponentNode] = {}

    def collect(current: StructuredGraph) -> None:
        for node in current.nodes:
            nodes_by_id.setdefault(node.component_id, node)
        for region_node in current.region_nodes:
            if isinstance(region_node.region.body, StructuredGraph):
                collect(region_node.region.body)
            otherwise = getattr(region_node.region, "otherwise", None)
            if isinstance(otherwise, StructuredGraph):
                collect(otherwise)

    collect(graph)

    def rebind(current: StructuredGraph) -> StructuredGraph:
        regions: list[RegionNode] = []
        for region_node in current.region_nodes:
            region = region_node.region
            body = (
                rebind(region.body)
                if isinstance(region.body, StructuredGraph)
                else region.body
            )
            otherwise = getattr(region, "otherwise", None)
            if isinstance(otherwise, StructuredGraph):
                otherwise = rebind(otherwise)
            if otherwise is not None and hasattr(region, "otherwise"):
                region = replace(region, body=body, otherwise=otherwise)
            else:
                region = replace(region, body=body)
            regions.append(replace(region_node, region=region))
        regions_by_id = {
            region_node.region.qualified_id: region_node for region_node in regions
        }
        operations: list[ComponentNode | RegionNode] = []
        for operation in current.operations:
            if isinstance(operation, ComponentNode):
                operations.append(
                    nodes_by_id.get(operation.component_id, operation)
                )
            else:
                operations.append(
                    regions_by_id.get(operation.region.qualified_id, operation)
                )
        return replace(
            current,
            region_nodes=tuple(regions),
            operations=tuple(operations),
        )

    return rebind(graph)


def _execution_signature(graph: StructuredGraph) -> tuple[object, ...]:
    """Return the executable tree shape, excluding semantic metadata."""

    def region_signature(region_node: RegionNode) -> tuple[object, ...]:
        region = region_node.region
        body = region.body
        otherwise = getattr(region, "otherwise", None)
        return (
            type(region).__name__,
            region.qualified_id,
            _execution_signature(body) if isinstance(body, StructuredGraph) else None,
            _execution_signature(otherwise)
            if isinstance(otherwise, StructuredGraph)
            else None,
        )

    operations = tuple(
        ("component", operation.component_id)
        if isinstance(operation, ComponentNode)
        else ("region", operation.region.qualified_id)
        for operation in graph.operations
    )
    regions = tuple(region_signature(region) for region in graph.region_nodes)
    node_ids = tuple(sorted(node.component_id for node in graph.nodes))
    return (node_ids, operations, regions)
