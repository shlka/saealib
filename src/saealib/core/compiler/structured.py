"""Structured graph representation retained alongside ordinary graph edges."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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
        """Raise when this graph or one of its nested bodies is invalid."""
        ids = {node.component_id for node in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValidationError("StructuredGraph component ids must be unique")
        for region_node in self.region_nodes:
            body = region_node.region.body
            if isinstance(body, StructuredGraph):
                body.validate()
            otherwise = getattr(region_node.region, "otherwise", None)
            if isinstance(otherwise, StructuredGraph):
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
