"""Internal graph traversal helpers for compiler semantic diagnostics."""

from __future__ import annotations

from collections.abc import Collection, Mapping

from saealib.core.compiler.graph import ComponentGraph

__all__ = [
    "data_reachable",
    "data_reachable_consumers",
    "owner_id",
    "owner_node_ids",
    "requires_sequential_decisions",
]


def owner_node_ids(graph: ComponentGraph, owner_id: str) -> frozenset[str]:
    """Return a graph node and all decomposed nodes owned by it."""
    prefix = f"{owner_id}__"
    return frozenset(
        node.component_id
        for node in graph.nodes
        if node.component_id == owner_id or node.component_id.startswith(prefix)
    )


def owner_id(graph: ComponentGraph, component_id: str) -> str:
    """Map a decomposed graph node back to its owning component node."""
    matches = [
        node.component_id
        for node in graph.nodes
        if component_id == node.component_id
        or component_id.startswith(f"{node.component_id}__")
    ]
    return min(matches, key=len) if matches else component_id


def requires_sequential_decisions(acquisition: object) -> bool:
    """Return whether an acquisition tree requires sequential decisions."""
    if getattr(acquisition, "requires_sequential_decisions", False):
        return True
    children = getattr(acquisition, "acquisitions", None)
    if isinstance(children, Mapping):
        return any(requires_sequential_decisions(child) for child in children.values())
    return False


def data_reachable(graph: ComponentGraph, starts: Collection[str]) -> frozenset[str]:
    """Follow data edges from the supplied graph nodes."""
    reachable = set(starts)
    changed = True
    while changed:
        changed = False
        for edge in graph.data_edges:
            source = edge.source.component_id
            target = edge.target.component_id
            if source in reachable and target not in reachable:
                reachable.add(target)
                changed = True
    return frozenset(reachable)


def data_reachable_consumers(
    graph: ComponentGraph,
    starts: Collection[str],
    consumers: Collection[str],
) -> frozenset[str]:
    """Return consumer component IDs reached by data flow from source IDs.

    Both source and consumer IDs may refer to a top-level component node or to
    one of its decomposed part nodes.  Top-level IDs are expanded to their
    owned nodes before traversal so callers do not need to know the graph's
    decomposition detail.
    """
    start_nodes = frozenset(
        node_id for start in starts for node_id in owner_node_ids(graph, start)
    )
    reachable = data_reachable(graph, start_nodes)
    return frozenset(
        consumer_id
        for consumer_id in consumers
        if reachable & owner_node_ids(graph, consumer_id)
    )
