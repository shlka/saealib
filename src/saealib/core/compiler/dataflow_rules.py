"""Resolution of port dataflow in structured graphs."""

from __future__ import annotations

from dataclasses import replace

from saealib.core.compiler.diagnostics import (
    ContractPath,
    Diagnostic,
    Severity,
)
from saealib.core.compiler.graph import DataEdge, NodeRef
from saealib.core.compiler.structured import StructuredGraph
from saealib.core.contracts.ports import (
    PortDirection,
    PortSpec,
    check_port_compatibility,
)

__all__ = ["StructuredDataflowRule"]


def _edge_key(edge: DataEdge) -> str:
    def ref_key(reference: NodeRef) -> str:
        role = f"[{reference.role}]" if reference.role is not None else ""
        return f"{reference.component_id}{role}"

    return (
        f"{ref_key(edge.source)}.{edge.source_port}->"
        f"{ref_key(edge.target)}.{edge.target_port}"
    )


def _port_specs(node, direction: PortDirection) -> tuple[tuple[str, PortSpec], ...]:
    result = [
        (role, port)
        for role, contract in sorted(node.contract.ports.items())
        for port in (
            contract.outputs if direction is PortDirection.OUTPUT else contract.inputs
        )
    ]
    return tuple(sorted(result, key=lambda item: (item[0], item[1].name)))


def _reachable_upstream(graph: StructuredGraph, target: str) -> frozenset[str]:
    predecessors: dict[str, set[str]] = {}
    for edge in graph.control_edges:
        predecessors.setdefault(edge.target.component_id, set()).add(
            edge.source.component_id
        )
    reachable: set[str] = set()
    pending = list(sorted(predecessors.get(target, ())))
    while pending:
        source = pending.pop(0)
        if source in reachable:
            continue
        reachable.add(source)
        pending.extend(sorted(predecessors.get(source, ())))
    return frozenset(reachable)


def _target_is_connected(
    edge: DataEdge, target_id: str, target_role: str, target_port: str
) -> bool:
    return (
        edge.target.component_id == target_id
        and edge.target_port == target_port
        and (edge.target.role == target_role or edge.target.role is None)
    )


class StructuredDataflowRule:
    """Infer compatible data edges between control-ordered component ports."""

    namespace = "core"
    name = "structured_dataflow"
    phase = "resolution"

    def apply(self, context):
        """Add uniquely resolved data edges and report unresolved inputs."""
        from saealib.core.compiler.compiler import ResolutionResult

        graph = context.graph
        if not isinstance(graph, StructuredGraph):
            return ResolutionResult(graph=graph)

        added: list[DataEdge] = []
        findings: list[Diagnostic] = []
        for target in sorted(graph.nodes, key=lambda node: node.component_id):
            upstream = _reachable_upstream(graph, target.component_id)
            connected = graph.data_edges
            for target_role, target_port in _port_specs(target, PortDirection.INPUT):
                if any(
                    _target_is_connected(
                        edge, target.component_id, target_role, target_port.name
                    )
                    for edge in connected
                ):
                    continue
                candidates: list[DataEdge] = []
                for source in sorted(
                    (node for node in graph.nodes if node.component_id in upstream),
                    key=lambda node: node.component_id,
                ):
                    for source_role, source_port in _port_specs(
                        source, PortDirection.OUTPUT
                    ):
                        if check_port_compatibility(
                            source_port, target_port
                        ).compatible:
                            candidates.append(
                                DataEdge(
                                    source=NodeRef(
                                        component_id=source.component_id,
                                        role=source_role,
                                    ),
                                    target=NodeRef(
                                        component_id=target.component_id,
                                        role=target_role,
                                    ),
                                    source_port=source_port.name,
                                    target_port=target_port.name,
                                )
                            )
                candidates = sorted(candidates, key=_edge_key)
                path = ContractPath(
                    components=(target.component_id,),
                    role=target_role,
                    port=target_port.name,
                )
                if len(candidates) == 1:
                    edge = candidates[0]
                    if edge not in (*connected, *added):
                        added.append(edge)
                        context.claim("data_edge", _edge_key(edge))
                elif len(candidates) == 0 and not target_port.optional:
                    findings.append(
                        Diagnostic(
                            severity=Severity.ERROR,
                            code="unresolved_input",
                            message=(
                                f"Input {path} has no compatible upstream producer."
                            ),
                            path=path,
                            resolutions=(
                                "Provide one control-ordered compatible output port.",
                            ),
                        )
                    )
                elif len(candidates) > 1:
                    findings.append(
                        Diagnostic(
                            severity=Severity.ERROR,
                            code="ambiguous_input",
                            message=(
                                f"Input {path} has multiple compatible upstream "
                                "producers."
                            ),
                            path=path,
                            related=tuple(
                                ContractPath(
                                    components=(edge.source.component_id,),
                                    role=edge.source.role,
                                    port=edge.source_port,
                                )
                                for edge in candidates
                            ),
                            resolutions=(
                                "Add an explicit target-port data edge or remove "
                                "the competing producer.",
                            ),
                        )
                    )
        if not added:
            return ResolutionResult(graph=graph, diagnostics=tuple(findings))
        return ResolutionResult(
            graph=replace(
                graph,
                data_edges=(*graph.data_edges, *sorted(added, key=_edge_key)),
            ),
            diagnostics=tuple(findings),
        )
