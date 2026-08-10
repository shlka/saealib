"""Resolution of port dataflow in structured graphs."""

from __future__ import annotations

from dataclasses import replace

from saealib.core.compiler.diagnostics import (
    ContractPath,
    Diagnostic,
    Severity,
)
from saealib.core.compiler.graph import DataEdge, NodeRef
from saealib.core.compiler.regions import (
    BranchRegion,
    LoopRegion,
    RegionNode,
    RepeatRegion,
)
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
        nodes_by_id = {node.component_id: node for node in graph.nodes}

        def resolve_target(target, available: set[str]) -> None:
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
                    (nodes_by_id[node_id] for node_id in available),
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

        def sequence(current: StructuredGraph, available: set[str]) -> set[str]:
            current_available = set(available)
            for operation in current.operations:
                if not isinstance(operation, RegionNode):
                    resolve_target(operation, current_available)
                    current_available.add(operation.component_id)
                    continue
                region = operation.region
                body = region.body
                if not isinstance(body, StructuredGraph):
                    continue
                if isinstance(region, RepeatRegion):
                    body_available = sequence(body, current_available)
                    if isinstance(region.count, int) and region.count > 0:
                        current_available.update(body_available)
                    continue
                if isinstance(region, LoopRegion):
                    sequence(body, current_available)
                    continue
                if isinstance(region, BranchRegion):
                    then_available = sequence(body, current_available)
                    otherwise = getattr(region, "otherwise", None)
                    else_available = (
                        sequence(otherwise, current_available)
                        if isinstance(otherwise, StructuredGraph)
                        else set(current_available)
                    )
                    current_available.update(then_available & else_available)
                    continue
                current_available.update(sequence(body, current_available))
            return current_available

        sequence(graph, set())
        if not added:
            return ResolutionResult(graph=graph, diagnostics=tuple(findings))
        return ResolutionResult(
            graph=replace(
                graph,
                data_edges=(*graph.data_edges, *sorted(added, key=_edge_key)),
            ),
            diagnostics=tuple(findings),
        )
