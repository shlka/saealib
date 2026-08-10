"""Static verification of graph state effects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from saealib.core.compiler.diagnostics import ContractPath, Diagnostic, Severity
from saealib.core.compiler.graph import ComponentGraph, ComponentNode, NodeRef
from saealib.core.state.keys import StateKey

if TYPE_CHECKING:
    from saealib.core.compiler.compiler import RuleContext, VerificationResult

__all__ = ["StateEffectRule"]


def _path(node: NodeRef) -> ContractPath:
    return ContractPath(components=(node.component_id,), role=node.role)


def _node_ref(node: ComponentNode) -> NodeRef:
    return NodeRef(component_id=node.component_id, role=node.role)


def _canonical_ref(graph: ComponentGraph, reference: NodeRef) -> NodeRef:
    try:
        node = graph.node_by_id(reference.component_id)
    except KeyError:
        return NodeRef(component_id=reference.component_id)
    return _node_ref(node)


def _reachable(graph: ComponentGraph) -> dict[NodeRef, set[NodeRef]]:
    successors: dict[NodeRef, set[NodeRef]] = {
        _node_ref(node): set() for node in graph.nodes
    }
    for edge in (*graph.data_edges, *graph.control_edges):
        source = _canonical_ref(graph, edge.source)
        target = _canonical_ref(graph, edge.target)
        if source in successors and target in successors:
            successors[source].add(target)
    for source in tuple(successors):
        pending = list(successors[source])
        while pending:
            target = pending.pop()
            for successor in successors.get(target, ()):
                if successor not in successors[source]:
                    successors[source].add(successor)
                    pending.append(successor)
    return successors


def _resolve_keys(
    bindings: tuple[StateKey[object], ...],
    keys: tuple[StateKey[object], ...],
) -> set[StateKey[object]]:
    resolved: set[StateKey[object]] = set()
    for key in keys:
        candidates = [
            binding for binding in bindings if binding.namespace == key.namespace
        ]
        exact = [binding for binding in candidates if binding.name == key.name]
        if len(exact) == 1:
            resolved.add(exact[0])
        elif len(candidates) == 1:
            resolved.add(candidates[0])
        else:
            resolved.add(key)
    return resolved


class StateEffectRule:
    """Verify initialization and ordering of declared state effects."""

    namespace = "core"
    name = "state_effect"
    phase: Literal["verification"] = "verification"

    def apply(self, context: RuleContext) -> VerificationResult:
        """Collect state initialization and ordering diagnostics."""
        from saealib.core.compiler.compiler import VerificationResult

        graph = context.graph
        order = _reachable(graph)
        refs = tuple(_node_ref(node) for node in graph.nodes)
        initial = set(context.compile_context.initial_state_keys)
        reads: dict[NodeRef, set[StateKey[object]]] = {}
        writes: dict[NodeRef, set[StateKey[object]]] = {}
        universe = set(initial)
        for node in graph.nodes:
            ref = _node_ref(node)
            bindings = tuple(
                binding.state_key
                for binding in graph.state_bindings
                if _canonical_ref(graph, binding.node) == ref
            )
            reads[ref] = _resolve_keys(bindings, node.contract.state.reads)
            writes[ref] = _resolve_keys(bindings, node.contract.state.writes)
            exports = _resolve_keys(bindings, node.contract.state.exports)
            universe.update(reads[ref] | writes[ref] | exports)
        universe.update(binding.state_key for binding in graph.state_bindings)
        for node in graph.nodes:
            ref = _node_ref(node)
            if not node.contract.state.reads_enumerable:
                reads[ref] = set(universe)

        findings: list[Diagnostic] = []

        def add(code: str, node: NodeRef, key: StateKey[object], related=()) -> None:
            findings.append(
                Diagnostic(
                    severity=Severity.ERROR,
                    code=code,
                    message=(
                        f"State key {key.namespace}:{key.name}:{key.schema_version} "
                        f"at {_path(node)} has an invalid effect ordering."
                    ),
                    path=_path(node),
                    related=tuple(_path(item) for item in related),
                    resolutions=(
                        "Add an initial state key or an ordering edge, and keep "
                        "the state declaration node-qualified.",
                    ),
                )
            )

        for ref in refs:
            for key in reads[ref]:
                writers = [
                    other for other in refs if key in writes[other] and other != ref
                ]
                if key in initial:
                    continue
                if key in writes[ref] and not writers:
                    add("uninitialized_state_write", ref, key)
                    continue
                if any(ref in order.get(other, set()) for other in writers):
                    continue
                if writers and all(other in order.get(ref, set()) for other in writers):
                    code = "unreachable_state_read"
                else:
                    code = (
                        "concurrent_state_read_write"
                        if writers
                        else "unreachable_state_read"
                    )
                add(code, ref, key, writers)
        for index, left in enumerate(refs):
            for right in refs[index + 1 :]:
                for key in writes[left] & writes[right]:
                    if right not in order.get(left, set()) and left not in order.get(
                        right, set()
                    ):
                        add("concurrent_state_write", left, key, (right,))
        return VerificationResult(diagnostics=tuple(findings))
