"""Verification rules for persistence and runtime capability boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from saealib.core.compiler.diagnostics import ContractPath, Diagnostic, Severity
from saealib.core.compiler.graph import ComponentGraph, ComponentNode
from saealib.core.contracts.contract import ComponentContract
from saealib.core.state.keys import StateKey

if TYPE_CHECKING:
    from saealib.core.compiler.compiler import RuleContext, VerificationResult

__all__ = ["PersistenceRule", "RuntimeCompatibilityRule"]


def _node_path(node: ComponentNode, part_path: tuple[str, ...] = ()) -> ContractPath:
    return ContractPath(
        components=(node.component_id, *part_path),
        role=node.role,
    )


def _has_genome_codec(context: RuleContext) -> bool:
    space = context.compile_context.space
    services = getattr(space, "services", space)
    get = getattr(services, "get", None)
    return callable(get) and get("GenomeCodec") is not None


def _exported_state_keys(
    contract: ComponentContract, part_path: tuple[str, ...] = ()
) -> tuple[tuple[tuple[str, ...], StateKey[object]], ...]:
    exports: list[tuple[tuple[str, ...], StateKey[object]]] = []
    for key in contract.state.exports:
        exports.append((part_path, key))
    for part in contract.parts:
        exports.extend(
            _exported_state_keys(
                part.contract,
                (*part_path, part.name),
            )
        )
    return tuple(exports)


def _population_state_names(graph: ComponentGraph) -> dict[ContractPath, set[str]]:
    names: dict[ContractPath, set[str]] = {}
    for node in graph.nodes:
        for part_path, key in _exported_state_keys(node.contract):
            if key.namespace == "populations":
                names.setdefault(_node_path(node, part_path), set()).add(key.name)

    for binding in graph.state_bindings:
        if binding.state_key.namespace == "populations":
            path = ContractPath(
                components=(binding.node.component_id,),
                role=binding.node.role,
            )
            names.setdefault(path, set()).add(binding.state_key.name)
    return names


class PersistenceRule:
    """Verify services needed to persist portable population state."""

    namespace = "core"
    name = "persistence"
    phase: Literal["verification"] = "verification"

    def apply(self, context: RuleContext) -> VerificationResult:
        """Check portable population exports without changing the graph."""
        from saealib.core.compiler.compiler import VerificationResult

        if not context.compile_context.portability_required:
            return VerificationResult()
        if _has_genome_codec(context):
            return VerificationResult()

        exported = _population_state_names(context.graph)
        findings: list[Diagnostic] = []
        for path, state_names in sorted(
            exported.items(), key=lambda item: str(item[0])
        ):
            findings.append(
                Diagnostic(
                    severity=Severity.ERROR,
                    code="missing_genome_codec",
                    message=(
                        f"Node {path} exports population state "
                        f"{', '.join(sorted(state_names))}; "
                        "portable checkpointing requires the 'GenomeCodec' "
                        "service, but the configured space does not offer it."
                    ),
                    path=path,
                    resolutions=(
                        "Register a GenomeCodec with encode/decode in the "
                        "configured space services, or disable portable "
                        "checkpointing for this compile.",
                    ),
                )
            )
        return VerificationResult(diagnostics=tuple(findings))


class RuntimeCompatibilityRule:
    """Verify required runtime capabilities by set containment only."""

    namespace = "core"
    name = "runtime_compatibility"
    phase: Literal["verification"] = "verification"

    def apply(self, context: RuleContext) -> VerificationResult:
        """Report required capabilities absent from the effective offer."""
        from saealib.core.compiler.compiler import VerificationResult

        offered = set(context.compile_context.offered_runtime_capabilities)
        offered_rendered = ", ".join(sorted(offered)) or "none"
        findings: list[Diagnostic] = []
        emitted: set[tuple[ContractPath, str]] = set()
        for node in context.graph.nodes:
            path = _node_path(node)
            required = set(node.contract.execution.required_runtime_capabilities)
            for capability in sorted(required - offered):
                key = (path, capability)
                if key in emitted:
                    continue
                emitted.add(key)
                findings.append(
                    Diagnostic(
                        severity=Severity.ERROR,
                        code="missing_runtime_capability",
                        message=(
                            f"Node {path} requires runtime capability "
                            f"{capability!r}; the effective runtime offers "
                            f"[{offered_rendered}]."
                        ),
                        path=path,
                        resolutions=(
                            f"Provide a runtime offering {capability!r}, or change "
                            "the component's execution contract.",
                        ),
                    )
                )
        return VerificationResult(diagnostics=tuple(findings))
