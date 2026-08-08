"""Rule-based graph compilation.

This module deliberately depends only on the graph and contract vocabulary.  Concrete
component packages must remain behind their contracts at the compiler boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol, TypeAlias, TypeVar, cast, runtime_checkable

from saealib.core.compiler.diagnostics import (
    ContractPath,
    Diagnostic,
    DiagnosticBag,
    Severity,
)
from saealib.core.compiler.graph import (
    ComponentGraph,
    ControlEdge,
    DataEdge,
    IdentityRule,
    NodeRef,
    ReachabilityRule,
    StateBinding,
)
from saealib.core.contracts.execution import RuntimeCapability
from saealib.core.contracts.vocabulary import validate_name
from saealib.exceptions import ConfigurationError, ValidationError

Phase = Literal["verification", "resolution"]
_ValueT = TypeVar("_ValueT")


@dataclass(frozen=True, kw_only=True)
class CompileContext:
    """Configuration supplied by the caller, kept separate from rule state."""

    enabled_rule_namespaces: frozenset[str] = frozenset()
    space: object | None = None
    problem: object | None = None
    offered_runtime_capabilities: frozenset[RuntimeCapability] = frozenset()
    portability_required: bool = False

    def __post_init__(self) -> None:
        """Normalize caller-provided collections."""
        namespaces = frozenset(self.enabled_rule_namespaces)
        capabilities = frozenset(self.offered_runtime_capabilities)
        for value in (*namespaces, *capabilities):
            validate_name(value)
        if not isinstance(self.portability_required, bool):
            raise ValidationError("portability_required must be a boolean")
        object.__setattr__(self, "enabled_rule_namespaces", namespaces)
        object.__setattr__(self, "offered_runtime_capabilities", capabilities)


@dataclass(frozen=True, order=True, kw_only=True)
class RewriteClaim:
    """A canonical graph location owned by a resolution proposal.

    The compiler maps ``node`` keys to component ids, ``data_edge`` keys to
    ``source[role].port->target[role].port``, ``control_edge`` keys to
    ``source[role]->target[role]``, ``state_binding`` keys to
    ``node[role]=namespace:name:version``, and ``entry_point`` keys to the
    referenced node (including its optional role).
    """

    kind: str
    key: str

    def __post_init__(self) -> None:
        """Validate the claim location."""
        validate_name(self.kind)
        if not isinstance(self.key, str) or not self.key:
            raise ValidationError("RewriteClaim key must be a non-empty string")

    def __str__(self) -> str:
        """Render the canonical claim location."""
        return f"{self.kind}:{self.key}"


@dataclass
class RuleContext:
    """State and services exposed to one rule invocation."""

    graph: ComponentGraph
    compile_context: CompileContext
    diagnostics: DiagnosticBag
    _declared_claims: set[RewriteClaim] = field(default_factory=set, repr=False)

    def claim(self, kind: str, key: str) -> RewriteClaim:
        """Declare a location this rule's proposed rewrite may change."""
        claim = RewriteClaim(kind=kind, key=key)
        self._declared_claims.add(claim)
        return claim

    @property
    def claims(self) -> frozenset[RewriteClaim]:
        """Return an immutable snapshot of declared claims."""
        return frozenset(self._declared_claims)


@dataclass(frozen=True, kw_only=True)
class VerificationResult:
    """Diagnostics produced by a verification rule."""

    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        """Normalize diagnostics to an immutable tuple."""
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


@dataclass(frozen=True, kw_only=True)
class ResolutionResult:
    """A proposed graph rewrite and its claimed locations."""

    graph: ComponentGraph
    claims: frozenset[RewriteClaim] = frozenset()
    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        """Validate and freeze the result collections."""
        if not isinstance(self.graph, ComponentGraph):
            raise ValidationError("ResolutionResult graph must be a ComponentGraph")
        object.__setattr__(self, "claims", frozenset(self.claims))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


RuleResult: TypeAlias = VerificationResult | ResolutionResult


@runtime_checkable
class CompilationRule(Protocol):
    """Common metadata and application boundary for compiler rules."""

    namespace: str
    name: str
    phase: Phase

    def apply(self, context: RuleContext) -> RuleResult:
        """Apply this rule to the current rule context."""
        ...


@runtime_checkable
class VerificationRule(CompilationRule, Protocol):
    """A rule that only observes the resolved graph."""

    phase: Literal["verification"]

    def apply(self, context: RuleContext) -> VerificationResult:
        """Collect diagnostics without changing the graph."""
        ...


@runtime_checkable
class ResolutionRule(CompilationRule, Protocol):
    """A rule that proposes graph changes through claims."""

    phase: Literal["resolution"]

    def apply(self, context: RuleContext) -> ResolutionResult:
        """Return a claimed graph rewrite."""
        ...


@dataclass(frozen=True, kw_only=True)
class RuleRegistration:
    """Stable metadata for one registered rule."""

    rule: CompilationRule
    namespace: str
    name: str
    phase: Phase


class RuleRegistry:
    """Own rule registrations independently for each compiler instance."""

    def __init__(self, registrations: Iterable[object] = ()) -> None:
        self._registrations: dict[tuple[str, str], RuleRegistration] = {}
        for registration in registrations:
            if isinstance(registration, RuleRegistration):
                self.register(
                    registration.rule,
                    namespace=registration.namespace,
                    name=registration.name,
                    phase=registration.phase,
                )
            else:
                self.register(registration)

    def register(
        self,
        rule: object,
        *,
        namespace: str | None = None,
        name: str | None = None,
        phase: Phase | None = None,
    ) -> RuleRegistration:
        """Register one rule, rejecting duplicate names."""
        resolved_namespace = namespace or getattr(rule, "namespace", None)
        resolved_name = name or getattr(rule, "name", None)
        resolved_phase = phase or getattr(rule, "phase", None)
        if not isinstance(resolved_namespace, str) or not isinstance(
            resolved_name, str
        ):
            raise ValidationError("Rules require namespace and name")
        validate_name(resolved_namespace)
        validate_name(resolved_name)
        if resolved_phase not in ("verification", "resolution"):
            raise ValidationError("Rule phase must be verification or resolution")
        if not callable(getattr(rule, "apply", None)):
            raise ValidationError("Rules must provide apply(RuleContext)")
        key = (resolved_namespace, resolved_name)
        if key in self._registrations:
            raise ConfigurationError(
                f"Rule is already registered: {resolved_namespace}:{resolved_name}"
            )
        registration = RuleRegistration(
            rule=cast(CompilationRule, rule),
            namespace=resolved_namespace,
            name=resolved_name,
            phase=resolved_phase,
        )
        self._registrations[key] = registration
        return registration

    def registrations(self) -> tuple[RuleRegistration, ...]:
        """Return registrations in insertion order."""
        return tuple(self._registrations.values())


def _rule_namespaces(graph: ComponentGraph) -> frozenset[str]:
    values: set[str] = set()

    def add(value: str) -> None:
        if ":" in value:
            values.add(value.split(":", 1)[0])

    for node in graph.nodes:
        add(node.component_id)
        for port in node.contract.ports.values():
            for spec in (*port.inputs, *port.outputs):
                add(spec.data.kind)
                for service in spec.required_services:
                    add(service.name)
        for key in (*node.contract.state.reads, *node.contract.state.writes):
            add(key.namespace)
    for edge in graph.data_edges:
        add(edge.source_port)
        add(edge.target_port)
    return frozenset(values)


def _diagnostic_sort_key(diagnostic: Diagnostic) -> tuple[str, ...]:
    return (
        diagnostic.severity.value,
        diagnostic.code,
        str(diagnostic.path),
        diagnostic.message,
        *diagnostic.resolutions,
    )


def _node_ref_key(reference: NodeRef) -> str:
    """Render the stable identity used for edge and entry-point claims."""
    rendered = reference.component_id
    if reference.role is not None:
        rendered += f"[{reference.role}]"
    return rendered


def _data_edge_key(edge: DataEdge) -> str:
    """Render a data-edge location for claim comparison."""
    return (
        f"{_node_ref_key(edge.source)}.{edge.source_port}->"
        f"{_node_ref_key(edge.target)}.{edge.target_port}"
    )


def _control_edge_key(edge: ControlEdge) -> str:
    """Render a control-edge location for claim comparison."""
    return f"{_node_ref_key(edge.source)}->{_node_ref_key(edge.target)}"


def _state_binding_key(binding: StateBinding) -> str:
    """Render a state-binding location for claim comparison."""
    state_key = binding.state_key
    return (
        f"{_node_ref_key(binding.node)}="
        f"{state_key.namespace}:{state_key.name}:{state_key.schema_version}"
    )


def _changed_locations(
    before: ComponentGraph, after: ComponentGraph
) -> frozenset[RewriteClaim]:
    """Return graph locations whose values differ between two graph snapshots."""

    def compare(
        kind: str,
        before_values: Sequence[_ValueT],
        after_values: Sequence[_ValueT],
        key: Callable[[_ValueT], str],
    ) -> set[RewriteClaim]:
        def grouped(values: Sequence[_ValueT]) -> dict[str, list[_ValueT]]:
            result: dict[str, list[_ValueT]] = {}
            for value in values:
                result.setdefault(key(value), []).append(value)
            return result

        before_groups = grouped(before_values)
        after_groups = grouped(after_values)
        changes = set()
        for location in before_groups.keys() | after_groups.keys():
            if before_groups.get(location, []) != after_groups.get(location, []):
                changes.add(RewriteClaim(kind=kind, key=location))
        return changes

    changes: set[RewriteClaim] = set()
    changes.update(
        compare(
            "node",
            before.nodes,
            after.nodes,
            lambda node: node.component_id,
        )
    )
    changes.update(
        compare(
            "data_edge",
            before.data_edges,
            after.data_edges,
            _data_edge_key,
        )
    )
    changes.update(
        compare(
            "control_edge",
            before.control_edges,
            after.control_edges,
            _control_edge_key,
        )
    )
    changes.update(
        compare(
            "state_binding",
            before.state_bindings,
            after.state_bindings,
            _state_binding_key,
        )
    )
    changes.update(
        compare(
            "entry_point",
            before.entry_points,
            after.entry_points,
            _node_ref_key,
        )
    )
    return frozenset(changes)


def _merge_graphs(
    base: ComponentGraph, proposals: Sequence[ResolutionResult]
) -> ComponentGraph:
    """Merge independent proposals by their graph collections deterministically."""
    graph = base

    def merge_values(
        original: tuple[_ValueT, ...],
        current: tuple[_ValueT, ...],
        candidate: tuple[_ValueT, ...],
    ) -> tuple[_ValueT, ...]:
        """Merge a full collection proposal while preserving replacement slots."""
        merged = list(current)

        def same_slot(left: _ValueT, right: _ValueT) -> bool:
            """Match graph values by stable identity when their payload changes."""
            if type(left) is type(right) and hasattr(left, "component_id"):
                return getattr(left, "component_id") == getattr(right, "component_id")
            return left == right

        matched: dict[int, int] = {}
        used_candidates: set[int] = set()
        for original_index, original_value in enumerate(original):
            for candidate_index, candidate_value in enumerate(candidate):
                if candidate_index in used_candidates:
                    continue
                if same_slot(original_value, candidate_value):
                    matched[original_index] = candidate_index
                    used_candidates.add(candidate_index)
                    break

        for original_index in reversed(range(len(original))):
            if original_index in matched:
                continue
            original_value = original[original_index]
            for current_index, current_value in enumerate(merged):
                if same_slot(original_value, current_value):
                    del merged[current_index]
                    break

        for original_index, candidate_index in matched.items():
            original_value = original[original_index]
            candidate_value = candidate[candidate_index]
            if candidate_value == original_value:
                continue
            for current_index, current_value in enumerate(merged):
                if same_slot(original_value, current_value):
                    merged[current_index] = candidate_value
                    break

        for candidate_index, candidate_value in enumerate(candidate):
            if candidate_index in used_candidates:
                continue
            if not any(same_slot(candidate_value, value) for value in merged):
                merged.append(candidate_value)
        return tuple(merged)

    for result in proposals:
        candidate = result.graph
        graph = replace(
            graph,
            nodes=merge_values(base.nodes, graph.nodes, candidate.nodes),
            data_edges=merge_values(
                base.data_edges, graph.data_edges, candidate.data_edges
            ),
            control_edges=merge_values(
                base.control_edges, graph.control_edges, candidate.control_edges
            ),
            state_bindings=merge_values(
                base.state_bindings, graph.state_bindings, candidate.state_bindings
            ),
            entry_points=merge_values(
                base.entry_points, graph.entry_points, candidate.entry_points
            ),
        )
    return graph


@dataclass(frozen=True, kw_only=True)
class ExecutablePlan:
    """Immutable result of compilation; execution belongs to a later phase."""

    graph: ComponentGraph
    diagnostics: tuple[Diagnostic, ...]
    required_runtime_capabilities: frozenset[RuntimeCapability]
    active_rule_namespaces: frozenset[str]
    active_rule_names: tuple[str, ...]

    def describe(self) -> str:
        """Return a concise human-readable plan summary."""
        codes = ", ".join(d.code for d in self.diagnostics) or "none"
        namespaces = ", ".join(sorted(self.active_rule_namespaces)) or "none"
        capabilities = ", ".join(sorted(self.required_runtime_capabilities)) or "none"
        return (
            f"ExecutablePlan(nodes={len(self.graph.nodes)}, "
            f"active_rule_namespaces=[{namespaces}], "
            f"required_runtime_capabilities=[{capabilities}], diagnostics=[{codes}])"
        )


class Compiler:
    """Compile a graph through resolution and then verification."""

    MAX_RESOLUTION_ITERATIONS = 32

    def __init__(self, registry: RuleRegistry | None = None) -> None:
        registrations_by_key = {
            (registration.namespace, registration.name): registration
            for registration in DEFAULT_RULE_REGISTRY.registrations()
        }
        if registry is not None:
            for registration in registry.registrations():
                key = (registration.namespace, registration.name)
                if key in registrations_by_key:
                    raise ConfigurationError(
                        "Rule conflicts with an existing registration: "
                        f"{registration.namespace}:{registration.name}"
                    )
                registrations_by_key[key] = registration
        self.registry = RuleRegistry(registrations_by_key.values())

    def compile(
        self, graph: ComponentGraph, context: CompileContext | None = None
    ) -> ExecutablePlan:
        """Resolve and verify a graph, returning an execution-free plan."""
        if not isinstance(graph, ComponentGraph):
            raise ValidationError("Compiler.compile graph must be a ComponentGraph")
        compile_context = CompileContext() if context is None else context
        structural = list(graph.well_formedness())
        registrations = self.registry.registrations()
        resolution = tuple(
            registration
            for registration in registrations
            if registration.phase == "resolution"
            and (
                registration.namespace == "core"
                or registration.namespace in compile_context.enabled_rule_namespaces
            )
        )
        diagnostics = DiagnosticBag(structural)
        current = graph
        unstable_names: set[str] = set()
        reported_conflicts: set[RewriteClaim] = set()
        for _ in range(self.MAX_RESOLUTION_ITERATIONS):
            proposals: list[ResolutionResult] = []
            conflicted: set[int] = set()
            claimants: dict[
                RewriteClaim, list[tuple[int, RuleRegistration, ResolutionResult]]
            ] = {}
            for registration in resolution:
                rule_context = RuleContext(current, compile_context, diagnostics)
                result = registration.rule.apply(rule_context)
                if not isinstance(result, ResolutionResult):
                    raise ValidationError(
                        "Resolution rule "
                        f"{registration.name!r} returned the wrong result type"
                    )
                claims = result.claims or rule_context.claims
                result = replace(result, claims=claims)
                diagnostics.extend(result.diagnostics)
                proposal_index = len(proposals)
                proposals.append(result)
                unclaimed = _changed_locations(current, result.graph) - set(claims)
                if unclaimed:
                    locations = ", ".join(sorted(map(str, unclaimed)))
                    diagnostics.append(
                        Diagnostic(
                            severity=Severity.ERROR,
                            code="unclaimed_rewrite",
                            message=(
                                f"Rule {registration.namespace}:{registration.name} "
                                f"changed unclaimed locations: {locations}."
                            ),
                            path=ContractPath(
                                components=(next(iter(sorted(unclaimed))).key,)
                            ),
                            resolutions=(
                                "Declare every changed graph location with "
                                "RuleContext.claim().",
                            ),
                        )
                    )
                    conflicted.add(proposal_index)
                else:
                    for claim in claims:
                        claimants.setdefault(claim, []).append(
                            (proposal_index, registration, result)
                        )
            for claim, owners in sorted(
                claimants.items(), key=lambda item: str(item[0])
            ):
                if len(owners) > 1:
                    if claim in reported_conflicts:
                        conflicted.update(index for index, _, _ in owners)
                        continue
                    names = ", ".join(
                        sorted(f"{r.namespace}:{r.name}" for _, r, _ in owners)
                    )
                    diagnostics.append(
                        Diagnostic(
                            severity=Severity.ERROR,
                            code="conflicting_rewrite",
                            message=f"Rules {names} claimed {claim}.",
                            path=ContractPath(components=(claim.key.split(":", 1)[0],)),
                            resolutions=(
                                "Narrow claims so each graph location has one owner.",
                            ),
                        )
                    )
                    reported_conflicts.add(claim)
                    conflicted.update(index for index, _, _ in owners)
            usable = [
                proposal
                for index, proposal in enumerate(proposals)
                if index not in conflicted
            ]
            next_graph = _merge_graphs(current, usable)
            changed = next_graph != current
            if changed:
                unstable_names.update(
                    registration.name
                    for registration in resolution
                    if any(proposal.graph != current for proposal in usable)
                )
                current = next_graph
                continue
            break
        else:
            names = ", ".join(sorted(unstable_names)) or ", ".join(
                item.name for item in resolution
            )
            diagnostics.append(
                Diagnostic(
                    severity=Severity.ERROR,
                    code="unstable_compilation",
                    message=f"Resolution rules did not reach a fixed point: {names}.",
                    path=ContractPath(components=("graph",)),
                    resolutions=("Make resolution rules terminating and idempotent.",),
                )
            )
        referenced = _rule_namespaces(current)
        verification = tuple(
            registration
            for registration in registrations
            if registration.phase == "verification"
            and (
                registration.namespace == "core"
                or registration.namespace in compile_context.enabled_rule_namespaces
                or registration.namespace in referenced
            )
        )
        active = tuple(
            sorted(
                (*resolution, *verification),
                key=lambda item: (item.namespace, item.name, item.phase),
            )
        )
        active_names = tuple(f"{item.namespace}:{item.name}" for item in active)
        for registration in verification:
            rule_context = RuleContext(current, compile_context, diagnostics)
            result = registration.rule.apply(rule_context)
            if not isinstance(result, VerificationResult):
                raise ValidationError(
                    "Verification rule "
                    f"{registration.name!r} returned the wrong result type"
                )
            diagnostics.extend(result.diagnostics)
        required = frozenset(
            capability
            for node in current.nodes
            for capability in node.contract.execution.required_runtime_capabilities
        )
        return ExecutablePlan(
            graph=current,
            diagnostics=tuple(sorted(diagnostics, key=_diagnostic_sort_key)),
            required_runtime_capabilities=required,
            active_rule_namespaces=frozenset(item.namespace for item in active),
            active_rule_names=active_names,
        )


DEFAULT_RULE_REGISTRY = RuleRegistry()
DEFAULT_RULE_REGISTRY.register(cast(CompilationRule, IdentityRule()))
DEFAULT_RULE_REGISTRY.register(cast(CompilationRule, ReachabilityRule()))

__all__ = [
    "DEFAULT_RULE_REGISTRY",
    "CompilationRule",
    "CompileContext",
    "Compiler",
    "ExecutablePlan",
    "ResolutionResult",
    "ResolutionRule",
    "RewriteClaim",
    "RuleContext",
    "RuleRegistration",
    "RuleRegistry",
    "RuleResult",
    "VerificationResult",
    "VerificationRule",
]
