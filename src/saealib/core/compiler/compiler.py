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
    ComponentNode,
    ControlEdge,
    DataEdge,
    IdentityRule,
    NodeRef,
    ReachabilityRule,
    StateBinding,
)
from saealib.core.contracts.contract import ComponentContract
from saealib.core.contracts.execution import RuntimeCapability
from saealib.core.contracts.ports import (
    SERVICE_VOCABULARY,
    PortCompatibility,
    PortDirection,
    PortSpec,
    check_port_compatibility,
)
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
    adapter_registry: object | None = None

    def __post_init__(self) -> None:
        """Normalize caller-provided collections."""
        namespaces = frozenset(self.enabled_rule_namespaces)
        capabilities = frozenset(self.offered_runtime_capabilities)
        for value in (*namespaces, *capabilities):
            validate_name(value)
        if not isinstance(self.portability_required, bool):
            raise ValidationError("portability_required must be a boolean")
        if self.adapter_registry is not None and not callable(
            getattr(self.adapter_registry, "candidates", None)
        ):
            raise ValidationError("adapter_registry must provide candidates()")
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


def _iter_port_specs(
    contract: ComponentContract,
    part_path: tuple[str, ...] = (),
) -> Iterable[tuple[tuple[str, ...], str, PortSpec]]:
    """Yield every port in a contract, including recursively declared parts."""
    for role, role_contract in contract.ports.items():
        for port in (*role_contract.inputs, *role_contract.outputs):
            yield part_path, role, port
    for part in contract.parts:
        yield from _iter_port_specs(part.contract, (*part_path, part.name))


def _service_path(
    node: ComponentNode,
    part_path: tuple[str, ...],
    role: str,
    port: PortSpec,
) -> ContractPath:
    """Build the contract path for one required service declaration."""
    return ContractPath(
        components=(node.component_id, *part_path),
        role=role,
        port=port.name,
    )


def _service_registry(provider: object) -> object | None:
    """Return a service registry from either a registry or an owning object."""
    if provider is None:
        return None
    services = getattr(provider, "services", None)
    return provider if services is None else services


def _lookup_service(
    compile_context: CompileContext,
    provider_name: str,
    service_name: str,
) -> object | None:
    """Resolve one service from its declared provider without registry mutation."""
    provider = (
        compile_context.space if provider_name == "space" else compile_context.problem
    )
    registry = _service_registry(provider)
    getter = getattr(registry, "get", None)
    if callable(getter):
        service = getter(service_name)
        if service is not None:
            return service
    # Problem owns ComparisonService in ADR-0003.  Problem currently exposes
    # its comparator directly; the compiler keeps that object as the bound
    # direct reference and does not add a runtime registry lookup.
    if provider_name == "problem" and service_name == "ComparisonService":
        return getattr(provider, "comparator", None)
    return None


def _service_diagnostic(
    *,
    code: str,
    path: ContractPath,
    service_name: str,
    message: str,
    resolution: str,
) -> Diagnostic:
    """Create a consistent diagnostic for a service declaration."""
    return Diagnostic(
        severity=Severity.ERROR,
        code=code,
        message=(f"{path} requires service {service_name!r}. {message}"),
        path=path,
        resolutions=(resolution,),
    )


class ServiceResolutionRule:
    """Bind declared port services to compile-time provider references."""

    namespace = "core"
    name = "service_resolution"
    phase: Literal["resolution"] = "resolution"

    def apply(self, context: RuleContext) -> ResolutionResult:
        """Resolve service requirements and claim changed component nodes."""
        findings: list[Diagnostic] = []
        updated_nodes: list[ComponentNode] = []
        claims: set[RewriteClaim] = set()
        for node in context.graph.nodes:
            resolved: dict[str, object] = {}
            for part_path, role, port in _iter_port_specs(node.contract):
                path = _service_path(node, part_path, role, port)
                for requirement in port.required_services:
                    descriptor = SERVICE_VOCABULARY.get(requirement.name)
                    if descriptor is None:
                        findings.append(
                            _service_diagnostic(
                                code="unknown_service",
                                path=path,
                                service_name=requirement.name,
                                message=(
                                    "The service is not in the core service vocabulary."
                                ),
                                resolution=(
                                    f"Register {requirement.name!r} in "
                                    "SERVICE_VOCABULARY before requiring it."
                                ),
                            )
                        )
                        continue
                    provider_name = getattr(descriptor, "provider", None)
                    if provider_name not in {"space", "problem"}:
                        findings.append(
                            _service_diagnostic(
                                code="unresolved_service",
                                path=path,
                                service_name=requirement.name,
                                message=(
                                    "Its provider descriptor is invalid or missing."
                                ),
                                resolution=(
                                    "Give the service descriptor a valid provider "
                                    "identity (space or problem)."
                                ),
                            )
                        )
                        continue
                    provider_name = cast(str, provider_name)
                    service = _lookup_service(
                        context.compile_context,
                        provider_name,
                        requirement.name,
                    )
                    if service is None:
                        findings.append(
                            _service_diagnostic(
                                code="unresolved_service",
                                path=path,
                                service_name=requirement.name,
                                message=(
                                    f"No {provider_name} provider is available in "
                                    "the compile context."
                                ),
                                resolution=(
                                    f"Provide {requirement.name!r} through the "
                                    f"bound {provider_name}, or choose a component "
                                    "whose port does not require it."
                                ),
                            )
                        )
                        continue
                    resolved.setdefault(requirement.name, service)
            if resolved != dict(node.resolved_services):
                updated_nodes.append(replace(node, resolved_services=resolved))
                claims.add(context.claim("node", node.component_id))
            else:
                updated_nodes.append(node)
        return ResolutionResult(
            graph=replace(context.graph, nodes=tuple(updated_nodes)),
            claims=frozenset(claims),
            diagnostics=tuple(findings),
        )


def _endpoint_path(reference: NodeRef, role: str | None, port: str) -> ContractPath:
    """Build a path for one side of a data connection."""
    return ContractPath(components=(reference.component_id,), role=role, port=port)


@dataclass(frozen=True, kw_only=True)
class _PortResolution:
    """Result of resolving one graph edge endpoint to a declared port."""

    status: Literal["resolved", "missing", "ambiguous"]
    role: str | None = None
    spec: PortSpec | None = None


def _resolve_port(
    node: ComponentNode,
    reference: NodeRef,
    port_name: str,
    direction: PortDirection,
) -> _PortResolution:
    """Resolve one endpoint by role, direction, and unique port name."""
    selected_role = reference.role
    if selected_role is not None:
        role_contract = node.contract.ports.get(selected_role)
        role_items = () if role_contract is None else ((selected_role, role_contract),)
    else:
        role_items = tuple(sorted(node.contract.ports.items()))
    candidates: list[tuple[str, PortSpec]] = []
    for role, role_contract in role_items:
        ports = (
            role_contract.outputs
            if direction is PortDirection.OUTPUT
            else role_contract.inputs
        )
        candidates.extend((role, port) for port in ports if port.name == port_name)
    if not candidates:
        return _PortResolution(status="missing")
    if len(candidates) > 1:
        return _PortResolution(status="ambiguous")
    role, spec = candidates[0]
    return _PortResolution(status="resolved", role=role, spec=spec)


def _compatibility_details(compatibility: PortCompatibility) -> str:
    """Describe the failed checks exposed by PortCompatibility."""
    details: list[str] = []
    if not compatibility.kind_ok:
        details.append("data kinds are incompatible")
    if not compatibility.cardinality_ok:
        details.append("cardinalities are incompatible")
    if not compatibility.direction_ok:
        details.append("port directions are incompatible")
    if not compatibility.schema_ok:
        details.append("schema bindings do not unify")
    if compatibility.unknown_kinds:
        details.append("unknown data kinds: " + ", ".join(compatibility.unknown_kinds))
    if compatibility.unknown_cardinalities:
        details.append(
            "unknown cardinalities: " + ", ".join(compatibility.unknown_cardinalities)
        )
    return "; ".join(details) or "the port compatibility check failed"


class PortCompatibilityRule:
    """Verify every graph data edge against its producer and consumer ports."""

    namespace = "core"
    name = "port_compatibility"
    phase: Literal["verification"] = "verification"

    def apply(self, context: RuleContext) -> VerificationResult:
        """Collect connection diagnostics without rewriting the graph."""
        findings: list[Diagnostic] = []
        for edge in context.graph.data_edges:
            try:
                producer = context.graph.node_by_id(edge.source.component_id)
                consumer = context.graph.node_by_id(edge.target.component_id)
            except KeyError:
                # ComponentGraph.well_formedness already reports missing nodes.
                continue
            producer_port = _resolve_port(
                producer, edge.source, edge.source_port, PortDirection.OUTPUT
            )
            consumer_port = _resolve_port(
                consumer, edge.target, edge.target_port, PortDirection.INPUT
            )
            source_role = edge.source.role or producer.role or producer_port.role
            target_role = edge.target.role or consumer.role or consumer_port.role
            source_path = _endpoint_path(edge.source, source_role, edge.source_port)
            target_path = _endpoint_path(edge.target, target_role, edge.target_port)
            connection = f"{source_path} -> {target_path}"
            unresolved = (
                ("source", producer_port)
                if producer_port.status != "resolved"
                else (
                    ("target", consumer_port)
                    if consumer_port.status != "resolved"
                    else None
                )
            )
            if unresolved is not None:
                endpoint_name, resolution = unresolved
                if resolution.status == "ambiguous":
                    code = "ambiguous_port"
                    port_name = (
                        edge.source_port
                        if endpoint_name == "source"
                        else edge.target_port
                    )
                    message = (
                        f"Connection {connection} names {endpoint_name} port "
                        f"{port_name!r}, "
                        "but multiple roles declare that directional port."
                    )
                    resolution_advice = (
                        "Specify the NodeRef role for the ambiguous endpoint."
                    )
                else:
                    code = "unknown_port"
                    port_name = (
                        edge.source_port
                        if endpoint_name == "source"
                        else edge.target_port
                    )
                    message = (
                        f"Connection {connection} names an undeclared "
                        f"{endpoint_name} port {port_name!r}."
                    )
                    resolution_advice = (
                        "Correct the edge port name or declare that directional "
                        "port in the endpoint contract."
                    )
                findings.append(
                    Diagnostic(
                        severity=Severity.ERROR,
                        code=code,
                        message=message,
                        path=source_path,
                        related=(target_path,),
                        resolutions=(resolution_advice,),
                    )
                )
                continue
            assert producer_port.spec is not None
            assert consumer_port.spec is not None
            producer_spec = producer_port.spec
            consumer_spec = consumer_port.spec
            compatibility = check_port_compatibility(producer_spec, consumer_spec)
            if compatibility.compatible:
                continue
            if (
                not compatibility.schema_ok
                and compatibility.kind_ok
                and compatibility.cardinality_ok
                and compatibility.direction_ok
            ):
                # SchemaBindingRule owns graph-wide schema substitution and
                # reports its conflicts/deferred variables with their causes.
                continue
            source_path = _endpoint_path(
                edge.source, edge.source.role or producer_port.role, producer_spec.name
            )
            target_path = _endpoint_path(
                edge.target, edge.target.role or consumer_port.role, consumer_spec.name
            )
            connection = f"{source_path} -> {target_path}"
            if any(
                diagnostic.code in {"ambiguous_adapter", "incompatible_representation"}
                and diagnostic.path == source_path
                and target_path in diagnostic.related
                for diagnostic in context.diagnostics
            ):
                # A resolution diagnostic already identifies the actionable
                # cause; do not add a generic incompatible-port shadow.
                continue
            findings.append(
                Diagnostic(
                    severity=Severity.ERROR,
                    code="incompatible_port",
                    message=(
                        f"Connection {connection} is incompatible: "
                        f"{_compatibility_details(compatibility)}."
                    ),
                    path=source_path,
                    related=(target_path,),
                    resolutions=(
                        "Connect compatible producer and consumer ports, or "
                        "change the component contract declarations.",
                    ),
                )
            )
        return VerificationResult(diagnostics=tuple(findings))


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

        def stable_value_key(value: _ValueT) -> tuple[str, str, str]:
            """Order newly added values without consulting rule enumeration."""
            return (
                type(value).__name__,
                str(getattr(value, "component_id", "")),
                repr(value),
            )

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
        existing = [
            value
            for value in merged
            if any(same_slot(value, original_value) for original_value in original)
        ]
        added = [
            value
            for value in merged
            if not any(same_slot(value, original_value) for original_value in original)
        ]
        return tuple((*existing, *sorted(added, key=stable_value_key)))

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
    inserted_adapters: tuple[object, ...] = ()

    def describe(self) -> str:
        """Return a concise human-readable plan summary."""
        codes = ", ".join(d.code for d in self.diagnostics) or "none"
        namespaces = ", ".join(sorted(self.active_rule_namespaces)) or "none"
        capabilities = ", ".join(sorted(self.required_runtime_capabilities)) or "none"
        insertions = ", ".join(map(str, self.inserted_adapters)) or "none"
        return (
            f"ExecutablePlan(nodes={len(self.graph.nodes)}, "
            f"active_rule_namespaces=[{namespaces}], "
            f"required_runtime_capabilities=[{capabilities}], "
            f"diagnostics=[{codes}], inserted_adapters=[{insertions}])"
        )


class Compiler:
    """Compile a graph through resolution and then verification."""

    MAX_RESOLUTION_ITERATIONS = 32

    def __init__(
        self,
        registry: RuleRegistry | None = None,
        *,
        adapter_registry: object | None = None,
    ) -> None:
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
        self.adapter_registry = adapter_registry

    def compile(
        self, graph: ComponentGraph, context: CompileContext | None = None
    ) -> ExecutablePlan:
        """Resolve and verify a graph, returning an execution-free plan."""
        if not isinstance(graph, ComponentGraph):
            raise ValidationError("Compiler.compile graph must be a ComponentGraph")
        compile_context = (
            CompileContext(adapter_registry=self.adapter_registry)
            if context is None
            else context
        )
        if (
            self.adapter_registry is not None
            and compile_context.adapter_registry is None
        ):
            compile_context = replace(
                compile_context, adapter_registry=self.adapter_registry
            )
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
        inserted_adapters = tuple(
            sorted(
                (
                    insertion
                    for node in current.nodes
                    if (insertion := getattr(node.component, "insertion", None))
                    is not None
                ),
                key=str,
            )
        )
        return ExecutablePlan(
            graph=current,
            diagnostics=tuple(
                sorted(dict.fromkeys(diagnostics), key=_diagnostic_sort_key)
            ),
            required_runtime_capabilities=required,
            active_rule_namespaces=frozenset(item.namespace for item in active),
            active_rule_names=active_names,
            inserted_adapters=inserted_adapters,
        )


DEFAULT_RULE_REGISTRY = RuleRegistry()
DEFAULT_RULE_REGISTRY.register(cast(CompilationRule, IdentityRule()))
DEFAULT_RULE_REGISTRY.register(cast(CompilationRule, ReachabilityRule()))
DEFAULT_RULE_REGISTRY.register(cast(CompilationRule, ServiceResolutionRule()))
DEFAULT_RULE_REGISTRY.register(cast(CompilationRule, PortCompatibilityRule()))
from saealib.core.compiler.adapters import (  # noqa: E402  # registration boundary
    DEFAULT_ADAPTER_REGISTRY,
    LosslessAdapterRule,
)
from saealib.core.compiler.lifecycle_rules import (  # noqa: E402
    FeedbackAccumulatorRule,
    LifecycleCompatibilityRule,
)
from saealib.core.compiler.persistence_runtime_rules import (  # noqa: E402
    PersistenceRule,
    RuntimeCompatibilityRule,
)
from saealib.core.compiler.schema_rules import SchemaBindingRule  # noqa: E402

DEFAULT_RULE_REGISTRY.register(cast(CompilationRule, SchemaBindingRule()))
DEFAULT_RULE_REGISTRY.register(cast(CompilationRule, FeedbackAccumulatorRule()))
DEFAULT_RULE_REGISTRY.register(cast(CompilationRule, LosslessAdapterRule()))
DEFAULT_RULE_REGISTRY.register(cast(CompilationRule, PersistenceRule()))
DEFAULT_RULE_REGISTRY.register(cast(CompilationRule, RuntimeCompatibilityRule()))
DEFAULT_RULE_REGISTRY.register(cast(CompilationRule, LifecycleCompatibilityRule()))

__all__ = [
    "DEFAULT_ADAPTER_REGISTRY",
    "DEFAULT_RULE_REGISTRY",
    "CompilationRule",
    "CompileContext",
    "Compiler",
    "ExecutablePlan",
    "FeedbackAccumulatorRule",
    "LifecycleCompatibilityRule",
    "LosslessAdapterRule",
    "PersistenceRule",
    "PortCompatibilityRule",
    "ResolutionResult",
    "ResolutionRule",
    "RewriteClaim",
    "RuleContext",
    "RuleRegistration",
    "RuleRegistry",
    "RuleResult",
    "RuntimeCompatibilityRule",
    "SchemaBindingRule",
    "ServiceResolutionRule",
    "VerificationResult",
    "VerificationRule",
]
