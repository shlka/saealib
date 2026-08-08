from __future__ import annotations

import ast
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import pytest

from saealib.core.compiler import (
    CompileContext,
    Compiler,
    ResolutionResult,
    RuleContext,
    RuleRegistry,
    VerificationResult,
)
from saealib.core.compiler.diagnostics import ContractPath, Diagnostic, Severity
from saealib.core.compiler.graph import ComponentGraph, ComponentNode, NodeRef
from saealib.core.contracts import ComponentContract, ExecutionContract
from saealib.exceptions import ConfigurationError


class _Component:
    def contract(self) -> ComponentContract:
        return ComponentContract()


def _graph() -> ComponentGraph:
    return ComponentGraph(
        nodes=(ComponentNode(component_id="start", component=_Component()),),
        entry_points=(NodeRef(component_id="start"),),
    )


def test_conflicting_claims_apply_neither_rewrite() -> None:
    graph = _graph()
    registry = RuleRegistry(
        [
            _ClaimingRule("a", role="left"),
            _ClaimingRule("b", role="right"),
        ]
    )
    plan = Compiler(registry).compile(
        graph, CompileContext(enabled_rule_namespaces=frozenset({"test"}))
    )

    assert any(
        diagnostic.code == "conflicting_rewrite" for diagnostic in plan.diagnostics
    )
    assert plan.graph is graph
    assert graph.nodes[0].role is None


@dataclass
class _ClaimingRule:
    name: str
    role: str = "left"
    namespace: str = "test"
    phase: Literal["resolution"] = "resolution"

    def apply(self, context: RuleContext) -> ResolutionResult:
        node = replace(context.graph.nodes[0], role=self.role)
        return ResolutionResult(
            graph=replace(context.graph, nodes=(node,)),
            claims=frozenset({context.claim("node", "start")}),
        )


@dataclass
class _UnclaimedRule:
    name: str = "unclaimed"
    namespace: str = "test"
    phase: Literal["resolution"] = "resolution"

    def apply(self, context: RuleContext) -> ResolutionResult:
        node = replace(context.graph.nodes[0], role="unclaimed")
        return ResolutionResult(graph=replace(context.graph, nodes=(node,)))


def test_unclaimed_graph_changes_are_rejected() -> None:
    graph = _graph()
    plan = Compiler(RuleRegistry([_UnclaimedRule()])).compile(
        graph, CompileContext(enabled_rule_namespaces=frozenset({"test"}))
    )

    assert [
        diagnostic.code
        for diagnostic in plan.diagnostics
        if diagnostic.code == "unclaimed_rewrite"
    ] == ["unclaimed_rewrite"]
    assert plan.graph is graph


@dataclass
class _VerificationRule:
    name: str
    namespace: str = "test"
    phase: Literal["verification"] = "verification"

    def apply(self, context: RuleContext) -> VerificationResult:
        return VerificationResult()


def test_compiler_result_is_execution_free_and_describable() -> None:
    registry = RuleRegistry([_VerificationRule("check")])
    plan = Compiler(registry).compile(
        _graph(), CompileContext(enabled_rule_namespaces=frozenset({"test"}))
    )

    assert "ExecutablePlan" in plan.describe()
    assert not hasattr(plan, "execute")


@dataclass
class _FindingRule:
    name: str
    namespace: str = "test"
    phase: Literal["verification"] = "verification"

    def apply(self, context: RuleContext) -> VerificationResult:
        return VerificationResult(
            diagnostics=(
                Diagnostic(
                    severity=Severity.WARNING,
                    code="unknown_data_spec",
                    message=f"finding from {self.name}",
                    path=ContractPath(components=("start",)),
                    resolutions=("Inspect the test rule.",),
                ),
            )
        )


def _diagnostic_signature(plan) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (diagnostic.code, str(diagnostic.path), diagnostic.message)
        for diagnostic in plan.diagnostics
    )


def test_rule_enumeration_order_is_deterministic() -> None:
    rules = [_FindingRule("a"), _FindingRule("b"), _FindingRule("c")]
    expected = None
    for seed in range(8):
        shuffled = list(rules)
        random.Random(seed).shuffle(shuffled)
        plan = Compiler(RuleRegistry(shuffled)).compile(
            _graph(), CompileContext(enabled_rule_namespaces=frozenset({"test"}))
        )
        signature = (_diagnostic_signature(plan), plan.active_rule_names)
        if expected is None:
            expected = signature
        else:
            assert signature == expected


@dataclass
class _DisjointRule:
    name: str
    target: str
    role: str
    namespace: str = "test"
    phase: Literal["resolution"] = "resolution"

    def apply(self, context: RuleContext) -> ResolutionResult:
        nodes = tuple(
            replace(node, role=self.role) if node.component_id == self.target else node
            for node in context.graph.nodes
        )
        return ResolutionResult(
            graph=replace(context.graph, nodes=nodes),
            claims=frozenset({context.claim("node", self.target)}),
        )


def test_disjoint_resolution_claims_are_order_independent() -> None:
    graph = replace(
        _graph(),
        nodes=(
            ComponentNode(component_id="start", component=_Component()),
            ComponentNode(component_id="other", component=_Component()),
        ),
    )
    rules = [
        _DisjointRule("a", "start", "first"),
        _DisjointRule("b", "other", "second"),
    ]
    expected = None
    for seed in range(8):
        shuffled = list(rules)
        random.Random(seed).shuffle(shuffled)
        plan = Compiler(RuleRegistry(shuffled)).compile(
            graph, CompileContext(enabled_rule_namespaces=frozenset({"test"}))
        )
        signature = tuple((node.component_id, node.role) for node in plan.graph.nodes)
        if expected is None:
            expected = signature
        else:
            assert signature == expected
    assert expected == (("start", "first"), ("other", "second"))


@dataclass
class _AddingRule:
    name: str
    target: str
    namespace: str = "test"
    phase: Literal["resolution"] = "resolution"

    def apply(self, context: RuleContext) -> ResolutionResult:
        if any(node.component_id == self.target for node in context.graph.nodes):
            return ResolutionResult(
                graph=context.graph,
                claims=frozenset(
                    {
                        context.claim("node", self.target),
                        context.claim("entry_point", self.target),
                    }
                ),
            )
        node = ComponentNode(component_id=self.target, component=_Component())
        return ResolutionResult(
            graph=replace(
                context.graph,
                nodes=(*context.graph.nodes, node),
                entry_points=(
                    *context.graph.entry_points,
                    NodeRef(component_id=self.target),
                ),
            ),
            claims=frozenset(
                {
                    context.claim("node", self.target),
                    context.claim("entry_point", self.target),
                }
            ),
        )


def test_disjoint_resolution_additions_are_order_independent() -> None:
    graph = _graph()
    rules = [_AddingRule("add_b", "b"), _AddingRule("add_a", "a")]
    expected = None
    for seed in range(8):
        shuffled = list(rules)
        random.Random(seed).shuffle(shuffled)
        plan = Compiler(RuleRegistry(shuffled)).compile(
            graph, CompileContext(enabled_rule_namespaces=frozenset({"test"}))
        )
        signature = tuple(node.component_id for node in plan.graph.nodes)
        if expected is None:
            expected = signature
        else:
            assert signature == expected
    assert expected == ("start", "a", "b")


@dataclass
class _OneShotRule:
    name: str = "one_shot"
    namespace: str = "test"
    phase: Literal["resolution"] = "resolution"
    calls: int = 0

    def apply(self, context: RuleContext) -> ResolutionResult:
        self.calls += 1
        if any(node.component_id == "added" for node in context.graph.nodes):
            return ResolutionResult(
                graph=context.graph,
                claims=frozenset(
                    {
                        context.claim("node", "added"),
                        context.claim("entry_point", "added"),
                    }
                ),
            )
        added = ComponentNode(component_id="added", component=_Component())
        return ResolutionResult(
            graph=replace(
                context.graph,
                nodes=(*context.graph.nodes, added),
                entry_points=(
                    *context.graph.entry_points,
                    NodeRef(component_id="added"),
                ),
            ),
            claims=frozenset(
                {
                    context.claim("node", "added"),
                    context.claim("entry_point", "added"),
                }
            ),
        )


def test_resolution_reaches_fixed_point() -> None:
    rule = _OneShotRule()
    plan = Compiler(RuleRegistry([rule])).compile(
        _graph(), CompileContext(enabled_rule_namespaces=frozenset({"test"}))
    )

    assert rule.calls == 2
    assert {node.component_id for node in plan.graph.nodes} == {"start", "added"}
    assert not any(
        diagnostic.code == "unstable_compilation" for diagnostic in plan.diagnostics
    )


@dataclass
class _NeverStableRule:
    name: str = "never_stable"
    namespace: str = "test"
    phase: Literal["resolution"] = "resolution"
    calls: int = 0

    def apply(self, context: RuleContext) -> ResolutionResult:
        self.calls += 1
        node = replace(
            context.graph.nodes[0],
            role=f"role_{self.calls}",
        )
        return ResolutionResult(
            graph=replace(context.graph, nodes=(node,)),
            claims=frozenset({context.claim("node", "start")}),
        )


class _ShortCompiler(Compiler):
    MAX_RESOLUTION_ITERATIONS = 3


def test_resolution_iteration_cap_reports_unstable_compilation() -> None:
    rule = _NeverStableRule()
    plan = _ShortCompiler(RuleRegistry([rule])).compile(
        _graph(), CompileContext(enabled_rule_namespaces=frozenset({"test"}))
    )

    assert rule.calls == 3
    assert [
        diagnostic.code
        for diagnostic in plan.diagnostics
        if diagnostic.code == "unstable_compilation"
    ] == ["unstable_compilation"]


def test_verification_diagnostics_are_collected() -> None:
    plan = Compiler(RuleRegistry([_FindingRule("a"), _FindingRule("b")])).compile(
        _graph(), CompileContext(enabled_rule_namespaces=frozenset({"test"}))
    )

    assert [
        diagnostic.message
        for diagnostic in plan.diagnostics
        if diagnostic.message.startswith("finding")
    ] == ["finding from a", "finding from b"]


def test_core_rules_are_registered_once_and_not_duplicated_by_well_formedness() -> None:
    graph = ComponentGraph(
        nodes=(
            ComponentNode(component_id="start", component=_Component()),
            ComponentNode(component_id="start", component=_Component()),
            ComponentNode(component_id="orphan", component=_Component()),
        ),
        entry_points=(NodeRef(component_id="start"),),
    )
    plan = Compiler().compile(graph)

    assert [
        diagnostic.code
        for diagnostic in plan.diagnostics
        if diagnostic.code == "duplicate_component_id"
    ] == ["duplicate_component_id"]
    assert [
        diagnostic.code
        for diagnostic in plan.diagnostics
        if diagnostic.code == "unreachable_node"
    ] == ["unreachable_node"]


def test_compiler_rejects_registration_collision_with_core_rule() -> None:
    with pytest.raises(ConfigurationError, match="core:identity"):
        Compiler(RuleRegistry([_VerificationRule(name="identity", namespace="core")]))


@dataclass
class _NamespacedVerificationRule:
    name: str = "verification"
    namespace: str = "extension"
    phase: Literal["verification"] = "verification"
    calls: int = 0

    def apply(self, context: RuleContext) -> VerificationResult:
        self.calls += 1
        return VerificationResult()


@dataclass
class _NamespacedResolutionRule:
    name: str = "resolution"
    namespace: str = "extension"
    phase: Literal["resolution"] = "resolution"
    calls: int = 0

    def apply(self, context: RuleContext) -> ResolutionResult:
        self.calls += 1
        return ResolutionResult(
            graph=context.graph,
            claims=frozenset({context.claim("node", "start")}),
        )


def test_third_party_rule_activation_differs_by_phase() -> None:
    verification = _NamespacedVerificationRule()
    resolution = _NamespacedResolutionRule()
    registry = RuleRegistry([verification, resolution])
    plain = _graph()
    namespaced = replace(
        plain,
        nodes=(ComponentNode(component_id="extension:start", component=_Component()),),
        entry_points=(NodeRef(component_id="extension:start"),),
    )

    Compiler(registry).compile(namespaced)
    assert verification.calls == 1
    assert resolution.calls == 0
    Compiler(registry).compile(plain)
    assert verification.calls == 1
    assert resolution.calls == 0
    Compiler(registry).compile(
        plain, CompileContext(enabled_rule_namespaces=frozenset({"extension"}))
    )
    assert verification.calls == 2
    assert resolution.calls == 1


def test_compiler_package_does_not_import_concrete_component_packages() -> None:
    compiler_root = Path(__file__).parents[3] / "src" / "saealib" / "core" / "compiler"
    imports: list[str] = []
    for path in compiler_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append(node.module)

    saealib_imports = tuple(
        imported
        for imported in imports
        if imported == "saealib" or imported.startswith("saealib.")
    )
    allowed = (
        "saealib.core",
        "saealib.exceptions",
    )
    unexpected = tuple(
        imported
        for imported in saealib_imports
        if not any(
            imported == package or imported.startswith(f"{package}.")
            for package in allowed
        )
    )
    assert not unexpected, (
        f"compiler package crossed the contract boundary: {unexpected}"
    )


def test_executable_plan_aggregates_required_capabilities_without_runtime_check() -> (
    None
):
    component = type(
        "CapabilityComponent",
        (),
        {
            "contract": lambda self: ComponentContract(
                execution=ExecutionContract(
                    required_runtime_capabilities=("partial_feedback",),
                )
            )
        },
    )()
    graph = ComponentGraph(
        nodes=(ComponentNode(component_id="start", component=component),),
        entry_points=(NodeRef(component_id="start"),),
    )

    plan = Compiler().compile(graph)

    assert plan.required_runtime_capabilities == frozenset({"partial_feedback"})
    assert not any(
        diagnostic.code == "unknown_runtime_capability"
        for diagnostic in plan.diagnostics
    )
