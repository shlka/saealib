from __future__ import annotations

from saealib.core.compiler.compiler import (
    DEFAULT_RULE_REGISTRY,
    CompilationRule,
    CompileContext,
    Compiler,
    ExecutablePlan,
    PortCompatibilityRule,
    ResolutionResult,
    ResolutionRule,
    RewriteClaim,
    RuleContext,
    RuleRegistration,
    RuleRegistry,
    RuleResult,
    ServiceResolutionRule,
    StateEffectRule,
    VerificationResult,
    VerificationRule,
)
from saealib.core.compiler.contract_diagnostics import check_component_contract
from saealib.core.compiler.diagnostics import (
    DIAGNOSTIC_CODES,
    ContractPath,
    Diagnostic,
    DiagnosticBag,
    DiagnosticCodeVocabulary,
    Severity,
)
from saealib.core.compiler.graph import (
    ComponentBindings,
    ComponentGraph,
    ComponentId,
    ComponentNode,
    ControlEdge,
    DataEdge,
    GraphTemplate,
    IdentityRule,
    NodeRef,
    ReachabilityRule,
    StateBinding,
)

__all__ = [
    "DIAGNOSTIC_CODES",
    "CompilationRule",
    "CompileContext",
    "ComponentBindings",
    "ComponentGraph",
    "ComponentId",
    "ComponentNode",
    "ContractPath",
    "ControlEdge",
    "DataEdge",
    "Diagnostic",
    "DiagnosticBag",
    "DiagnosticCodeVocabulary",
    "ExecutablePlan",
    "GraphTemplate",
    "NodeRef",
    "Severity",
    "StateBinding",
]

# Implementation names remain importable for the package's own compatibility
# tests and internal modules, but are intentionally absent from ``__all__``.
from saealib.core.compiler.lifecycle_rules import (
    FeedbackAccumulatorRule,
    LifecycleCompatibilityRule,
)
from saealib.core.compiler.persistence_runtime_rules import (
    PersistenceRule,
    RuntimeCompatibilityRule,
)
from saealib.core.compiler.schema_rules import SchemaBindingRule
