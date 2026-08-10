"""Tests for the canonical framework extension facade."""

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import saealib.core as core


PUBLIC_EXPORTS = {
    "Component": ("saealib.core.component", "Component"),
    "ComponentContract": ("saealib.core.contracts.contract", "ComponentContract"),
    "DataSpec": ("saealib.core.contracts.data", "DataSpec"),
    "PortSpec": ("saealib.core.contracts.ports", "PortSpec"),
    "ComponentGraph": ("saealib.core.compiler.graph", "ComponentGraph"),
    "GraphTemplate": ("saealib.core.compiler.graph", "GraphTemplate"),
    "CompilationRule": ("saealib.core.compiler.compiler", "CompilationRule"),
    "ExecutablePlan": ("saealib.core.compiler.compiler", "ExecutablePlan"),
    "StateStore": ("saealib.core.state.store", "StateStore"),
    "StateView": ("saealib.core.state.store", "StateView"),
    "StatePatch": ("saealib.core.state.patch", "StatePatch"),
    "ExecutionRuntime": ("saealib.core.runtime", "ExecutionRuntime"),
}


def test_core_facade_exports_are_importable_and_canonical() -> None:
    assert set(core.__all__) == set(PUBLIC_EXPORTS)
    for name, (module_name, symbol_name) in PUBLIC_EXPORTS.items():
        assert getattr(core, name) is getattr(
            importlib.import_module(module_name), symbol_name
        )


def test_representative_extension_uses_only_facade_names() -> None:
    extension = {}
    exec(
        """
from saealib.core import (
    Component,
    ComponentContract,
    DataSpec,
    PortSpec,
    ComponentGraph,
    GraphTemplate,
    CompilationRule,
    ExecutablePlan,
    StateStore,
    StateView,
    StatePatch,
    ExecutionRuntime,
)

class ExampleComponent:
    def contract(self) -> ComponentContract:
        return ComponentContract()

class ExampleTemplate(GraphTemplate):
    def build_graph(self, bindings):
        return ComponentGraph(nodes=())
        """,
        extension,
    )

    assert extension["ExampleComponent"]().contract() == core.ComponentContract()
    assert extension["ExampleTemplate"]().build_graph(None) == core.ComponentGraph(
        nodes=()
    )
    assert extension["Component"] is core.Component
    assert extension["StatePatch"] is core.StatePatch
    assert extension["ExecutionRuntime"] is core.ExecutionRuntime


def test_core_facade_resolves_lazily_on_a_cold_import() -> None:
    check = """
import sys
import saealib.core as core

targets = {
    "saealib.core.component",
    "saealib.core.contracts.contract",
    "saealib.core.compiler.graph",
    "saealib.core.compiler.compiler",
    "saealib.core.state.store",
    "saealib.core.state.patch",
    "saealib.core.runtime",
}
# The top-level package may have imported implementation modules for its own
# compatibility surface.  Reset those modules and facade cache entries to
# exercise the facade's resolution path itself.
for target in targets:
    sys.modules.pop(target, None)
for name in core.__all__:
    core.__dict__.pop(name, None)
assert "Component" not in core.__dict__
assert core.Component.__module__ == "saealib.core.component"
assert "saealib.core.component" in sys.modules
assert "ComponentContract" not in core.__dict__
"""
    subprocess.run([sys.executable, "-c", check], check=True)


def test_facade_does_not_expose_engine_or_runtime_registry_internals() -> None:
    forbidden = {"Compiler", "RuleRegistry", "DEFAULT_RULE_REGISTRY", "NodeStatus"}
    assert forbidden.isdisjoint(core.__all__)
    assert forbidden.isdisjoint(dir(core))


def test_facade_stub_exports_match_runtime_exports() -> None:
    stub_path = Path(core.__file__).with_suffix(".pyi")
    tree = ast.parse(stub_path.read_text(encoding="utf-8"))
    stub_exports = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name != "*"
    }
    assert stub_exports == set(core.__all__)
