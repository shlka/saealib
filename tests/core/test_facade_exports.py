import importlib

import saealib.core as core


def test_core_facade_exports_are_importable_and_canonical() -> None:
    expected = {
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

    assert set(core.__all__) == set(expected)
    for name, (module_name, symbol_name) in expected.items():
        assert getattr(core, name) is getattr(
            importlib.import_module(module_name), symbol_name
        )
