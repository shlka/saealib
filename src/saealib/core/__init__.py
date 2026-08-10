"""Public facade for the framework's core vocabulary.

The implementation modules are loaded on first attribute access.  Keeping the
facade lazy avoids making the core package import order part of the API.
"""

from __future__ import annotations

from importlib import import_module
from typing import Final

_LAZY_EXPORTS: Final[dict[str, str]] = {
    "Component": "saealib.core.component",
    "ComponentContract": "saealib.core.contracts.contract",
    "DataSpec": "saealib.core.contracts.data",
    "PortSpec": "saealib.core.contracts.ports",
    "ComponentGraph": "saealib.core.compiler.graph",
    "GraphTemplate": "saealib.core.compiler.graph",
    "CompilationRule": "saealib.core.compiler.compiler",
    "ExecutablePlan": "saealib.core.compiler.compiler",
    "StateStore": "saealib.core.state.store",
    "StateView": "saealib.core.state.store",
    "StatePatch": "saealib.core.state.patch",
    "ExecutionRuntime": "saealib.core.runtime",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> object:
    """Resolve a public core symbol lazily and cache it on the facade."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public symbols in interactive discovery."""
    return sorted(set(__all__) | set(_LAZY_EXPORTS))
