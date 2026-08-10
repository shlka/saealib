"""Type stubs for the public core facade."""

from saealib.core.compiler.compiler import CompilationRule as CompilationRule
from saealib.core.compiler.compiler import ExecutablePlan as ExecutablePlan
from saealib.core.compiler.graph import ComponentGraph as ComponentGraph
from saealib.core.compiler.graph import GraphTemplate as GraphTemplate
from saealib.core.component import Component as Component
from saealib.core.contracts.contract import ComponentContract as ComponentContract
from saealib.core.contracts.data import DataSpec as DataSpec
from saealib.core.contracts.ports import PortSpec as PortSpec
from saealib.core.runtime import ExecutionRuntime as ExecutionRuntime
from saealib.core.state.patch import StatePatch as StatePatch
from saealib.core.state.store import StateStore as StateStore
from saealib.core.state.store import StateView as StateView

__all__: list[str]
