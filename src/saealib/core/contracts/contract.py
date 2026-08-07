from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from saealib.core.contracts.assumptions import AssumptionSet
from saealib.core.contracts.execution import ExecutionContract
from saealib.core.contracts.lifecycle import LifecycleContract
from saealib.core.contracts.ports import PortContract
from saealib.core.contracts.roles import RoleName
from saealib.core.contracts.state import StateContract
from saealib.core.contracts.vocabulary import validate_identifier, validate_name
from saealib.exceptions import ValidationError

__all__ = ["ComponentContract", "PartSpec"]


@dataclass(frozen=True, kw_only=True)
class PartSpec:
    """Declare one constructor-held component contract."""

    name: str
    contract: ComponentContract
    optional: bool = False

    def __post_init__(self) -> None:
        """Validate the part declaration."""
        validate_identifier(self.name)
        if not isinstance(self.contract, ComponentContract):
            raise ValidationError("Part contracts must be ComponentContract values")
        if not isinstance(self.optional, bool):
            raise ValidationError("Part optionality must be a boolean")


@dataclass(frozen=True, kw_only=True)
class ComponentContract:
    """Declare a component's ports, parts, and sub-contracts."""

    ports: Mapping[RoleName, PortContract] = field(default_factory=dict)
    parts: tuple[PartSpec, ...] = ()
    lifecycle: LifecycleContract = field(default_factory=LifecycleContract)
    state: StateContract = field(default_factory=StateContract)
    execution: ExecutionContract = field(default_factory=ExecutionContract)
    assumptions: AssumptionSet = field(default_factory=AssumptionSet.empty)

    def __post_init__(self) -> None:
        """Validate and protect component contract declarations."""
        if not isinstance(self.ports, Mapping):
            raise ValidationError("Component contract ports must be a mapping")
        ports = dict(self.ports)
        for role, port_contract in ports.items():
            if not isinstance(role, str):
                raise ValidationError("Component contract role keys must be strings")
            validate_name(role)
            if not isinstance(port_contract, PortContract):
                raise ValidationError(
                    "Component contract ports must contain PortContract values"
                )
        object.__setattr__(self, "ports", MappingProxyType(ports))

        parts = tuple(self.parts)
        if any(not isinstance(part, PartSpec) for part in parts):
            raise ValidationError(
                "Component contract parts must contain PartSpec values"
            )
        object.__setattr__(self, "parts", parts)

        if not isinstance(self.lifecycle, LifecycleContract):
            raise ValidationError("Component lifecycle must be a LifecycleContract")
        if not isinstance(self.state, StateContract):
            raise ValidationError("Component state must be a StateContract")
        if not isinstance(self.execution, ExecutionContract):
            raise ValidationError("Component execution must be an ExecutionContract")
        if not isinstance(self.assumptions, AssumptionSet):
            raise ValidationError("Component assumptions must be an AssumptionSet")
