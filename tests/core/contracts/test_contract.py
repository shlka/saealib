from typing import Any, cast

import pytest

from saealib.core.contracts.assumptions import AssumptionSet
from saealib.core.contracts.contract import ComponentContract, PartSpec
from saealib.core.contracts.data import DataSpec
from saealib.core.contracts.ports import PortContract, PortDirection, PortSpec
from saealib.exceptions import ValidationError


def _port(name: str, direction: PortDirection) -> PortSpec:
    return PortSpec(
        name=name,
        direction=direction,
        data=DataSpec(kind="GenomeBatch"),
        cardinality="ONE",
    )


def test_empty_component_contract_has_all_defaults() -> None:
    contract = ComponentContract()

    assert contract.ports == {}
    assert contract.parts == ()
    assert contract.lifecycle.events == ()
    assert contract.state.reads == ()
    assert contract.execution.required_runtime_capabilities == ()
    assert contract.execution.offered_runtime_capabilities == ()
    assert contract.assumptions["observation_schema.fixed"] is True


def test_component_contract_detaches_and_protects_ports() -> None:
    ports = {"future_role": PortContract(outputs=(_port("out", PortDirection.OUTPUT),))}
    contract = ComponentContract(ports=ports)
    ports["future_role"] = PortContract()

    assert tuple(contract.ports) == ("future_role",)
    with pytest.raises(TypeError):
        cast(Any, contract.ports)["other_role"] = PortContract()


def test_parts_hold_nested_contracts_without_merging_declarations() -> None:
    child = ComponentContract(
        ports={
            "child_role": PortContract(outputs=(_port("child", PortDirection.OUTPUT),))
        },
        assumptions=AssumptionSet({"population.fixed_size": False}),
    )
    parent = ComponentContract(
        ports={
            "parent_role": PortContract(inputs=(_port("parent", PortDirection.INPUT),))
        },
        parts=(PartSpec(name="child", contract=child),),
    )

    assert parent.parts[0].contract == child
    assert tuple(parent.ports) == ("parent_role",)
    assert tuple(parent.parts[0].contract.ports) == ("child_role",)
    assert parent.assumptions["population.fixed_size"] is True


def test_component_contract_rejects_invalid_role_shape() -> None:
    with pytest.raises(ValidationError):
        ComponentContract(ports={"invalid role": PortContract()})


def test_part_spec_rejects_namespace_name() -> None:
    with pytest.raises(ValidationError):
        PartSpec(name="child:part", contract=ComponentContract())
