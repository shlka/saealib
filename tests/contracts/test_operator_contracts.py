from __future__ import annotations

import numpy as np
import pytest

from saealib.core.contracts import (
    SERVICE_VOCABULARY,
    Fixed,
    PortDirection,
    StateContract,
    Var,
)
from saealib.core.state import RUNTIME_RNG
from saealib.operators.crossover import (
    Crossover,
    CrossoverCategorical,
    CrossoverIntegerSBX,
    CrossoverOnePoint,
    CrossoverSBX,
    CrossoverTwoPoint,
    CrossoverUniform,
)
from saealib.operators.dedup import DuplicateElimination
from saealib.operators.mutation import (
    MutationCategorical,
    MutationGaussian,
    MutationIntegerUniform,
    MutationPolynomial,
    MutationUniform,
)
from saealib.operators.pymoo_crossover import PymooCrossover
from saealib.operators.pymoo_mutation import PymooMutation
from saealib.operators.selection import (
    LinearRankSelection,
    ParentSelection,
    SequentialSelection,
    TournamentSelection,
    TruncationSelection,
)


def _port(contract, role: str, direction: PortDirection):
    role_contract = contract.ports[role]
    return (
        role_contract.inputs[0]
        if direction is PortDirection.INPUT
        else role_contract.outputs[0]
    )


def _service_names(port) -> tuple[str, ...]:
    return tuple(service.name for service in port.required_services)


@pytest.mark.parametrize(
    ("operator", "representation", "bounds"),
    [
        (CrossoverSBX(prob=1.0, eta=20.0), "real", True),
        (CrossoverIntegerSBX(prob=1.0, eta=20.0), "integer", True),
        (CrossoverCategorical(prob=1.0), "categorical", False),
    ],
)
def test_specialized_crossover_contracts(operator, representation, bounds) -> None:
    contract = operator.contract()
    inputs = _port(contract, "crossover", PortDirection.INPUT)
    outputs = _port(contract, "crossover", PortDirection.OUTPUT)

    assert inputs.data.bindings["representation"] == Fixed(value=representation)
    assert outputs.data.bindings["representation"] == Fixed(value=representation)
    assert _service_names(inputs) == (("BoundsService",) if bounds else ())
    assert _service_names(outputs) == ()


def test_representation_independent_crossovers_inherit_contract() -> None:
    assert CrossoverUniform.contract is Crossover.contract
    assert CrossoverOnePoint.contract is Crossover.contract
    assert CrossoverTwoPoint.contract is Crossover.contract


def test_pymoo_crossover_contract_uses_effective_arity() -> None:
    operator = PymooCrossover(_PymooCrossoverStub())
    contract = operator.contract()
    inputs = _port(contract, "crossover", PortDirection.INPUT)
    outputs = _port(contract, "crossover", PortDirection.OUTPUT)

    assert isinstance(inputs.data.bindings["representation"], Var)
    assert isinstance(outputs.data.bindings["representation"], Var)
    assert inputs.data.bindings["parent_count"] == Fixed(value=3)
    assert outputs.data.bindings["candidate_count"] == Fixed(value=4)
    assert _service_names(inputs) == ("BoundsService",)
    assert _service_names(outputs) == ()


@pytest.mark.parametrize(
    ("operator", "representation", "input_services"),
    [
        (MutationUniform(), "real", ("BoundsService",)),
        (MutationPolynomial(eta=20.0), "real", ("BoundsService",)),
        (MutationGaussian(sigma=1.0), "real", ()),
        (MutationIntegerUniform(), "integer", ("BoundsService",)),
        (MutationCategorical(), "categorical", ("BoundsService",)),
    ],
)
def test_mutation_contracts_declare_representation(
    operator, representation, input_services
) -> None:
    contract = operator.contract()
    inputs = _port(contract, "mutation", PortDirection.INPUT)
    outputs = _port(contract, "mutation", PortDirection.OUTPUT)

    assert inputs.data.bindings["representation"] == Fixed(value=representation)
    assert outputs.data.bindings["representation"] == Fixed(value=representation)
    assert _service_names(inputs) == input_services
    assert _service_names(outputs) == ()


def test_pymoo_mutation_inherits_generic_contract() -> None:
    contract = PymooMutation(_PymooMutationStub()).contract()
    inputs = _port(contract, "mutation", PortDirection.INPUT)
    outputs = _port(contract, "mutation", PortDirection.OUTPUT)

    assert isinstance(inputs.data.bindings["representation"], Var)
    assert isinstance(outputs.data.bindings["representation"], Var)
    assert _service_names(inputs) == ("BoundsService",)
    assert _service_names(outputs) == ()


def test_comparison_service_is_registered_without_provider_metadata() -> None:
    descriptor = SERVICE_VOCABULARY.get("ComparisonService")

    assert descriptor is not None
    assert descriptor.description == "order candidates by their objective values"
    assert not hasattr(descriptor, "provider")


@pytest.mark.parametrize("operator", [TournamentSelection(2), LinearRankSelection()])
def test_comparing_parent_selections_require_comparison_service(operator) -> None:
    contract = operator.contract()
    inputs = _port(contract, "parent_selection", PortDirection.INPUT)
    outputs = _port(contract, "parent_selection", PortDirection.OUTPUT)

    assert _service_names(inputs) == ("ComparisonService",)
    assert _service_names(outputs) == ()


def test_sequential_selection_does_not_require_comparison_service() -> None:
    assert SequentialSelection.contract is ParentSelection.contract
    contract = SequentialSelection().contract()

    assert (
        _service_names(_port(contract, "parent_selection", PortDirection.INPUT)) == ()
    )


@pytest.mark.parametrize("randomize_ties", [False, True])
def test_truncation_selection_declares_comparison_and_optional_rng_state(
    randomize_ties: bool,
) -> None:
    contract = TruncationSelection(randomize_ties=randomize_ties).contract()
    inputs = _port(contract, "survivor_selection", PortDirection.INPUT)
    outputs = _port(contract, "survivor_selection", PortDirection.OUTPUT)

    assert _service_names(inputs) == ("ComparisonService",)
    assert _service_names(outputs) == ()
    assert contract.state == (
        StateContract(reads=(RUNTIME_RNG,), writes=(RUNTIME_RNG,))
        if randomize_ties
        else StateContract()
    )


def test_duplicate_elimination_contract_declares_two_genome_inputs() -> None:
    contract = DuplicateElimination().contract()
    role = contract.ports["duplicate_filter"]
    offspring, population = role.inputs
    duplicates = role.outputs[0]

    assert (offspring.name, population.name) == ("offspring", "population")
    assert offspring.data.kind == population.data.kind == "GenomeBatch"
    assert (
        offspring.data.bindings["representation"]
        is population.data.bindings["representation"]
    )
    assert isinstance(offspring.data.bindings["representation"], Var)
    assert offspring.cardinality == population.cardinality == "MANY"
    assert duplicates.name == "duplicates"
    assert duplicates.data.kind == "RowPredicate"
    assert duplicates.cardinality == "MANY"
    assert offspring.required_services == ()
    assert population.required_services == ()
    assert duplicates.required_services == ()
    assert contract.state == StateContract()


class _PymooVariableStub:
    value = 0.5


class _PymooCrossoverStub:
    n_parents = 3
    n_offsprings = 4
    prob = _PymooVariableStub()

    def _do(
        self,
        problem: object,
        x: np.ndarray,
        *args: object,
        random_state: object = None,
        **kwargs: object,
    ) -> np.ndarray:
        return x


class _PymooMutationStub:
    def _do(
        self,
        problem: object,
        x: np.ndarray,
        *args: object,
        random_state: object = None,
        **kwargs: object,
    ) -> np.ndarray:
        return x
