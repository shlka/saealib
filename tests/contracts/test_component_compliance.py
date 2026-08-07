from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest

import saealib
from saealib.core.component import Component
from saealib.core.contracts import (
    ASSUMPTION_KEYS,
    CARDINALITIES,
    DATA_SPEC_KINDS,
    ROLES,
    RUNTIME_CAPABILITIES,
    SCHEMA_VARIABLES,
    SERVICE_VOCABULARY,
    AssumptionSet,
    ComponentContract,
    DataSpec,
    ExecutionContract,
    Fixed,
    PartSpec,
    PortContract,
    PortDirection,
    PortSpec,
    Product,
    ServiceRequirement,
    Var,
)
from saealib.registry import build, to_spec
from saealib.surrogate.rbf import gaussian_kernel

_REGISTRY = saealib.registry._REGISTRY
_NON_CLASS_REGISTRY_NAMES = frozenset({"max_fe", "max_gen", "f_target", "stalled"})


def _recipe_local_surrogate_manager() -> Any:
    return build(
        {
            "type": "LocalSurrogateManager",
            "params": {
                "surrogate": {
                    "type": "RBFSurrogate",
                    "params": {"kernel": gaussian_kernel, "dim": 5},
                },
                "training_set": {
                    "type": "KNNObjectiveSet",
                    "params": {"n_neighbors": 50},
                },
            },
        }
    )


def _recipe_rbf_surrogate() -> Any:
    return build(
        {
            "type": "RBFSurrogate",
            "params": {"kernel": gaussian_kernel, "dim": 5},
        }
    )


def _recipe_ga() -> Any:
    return build(
        {
            "type": "GA",
            "params": {
                "crossover": {
                    "type": "CrossoverBLXAlpha",
                    "params": {"prob": 0.7, "alpha": 0.4},
                },
                "mutation": {
                    "type": "MutationUniform",
                    "params": {"prob_var": 0.3},
                },
                "parent_selection": {"type": "SequentialSelection"},
                "survivor_selection": {"type": "TruncationSelection"},
            },
        }
    )


def _recipe_termination() -> Any:
    from saealib.termination import Termination, max_fe

    return Termination(max_fe(100))


_RECIPES: dict[str, Callable[[], Any]] = {
    "LocalSurrogateManager": _recipe_local_surrogate_manager,
    "RBFSurrogate": _recipe_rbf_surrogate,
    "CORSDistance": lambda: build({"type": "CORSDistance", "params": {"delta": 10.0}}),
    "CrossoverBLXAlpha": lambda: build(
        {
            "type": "CrossoverBLXAlpha",
            "params": {"prob": 0.7, "alpha": 0.4},
        }
    ),
    "GA": _recipe_ga,
    "TopKEvaluation": lambda: build({"type": "TopKEvaluation", "params": {"k": 1}}),
    "RatioEvaluation": lambda: build(
        {"type": "RatioEvaluation", "params": {"ratio": 0.5}}
    ),
    "Termination": _recipe_termination,
    "GenerationBasedStrategy": lambda: build(
        {"type": "GenerationBasedStrategy", "params": {"gen_ctrl": 5}}
    ),
    "PreSelectionStrategy": lambda: build(
        {
            "type": "PreSelectionStrategy",
            "params": {"n_candidates": 40, "n_select": 4},
        }
    ),
}


def _builtin_classes() -> dict[str, type[Any]]:
    return {
        name: obj
        for name, obj in _REGISTRY.items()
        if inspect.isclass(obj) and obj.__module__.startswith("saealib.")
    }


_BUILTIN_CLASSES = _builtin_classes()
_CONSTRUCTION_FAILURES: dict[str, type[BaseException]] = {}


def _construct_builtin(name: str) -> Any | None:
    try:
        return build(name)
    except (TypeError, ValueError) as error:
        _CONSTRUCTION_FAILURES[name] = type(error)
        recipe = _RECIPES.get(name)
        return None if recipe is None else recipe()


_BUILTIN_INSTANCES = tuple(
    (name, _construct_builtin(name)) for name in _BUILTIN_CLASSES
)


def _implements_contract(instance: Any) -> bool:
    method = getattr(type(instance), "contract", None)
    return callable(method) and method is not Component.contract


_IMPLEMENTED_COMPONENTS = tuple(
    (name, instance)
    for name, instance in _BUILTIN_INSTANCES
    if instance is not None and _implements_contract(instance)
)


def _freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                ((repr(key), _freeze(item)) for key, item in value.items()),
                key=lambda item: item[0],
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _snapshot(instance: Any) -> Any:
    try:
        return ("to_spec", _freeze(to_spec(instance)))
    except Exception:
        for name in ("state_export", "export_state"):
            exporter = getattr(instance, name, None)
            if callable(exporter):
                try:
                    return (name, _freeze(exporter()))
                except Exception:
                    break
        return ("vars", _freeze(vars(instance)))


def _check_purity(instance: Any) -> None:
    before = _snapshot(instance)
    first = instance.contract()
    between = _snapshot(instance)
    second = instance.contract()
    after = _snapshot(instance)
    assert first == second
    assert before == between == after


class _PoisonedGenerator:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"contract accessed poisoned RNG attribute {name!r}")


@contextmanager
def _poisoned_generators(root: Any) -> Iterator[None]:
    replacements: list[tuple[Any, str, np.random.Generator]] = []
    visited: set[int] = set()

    def visit(value: Any) -> None:
        if id(value) in visited:
            return
        visited.add(id(value))
        try:
            attributes = tuple(vars(value).items())
        except TypeError:
            return
        for name, attribute in attributes:
            if isinstance(attribute, np.random.Generator):
                setattr(value, name, _PoisonedGenerator())
                replacements.append((value, name, attribute))
            elif callable(getattr(attribute, "contract", None)):
                visit(attribute)

    try:
        visit(root)
        yield
    finally:
        for owner, name, generator in reversed(replacements):
            setattr(owner, name, generator)


def _check_state_free(instance: Any) -> None:
    with _poisoned_generators(instance):
        instance.contract()


def _check_port_names(instance: Any) -> None:
    contract = instance.contract()
    for role_contract in contract.ports.values():
        for ports in (role_contract.inputs, role_contract.outputs):
            names = tuple(port.name for port in ports)
            assert all(isinstance(name, str) and name for name in names)
            assert len(names) == len(set(names))
    for part in contract.parts:
        child = getattr(instance, part.name)
        _check_port_names(child)


def _check_vocabularies(instance: Any) -> None:
    contract = instance.contract()
    for role, role_contract in contract.ports.items():
        assert ROLES.get(role) is not None
        for port in (*role_contract.inputs, *role_contract.outputs):
            assert DATA_SPEC_KINDS.get(port.data.kind) is not None
            assert CARDINALITIES.get(port.cardinality) is not None
            for variable in port.data.bindings:
                assert SCHEMA_VARIABLES.get(variable) is not None
            for service in port.required_services:
                assert SERVICE_VOCABULARY.get(service.name) is not None
    for name in contract.assumptions:
        assert ASSUMPTION_KEYS.get(name) is not None
    for capability in contract.execution.required_runtime_capabilities:
        assert RUNTIME_CAPABILITIES.get(capability) is not None
    for part in contract.parts:
        child = getattr(instance, part.name)
        _check_vocabularies(child)


def _check_parts(instance: Any) -> None:
    contract = instance.contract()
    for part in contract.parts:
        assert isinstance(part.contract, ComponentContract)
        try:
            child = getattr(instance, part.name)
        except AttributeError as error:
            raise AssertionError(f"missing part attribute {part.name!r}") from error
        assert callable(getattr(child, "contract", None))
        assert part.contract == child.contract()
        _check_parts(child)


def _port(name: str, direction: PortDirection) -> PortSpec:
    return PortSpec(
        name=name,
        direction=direction,
        data=DataSpec(kind="GenomeBatch"),
        cardinality="ONE",
    )


class _PureDummy:
    def __init__(self, value: int = 1) -> None:
        self.value = value

    def contract(self) -> ComponentContract:
        return ComponentContract()


class _ImpureDummy:
    def __init__(self, value: int = 1) -> None:
        self.value = value

    def contract(self) -> ComponentContract:
        self.value += 1
        return ComponentContract()


class _StateFreeChild:
    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)

    def contract(self) -> ComponentContract:
        return ComponentContract()


class _StateReadingChild(_StateFreeChild):
    def contract(self) -> ComponentContract:
        self.rng.random()
        return ComponentContract()


class _StateFreeParent:
    def __init__(self) -> None:
        self.child = _StateFreeChild()

    def contract(self) -> ComponentContract:
        return self.child.contract()


class _StateReadingParent:
    def __init__(self) -> None:
        self.child = _StateReadingChild()

    def contract(self) -> ComponentContract:
        return self.child.contract()


class _PortNamesPassDummy:
    def contract(self) -> ComponentContract:
        port = _port("input", PortDirection.INPUT)
        return ComponentContract(
            ports={
                "proposer": PortContract(inputs=(port,)),
                "predictor": PortContract(inputs=(port,)),
            }
        )


class _PortNamesFailDummy:
    def contract(self) -> ComponentContract:
        port = _port("input", PortDirection.INPUT)
        role_contract = PortContract(inputs=(port,))
        object.__setattr__(role_contract, "inputs", (port, port))
        return ComponentContract(ports={"proposer": role_contract})


class _VocabularyChild:
    def contract(self) -> ComponentContract:
        return ComponentContract()


class _VocabularyPassDummy:
    def __init__(self) -> None:
        self.child = _VocabularyChild()

    def contract(self) -> ComponentContract:
        return ComponentContract(
            ports=_vocabulary_contract().ports,
            parts=(PartSpec(name="child", contract=self.child.contract()),),
            assumptions=_vocabulary_contract().assumptions,
        )


class _VocabularyDummy:
    def __init__(self, component_contract: ComponentContract) -> None:
        self.component_contract = component_contract

    def contract(self) -> ComponentContract:
        return self.component_contract


def _vocabulary_contract(
    *,
    role: str = "proposer",
    kind: str = "GenomeBatch",
    bindings: Mapping[str, Any] | None = None,
    cardinality: str = "ONE",
    services: tuple[str, ...] = ("SamplingService",),
    assumptions: Mapping[str, bool] | None = None,
    capabilities: tuple[str, ...] = (),
) -> ComponentContract:
    if bindings is None:
        bindings = {
            "representation": Product(
                elements=(
                    Var(name="feature_schema"),
                    Product(
                        elements=(
                            Var(name="objective_schema"),
                            Fixed(value="dense"),
                        )
                    ),
                )
            )
        }
    if assumptions is None:
        assumptions = {"evaluation.deterministic": False}
    port = PortSpec(
        name="input",
        direction=PortDirection.INPUT,
        data=DataSpec(kind=kind, bindings=bindings),
        cardinality=cardinality,
        required_services=tuple(
            ServiceRequirement(name=service) for service in services
        ),
    )
    return ComponentContract(
        ports={role: PortContract(inputs=(port,))},
        execution=ExecutionContract(required_runtime_capabilities=capabilities),
        assumptions=AssumptionSet(assumptions),
    )


def _unknown_role_contract() -> ComponentContract:
    return _vocabulary_contract(role="unknown_role")


def _unknown_data_kind_contract() -> ComponentContract:
    return _vocabulary_contract(kind="unknown_kind")


def _unknown_cardinality_contract() -> ComponentContract:
    return _vocabulary_contract(cardinality="UNKNOWN_CARDINALITY")


def _unknown_top_level_schema_variable_contract() -> ComponentContract:
    return _vocabulary_contract(
        bindings={
            "unknown_variable": Product(
                elements=(
                    Var(name="anything"),
                    Fixed(value="dense"),
                )
            )
        }
    )


def _unknown_required_service_contract() -> ComponentContract:
    return _vocabulary_contract(services=("UnknownService",))


def _unknown_assumption_contract() -> ComponentContract:
    return _vocabulary_contract(assumptions={"unknown.assumption": True})


def _unknown_runtime_capability_contract() -> ComponentContract:
    return _vocabulary_contract(capabilities=("missing_capability",))


class _PartsChild:
    def contract(self) -> ComponentContract:
        return ComponentContract(
            assumptions=AssumptionSet({"evaluation.deterministic": False})
        )


class _PartsPassDummy:
    def __init__(self) -> None:
        self.child = _PartsChild()

    def contract(self) -> ComponentContract:
        return ComponentContract(
            parts=(PartSpec(name="child", contract=self.child.contract()),)
        )


class _PartsMismatchDummy(_PartsPassDummy):
    def contract(self) -> ComponentContract:
        return ComponentContract(
            parts=(PartSpec(name="child", contract=ComponentContract()),)
        )


class _PartsMissingDummy:
    def contract(self) -> ComponentContract:
        return ComponentContract(
            parts=(PartSpec(name="child", contract=ComponentContract()),)
        )


def test_builtin_contract_implementation_set_is_not_empty() -> None:
    assert _IMPLEMENTED_COMPONENTS


def test_non_class_registry_entries_are_explicitly_allowed() -> None:
    actual = {
        name
        for name, obj in _REGISTRY.items()
        if not inspect.isclass(obj)
        and getattr(obj, "__module__", "").startswith("saealib.")
    }
    assert actual == _NON_CLASS_REGISTRY_NAMES


def test_every_builtin_class_has_a_construction_path() -> None:
    constructed = {
        name for name, instance in _BUILTIN_INSTANCES if instance is not None
    }
    assert set(_CONSTRUCTION_FAILURES) <= set(_RECIPES)
    assert set(_RECIPES) <= set(_BUILTIN_CLASSES)
    assert constructed == set(_BUILTIN_CLASSES)


@pytest.mark.parametrize(
    ("name", "instance"),
    _IMPLEMENTED_COMPONENTS,
    ids=[name for name, _ in _IMPLEMENTED_COMPONENTS],
)
def test_builtin_contract_is_state_free(name: str, instance: Any) -> None:
    del name
    _check_state_free(instance)


@pytest.mark.parametrize(
    ("name", "instance"),
    _IMPLEMENTED_COMPONENTS,
    ids=[name for name, _ in _IMPLEMENTED_COMPONENTS],
)
def test_builtin_contract_is_pure(name: str, instance: Any) -> None:
    del name
    _check_purity(instance)


@pytest.mark.parametrize(
    ("name", "instance"),
    _IMPLEMENTED_COMPONENTS,
    ids=[name for name, _ in _IMPLEMENTED_COMPONENTS],
)
def test_builtin_port_names_are_valid(name: str, instance: Any) -> None:
    del name
    _check_port_names(instance)


@pytest.mark.parametrize(
    ("name", "instance"),
    _IMPLEMENTED_COMPONENTS,
    ids=[name for name, _ in _IMPLEMENTED_COMPONENTS],
)
def test_builtin_contract_vocabulary_references_are_registered(
    name: str, instance: Any
) -> None:
    del name
    _check_vocabularies(instance)


@pytest.mark.parametrize(
    ("name", "instance"),
    _IMPLEMENTED_COMPONENTS,
    ids=[name for name, _ in _IMPLEMENTED_COMPONENTS],
)
def test_builtin_parts_match_held_components(name: str, instance: Any) -> None:
    del name
    _check_parts(instance)


def test_purity_check_accepts_dummy() -> None:
    _check_purity(_PureDummy())


def test_purity_check_rejects_dummy() -> None:
    with pytest.raises(AssertionError):
        _check_purity(_ImpureDummy())


def test_state_free_check_accepts_nested_dummy() -> None:
    _check_state_free(_StateFreeParent())


def test_state_free_check_rejects_nested_dummy() -> None:
    with pytest.raises(AssertionError):
        _check_state_free(_StateReadingParent())


def test_port_name_check_accepts_same_name_in_distinct_roles() -> None:
    _check_port_names(_PortNamesPassDummy())


def test_port_name_check_rejects_duplicate_name_in_direction() -> None:
    with pytest.raises(AssertionError):
        _check_port_names(_PortNamesFailDummy())


def test_vocabulary_check_accepts_nested_registered_references() -> None:
    _check_vocabularies(_VocabularyPassDummy())


def test_vocabulary_check_ignores_nested_variable_names() -> None:
    _check_vocabularies(
        _VocabularyDummy(
            _vocabulary_contract(
                bindings={
                    "representation": Product(
                        elements=(
                            Var(name="anything"),
                            Fixed(value="dense"),
                        )
                    )
                }
            )
        )
    )


@pytest.mark.parametrize(
    ("vocabulary_name", "contract_factory"),
    [
        ("role", _unknown_role_contract),
        ("data kind", _unknown_data_kind_contract),
        ("cardinality", _unknown_cardinality_contract),
        ("top-level schema variable", _unknown_top_level_schema_variable_contract),
        ("required service", _unknown_required_service_contract),
        ("assumption key", _unknown_assumption_contract),
        ("runtime capability", _unknown_runtime_capability_contract),
    ],
)
def test_vocabulary_check_rejects_each_unregistered_reference(
    vocabulary_name: str, contract_factory: Callable[[], ComponentContract]
) -> None:
    del vocabulary_name
    with pytest.raises(AssertionError):
        _check_vocabularies(_VocabularyDummy(contract_factory()))


def test_parts_check_accepts_matching_child_contract() -> None:
    _check_parts(_PartsPassDummy())


def test_parts_check_rejects_mismatched_child_contract() -> None:
    with pytest.raises(AssertionError):
        _check_parts(_PartsMismatchDummy())


def test_parts_check_rejects_missing_child_attribute() -> None:
    with pytest.raises(AssertionError):
        _check_parts(_PartsMissingDummy())


@pytest.mark.skip(
    reason="ADR-0006 instrumented state store is not implemented until Phase 2"
)
def test_declared_writes_match_instrumented_state_store() -> None:
    pass


@pytest.mark.skip(reason="CompilationRule is not implemented until Phase 2")
def test_custom_diagnostics_have_paths_and_registered_codes() -> None:
    pass
