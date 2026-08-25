"""Failure and composition semantics for semantic default resolution."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from saealib.context import OptimizationState
from saealib.core.contracts import ComponentContract, PartSpec
from saealib.defaults.context import DefaultContext
from saealib.defaults.keys import (
    INITIAL_ARCHIVE_SIZE,
    MAX_EVALUATIONS,
    POPULATION_SIZE,
    DefaultKey,
)
from saealib.defaults.model import DefaultHint, DefaultStrength
from saealib.defaults.resolver import DEFAULT_RESOLVER
from saealib.exceptions import ConfigurationError, ValidationError
from saealib.execution.initializer import LHSInitializer
from saealib.optimizer import Optimizer
from saealib.problem import Problem
from saealib.strategies.base import OptimizationStrategy
from saealib.termination import Termination


def _problem(dim: int = 3) -> Problem:
    return Problem(
        func=lambda x: np.sum(x),
        dim=dim,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0] * dim,
        ub=[1.0] * dim,
    )


class _StaticProvider:
    def __init__(self, *hints: DefaultHint) -> None:
        self.hints = hints

    def default_hints(self, ctx: DefaultContext) -> tuple[DefaultHint, ...]:
        return self.hints


def test_same_strength_different_values_raise_configuration_error() -> None:
    ctx = DefaultContext(problem=_problem())
    providers = [
        _StaticProvider(
            DefaultHint(
                key=POPULATION_SIZE,
                value=8,
                strength=DefaultStrength.RECOMMENDED,
                source="first",
            )
        ),
        _StaticProvider(
            DefaultHint(
                key=POPULATION_SIZE,
                value=12,
                strength=DefaultStrength.RECOMMENDED,
                source="second",
            )
        ),
    ]

    with pytest.raises(ConfigurationError, match=r"Conflict for population\.size"):
        DEFAULT_RESOLVER.resolve(ctx, providers)


def test_same_strength_same_value_is_resolved_once() -> None:
    ctx = DefaultContext(problem=_problem())
    hint = DefaultHint(
        key=POPULATION_SIZE,
        value=8,
        strength=DefaultStrength.RECOMMENDED,
        source="provider",
    )

    resolution = DEFAULT_RESOLVER.resolve(
        ctx, [_StaticProvider(hint), _StaticProvider(hint)]
    )

    assert resolution.get(POPULATION_SIZE) == 8
    assert resolution.resolved[POPULATION_SIZE].selected_hint is hint


def test_provider_exception_fails_fast_as_configuration_error() -> None:
    class ExplodingProvider:
        def default_hints(self, ctx: DefaultContext) -> tuple[DefaultHint, ...]:
            raise RuntimeError("provider bug")

    with pytest.raises(ConfigurationError, match="provider bug"):
        DEFAULT_RESOLVER.resolve(
            DefaultContext(problem=_problem()), [ExplodingProvider()]
        )


def test_default_hint_rejects_wrong_types_and_bool_integer_values() -> None:
    with pytest.raises(ValidationError, match="not bool"):
        DefaultHint(
            key=POPULATION_SIZE,
            value=True,
            strength=DefaultStrength.FALLBACK,
            source="test",
        )

    with pytest.raises(ValidationError, match="must be an int"):
        DefaultHint(
            key=POPULATION_SIZE,
            value=4.0,
            strength=DefaultStrength.FALLBACK,
            source="test",
        )


class _NestedLeaf:
    def default_hints(self, ctx: DefaultContext) -> tuple[DefaultHint, ...]:
        return (
            DefaultHint(
                key=POPULATION_SIZE,
                value=92,
                strength=DefaultStrength.RECOMMENDED,
                source="nested-leaf",
            ),
            DefaultHint(
                key=INITIAL_ARCHIVE_SIZE,
                value=80,
                strength=DefaultStrength.RECOMMENDED,
                source="nested-leaf",
            ),
        )


class _NestedMiddle:
    def __init__(self) -> None:
        self.leaf = _NestedLeaf()


class _NestedRoot(OptimizationStrategy):
    def __init__(self) -> None:
        self.nested = _NestedMiddle()
        self.context: DefaultContext | None = None

    def default_hints(self, ctx: DefaultContext) -> tuple[DefaultHint, ...]:
        self.context = ctx
        return (
            DefaultHint(
                key=MAX_EVALUATIONS,
                value=37,
                strength=DefaultStrength.RECOMMENDED,
                source="custom-root",
            ),
        )

    def contract(self) -> ComponentContract:
        return ComponentContract(
            parts=(
                PartSpec(
                    name="nested",
                    contract=ComponentContract(
                        parts=(
                            PartSpec(
                                name="leaf",
                                contract=ComponentContract(),
                            ),
                        )
                    ),
                ),
            )
        )


class _RequiredArchiveRoot(_NestedRoot):
    def default_hints(self, ctx: DefaultContext) -> tuple[DefaultHint, ...]:
        return (
            DefaultHint(
                key=POPULATION_SIZE,
                value=92,
                strength=DefaultStrength.RECOMMENDED,
                source="required-archive-root",
            ),
            DefaultHint(
                key=INITIAL_ARCHIVE_SIZE,
                value=80,
                strength=DefaultStrength.REQUIRED,
                source="required-archive-root",
            ),
        )


class _InitializerHintProvider(LHSInitializer):
    def default_hints(self, ctx: DefaultContext) -> tuple[DefaultHint, ...]:
        return (
            DefaultHint(
                key=MAX_EVALUATIONS,
                value=41,
                strength=DefaultStrength.RECOMMENDED,
                source="initializer-root",
            ),
        )


def test_optimizer_discovers_custom_nested_providers_and_uses_all_keys() -> None:
    root = _NestedRoot()
    optimizer = Optimizer(_problem(dim=25)).set_strategy(root)

    optimizer.resolve_defaults()

    initializer = cast(LHSInitializer, optimizer.initializer)
    termination = cast(Termination, optimizer.termination)
    assert initializer.n_init_population == 92
    assert initializer.n_init_archive == 92
    assert (
        termination.is_terminated(cast(OptimizationState, SimpleNamespace(fe=36)))
        is False
    )
    assert (
        termination.is_terminated(cast(OptimizationState, SimpleNamespace(fe=37)))
        is True
    )

    resolution = optimizer._default_resolution
    assert resolution is not None
    assert resolution.resolved[POPULATION_SIZE].selected_hint.source == "nested-leaf"
    assert resolution.get(INITIAL_ARCHIVE_SIZE) == 92
    assert resolution.resolved[INITIAL_ARCHIVE_SIZE].selected_hint.source == "optimizer"
    assert (
        any(
            hint.source == "nested-leaf"
            for hint in resolution.resolved[INITIAL_ARCHIVE_SIZE].alternatives
        )
        is True
    )
    assert resolution.resolved[MAX_EVALUATIONS].selected_hint.source == "custom-root"
    assert root.context is not None
    assert root.context.components is not None
    assert root.context.components["strategy.nested.leaf"] is root.nested.leaf


def test_resolution_normalization_is_idempotent_and_matches_initializer() -> None:
    optimizer = Optimizer(_problem(dim=25)).set_strategy(_NestedRoot())

    optimizer.resolve_defaults()
    first = optimizer._default_resolution
    assert first is not None
    initializer = cast(LHSInitializer, optimizer.initializer)
    assert first.get(POPULATION_SIZE) == initializer.n_init_population
    assert first.get(INITIAL_ARCHIVE_SIZE) == initializer.n_init_archive

    optimizer.resolve_defaults()
    second = optimizer._default_resolution
    assert second is not None
    assert second == first
    assert second.get(INITIAL_ARCHIVE_SIZE) == 92


def test_required_archive_smaller_than_population_is_an_error() -> None:
    optimizer = Optimizer(_problem(dim=25)).set_strategy(_RequiredArchiveRoot())

    with pytest.raises(ConfigurationError, match=r"REQUIRED.*archive\.initial_size"):
        optimizer.resolve_defaults()


def test_added_root_component_provider_is_discovered() -> None:
    initializer = _InitializerHintProvider(n_init_archive=10, n_init_population=10)
    optimizer = Optimizer(_problem()).set_initializer(initializer)

    optimizer.resolve_defaults()

    resolution = optimizer._default_resolution
    assert resolution is not None
    assert resolution.get(MAX_EVALUATIONS) == 41
    assert resolution.resolved[MAX_EVALUATIONS].selected_hint.source == (
        "initializer-root"
    )
    termination = cast(Termination, optimizer.termination)
    assert termination.is_terminated(cast(OptimizationState, SimpleNamespace(fe=41)))


def test_semantic_resolution_runs_with_explicit_initializer() -> None:
    root = _NestedRoot()
    explicit = LHSInitializer(n_init_archive=100, n_init_population=80)
    optimizer = Optimizer(_problem(dim=25)).set_strategy(root).set_initializer(explicit)

    optimizer.resolve_defaults()

    assert optimizer.initializer is explicit
    assert optimizer._default_resolution is not None
    assert optimizer._default_resolution.get(POPULATION_SIZE) == 92
    assert optimizer._default_resolution.get(INITIAL_ARCHIVE_SIZE) == 92


def test_default_resolution_uses_default_key_mapping() -> None:
    custom_key = DefaultKey("test.value", str)
    hint = DefaultHint(
        key=custom_key,
        value="ok",
        strength=DefaultStrength.FALLBACK,
        source="test",
    )

    resolution = DEFAULT_RESOLVER.resolve(
        DefaultContext(problem=_problem()), [_StaticProvider(hint)]
    )

    assert custom_key in resolution.values
    assert resolution.resolved[custom_key].value == "ok"
