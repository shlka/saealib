"""Tests for the move-based state store."""

from dataclasses import dataclass

import numpy as np
import pytest

from saealib.core.contracts.state import StateContract
from saealib.core.state import (
    PopulationRowUpdate,
    StateKey,
    StatePatch,
    StateStore,
    StateUpdate,
)
from saealib.exceptions import ValidationError
from saealib.population import Population, PopulationAttribute


def key(name: str, version: int = 1) -> StateKey[object]:
    """Build a test state key."""
    return StateKey(namespace="user", name=name, schema_version=version)


def population() -> Population:
    """Build a small population with one numeric column."""
    pop = Population([PopulationAttribute(name="x", dtype=np.float64)])
    pop.append({"x": 1.0})
    pop.append({"x": 2.0})
    return pop


def test_patch_replaces_deletes_and_distinguishes_schema_versions() -> None:
    v1 = key("value", 1)
    v2 = key("value", 2)
    remove = key("remove")
    store = StateStore({v1: "old", v2: "new", remove: True})

    next_store = store.apply_patch(
        StatePatch(writes={v1: "replacement"}, deletes=frozenset({remove}))
    )

    assert next_store.get(v1) == "replacement"
    assert next_store.get(v2) == "new"
    assert not next_store.contains(remove)


def test_subclass_clone_hook_preserves_custom_constructor_state() -> None:
    state_key = key("value")

    class CustomStore(StateStore):
        def __init__(self, initial=None, *, generation=0, token="token"):
            super().__init__(initial, generation=generation)
            self.token = token

        def _clone(self, values, *, generation):
            return type(self)(values, generation=generation, token=self.token)

    store = CustomStore({state_key: 1}, token="observer")
    updated = store.apply_patch(StatePatch(writes={state_key: 2}))

    assert isinstance(updated, CustomStore)
    assert updated.token == "observer"
    assert updated.get(state_key) == 2


def test_old_store_raises_after_successful_patch() -> None:
    state_key = key("value")
    store = StateStore({state_key: 1})
    store.apply_patch(StatePatch(writes={state_key: 3}))

    with pytest.raises(RuntimeError):
        store.get(state_key)


def test_old_view_raises_after_successful_patch() -> None:
    state_key = key("value")
    store = StateStore({state_key: np.array([1, 2])})
    view = store.view((state_key,))
    store.apply_patch(StatePatch(writes={state_key: 3}))

    with pytest.raises(RuntimeError, match=r"StateView.*patch"):
        view.get(state_key)


def test_view_requires_declared_keys_and_returns_read_only_arrays() -> None:
    allowed = key("allowed")
    denied = key("denied")
    array = np.array([1.0, 2.0])
    view = StateStore({allowed: array, denied: 4}).view(StateContract(reads=(allowed,)))

    result = view.get(allowed)
    assert isinstance(result, np.ndarray)
    assert not result.flags.writeable
    with pytest.raises(ValueError):
        result[0] = 9.0
    with pytest.raises(KeyError, match="not declared"):
        view.get(denied)


def test_population_row_update_is_atomic_and_bumps_once() -> None:
    state_key = key("population")
    pop = population()
    before = pop.get_array("x").copy()
    version = pop.value_version
    store = StateStore({state_key: pop})
    invalid = PopulationRowUpdate(
        indices=np.array([0, 1]),
        values={
            "x": np.array([10.0, 20.0]),
            "missing": np.array([1.0, 2.0]),
        },
    )

    with pytest.raises(ValidationError):
        store.apply_patch(StatePatch(writes={state_key: invalid}))
    assert np.array_equal(pop.get_array("x"), before)
    assert pop.value_version == version

    updated = store.apply_patch(
        StatePatch(
            writes={
                state_key: PopulationRowUpdate(
                    indices=np.array([0, 1]),
                    values={"x": np.array([10.0, 20.0])},
                )
            }
        )
    )
    assert np.array_equal(pop.get_array("x")[:2], [10.0, 20.0])
    assert pop.value_version == version + 1
    assert updated.get(state_key) is pop


def test_multiple_row_updates_are_not_atomic_across_the_patch() -> None:
    first_key = key("first_population")
    second_key = key("second_population")
    first = population()
    second = population()
    before_second = second.get_array("x").copy()
    store = StateStore({first_key: first, second_key: second})

    with pytest.raises(ValidationError):
        store.apply_patch(
            StatePatch(
                writes={
                    first_key: PopulationRowUpdate(
                        indices=np.array([0]),
                        values={"x": np.array([10.0])},
                    ),
                    second_key: PopulationRowUpdate(
                        indices=np.array([0]),
                        values={"x": np.array([20.0, 30.0])},
                    ),
                }
            )
        )

    assert np.array_equal(first.get_array("x")[:1], [10.0])
    assert np.array_equal(second.get_array("x"), before_second)
    assert store.get(first_key) is first


@dataclass(frozen=True, kw_only=True)
class UnknownUpdate(StateUpdate):
    """An update type not supported by the built-in store."""


def test_unknown_state_update_is_rejected_without_consuming_store() -> None:
    state_key = key("value")
    store = StateStore({state_key: object()})

    with pytest.raises(ValidationError, match="Unsupported StateUpdate"):
        store.apply_patch(StatePatch(writes={state_key: UnknownUpdate()}))
    assert store.contains(state_key)
