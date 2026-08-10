"""Move-based state storage and restricted read views."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

from saealib.core.state.keys import StateKey
from saealib.core.state.patch import (
    PopulationRowUpdate,
    StatePatch,
    StateUpdate,
)
from saealib.exceptions import ValidationError
from saealib.population import Population

if TYPE_CHECKING:
    from saealib.core.contracts.state import StateContract

__all__ = ["StateStore", "StateView"]

ValueT = TypeVar("ValueT")


class StateStore:
    """Own versioned state values and consume each instance after a move."""

    def __init__(
        self,
        initial: Mapping[StateKey, object] | None = None,
        *,
        generation: int = 0,
    ) -> None:
        if generation < 0:
            raise ValidationError("StateStore generation must be non-negative")
        values = {} if initial is None else dict(initial)
        self._validate_keys(values)
        self._values: Mapping[StateKey, object] = MappingProxyType(values)
        self._generation = generation
        self._moved = False

    def __copy__(self) -> StateStore:
        result = type(self).__new__(type(self))
        result.__dict__ = self.__dict__.copy()
        return result

    @property
    def generation(self) -> int:
        """Return this store's generation while it is still usable."""
        self._check_live()
        return self._generation

    def get(self, key: StateKey[ValueT]) -> ValueT:
        """Return the value for ``key`` or raise ``KeyError`` if absent."""
        self._check_live()
        self._validate_key(key)
        return cast(ValueT, self._values[key])

    def contains(self, key: StateKey) -> bool:
        """Return whether ``key`` is present in this store."""
        self._check_live()
        self._validate_key(key)
        return key in self._values

    def view(
        self,
        reads: Iterable[StateKey] | StateContract,
        *,
        context: object | None = None,
        dispatch: Callable[[object], None] | None = None,
    ) -> StateView:
        """Create a read-only view limited to the supplied declared keys."""
        self._check_live()
        if hasattr(reads, "reads"):
            declared = tuple(cast(Iterable[StateKey], reads.reads))
        else:
            declared = tuple(reads)
        return StateView(self, declared, context=context, dispatch=dispatch)

    def apply_patch(self, patch: StatePatch) -> StateStore:
        """Apply ``patch`` and return the new store, consuming this store.

        Each individual ``StateUpdate`` inherits the atomicity of the
        operation it delegates to, such as ``Population.update_rows``.  A
        patch containing multiple ``StateUpdate`` values is not atomic across
        those updates: if a later update fails, earlier updates remain applied
        and this store remains live.  Keeping validation in the delegated
        operation avoids duplicating its row-validation rules here.
        """
        self._check_live()
        if not isinstance(patch, StatePatch):
            raise ValidationError("apply_patch() expects a StatePatch")
        writes = dict(patch.writes)
        deletes = frozenset(patch.deletes)
        self._validate_keys(writes)
        self._validate_keys({key: None for key in deletes})
        if writes.keys() & deletes:
            raise ValidationError("A StatePatch cannot write and delete the same key")

        if (
            type(self) is StateStore
            and not deletes
            and not any(isinstance(value, StateUpdate) for value in writes.values())
        ):
            values = dict(self._values)
            values.update(writes)
            self._moved = True
            result = StateStore.__new__(StateStore)
            result._values = MappingProxyType(values)
            result._generation = self._generation + 1
            result._moved = False
            return result

        values = dict(self._values)
        self._preflight_update_kinds(writes, values)
        for key, update in writes.items():
            if isinstance(update, StateUpdate):
                current = values[key]
                if isinstance(update, PopulationRowUpdate):
                    cast(Population, current).update_rows(
                        update.indices,
                        dict(update.values),
                        genome=update.genome,
                    )
            else:
                values[key] = update
        for key in deletes:
            values.pop(key, None)

        self._moved = True
        return self._clone(values, generation=self._generation + 1)

    def _clone(
        self, values: Mapping[StateKey, object], *, generation: int
    ) -> StateStore:
        return type(self)(values, generation=generation)

    @staticmethod
    def _preflight_update_kinds(
        writes: Mapping[StateKey, object],
        values: Mapping[StateKey, object],
    ) -> None:
        """Validate update dispatch before mutating any owned object.

        Row shape, dtype, and index validation deliberately remain in
        ``Population.update_rows``.  That method is the single source of truth
        for row-update semantics and atomicity.
        """
        for key, update in writes.items():
            if not isinstance(update, StateUpdate):
                continue
            if key not in values:
                raise ValidationError(f"Cannot update missing state key: {key!r}")
            if isinstance(update, PopulationRowUpdate):
                current = values[key]
                if not isinstance(current, Population):
                    raise ValidationError(
                        "PopulationRowUpdate requires a Population state value"
                    )
            else:
                raise ValidationError(
                    f"Unsupported StateUpdate type: {type(update).__name__}"
                )

    def _check_live(self) -> None:
        if self._moved:
            raise RuntimeError("StateStore has been moved; use the returned store")

    @staticmethod
    def _validate_key(key: object) -> None:
        if not isinstance(key, StateKey):
            raise ValidationError("State store keys must be StateKey values")

    @classmethod
    def _validate_keys(cls, keys: Iterable[object]) -> None:
        for key in keys:
            cls._validate_key(key)


class StateView:
    """Read-only projection of a store for declared state keys."""

    def __init__(
        self,
        store: StateStore,
        reads: Iterable[StateKey] | StateContract,
        *,
        context: object | None = None,
        dispatch: Callable[[object], None] | None = None,
    ) -> None:
        if hasattr(reads, "reads"):
            declared = tuple(cast(Iterable[StateKey], reads.reads))
        else:
            declared = tuple(reads)
        StateStore._validate_keys(declared)
        self._store = store
        store._check_live()
        self._reads = frozenset(declared)
        self._context = context
        self._dispatch = dispatch

    def get(self, key: StateKey[ValueT]) -> ValueT:
        """Return a declared value, making arrays read-only for the caller."""
        self._check_live()
        StateStore._validate_key(key)
        if key not in self._reads:
            raise KeyError(f"State key was not declared as a read: {key!r}")
        value = self._store.get(key)
        if isinstance(value, np.ndarray):
            readonly = value.view()
            readonly.setflags(write=False)
            return cast(ValueT, readonly)
        return value

    def contains(self, key: StateKey) -> bool:
        """Return presence for a declared key."""
        self._check_live()
        StateStore._validate_key(key)
        if key not in self._reads:
            raise KeyError(f"State key was not declared as a read: {key!r}")
        return self._store.contains(key)

    @property
    def context(self) -> Any:
        """Return the runtime context bound by the execution owner."""
        self._check_live()
        if self._context is None:
            raise ValidationError("StateView has no runtime context")
        return self._context

    def dispatch(self, event: object) -> None:
        """Dispatch a component event through the bound runtime callback."""
        self._check_live()
        if self._dispatch is not None:
            self._dispatch(event)

    def _check_live(self) -> None:
        try:
            self._store._check_live()
        except RuntimeError as exc:
            raise RuntimeError("StateView is stale after a state patch") from exc
