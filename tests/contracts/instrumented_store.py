"""Test-only state access instrumentation for execution-contract checks.

The production state objects deliberately have no tracing hooks.  This module
keeps the tracing boundary in ``tests/`` while preserving the public
``StateStore`` move semantics and the state object's compatibility properties.
"""

from __future__ import annotations

import contextvars
import copy
import functools
import pickle
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import numpy as np

import saealib.context as context_module
from saealib.context import OptimizationState
from saealib.core.state import (
    ARCHIVES_MAIN,
    ARCHIVES_PARETO,
    POPULATIONS_MAIN,
    StateKey,
)
from saealib.core.state.patch import StatePatch
from saealib.core.state.store import StateStore
from saealib.identity import IDAllocator
from saealib.population import Population

__all__ = [
    "Access",
    "InstrumentedStateStore",
    "Recorder",
    "attach_instrumentation",
    "instrumentation_scope",
    "instrumented",
    "instrumented_component",
]

_T = TypeVar("_T")
_MISSING = object()
_ACTIVE: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "saealib_test_instrumented_components", default=()
)
_CURRENT_RECORDER: contextvars.ContextVar[Recorder | None] = contextvars.ContextVar(
    "saealib_test_instrumented_recorder", default=None
)
_SUPPRESS_MECHANICAL_READS: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "saealib_test_suppress_mechanical_reads", default=False
)


@dataclass(frozen=True)
class Access:
    """One observed state operation."""

    operation: str
    key: StateKey
    component: str | None


def _fingerprint(value: object) -> object:
    """Return a cheap fingerprint for controlled in-place state values."""
    if isinstance(value, np.random.Generator):
        return ("rng", pickle.dumps(copy.deepcopy(value.bit_generator.state)))
    if isinstance(value, IDAllocator):
        return ("allocator", value.next_value)
    if isinstance(value, Population):
        return (
            "population",
            len(value),
            value._structure_version,
            value._value_version,
        )
    return None


class Recorder:
    """Accumulate accesses, including accesses outside all wrappers."""

    def __init__(self) -> None:
        self.events: list[Access] = []
        self.counts: Counter[tuple[str | None, str, StateKey]] = Counter()
        self._stores: list[InstrumentedStateStore] = []

    @property
    def unowned(self) -> Counter[tuple[str, StateKey]]:
        """Return unowned counts keyed by operation and state key."""
        return Counter(
            {
                (operation, key): count
                for (owner, operation, key), count in self.counts.items()
                if owner is None
            }
        )

    def record(self, operation: str, key: StateKey) -> None:
        """Record an operation for the innermost active wrapper."""
        stack = _ACTIVE.get()
        owner = stack[-1] if stack else None
        self.events.append(Access(operation, key, owner))
        self.counts[(owner, operation, key)] += 1

    def keys(
        self, component: str | None = None, operation: str | None = None
    ) -> set[StateKey]:
        """Return distinct observed keys, optionally filtered."""
        return {
            key
            for owner, op, key in self.counts
            if (component is None or owner == component)
            and (operation is None or op == operation)
        }

    def report_unowned(self) -> dict[str, dict[str, int]]:
        """Return a stable, failure-friendly summary of unowned accesses."""
        return {
            f"{operation}:{key.namespace}.{key.name}": {"count": count}
            for (operation, key), count in sorted(
                self.unowned.items(), key=lambda item: (item[0][0], repr(item[0][1]))
            )
        }

    def register_store(self, store: InstrumentedStateStore) -> None:
        """Keep every moved store available for in-place mutation checks."""
        self._stores.append(store)

    def flush(self) -> None:
        """Record controlled in-place mutations made since the last flush."""
        for store in tuple(self._stores):
            store._flush_mutations()


class InstrumentedStateStore(StateStore):
    """A test-only ``StateStore`` that preserves recorder identity on moves."""

    def __init__(
        self,
        initial: Mapping[StateKey, object] | None = None,
        *,
        generation: int = 0,
        recorder: Recorder | None = None,
    ) -> None:
        super().__init__(initial, generation=generation)
        self.recorder = recorder or _CURRENT_RECORDER.get()
        self._snapshots = {
            key: _fingerprint(value) for key, value in self._values.items()
        }
        if self.recorder is not None:
            self.recorder.register_store(self)

    def get(self, key: StateKey[_T]) -> _T:
        if not _SUPPRESS_MECHANICAL_READS.get() and self.recorder is not None:
            self.recorder.record("read", key)
        return super().get(key)

    def contains(self, key: StateKey) -> bool:
        if not _SUPPRESS_MECHANICAL_READS.get() and self.recorder is not None:
            self.recorder.record("read", key)
        return super().contains(key)

    def apply_patch(self, patch: StatePatch) -> InstrumentedStateStore:
        self._flush_mutations()
        moved = super().apply_patch(patch)
        if self.recorder is not None:
            for key in (*patch.writes, *patch.deletes):
                self.recorder.record("write", key)
        return type(self)(
            moved._values,
            generation=moved.generation,
            recorder=self.recorder,
        )

    def _flush_mutations(self) -> None:
        """Detect RNG, allocator, and collection changes made in place."""
        if self.recorder is None:
            return
        for key, value in self._values.items():
            before = self._snapshots.get(key, _MISSING)
            after = _fingerprint(value)
            if before is not _MISSING and before != after:
                self.recorder.record("write", key)
            self._snapshots[key] = after


_COLLECTION_KEYS = {
    ("populations", "main"): POPULATIONS_MAIN,
    ("archives", "main"): ARCHIVES_MAIN,
    ("archives", "pareto"): ARCHIVES_PARETO,
}
_ProductionCollection = context_module._NamedCollection


class _RecordingCollection(_ProductionCollection):
    """Observe compatibility-property reads without changing production code."""

    def __init__(
        self,
        kind: str,
        values: Mapping[str, Any],
        recorder: Recorder | None = None,
    ) -> None:
        self._recorder = recorder or _CURRENT_RECORDER.get()
        super().__init__(kind, dict(values))

    def __getitem__(self, name: str) -> Any:
        recorder = _CURRENT_RECORDER.get() or self._recorder
        if recorder is not None:
            key = _COLLECTION_KEYS.get((self._kind, name)) or StateKey(
                namespace=self._kind, name=name, schema_version=1
            )
            recorder.record("read", key)
        return super().__getitem__(name)


def _record_collection(kind: str, name: str) -> StateKey:
    return _COLLECTION_KEYS.get((kind, name)) or StateKey(
        namespace=kind, name=name, schema_version=1
    )


def _replace_collection_with_recorder(
    state: OptimizationState, attribute: str, kind: str
) -> None:
    collection = object.__getattribute__(state, attribute)
    if isinstance(collection, _RecordingCollection):
        return
    replacement = _RecordingCollection(
        kind,
        dict(collection),
        recorder=object.__getattribute__(state, "_store").recorder,
    )
    replacement._bind(getattr(collection, "_on_change", None))
    object.__setattr__(state, attribute, replacement)


def attach_instrumentation(
    state: OptimizationState, recorder: Recorder
) -> OptimizationState:
    """Install tracing on a pre-existing state and its compatibility collections."""
    store = object.__getattribute__(state, "_store")
    if isinstance(store, InstrumentedStateStore):
        if store.recorder is recorder:
            return state
        store = StateStore(store._values, generation=store.generation)
    object.__setattr__(
        state,
        "_store",
        InstrumentedStateStore(
            store._values, generation=store.generation, recorder=recorder
        ),
    )
    _replace_collection_with_recorder(state, "_population_collection", "populations")
    _replace_collection_with_recorder(state, "_archive_collection", "archives")
    return state


def _active_recorder(state: OptimizationState | None = None) -> Recorder | None:
    recorder = _CURRENT_RECORDER.get()
    if recorder is not None:
        return recorder
    if state is not None:
        store = object.__getattribute__(state, "_store")
        if isinstance(store, InstrumentedStateStore):
            return store.recorder
    return None


def _record_replace_delta(
    before: Mapping[StateKey, object],
    after: Mapping[StateKey, object],
    recorder: Recorder,
) -> None:
    """Record logical writes made by ``OptimizationState.replace``."""
    for key in before.keys() | after.keys():
        old = before.get(key, _MISSING)
        new = after.get(key, _MISSING)
        if old is _MISSING or new is _MISSING:
            recorder.record("write", key)
            continue
        if old is new:
            continue
        if isinstance(old, np.ndarray) and isinstance(new, np.ndarray):
            equal = np.array_equal(old, new)
        else:
            try:
                equal = bool(old == new)
            except (TypeError, ValueError):
                equal = False
        if not equal:
            recorder.record("write", key)


@contextmanager
def instrumented_component(
    name: str, recorder: Recorder | None = None
) -> Iterator[Recorder]:
    """Attribute accesses in the block to ``name``; the innermost wins."""
    active = recorder or Recorder()
    token = _ACTIVE.set((*_ACTIVE.get(), name))
    recorder_token = _CURRENT_RECORDER.set(active)
    try:
        yield active
    finally:
        active.flush()
        _CURRENT_RECORDER.reset(recorder_token)
        _ACTIVE.reset(token)


def instrumented(
    name: str, recorder: Recorder | None = None
) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Decorate a public component entrypoint with access attribution."""

    def decorate(function: Callable[..., _T]) -> Callable[..., _T]:
        @functools.wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> _T:
            with instrumented_component(name, recorder) as active:
                result = function(*args, **kwargs)
                if isinstance(result, OptimizationState):
                    attach_instrumentation(result, active)
                return result

        return wrapped

    return decorate


@contextmanager
def instrumentation_scope(recorder: Recorder) -> Iterator[Recorder]:
    """Make every new/replaced state inherit ``recorder``.

    ``OptimizationState.replace`` is a value-copy constructor, not a call to
    ``StateStore.apply_patch``.  The test shim therefore records its before /
    after store delta as logical writes and suppresses the dataclass's
    mechanical copy reads.  This keeps the compliance trace about the
    component's state effect rather than the transport implementation.
    """
    production_store = context_module.StateStore
    production_collection = context_module._NamedCollection
    production_replace = OptimizationState.replace
    instrumented_context = cast(Any, context_module)

    def traced_replace(self: OptimizationState, **kwargs: Any) -> OptimizationState:
        active = _active_recorder(self)
        if active is None:
            return production_replace(self, **kwargs)
        active.flush()
        before = dict(object.__getattribute__(self, "_store")._values)
        token = _SUPPRESS_MECHANICAL_READS.set(True)
        try:
            result = production_replace(self, **kwargs)
        finally:
            _SUPPRESS_MECHANICAL_READS.reset(token)
        after = dict(object.__getattribute__(result, "_store")._values)
        _record_replace_delta(before, after, active)
        active.flush()
        return result

    recorder_token = _CURRENT_RECORDER.set(recorder)
    instrumented_context.StateStore = InstrumentedStateStore
    instrumented_context._NamedCollection = _RecordingCollection
    OptimizationState.replace = traced_replace
    try:
        yield recorder
    finally:
        recorder.flush()
        instrumented_context.StateStore = production_store
        instrumented_context._NamedCollection = production_collection
        OptimizationState.replace = production_replace
        _CURRENT_RECORDER.reset(recorder_token)
