from __future__ import annotations

import copy
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from saealib.identity import IDAllocator
from saealib.population import Population

_MUTATING_METHODS = frozenset(
    {
        "add",
        "append",
        "clear",
        "delete",
        "extend",
        "mod_structure",
        "mod_value",
        "reorder",
        "_append_genomes",
        "_append_internal",
        "_assign_ids",
        "_extend_internal",
        "_replace_from_population",
        "_reorder_genomes",
        "set_cache",
        "truncate",
        "update_array",
        "update_rows",
    }
)


class _ReadOnlyFacade:
    __slots__ = ("_value",)

    def __init__(self, value: object) -> None:
        object.__setattr__(self, "_value", value)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"read-only value; cannot set {name!r}")

    def __getattribute__(self, name: str) -> Any:
        if name in {"_value", "_unwrap"}:
            raise AttributeError("read-only value has no raw storage attribute")
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        value = getattr(object.__getattribute__(self, "_value"), name)
        if name in _MUTATING_METHODS:
            raise AttributeError(f"read-only value has no mutating method {name!r}")
        if callable(value):

            def invoke(*args: Any, **kwargs: Any) -> Any:
                result = value(*args, **kwargs)
                raw = object.__getattribute__(self, "_value")
                if isinstance(result, Population) and result is not raw:
                    return result
                return _readonly_value(result)

            return invoke
        return _readonly_value(value)

    def __len__(self) -> int:
        return len(object.__getattribute__(self, "_value"))  # type: ignore[arg-type]

    def __getitem__(self, index: object) -> Any:
        value = object.__getattribute__(self, "_value")[index]  # type: ignore[index]
        return value if isinstance(value, Population) else _readonly_value(value)

    def __iter__(self):
        value = object.__getattribute__(self, "_value")
        return iter(_readonly_value(item) for item in value)  # type: ignore[union-attr]

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, "_value"))

    def _unwrap(self) -> object:
        return object.__getattribute__(self, "_value")


def _readonly_array(value: np.ndarray) -> np.ndarray:
    result = value.view()
    result.setflags(write=False)
    return result


def _readonly_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _readonly_array(value)
    if isinstance(value, Population):
        return _ReadOnlyFacade(value)
    if isinstance(value, np.random.Generator):
        return copy.deepcopy(value)
    if isinstance(value, IDAllocator):
        return IDAllocator(value.next_value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _readonly_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple, set)):
        return type(value)(_readonly_value(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_readonly_value(item) for item in value)
    if hasattr(value, "__dict__") and not isinstance(value, type):
        try:
            return copy.deepcopy(value)
        except (TypeError, ValueError):
            return _ReadOnlyFacade(value)
    return value


def _unwrap_readonly(value: Any) -> Any:
    if isinstance(value, _ReadOnlyFacade):
        return object.__getattribute__(value, "_unwrap")()
    if isinstance(value, Mapping):
        return {key: _unwrap_readonly(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_unwrap_readonly(item) for item in value)
    if isinstance(value, list):
        return [_unwrap_readonly(item) for item in value]
    if isinstance(value, set):
        return {_unwrap_readonly(item) for item in value}
    return value
