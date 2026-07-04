"""Registry: name -> class resolution and recursive construction from specs.

A "spec" is one of:

- a live instance, returned unchanged
- a string ``"Name"``, resolved via the registry (or as a dotted import path
  ``"module.submodule.ClassName"`` if not registered) and instantiated with
  no arguments
- a mapping ``{"type": "Name", "params": {...}}``, resolved the same way and
  instantiated with ``params`` (values that are themselves specs are built
  recursively first)
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, TypeVar

from saealib.exceptions import ValidationError

_REGISTRY: dict[str, type] = {}

T = TypeVar("T", bound=type)


def register(name: str | None = None) -> Callable[[T], T]:
    """Class decorator registering ``cls`` under ``name`` (default ``cls.__name__``)."""

    def _decorator(cls: T) -> T:
        _REGISTRY[name or cls.__name__] = cls
        return cls

    return _decorator


def get(name: str) -> type:
    """Resolve ``name`` to a class via the registry, or as a dotted import path."""
    if name in _REGISTRY:
        return _REGISTRY[name]
    if "." in name:
        module_path, _, cls_name = name.rpartition(".")
        try:
            module = importlib.import_module(module_path)
            return getattr(module, cls_name)
        except (ImportError, AttributeError) as e:
            raise ValidationError(f"Cannot resolve {name!r}: {e}") from e
    raise ValidationError(
        f"Unknown name: {name!r}. Registered names: {sorted(_REGISTRY)}"
    )


def _is_spec(value: Any) -> bool:
    return isinstance(value, dict) and "type" in value


def build(spec: Any) -> Any:
    """Recursively construct an object from a spec; non-spec values pass through."""
    if isinstance(spec, str):
        return get(spec)()
    if _is_spec(spec):
        cls = get(spec["type"])
        params = {
            k: build(v) if _is_spec(v) else v for k, v in spec.get("params", {}).items()
        }
        try:
            return cls(**params)
        except TypeError as e:
            raise ValidationError(f"Cannot construct {spec['type']!r}: {e}") from e
    return spec
