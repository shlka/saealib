"""Registry: name -> class resolution and recursive construction from specs.

A "spec" is one of:

- a live instance, returned unchanged
- a string ``"Name"``, resolved via the registry (or as a dotted import path
  ``"module.submodule.ClassName"`` if not registered) and instantiated with
  no arguments
- a mapping ``{"type": "Name", "params": {...}}``, resolved the same way and
  instantiated with ``params``. If ``params`` is a dict, it is passed as
  keyword arguments (``cls(**params)``); if it is a list, as positional
  arguments (``cls(*params)``), for constructors taking ``*args``. Values
  that are themselves specs are built recursively first.

``to_spec()`` is the reverse operation: it serializes a live instance back
into a spec by reflecting its constructor signature and reading same-named
attributes. A class opts out of this generic reflection by exposing a
``_registry_spec`` attribute (a spec, or ``None`` if it cannot be
serialized) which ``to_spec()`` uses directly instead.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

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
        params = spec.get("params", {})
        if isinstance(params, list):
            args = [build(v) if _is_spec(v) else v for v in params]
            try:
                return cls(*args)
            except TypeError as e:
                raise ValidationError(f"Cannot construct {spec['type']!r}: {e}") from e
        built_params = {k: build(v) if _is_spec(v) else v for k, v in params.items()}
        try:
            return cls(**built_params)
        except TypeError as e:
            raise ValidationError(f"Cannot construct {spec['type']!r}: {e}") from e
    return spec


def dotted_path(obj: Any) -> str:
    """Return the ``module.qualname`` import path of a class or function."""
    return f"{obj.__module__}.{obj.__qualname__}"


def _find_registered_name(cls: type) -> str | None:
    for name, registered_cls in _REGISTRY.items():
        if registered_cls is cls:
            return name
    return None


def to_spec(obj: Any) -> Any:
    """Recursively serialize ``obj`` into a spec (the reverse of ``build()``).

    Primitives, ``None``, lists/tuples, and dicts pass through structurally
    (numpy arrays become lists). Functions/builtins become a dotted import
    path string. Other objects are reflected via their constructor
    signature: for each ``__init__`` parameter, the same-named attribute on
    ``obj`` is read and serialized recursively. A single ``*args``
    (``VAR_POSITIONAL``) parameter is serialized as a params *list* instead
    of a dict, so it round-trips through ``build()``'s positional-call path.
    """
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list | tuple):
        return [to_spec(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_spec(v) for k, v in obj.items()}
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return dotted_path(obj)

    if hasattr(obj, "_registry_spec"):
        spec = obj._registry_spec
        if spec is None:
            raise ValidationError(
                f"{type(obj).__name__} instance has no recorded spec and "
                "cannot be serialized."
            )
        return spec

    cls = type(obj)
    type_name = _find_registered_name(cls) or dotted_path(cls)
    parameters = [
        p
        for name, p in inspect.signature(cls.__init__).parameters.items()
        if name != "self"
    ]

    var_positional = [p for p in parameters if p.kind is p.VAR_POSITIONAL]
    if var_positional:
        (param,) = var_positional
        if not hasattr(obj, param.name):
            raise ValidationError(
                f"Cannot serialize {type_name!r}: no attribute {param.name!r} "
                f"matching *{param.name} constructor parameter."
            )
        return {
            "type": type_name,
            "params": [to_spec(v) for v in getattr(obj, param.name)],
        }

    params: dict[str, Any] = {}
    for param in parameters:
        if param.kind is param.VAR_KEYWORD:
            continue
        if not hasattr(obj, param.name):
            if param.default is not param.empty:
                continue
            raise ValidationError(
                f"Cannot serialize {type_name!r}: no attribute {param.name!r} "
                "matching required constructor parameter."
            )
        params[param.name] = to_spec(getattr(obj, param.name))
    return {"type": type_name, "params": params}
