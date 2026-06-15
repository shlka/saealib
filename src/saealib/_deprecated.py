"""Deprecation utilities for backward-compatible API changes."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F", bound=Callable)


def warn_deprecated(old: str, replacement: str, version: str, stacklevel: int = 2) -> None:
    """Emit a FutureWarning for a deprecated name."""
    warnings.warn(
        f"'{old}' is deprecated and will be removed in {version}. "
        f"Use {replacement!r} instead.",
        FutureWarning,
        stacklevel=stacklevel + 1,
    )


def deprecated_param(old: str, new: str, version: str) -> Callable[[F], F]:
    """Decorator: transparently rename a kwarg with a FutureWarning."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if old in kwargs:
                warn_deprecated(old, new, version, stacklevel=2)
                kwargs.setdefault(new, kwargs.pop(old))
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def deprecated_class(replacement: str) -> Callable[[type], type]:
    """Decorator: emit FutureWarning when a deprecated class is instantiated."""

    def decorator(cls: type) -> type:
        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            warnings.warn(
                f"{cls.__name__} is deprecated. Use {replacement} instead.",
                FutureWarning,
                stacklevel=2,
            )
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls

    return decorator
