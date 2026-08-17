"""Candidate and request ID allocation."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from saealib.exceptions import ValidationError

__all__ = ["CandidateIds", "IDAllocator", "PopulationAttribute"]

# Keep the identity alias identical across the core and compatibility import paths.
CandidateIds = np.ndarray


@dataclass(frozen=True)
class PopulationAttribute:
    """Definition of one named attribute in a population-like table."""

    name: str
    dtype: type | np.dtype
    shape: tuple[int, ...] = ()
    default: Any = np.nan


class IDAllocator:
    """Allocate unique, monotonically increasing int64 IDs.

    Allocation mutates the allocator and is thread-safe.
    """

    def __init__(self, start: int = 0) -> None:
        if isinstance(start, (bool, np.bool_)) or not isinstance(
            start, (int, np.integer)
        ):
            raise ValidationError("IDAllocator start must be an integer")
        start = int(start)
        if start < 0 or start > np.iinfo(np.int64).max:
            raise ValidationError("IDAllocator start is outside the int64 range")
        self._next = start
        self._lock = threading.Lock()

    @property
    def next_value(self) -> int:
        """Return the next unallocated ID for serialization."""
        return self._next

    def allocate(self, n: int) -> np.ndarray:
        """Allocate ``n`` sequential int64 IDs and advance the allocator."""
        if n < 0:
            raise ValidationError("IDAllocator.allocate(n) requires n >= 0")
        with self._lock:
            if n == 0:
                return np.empty(0, dtype=np.int64)
            if self._next + (n - 1) > np.iinfo(np.int64).max:
                raise OverflowError("IDAllocator exhausted the int64 ID space")
            ids = np.arange(self._next, self._next + n, dtype=np.int64)
            self._next += n
        return ids

    def __getstate__(self) -> dict:
        return {"_next": self._next}

    def __setstate__(self, state: dict) -> None:
        self._next = state["_next"]
        self._lock = threading.Lock()
