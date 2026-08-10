"""Candidate and request ID allocation.

Leaf module: its dependencies are limited to standard-library helpers,
``numpy``, and ``saealib.exceptions.ValidationError`` to avoid import-cycle
risk, since ``context.py`` needs a runtime top-level import of ``IDAllocator``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from saealib.exceptions import ValidationError

CandidateIds = np.ndarray


@dataclass(frozen=True)
class PopulationAttribute:
    """Definition of one named attribute in a population-like table."""

    name: str
    dtype: type | np.dtype
    shape: tuple[int, ...] = ()
    default: Any = np.nan


class IDAllocator:
    """Allocates unique, monotonically increasing int64 IDs.

    A controlled mutable exception, like OptimizationState.rng/archive:
    advances its internal state as a side effect. Thread-safe via an
    internal lock. Not part of saealib's public top-level export surface —
    accessed only through OptimizationState.candidate_id_allocator /
    .request_id_allocator.
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
        """The next ID that will be allocated (for checkpoint serialization only)."""
        return self._next

    def allocate(self, n: int) -> np.ndarray:
        """Allocate ``n`` sequential unique int64 IDs and advance internal state.

        Parameters
        ----------
        n : int
            Number of IDs to allocate. Must be non-negative.

        Returns
        -------
        np.ndarray
            Owned ``(n,)`` int64 array of newly allocated IDs.
        """
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
        """Return pickle state, excluding the unpicklable lock."""
        return {"_next": self._next}

    def __setstate__(self, state: dict) -> None:
        """Restore pickle state, reconstructing a fresh lock."""
        self._next = state["_next"]
        self._lock = threading.Lock()
