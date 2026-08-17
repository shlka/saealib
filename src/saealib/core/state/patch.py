"""Declarative updates to values held by a :class:`StateStore`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from saealib.core.state.keys import StateKey
from saealib.population.genome import GenomeBatch

__all__ = ["PopulationRowUpdate", "StatePatch", "StateUpdate"]


@dataclass(frozen=True, kw_only=True)
class StateUpdate:
    """Base class for updates applied to an existing state value."""


@dataclass(frozen=True, kw_only=True)
class PopulationRowUpdate(StateUpdate):
    """Update selected rows and columns of a ``Population`` atomically."""

    indices: np.ndarray
    values: Mapping[str, np.ndarray]
    genome: GenomeBatch | np.ndarray | None = None


@dataclass(frozen=True, kw_only=True)
class StatePatch:
    """Describe replacement, in-place, and deletion operations on state."""

    writes: Mapping[StateKey, object]
    deletes: frozenset[StateKey] = frozenset()
