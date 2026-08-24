"""Semantic key vocabulary for default resolution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DefaultKey:
    """A semantic key for a default value."""

    name: str
    value_type: type

    def __repr__(self) -> str:
        return f"DefaultKey({self.name!r})"


# Population size keys
POPULATION_SIZE = DefaultKey("population.size", int)
INITIAL_ARCHIVE_SIZE = DefaultKey("archive.initial_size", int)

# Termination keys
MAX_EVALUATIONS = DefaultKey("termination.max_evaluations", int)
