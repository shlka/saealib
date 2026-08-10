"""ObjectSpace: search space for arbitrary Python objects."""

from __future__ import annotations

import numpy as np

from saealib.core.contracts.representation import RepresentationSpec
from saealib.exceptions import ValidationError
from saealib.population.genome import GenomeBatch, ObjectBatch
from saealib.space.space import ServiceRegistry, ValidationResult

__all__ = ["ObjectSpace"]


class ObjectSpace:
    """A search space for arbitrary Python objects.

    An ObjectSpace registers no services by default. It can be sampled,
    selected, and concatenated; components requiring richer operations
    will be rejected at compile time due to missing services.
    """

    def __init__(self, representation: RepresentationSpec) -> None:
        if not isinstance(representation, RepresentationSpec):
            raise ValidationError(
                "ObjectSpace requires a RepresentationSpec, got "
                f"{type(representation).__name__}"
            )
        self._representation = representation
        self._services = ServiceRegistry()

    @property
    def representation(self) -> RepresentationSpec:
        """Return RepresentationSpec of this object space."""
        return self._representation

    @property
    def services(self) -> ServiceRegistry:
        """Return ServiceRegistry offered by this object space (empty by default)."""
        return self._services

    def sample(self, n: int, rng: np.random.Generator | None = None) -> ObjectBatch:
        """Sample n empty object slots (represented by None placeholders)."""
        if not isinstance(n, int) or n < 0:
            raise ValidationError(f"n must be a non-negative integer, got {n!r}")
        return ObjectBatch([None] * n)

    def validate(self, genomes: GenomeBatch) -> ValidationResult:
        """Check if genomes batch is an ObjectBatch."""
        if not isinstance(genomes, ObjectBatch):
            raise ValidationError(
                "ObjectSpace validation requires ObjectBatch, got "
                f"{type(genomes).__name__}"
            )
        n = len(genomes)
        return ValidationResult(
            valid_mask=tuple([True] * n),
            errors=(),
        )
