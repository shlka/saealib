"""Service protocols for SearchSpace requirements.

This module defines the Protocol interfaces for the 10 core services declared in
SERVICE_VOCABULARY (src/saealib/core/contracts/ports.py).
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Protocol, runtime_checkable

import numpy as np

from saealib.population.genome import GenomeBatch

__all__ = [
    "BoundsService",
    "CloneService",
    "ComparisonService",
    "DenseNumericView",
    "DistanceService",
    "EquivalenceService",
    "FingerprintService",
    "GenomeCodec",
    "SamplingService",
    "ValidationService",
]


@runtime_checkable
class SamplingService(Protocol):
    """Service for drawing new genomes from a SearchSpace."""

    def sample(self, n: int, rng: np.random.Generator | None = None) -> GenomeBatch:
        """Draw n genomes from the space."""
        ...


@runtime_checkable
class ValidationService(Protocol):
    """Service for checking if genomes are well-formed for a space."""

    def validate(self, genomes: GenomeBatch) -> Any:
        """Validate a batch of genomes."""
        ...


@runtime_checkable
class CloneService(Protocol):
    """Service for producing independent copies of genomes."""

    def clone(self, genomes: GenomeBatch) -> GenomeBatch:
        """Produce an independent copy of the given genomes."""
        ...


@runtime_checkable
class FingerprintService(Protocol):
    """Service for producing an exact, canonical, hashable identity for genomes."""

    def fingerprint(self, genomes: GenomeBatch) -> tuple[Hashable, ...]:
        """Return a tuple of exact, hashable identity fingerprints."""
        ...


@runtime_checkable
class EquivalenceService(Protocol):
    """Service for approximate duplicate matching of genomes."""

    def find_duplicates(self, genomes: GenomeBatch) -> np.ndarray:
        """Return a boolean array marking duplicate rows within a batch."""
        ...


@runtime_checkable
class GenomeCodec(Protocol):
    """Service for encoding genomes into persistable primitives and decoding back."""

    def encode(self, genomes: GenomeBatch) -> dict[str, Any]:
        """Encode genomes into a dictionary of persistable primitives."""
        ...

    def decode(self, payload: dict[str, Any]) -> GenomeBatch:
        """Decode a dictionary payload back into a GenomeBatch."""
        ...


@runtime_checkable
class DenseNumericView(Protocol):
    """Service providing zero-copy dense numeric array access for vector spaces."""

    def get_view(self, genomes: GenomeBatch) -> np.ndarray:
        """Return a zero-copy 2D float64 NumPy array view of the genomes."""
        ...


@runtime_checkable
class BoundsService(Protocol):
    """Service providing lower and upper variable bounds for vector spaces."""

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lb, ub) array pair of lower and upper bounds."""
        ...


@runtime_checkable
class DistanceService(Protocol):
    """Service for computing distances between genomes."""

    def pairwise_distance(
        self, batch1: GenomeBatch, batch2: GenomeBatch | None = None
    ) -> np.ndarray:
        """Compute pairwise distance matrix between batch1 and batch2."""
        ...


@runtime_checkable
class ComparisonService(Protocol):
    """Service for comparing objective vectors.

    Note: Provider is Problem, not SearchSpace (ADR-0003 §4.1.0).
    """

    def compare(self, f1: np.ndarray, f2: np.ndarray) -> int:
        """Compare two objective vectors."""
        ...
