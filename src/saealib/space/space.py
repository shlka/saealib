"""SearchSpace protocol, ValidationResult, and ServiceRegistry.

SearchSpace owns exactly four things:
1. representation identity (RepresentationSpec)
2. offered services (ServiceRegistry)
3. sampling genomes (sample)
4. well-formedness validation (validate)

It owns NO mutation, crossover, selection, surrogate, acquisition, or algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from saealib.core.contracts.ports import SERVICE_VOCABULARY
from saealib.core.contracts.representation import RepresentationSpec
from saealib.exceptions import ValidationError
from saealib.population.genome import GenomeBatch

__all__ = ["SearchSpace", "ServiceRegistry", "ValidationResult", "encode_features"]


@dataclass(frozen=True, kw_only=True)
class ValidationResult:
    """The result of validating a GenomeBatch against a SearchSpace.

    Attributes
    ----------
    valid_mask : tuple[bool, ...]
        Per-genome boolean flag indicating validity of each row.
    errors : tuple[str, ...]
        Human-readable error messages for invalid genomes or batch-level failures.
    """

    valid_mask: tuple[bool, ...] = ()
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce sequence attributes to immutable tuples."""
        object.__setattr__(self, "valid_mask", tuple(self.valid_mask))
        object.__setattr__(self, "errors", tuple(self.errors))

    @property
    def valid(self) -> bool:
        """Return True iff there are no batch errors and all valid_mask items are True.

        If errors is non-empty (e.g. batch-level failure), valid is False.
        If errors is empty and valid_mask is empty (e.g. empty batch validation),
        valid is True.
        """
        if self.errors:
            return False
        if not self.valid_mask:
            return True
        return all(self.valid_mask)


class ServiceRegistry:
    """Registry of services offered by a SearchSpace."""

    def __init__(self) -> None:
        self._services: dict[str, object] = {}

    def register(self, name: str, service: object) -> None:
        """Register a service under a name defined in SERVICE_VOCABULARY."""
        if SERVICE_VOCABULARY.get(name) is None:
            raise ValidationError(f"Unknown service name: {name!r}")
        self._services[name] = service

    def get(self, name: str) -> object | None:
        """Return the service registered for name, or None if not present."""
        return self._services.get(name)

    def require(self, name: str) -> object:
        """Return the registered service, raising ValidationError if missing."""
        if name not in self._services:
            raise ValidationError(
                f"Required service {name!r} is not provided by this SearchSpace"
            )
        return self._services[name]

    def contains(self, name: str) -> bool:
        """Return True if a service is registered under name."""
        return name in self._services

    def __contains__(self, name: object) -> bool:
        """Return True if a service is registered under name."""
        return isinstance(name, str) and self.contains(name)

    def names(self) -> tuple[str, ...]:
        """Return names of all registered services."""
        return tuple(self._services)


def encode_features(space: SearchSpace, genomes: GenomeBatch) -> np.ndarray:
    """Encode genomes for a surrogate through an explicit space service."""
    feature_encoder = space.services.get("FeatureEncoder")
    if feature_encoder is not None:
        encode = getattr(feature_encoder, "encode", None)
        if callable(encode):
            return np.asarray(encode(genomes), dtype=np.float64)
    raise ValidationError("surrogate features require the FeatureEncoder service")


@runtime_checkable
class SearchSpace(Protocol):
    """Protocol for an optimization search space.

    A SearchSpace owns representation identity, service registry, genome sampling,
    and genome validation.
    """

    @property
    def representation(self) -> RepresentationSpec:
        """Return the RepresentationSpec of the space."""
        ...

    @property
    def services(self) -> ServiceRegistry:
        """Return the ServiceRegistry offered by the space."""
        ...

    def sample(self, n: int, rng: np.random.Generator | None = None) -> GenomeBatch:
        """Draw n sampled genomes from the space."""
        ...

    def validate(self, genomes: GenomeBatch) -> ValidationResult:
        """Validate a batch of genomes against space constraints."""
        ...


class DerivedSamplingService:
    """SamplingService view derived from SearchSpace.sample."""

    def __init__(self, space: SearchSpace) -> None:
        self._space = space

    def sample(self, n: int, rng: np.random.Generator | None = None) -> GenomeBatch:
        """Delegate to space.sample."""
        return self._space.sample(n, rng)


class DerivedValidationService:
    """ValidationService view derived from SearchSpace.validate."""

    def __init__(self, space: SearchSpace) -> None:
        self._space = space

    def validate(self, genomes: GenomeBatch) -> ValidationResult:
        """Delegate to space.validate."""
        return self._space.validate(genomes)
