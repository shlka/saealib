"""VectorSpace: search space for dense real-valued vectors (ADR-0003 §1.2)."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from saealib.core.contracts.data import Fixed
from saealib.core.contracts.representation import ParameterSpec, RepresentationSpec
from saealib.exceptions import ValidationError
from saealib.population.genome import DenseVectorBatch, GenomeBatch
from saealib.space.space import (
    DerivedSamplingService,
    DerivedValidationService,
    ServiceRegistry,
    ValidationResult,
)

if TYPE_CHECKING:
    pass

__all__ = ["VectorSpace"]


def _canonical_float(val: float) -> float | str:
    """Normalize floats to a canonical representation for fingerprinting.

    -0.0 is normalized to 0.0 (+0.0).
    NaN is normalized to the canonical string "__nan__" so that NaN == NaN
    and hash(NaN) == hash(NaN) holds for fingerprints.
    """
    fval = float(val)
    if math.isnan(fval):
        return "__nan__"
    if fval == 0.0:
        return 0.0
    return fval


class _VectorBoundsService:
    def __init__(self, lb: np.ndarray, ub: np.ndarray) -> None:
        self._lb = lb
        self._ub = ub

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self._lb, self._ub


class _VectorDenseNumericView:
    def get_view(self, genomes: GenomeBatch) -> np.ndarray:
        if not isinstance(genomes, DenseVectorBatch):
            raise ValidationError(
                "DenseNumericView requires DenseVectorBatch, got "
                f"{type(genomes).__name__}"
            )
        # Returns the read-only array view which shares memory with the batch
        return genomes.array


class _VectorCloneService:
    def clone(self, genomes: GenomeBatch) -> GenomeBatch:
        if not isinstance(genomes, DenseVectorBatch):
            raise ValidationError(
                f"CloneService requires DenseVectorBatch, got {type(genomes).__name__}"
            )
        # Produce an independent copy of the array data
        return DenseVectorBatch(genomes.array.copy())


class _VectorDistanceService:
    def pairwise_distance(
        self, batch1: GenomeBatch, batch2: GenomeBatch | None = None
    ) -> np.ndarray:
        if not isinstance(batch1, DenseVectorBatch):
            raise ValidationError(
                "DistanceService requires DenseVectorBatch for batch1"
            )
        x1 = batch1.array
        if batch2 is None:
            x2 = x1
        else:
            if not isinstance(batch2, DenseVectorBatch):
                raise ValidationError(
                    "DistanceService requires DenseVectorBatch for batch2"
                )
            x2 = batch2.array

        if x1.shape[1] != x2.shape[1]:
            raise ValidationError(
                f"Dimension mismatch in DistanceService: {x1.shape[1]} vs {x2.shape[1]}"
            )

        # Euclidean pairwise distance computation
        diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1))


class _VectorFingerprintService:
    def fingerprint(self, genomes: GenomeBatch) -> tuple[Hashable, ...]:
        if not isinstance(genomes, DenseVectorBatch):
            raise ValidationError(
                "FingerprintService requires DenseVectorBatch, got "
                f"{type(genomes).__name__}"
            )
        # Exact canonical hashable identity:
        # tuple of canonical float values for each row
        return tuple(
            tuple(_canonical_float(val) for val in row) for row in genomes.array
        )


class _VectorEquivalenceService:
    def __init__(self, atol: float = 1e-16, rtol: float = 0.0) -> None:
        self._atol = atol
        self._rtol = rtol

    def find_duplicates(self, genomes: GenomeBatch) -> np.ndarray:
        if not isinstance(genomes, DenseVectorBatch):
            raise ValidationError(
                "EquivalenceService requires DenseVectorBatch, got "
                f"{type(genomes).__name__}"
            )
        arr = genomes.array
        n = len(arr)
        is_dup = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dup[i]:
                continue
            for j in range(i + 1, n):
                if not is_dup[j] and np.all(
                    np.isclose(arr[i], arr[j], atol=self._atol, rtol=self._rtol)
                ):
                    is_dup[j] = True
        return is_dup


class VectorSpace:
    """A search space for continuous numeric vectors defined by variable bounds.

    Parameters
    ----------
    dim : int
        Dimension of the vector space.
    lb : Sequence[float] | np.ndarray
        Lower bounds per dimension.
    ub : Sequence[float] | np.ndarray
        Upper bounds per dimension.
    atol : float, optional
        Absolute tolerance for approximate equivalence matching (default: 1e-16).
    rtol : float, optional
        Relative tolerance for approximate equivalence matching (default: 0.0).
    """

    def __init__(
        self,
        dim: int,
        lb: Sequence[float] | np.ndarray,
        ub: Sequence[float] | np.ndarray,
        *,
        atol: float = 1e-16,
        rtol: float = 0.0,
    ) -> None:
        if not isinstance(dim, int) or dim <= 0:
            raise ValidationError(f"dim must be a positive integer, got {dim!r}")

        lb_arr = np.asarray(lb, dtype=np.float64)
        ub_arr = np.asarray(ub, dtype=np.float64)

        if lb_arr.shape != (dim,) or ub_arr.shape != (dim,):
            raise ValidationError(
                f"lb and ub must have shape ({dim},), got {lb_arr.shape} "
                f"and {ub_arr.shape}"
            )

        if np.any(lb_arr > ub_arr):
            raise ValidationError(
                "Lower bound (lb) cannot be greater than upper bound (ub)"
            )

        lb_arr.setflags(write=False)
        ub_arr.setflags(write=False)
        self._dim = dim
        self._lb = lb_arr
        self._ub = ub_arr
        self._atol = atol
        self._rtol = rtol

        self._representation = RepresentationSpec(
            kind="vector",
            parameters=(ParameterSpec(name="dim", value=Fixed(value=dim)),),
        )

        self._services = ServiceRegistry()
        self._services.register("SamplingService", DerivedSamplingService(self))
        self._services.register("ValidationService", DerivedValidationService(self))
        self._services.register(
            "BoundsService", _VectorBoundsService(self._lb, self._ub)
        )
        self._services.register("DenseNumericView", _VectorDenseNumericView())
        self._services.register("CloneService", _VectorCloneService())
        self._services.register("DistanceService", _VectorDistanceService())
        self._services.register("FingerprintService", _VectorFingerprintService())
        self._services.register(
            "EquivalenceService",
            _VectorEquivalenceService(atol=self._atol, rtol=self._rtol),
        )

    @property
    def dim(self) -> int:
        """Return space dimension."""
        return self._dim

    @property
    def lb(self) -> np.ndarray:
        """Return lower bounds array."""
        return self._lb

    @property
    def ub(self) -> np.ndarray:
        """Return upper bounds array."""
        return self._ub

    @property
    def atol(self) -> float:
        """Return absolute tolerance for equivalence matching."""
        return self._atol

    @property
    def rtol(self) -> float:
        """Return relative tolerance for equivalence matching."""
        return self._rtol

    @property
    def representation(self) -> RepresentationSpec:
        """Return RepresentationSpec of this vector space."""
        return self._representation

    @property
    def services(self) -> ServiceRegistry:
        """Return ServiceRegistry offered by this vector space."""
        return self._services

    def sample(
        self, n: int, rng: np.random.Generator | None = None
    ) -> DenseVectorBatch:
        """Sample n random genomes uniformly within bounds."""
        if not isinstance(n, int) or n < 0:
            raise ValidationError(f"n must be a non-negative integer, got {n!r}")
        generator = rng if rng is not None else np.random.default_rng()
        data = generator.uniform(self._lb, self._ub, size=(n, self._dim))
        return DenseVectorBatch(data)

    def validate(self, genomes: GenomeBatch) -> ValidationResult:
        """Check if genomes batch is well-formed and within bounds."""
        if not isinstance(genomes, DenseVectorBatch):
            raise ValidationError(
                "VectorSpace validation requires DenseVectorBatch, got "
                f"{type(genomes).__name__}"
            )

        arr = genomes.array
        if arr.shape[1] != self._dim:
            raise ValidationError(
                "VectorSpace dimension mismatch: expected "
                f"{self._dim}, got {arr.shape[1]}"
            )

        if len(arr) == 0:
            return ValidationResult(valid_mask=(), errors=())

        finite_mask = np.all(np.isfinite(arr), axis=1)
        in_bounds_mask = np.all((arr >= self._lb) & (arr <= self._ub), axis=1)
        valid_mask_arr = finite_mask & in_bounds_mask

        valid = bool(np.all(valid_mask_arr))
        errors = (
            () if valid else ("One or more genomes are out of bounds or non-finite",)
        )
        return ValidationResult(
            valid_mask=tuple(bool(v) for v in valid_mask_arr),
            errors=errors,
        )
