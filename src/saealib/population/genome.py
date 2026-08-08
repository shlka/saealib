"""GenomeBatch protocol and concrete implementations.

This module defines the ``GenomeBatch`` protocol — the minimal contract that a
batch of solution genomes must satisfy — along with two concrete implementations:

- ``DenseVectorBatch``: wraps a 2D float64 NumPy array for dense numeric vector spaces.
- ``ObjectBatch``: wraps arbitrary Python objects with no structural assumptions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from typing_extensions import Self

from saealib.exceptions import ValidationError

__all__ = ["DenseVectorBatch", "GenomeBatch", "ObjectBatch"]


@runtime_checkable
class GenomeBatch(Protocol):
    """Protocol for a batch of solution genomes.

    GenomeBatch is an immutable value representing a collection of genomes.
    Generic framework components count, select, and concatenate batches
    without inspecting their representation details.

    The protocol contains exactly three operations:
    - ``__len__()``: count genomes in the batch.
    - ``take(indices)``: pick a subset of genomes by row indices.
    - ``concat(batches)``: join multiple batches into one.

    No shape, arithmetic, distance, equality, or serialization methods are on
    this protocol; those are services provided by the SearchSpace.
    """

    def __len__(self) -> int:
        """Return the number of genomes in the batch."""
        ...

    def take(self, indices: Sequence[int] | np.ndarray) -> Self:
        """Return a new batch containing genomes selected by row indices."""
        ...

    @classmethod
    def concat(cls, batches: Sequence[Self]) -> Self:
        """Concatenate multiple batches into a single new batch.

        To obtain an empty batch, call ``batch.take([])`` on an existing batch
        instance rather than calling ``concat([])``.
        """
        ...


class DenseVectorBatch:
    """GenomeBatch implementation wrapping a 2D float64 NumPy matrix (n, dim).

    Attributes
    ----------
    array : np.ndarray
        A read-only, C-contiguous 2D float64 view of the backing data.
    """

    def __init__(self, data: Sequence[Sequence[float]] | np.ndarray) -> None:
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValidationError(
                f"DenseVectorBatch requires a 2D array, got shape {arr.shape}"
            )
        if arr.dtype != np.float64:
            arr = arr.astype(np.float64)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)

        readonly_view = arr.view()
        readonly_view.setflags(write=False)
        self._data: np.ndarray = readonly_view

    @property
    def array(self) -> np.ndarray:
        """Return the read-only 2D float64 array of shape (n, dim)."""
        return self._data

    def __len__(self) -> int:
        """Return the number of genomes in the batch."""
        return len(self._data)

    def take(self, indices: Sequence[int] | np.ndarray) -> Self:
        """Return a new DenseVectorBatch containing rows selected by indices."""
        idx = np.asarray(indices, dtype=np.intp)
        if len(self._data) == 0 and len(idx) > 0:
            raise ValidationError(
                "Cannot take non-empty indices from an empty DenseVectorBatch"
            )
        try:
            selected = self._data[idx]
        except IndexError as exc:
            raise ValidationError(f"Index out of bounds: {exc}") from exc
        return type(self)(selected)

    @classmethod
    def concat(cls, batches: Sequence[Self]) -> Self:
        """Concatenate multiple DenseVectorBatch instances into a new batch.

        To obtain an empty batch, call ``batch.take([])`` on an existing batch
        instance rather than calling ``concat([])``.
        """
        batches_tuple = tuple(batches)
        if not batches_tuple:
            raise ValidationError(
                "Cannot concat an empty sequence of DenseVectorBatch "
                "(dimension unknown)"
            )
        for b in batches_tuple:
            if not isinstance(b, DenseVectorBatch):
                raise ValidationError(
                    f"Expected DenseVectorBatch instance, got {type(b).__name__}"
                )
        dims = {b._data.shape[1] for b in batches_tuple}
        if len(dims) > 1:
            raise ValidationError(
                "Cannot concat DenseVectorBatch instances with mismatched "
                f"dimensions: {sorted(dims)}"
            )
        concatenated = np.concatenate([b._data for b in batches_tuple], axis=0)
        return cls(concatenated)


class ObjectBatch:
    """GenomeBatch implementation wrapping arbitrary Python objects.

    Holds objects in a plain Python tuple without converting to object-dtype
    NumPy arrays.
    """

    def __init__(self, items: Sequence[object] = ()) -> None:
        try:
            self._items: tuple[object, ...] = tuple(items)
        except TypeError as exc:
            raise ValidationError("ObjectBatch items must be iterable") from exc

    @property
    def items(self) -> tuple[object, ...]:
        """Return the backing tuple of objects."""
        return self._items

    def __len__(self) -> int:
        """Return the number of items in the batch."""
        return len(self._items)

    def take(self, indices: Sequence[int] | np.ndarray) -> Self:
        """Return a new ObjectBatch containing items selected by indices."""
        idx = np.asarray(indices, dtype=np.intp)
        n = len(self._items)
        selected: list[object] = []
        for i in idx:
            if i < -n or i >= n:
                raise ValidationError(
                    f"Index out of bounds for ObjectBatch of length {n}: {i}"
                )
            selected.append(self._items[i])
        return type(self)(selected)

    @classmethod
    def concat(cls, batches: Sequence[Self]) -> Self:
        """Concatenate multiple ObjectBatch instances into a new batch.

        To obtain an empty batch, call ``batch.take([])`` on an existing batch
        instance rather than calling ``concat([])``.
        """
        batches_tuple = tuple(batches)
        if not batches_tuple:
            raise ValidationError("Cannot concat an empty sequence of ObjectBatch")
        combined: list[object] = []
        for b in batches_tuple:
            if not isinstance(b, ObjectBatch):
                raise ValidationError(
                    f"Expected ObjectBatch instance, got {type(b).__name__}"
                )
            combined.extend(b._items)
        return cls(combined)
