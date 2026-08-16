"""GenomeBatch protocol and concrete implementations.

This module defines the ``GenomeBatch`` protocol — the minimal contract that a
batch of solution genomes must satisfy — along with concrete implementations:

- ``DenseVectorBatch``: wraps a 2D float64 NumPy array for dense numeric vector spaces.
- ``PermutationBatch``: wraps validated fixed-length integer permutations.
- ``VariableLengthBatch``: wraps arbitrary finite sequences.
- ``ObjectBatch``: wraps arbitrary Python objects with no structural assumptions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from typing_extensions import Self

from saealib.exceptions import ValidationError

__all__ = [
    "DenseVectorBatch",
    "GenomeBatch",
    "ObjectBatch",
    "PermutationBatch",
    "VariableLengthBatch",
]


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

    def __len__(self) -> int: ...

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


def genome_value(batch: GenomeBatch, index: int) -> object:
    """Return a scalar genome value when the batch exposes one structurally.

    The generic batch contract intentionally has no row-value method.  The
    standard built-in batches expose their values through ``array``,
    ``sequences``, or ``items``; opaque third-party batches remain a one-row
    ``GenomeBatch`` so their owner can interpret them explicitly.
    """
    selected = batch.take([index])
    for name in ("array", "sequences", "items"):
        values = getattr(selected, name, None)
        if values is not None:
            return values[0]
    return selected


class DenseVectorBatch:
    """GenomeBatch implementation wrapping a 2D float64 NumPy matrix (n, dim).

    Attributes
    ----------
    array : np.ndarray
        A read-only, C-contiguous 2D float64 view of the backing data.
    """

    def __init__(self, data: Sequence[Sequence[float]] | np.ndarray) -> None:
        self._set_data(self._normalize_array(data, copy=True))

    @staticmethod
    def _normalize_array(
        data: Sequence[Sequence[float]] | np.ndarray, *, copy: bool
    ) -> np.ndarray:
        """Validate and normalize dense storage, optionally taking ownership."""
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValidationError(
                f"DenseVectorBatch requires a 2D array, got shape {arr.shape}"
            )
        if arr.dtype != np.float64:
            arr = arr.astype(np.float64)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        if copy:
            arr = arr.copy()
        return arr

    def _set_data(self, arr: np.ndarray) -> None:
        readonly_view = arr.view()
        readonly_view.setflags(write=False)
        self._data: np.ndarray = readonly_view

    @classmethod
    def _from_borrowed_view(cls, array: np.ndarray) -> Self:
        """Build a batch from normalized storage without copying suitable input."""
        batch = cls.__new__(cls)
        batch._set_data(cls._normalize_array(array, copy=False))
        return batch

    @property
    def array(self) -> np.ndarray:
        """Return the read-only 2D float64 array of shape (n, dim)."""
        return self._data

    def __getitem__(self, index: Any) -> Any:
        """Provide read-only ndarray-style indexing for evaluator compatibility."""
        return self._data[index]

    def __len__(self) -> int:
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


class PermutationBatch:
    """GenomeBatch implementation for fixed-length permutations.

    Rows contain every integer in ``[0, length)`` exactly once.  The backing
    array is owned by the batch and exposed read-only, matching
    :class:`DenseVectorBatch` while keeping permutation validation local to
    this representation profile.
    """

    def __init__(
        self, data: Sequence[Sequence[int]] | np.ndarray, *, length: int | None = None
    ) -> None:
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValidationError(
                f"PermutationBatch requires a 2D array, got shape {arr.shape}"
            )
        if arr.dtype.kind not in "iu":
            raise ValidationError("PermutationBatch values must be integers")
        arr = np.array(arr, dtype=np.int64, order="C", copy=True)
        resolved_length = arr.shape[1] if length is None else length
        if not isinstance(resolved_length, int) or resolved_length < 0:
            raise ValidationError(
                "PermutationBatch length must be a non-negative integer"
            )
        if arr.shape[1] != resolved_length:
            raise ValidationError(
                f"PermutationBatch dimension {arr.shape[1]} does not match "
                f"length {resolved_length}"
            )
        expected = np.arange(resolved_length, dtype=np.int64)
        if len(arr) and not np.all(np.sort(arr, axis=1) == expected):
            raise ValidationError(
                "PermutationBatch rows must contain each integer in [0, length) once"
            )
        view = arr.view()
        view.setflags(write=False)
        self._data = view
        self._length = resolved_length

    @property
    def array(self) -> np.ndarray:
        """Return the read-only ``(n, length)`` integer array."""
        return self._data

    @property
    def length(self) -> int:
        """Return the permutation length."""
        return self._length

    def __len__(self) -> int:
        return len(self._data)

    def take(self, indices: Sequence[int] | np.ndarray) -> Self:
        """Return selected permutation rows."""
        idx = np.asarray(indices, dtype=np.intp)
        try:
            selected = self._data[idx]
        except IndexError as exc:
            raise ValidationError(f"Index out of bounds: {exc}") from exc
        return type(self)(selected, length=self._length)

    @classmethod
    def concat(cls, batches: Sequence[Self]) -> Self:
        """Concatenate permutation batches with one common length."""
        values = tuple(batches)
        if not values:
            raise ValidationError("Cannot concat an empty sequence of PermutationBatch")
        if any(not isinstance(batch, PermutationBatch) for batch in values):
            raise ValidationError(
                "PermutationBatch.concat received an incompatible batch"
            )
        lengths = {batch.length for batch in values}
        if len(lengths) != 1:
            raise ValidationError("Cannot concat permutations with different lengths")
        return cls(
            np.concatenate([batch.array for batch in values], axis=0),
            length=values[0].length,
        )


class VariableLengthBatch:
    """GenomeBatch implementation for arbitrary finite sequences.

    The batch owns an immutable tuple of tuple rows.  Elements are deliberately
    left opaque; :class:`~saealib.space.sequence.SequenceSpace` supplies the
    alphabet and validation policy.
    """

    def __init__(self, sequences: Sequence[Sequence[object]] = ()) -> None:
        try:
            rows = tuple(tuple(sequence) for sequence in sequences)
        except TypeError as exc:
            raise ValidationError("VariableLengthBatch requires sequences") from exc
        self._sequences = rows

    @property
    def sequences(self) -> tuple[tuple[object, ...], ...]:
        """Return the immutable sequence rows."""
        return self._sequences

    @property
    def items(self) -> tuple[tuple[object, ...], ...]:
        """Alias for the generic object-batch inspection convention."""
        return self._sequences

    def __len__(self) -> int:
        return len(self._sequences)

    def take(self, indices: Sequence[int] | np.ndarray) -> Self:
        """Return selected sequence rows."""
        idx = np.asarray(indices, dtype=np.intp)
        n = len(self._sequences)
        selected: list[tuple[object, ...]] = []
        for raw_index in idx:
            index = int(raw_index)
            if index < -n or index >= n:
                raise ValidationError(
                    f"Index out of bounds for VariableLengthBatch of length {n}: "
                    f"{index}"
                )
            selected.append(self._sequences[index])
        return type(self)(selected)

    @classmethod
    def concat(cls, batches: Sequence[Self]) -> Self:
        """Concatenate variable-length sequence batches."""
        values = tuple(batches)
        if not values:
            raise ValidationError(
                "Cannot concat an empty sequence of VariableLengthBatch"
            )
        if any(not isinstance(batch, VariableLengthBatch) for batch in values):
            raise ValidationError(
                "VariableLengthBatch.concat received an incompatible batch"
            )
        return cls(tuple(row for batch in values for row in batch.sequences))
