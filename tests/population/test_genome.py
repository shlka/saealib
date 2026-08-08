"""Tests for GenomeBatch protocol and implementations.

Covers DenseVectorBatch and ObjectBatch guarantees.
Each test covers specific requirements from Unit H3 and documents the
implementation-side mutation that would cause it to fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from saealib.exceptions import ValidationError
from saealib.population.genome import DenseVectorBatch, GenomeBatch, ObjectBatch

# ---------------------------------------------------------------------------
# Protocol runtime / static compatibility check
# ---------------------------------------------------------------------------


def test_implementations_satisfy_protocol() -> None:
    """DenseVectorBatch and ObjectBatch satisfy GenomeBatch protocol.

    Implementation mutation that would break this test:
        Rename `take` or `concat` in either class to something else (e.g. `take_rows`).
        `isinstance(obj, GenomeBatch)` or type checkers would fail.
    """
    dense = DenseVectorBatch([[1.0, 2.0], [3.0, 4.0]])
    obj = ObjectBatch(["a", "b"])

    assert isinstance(dense, GenomeBatch)
    assert isinstance(obj, GenomeBatch)


# ---------------------------------------------------------------------------
# 1. DenseVectorBatch array is read-only
# ---------------------------------------------------------------------------


def test_dense_vector_batch_array_is_readonly() -> None:
    """DenseVectorBatch array returned to callers rejects in-place mutation.

    Implementation mutation that would break this test:
        In `DenseVectorBatch.__init__`, remove `readonly_view.setflags(write=False)`.
        `batch.array[0, 0] = 999.0` would then succeed without raising ValueError.
    """
    batch = DenseVectorBatch([[1.0, 2.0], [3.0, 4.0]])

    assert not batch.array.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        batch.array[0, 0] = 999.0


# ---------------------------------------------------------------------------
# 2. Out-of-bounds take raises ValidationError
# ---------------------------------------------------------------------------


def test_dense_vector_batch_take_out_of_bounds_raises() -> None:
    """DenseVectorBatch.take with an out-of-bounds index raises ValidationError.

    Implementation mutation that would break this test:
        In `DenseVectorBatch.take`, catch `IndexError` and return `self`
        instead of re-raising `ValidationError`. The `pytest.raises` check fails.
    """
    batch = DenseVectorBatch([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(ValidationError, match="Index out of bounds"):
        batch.take([0, 5])


def test_object_batch_take_out_of_bounds_raises() -> None:
    """ObjectBatch.take with an out-of-bounds index raises ValidationError.

    Implementation mutation that would break this test:
        In `ObjectBatch.take`, remove the `if i < -n or i >= n:` bounds check.
        Taking index 10 from a length-2 ObjectBatch would raise IndexError
        instead of ValidationError.
    """
    batch = ObjectBatch(["a", "b"])

    with pytest.raises(ValidationError, match="Index out of bounds"):
        batch.take([10])


# ---------------------------------------------------------------------------
# 3. Mismatched dim concat raises ValidationError
# ---------------------------------------------------------------------------


def test_dense_vector_batch_concat_mismatched_dim_raises() -> None:
    """DenseVectorBatch.concat with mismatched dimensions raises.

    Implementation mutation that would break this test:
        In `DenseVectorBatch.concat`, remove the `if len(dims) > 1:` check.
        NumPy's `np.concatenate` raises ValueError instead of ValidationError.
    """
    b1 = DenseVectorBatch([[1.0, 2.0]])
    b2 = DenseVectorBatch([[3.0, 4.0, 5.0]])

    with pytest.raises(ValidationError, match="mismatched dimensions"):
        DenseVectorBatch.concat([b1, b2])


# ---------------------------------------------------------------------------
# 4. Empty concat behavior (both implementations raise ValidationError)
# ---------------------------------------------------------------------------


def test_dense_vector_batch_concat_empty_sequence_raises() -> None:
    """DenseVectorBatch.concat on an empty sequence raises ValidationError.

    Implementation mutation that would break this test:
        In `DenseVectorBatch.concat`, remove the `if not batches_tuple:` check.
        Calling `np.concatenate([])` raises ValueError instead of ValidationError.
    """
    with pytest.raises(ValidationError, match="empty sequence"):
        DenseVectorBatch.concat([])


def test_object_batch_concat_empty_sequence_raises() -> None:
    """ObjectBatch.concat on an empty sequence raises ValidationError.

    Implementation mutation that would break this test:
        In `ObjectBatch.concat`, remove the `if not batches_tuple:` check
        and return `cls(())`. `pytest.raises(ValidationError)` would fail.
    """
    with pytest.raises(ValidationError, match="empty sequence"):
        ObjectBatch.concat([])


def test_take_empty_indices_returns_empty_batch() -> None:
    """take([]) on an existing batch returns an empty batch for all implementations.

    This is the recommended idiom for obtaining an empty batch rather than
    calling concat([]).

    Implementation mutation that would break this test:
        In `DenseVectorBatch.take` or `ObjectBatch.take`, raise ValidationError
        when `len(idx) == 0`. This test would then fail.
    """
    dense_batch = DenseVectorBatch([[1.0, 2.0], [3.0, 4.0]])
    dense_empty = dense_batch.take([])
    assert isinstance(dense_empty, DenseVectorBatch)
    assert len(dense_empty) == 0
    assert dense_empty.array.shape == (0, 2)

    obj_batch = ObjectBatch(["a", "b"])
    obj_empty = obj_batch.take([])
    assert isinstance(obj_empty, ObjectBatch)
    assert len(obj_empty) == 0
    assert obj_empty.items == ()


# ---------------------------------------------------------------------------
# 5. ObjectBatch take/concat preserves object identity
# ---------------------------------------------------------------------------


def test_object_batch_preserves_object_identity() -> None:
    """ObjectBatch.take and concat preserve item identity (is).

    Implementation mutation that would break this test:
        In `ObjectBatch.take` or `concat`, apply `copy.deepcopy(item)` to items.
        The `item is obj` identity check would fail.
    """

    class CustomItem:
        pass

    obj1 = CustomItem()
    obj2 = CustomItem()

    batch = ObjectBatch([obj1, obj2])
    taken = batch.take([1, 0])

    assert taken.items[0] is obj2
    assert taken.items[1] is obj1

    b1 = ObjectBatch([obj1])
    b2 = ObjectBatch([obj2])
    concatenated = ObjectBatch.concat([b1, b2])

    assert concatenated.items[0] is obj1
    assert concatenated.items[1] is obj2


# ---------------------------------------------------------------------------
# 6. Immutability: take returns a new batch, original batch is unchanged
# ---------------------------------------------------------------------------


def test_dense_vector_batch_take_immutability() -> None:
    """DenseVectorBatch.take returns a new batch, leaving original unchanged.

    Implementation mutation that would break this test:
        In `DenseVectorBatch.take`, mutate `self._data` in place and return `self`.
        `len(batch)` or `batch.array` would change, breaking `len(batch) == 3`.
    """
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    batch = DenseVectorBatch(data)

    taken = batch.take([0, 2])

    assert isinstance(taken, DenseVectorBatch)
    assert taken is not batch
    assert len(batch) == 3
    assert len(taken) == 2
    np.testing.assert_array_equal(batch.array, data)
    np.testing.assert_array_equal(taken.array, [[1.0, 2.0], [5.0, 6.0]])


def test_object_batch_take_immutability() -> None:
    """ObjectBatch.take returns a new batch, leaving original unchanged.

    Implementation mutation that would break this test:
        In `ObjectBatch.take`, mutate `self._items` in place and return `self`.
        `len(batch) == 3` would fail after `take([0])`.
    """
    items = ("x", "y", "z")
    batch = ObjectBatch(items)

    taken = batch.take([0, 1])

    assert isinstance(taken, ObjectBatch)
    assert taken is not batch
    assert len(batch) == 3
    assert len(taken) == 2
    assert batch.items == items
    assert taken.items == ("x", "y")
