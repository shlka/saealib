"""Storage invariants for the dense observation column store."""

import numpy as np

from saealib.core.contracts.observation import OK, TRUE
from saealib.core.contracts.observations import (
    ObservationBatch,
    ObservationRecords,
    ObservationSchema,
)


def test_dense_storage_uses_native_columns() -> None:
    records = ObservationRecords.from_dense(
        np.array([10, 20], dtype=np.int64),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[0.5], [0.25]]),
        source=TRUE,
        status=OK,
    )
    assert records.columns["value"].dtype == np.dtype(np.float64)
    assert records.columns["subject_kind"].dtype == np.dtype(np.int8)
    assert records.columns["quantity_kind"].dtype == np.dtype(np.int8)
    assert records.columns["quantity_index"].dtype == np.dtype(np.int64)
    assert records.columns["status"].dtype == np.dtype(np.int8)
    assert records.columns["source"].dtype == np.dtype(np.int8)


def test_dense_storage_does_not_materialize_unused_optional_columns() -> None:
    records = ObservationRecords.from_dense(
        np.array([10], dtype=np.int64), np.array([[1.0]]), np.empty((1, 0))
    )
    assert "uncertainty" not in records.columns
    assert "provenance" not in records.columns
    try:
        records.column("uncertainty")
    except KeyError:
        pass
    else:
        raise AssertionError("unused optional column was materialized")
    assert records[0].uncertainty is None


def test_subject_payload_uses_descriptor_dtype() -> None:
    batch = ObservationBatch.from_dense(
        ObservationSchema(objective_count=1),
        np.array([10, 20], dtype=np.int64),
        np.array([[1.0], [2.0]]),
        np.empty((2, 0)),
    )
    payload = batch.records.columns["subject_payload"]
    assert payload.dtype == np.dtype(np.int64)
    assert payload.shape == (2, 1)
