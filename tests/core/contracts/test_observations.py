"""Contract tests for the J2 observation column store."""

from typing import cast

import numpy as np
import pytest

from saealib.core.contracts.observation import (
    CONSTRAINT,
    CV,
    OBJECTIVE,
    OBSERVATION_SUBJECT_KINDS,
    OK,
    SURROGATE,
    TRUE,
    ObservationSubjectKind,
)
from saealib.core.contracts.observations import (
    ObservationBatch,
    ObservationRecord,
    ObservationRecords,
    ObservationSchema,
)
from saealib.population.population import PopulationAttribute
from saealib.space.services import GenomeCodec
from saealib.space.space import ValidationResult


def _record(
    candidate: int,
    kind: str,
    index: int,
    value: float,
    *,
    source: str = TRUE,
    status: str = OK,
) -> ObservationRecord:
    return ObservationRecord(
        subject=("candidate", np.array([candidate], dtype=np.int64)),
        quantity=(kind, index),
        value=value,
        status=status,
        source=source,
    )


def _batch(records: list[ObservationRecord]) -> ObservationBatch:
    return ObservationBatch(
        schema=ObservationSchema(objective_count=2, constraint_count=1),
        records=ObservationRecords(records),
    )


def test_candidate_ids_use_registered_unknown_subject_kind() -> None:
    descriptor = ObservationSubjectKind(
        name="test:group",
        description="test group",
        arity="MANY",
        ordered=False,
        codec=cast(GenomeCodec, GenomeCodec),
        candidate_ids=lambda payload: np.asarray(payload, dtype=np.int64) + 10,
        validate=lambda payload: ValidationResult(valid_mask=(True,)),
        columns=(PopulationAttribute(name="ids", dtype=np.int64, default=-1),),
    )
    OBSERVATION_SUBJECT_KINDS.register("test:group", descriptor)
    try:
        records = ObservationRecords(
            (
                ObservationRecord(
                    subject=("test:group", np.array([7, 3])),
                    quantity=(OBJECTIVE, 0),
                    value=1,
                    status=OK,
                    source=TRUE,
                ),
            )
        )
        assert np.array_equal(
            ObservationBatch(
                schema=ObservationSchema(objective_count=1), records=records
            ).candidate_ids,
            [17, 13],
        )
    finally:
        del OBSERVATION_SUBJECT_KINDS._entries["test:group"]


def test_dense_rejects_each_incomplete_condition() -> None:
    complete = [
        _record(c, k, i, float(c + i))
        for c in (1, 2)
        for k, i in ((OBJECTIVE, 0), (OBJECTIVE, 1), (CONSTRAINT, 0))
    ]
    bad_subject = complete.copy()
    bad_subject[0] = ObservationRecord(
        subject=("candidate", np.array([1, 2])),
        quantity=(OBJECTIVE, 0),
        value=1,
        status=OK,
        source=TRUE,
    )
    with pytest.raises(ValueError, match="single-candidate"):
        _batch(bad_subject).f
    bad_coverage = complete[1:]
    with pytest.raises(ValueError, match="exactly one objective"):
        _batch(bad_coverage).f
    bad_constraint_coverage = complete[:-1]
    with pytest.raises(ValueError, match="exactly one constraint"):
        _batch(bad_constraint_coverage).f
    bad_status = complete.copy()
    bad_status[0] = _record(1, OBJECTIVE, 0, 1, status="failed")
    with pytest.raises(ValueError, match="statuses"):
        _batch(bad_status).f
    bad_source = complete.copy()
    bad_source[0] = _record(1, OBJECTIVE, 0, 1, source="surrogate")
    with pytest.raises(ValueError, match="common source"):
        _batch(bad_source).f


def test_column_operations_preserve_columns_and_values() -> None:
    records = ObservationRecords(tuple(_record(c, OBJECTIVE, 0, c) for c in (1, 2, 3)))
    selected = records.select(source=TRUE).take([2, 0])
    joined = ObservationRecords.concat((selected, records.take([1])))
    assert selected.column("value").tolist() == [3, 1]
    assert joined.column("value").tolist() == [3, 1, 2]
    assert [record.value for record in selected] == [3, 1]


def test_schema_declares_registered_extra_quantity_spaces() -> None:
    schema = ObservationSchema(
        objective_count=2,
        constraint_count=1,
        quantities={"feature": ("height", "width"), CV: 1},
        schema_version=3,
    )
    assert schema.schema_version == 3
    assert schema.quantity_kinds == (OBJECTIVE, CONSTRAINT, "feature", CV)
    assert schema.indices(OBJECTIVE) == (0, 1)
    assert schema.indices("feature") == ("height", "width")
    assert schema.indices(CV) == (0,)
    with pytest.raises(ValueError, match="not registered"):
        ObservationSchema(quantities={"not_registered": 1})


def test_mapping_input_supplies_optional_columns_and_quantity_selection() -> None:
    records = ObservationRecords(
        {
            "subject": [
                ("candidate", np.array([1], dtype=np.int64)),
                ("candidate", np.array([2], dtype=np.int64)),
            ],
            "quantity": [(OBJECTIVE, 0), (OBJECTIVE, 0)],
            "value": [10.0, 20.0],
            "status": [OK, OK],
            "source": [TRUE, SURROGATE],
        }
    )
    assert records.column("uncertainty").tolist() == [None, None]
    selected = records.select(quantity=(OBJECTIVE, 0), source=TRUE)
    assert selected.column("value").tolist() == [10.0]
    assert records.column("subject").shape == (2,)


def test_select_returns_self_or_a_slice_view_when_possible() -> None:
    records = ObservationRecords(
        tuple(
            _record(candidate, OBJECTIVE, 0, float(candidate), source=source)
            for candidate, source in ((1, TRUE), (2, TRUE), (3, SURROGATE))
        )
    )
    assert records.select(source=TRUE) is not records
    contiguous = records.select(subject=("candidate", np.array([2], dtype=np.int64)))
    assert contiguous.column("value").tolist() == [2.0]
    assert contiguous.column("value").base is records.column("value")
    assert records.select(status=OK) is records


def test_batch_builtins_do_not_use_getitem_on_the_record_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _batch(
        [
            _record(c, k, i, float(c + i))
            for c in (1, 2)
            for k, i in ((OBJECTIVE, 0), (OBJECTIVE, 1), (CONSTRAINT, 0))
        ]
    )

    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("batch built-in called __getitem__")

    monkeypatch.setattr(ObservationRecords, "__getitem__", fail)
    assert np.array_equal(batch.candidate_ids, [1, 2])
    assert batch.f.shape == (2, 2)
    assert batch.g.shape == (2, 1)
    assert batch.cv.shape == (2,)


def test_select_result_matches_source_rows() -> None:
    records = ObservationRecords(
        tuple(
            _record(candidate, OBJECTIVE, 0, float(candidate), source=source)
            for candidate, source in ((1, TRUE), (2, SURROGATE), (3, TRUE))
        )
    )
    selected = records.select(source=TRUE)
    assert selected.column("value").tolist() == [1.0, 3.0]
    assert selected.column("source").tolist() == [TRUE, TRUE]
    assert len(selected) == 2


def test_batch_dense_builtins_do_not_materialize_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _batch(
        [
            _record(c, k, i, float(c + i))
            for c in (1, 2)
            for k, i in ((OBJECTIVE, 0), (OBJECTIVE, 1), (CONSTRAINT, 0))
        ]
    )

    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("batch built-in called __getitem__")

    monkeypatch.setattr(batch.records, "__getitem__", fail)
    assert batch.f.shape == (2, 2)
    assert batch.g.shape == (2, 1)
    assert batch.cv.shape == (2,)
