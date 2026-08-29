"""Focused tests for the public History accessors."""

from __future__ import annotations

import numpy as np
import pytest

from saealib.exceptions import ValidationError
from saealib.execution.history import History


def test_get_selects_scalar_and_block_columns_as_read_only_views() -> None:
    history = History(("summary", "population"))
    history.append("summary", gen=3, score=1.5)
    history.append_block("population", {"x": np.array([[1.0], [2.0]])}, size=2)

    scalar = history.get("summary", "gen")
    block = history.get("population", "x")

    assert isinstance(scalar, np.ndarray)
    assert scalar.flags.writeable is False
    assert np.shares_memory(scalar, history.channel("summary")["gen"])
    assert isinstance(block, tuple)
    assert len(block) == 1
    assert block[0].flags.writeable is False
    assert np.shares_memory(block[0], np.asarray(history.blocks("population", "x")[0]))
    with pytest.raises(ValueError, match="read-only"):
        scalar[0] = 4
    with pytest.raises(ValueError, match="read-only"):
        block[0][0, 0] = 4


def test_get_reports_unknown_channel_and_scalar_and_block_columns() -> None:
    history = History(("summary",))
    history.append_block("summary", {"x": np.zeros((1, 1))}, gen=0)

    with pytest.raises(ValidationError, match="not enabled"):
        history.get("population", "x")
    with pytest.raises(ValidationError) as excinfo:
        history.get("summary", "missing")
    message = str(excinfo.value)
    assert "gen" in message
    assert "x" in message
    assert "scalar" in message
    assert "block" in message


def test_records_support_mixed_channels_and_preserve_row_count() -> None:
    history = History(("summary",))
    history.append_block("summary", {"x": np.array([[1.0], [2.0]])}, gen=0, size=2)
    history.append_block("summary", {"x": np.array([[3.0]])}, gen=1, size=1)

    records = list(history.records("summary"))

    assert len(records) == 2
    assert records[0]["gen"] == 0
    assert isinstance(records[0]["size"], int)
    np.testing.assert_array_equal(records[0]["x"], [[1.0], [2.0]])
    assert records[1]["x"].shape == (1, 1)
    assert records[0]["x"].flags.writeable is False
    with pytest.raises(TypeError):
        records[0]["gen"] = 10  # type: ignore[index]
    with pytest.raises(ValueError, match="read-only"):
        records[0]["x"][0, 0] = 10


def test_records_support_scalar_only_block_only_and_empty_channels() -> None:
    scalar = History(("summary",))
    scalar.append("summary", gen=0)
    assert [record["gen"] for record in scalar.records("summary")] == [0]

    block = History(("population",))
    block.append_block("population", {"x": np.zeros((0, 2))})
    block_records = list(block.records("population"))
    assert len(block_records) == 1
    assert block_records[0]["x"].shape == (0, 2)

    empty = History(("front",))
    assert list(empty.records("front")) == []
