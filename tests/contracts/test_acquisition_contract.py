"""Acquisition and archive boundary contracts."""

import warnings
from typing import Any, cast

import numpy as np
import pytest

from saealib.acquisition import MeanPrediction
from saealib.acquisition.base import (
    CompositeAcquisition,
)
from saealib.exceptions import ValidationError
from saealib.population import Archive, PopulationAttribute
from saealib.surrogate.manager import GlobalSurrogateManager
from saealib.surrogate.prediction import PredictionChannel, SurrogatePrediction


def _archive() -> Archive:
    attrs = [
        PopulationAttribute("x", np.float64, (1,)),
        PopulationAttribute("f", np.float64, (1,)),
        PopulationAttribute("id", np.int64, (), -1),
    ]
    return Archive(attrs, duplicate_policy="keep_first")


def test_prediction_owns_arrays_and_normalizes_dtype() -> None:
    value = np.array([[1]], dtype=np.int32)
    pred = SurrogatePrediction(channels={"objective": PredictionChannel(value)})
    value[0, 0] = 9
    assert pred.value.dtype == np.float64
    assert pred.value.flags.owndata
    assert pred.value[0, 0] == 1


def test_prediction_rejects_channel_row_mismatch() -> None:
    with pytest.raises(ValidationError):
        SurrogatePrediction(
            channels={
                "a": PredictionChannel(np.zeros((2, 1))),
                "b": PredictionChannel(np.zeros((3, 1))),
            }
        )


def test_manager_does_not_own_acquisition_and_provider_can_exchange_it() -> None:
    manager = GlobalSurrogateManager(cast(Any, object()))
    assert not hasattr(manager, "acquisition")
    assert not hasattr(manager, "iter_acquisitions")
    first = MeanPrediction()
    second = MeanPrediction()
    assert first is not second


def test_pointwise_empty_batch_is_owned_float64_and_does_not_prepare() -> None:
    class Counting(MeanPrediction):
        def __init__(self) -> None:
            super().__init__()
            self.prepares = 0

        def prepare(self, archive, ctx=None):
            self.prepares += 1
            return super().prepare(archive, ctx)

    acq = Counting()
    result = acq.evaluate(
        np.empty((0, 1)),
        SurrogatePrediction.objective(np.empty((0, 1))),
        _archive(),
    )
    assert result.scores is not None
    assert result.scores.shape == (0,)
    assert result.scores.dtype == np.float64
    assert acq.prepares == 0


def test_composite_rejects_missing_channel_before_combining() -> None:
    acq = CompositeAcquisition(
        {"constraint": MeanPrediction()}, lambda scores: scores[0]
    )
    pred = SurrogatePrediction.objective(np.zeros((2, 1)))
    with pytest.raises(ValidationError, match="missing configured channel"):
        acq.evaluate(np.zeros((2, 1)), pred, _archive())


@pytest.mark.parametrize(
    "policy, expected", [("keep_first", 1), ("replace", 1), ("append", 2)]
)
def test_archive_duplicate_policy(policy: str, expected: int) -> None:
    archive = Archive(
        [
            PopulationAttribute("x", np.float64, (1,)),
            PopulationAttribute("f", np.float64, (1,)),
            PopulationAttribute("id", np.int64, (), -1),
            PopulationAttribute("request_id", np.int64, (), -1),
        ],
        duplicate_policy=policy,
    )
    archive.add(x=[0.0], f=[1.0], id=10, request_id=20)
    archive.add(x=[0.0], f=[2.0], id=11, request_id=21)
    assert len(archive) == expected
    if policy == "replace":
        assert archive.id[0] == 11
        assert archive.f[0, 0] == 2.0


def test_duplicate_history_api_is_deprecated() -> None:
    archive = _archive()
    archive.add(x=[0.0], f=[1.0], id=1)
    archive.add(x=[0.0], f=[2.0], id=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        archive.get_duplicated_population()
    assert any(item.category is DeprecationWarning for item in caught)
