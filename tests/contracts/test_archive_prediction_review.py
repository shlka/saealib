import numpy as np
import pytest

from saealib.exceptions import ValidationError
from saealib.population import Archive, Population, PopulationAttribute
from saealib.surrogate import PredictionChannel, SurrogatePrediction


def _attrs():
    return [
        PopulationAttribute("x", np.float64, (2,), default=np.nan),
        PopulationAttribute("f", np.float64, (1,), default=np.nan),
        PopulationAttribute("g", np.float64, (1,), default=np.nan),
        PopulationAttribute("cv", np.float64, (), default=0.0),
        PopulationAttribute("id", np.int64, (), default=-1),
    ]


def _archive():
    archive = Archive(_attrs(), 4, duplicate_policy="replace")
    archive.add(
        id=np.int64(1),
        x=np.array([1.0, 2.0]),
        f=np.array([3.0]),
        g=np.array([4.0]),
        cv=np.array(0.0),
    )
    archive.add(
        id=np.int64(2),
        x=np.array([5.0, 6.0]),
        f=np.array([7.0]),
        g=np.array([8.0]),
        cv=np.array(0.0),
    )
    return archive


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"x": np.array([1.0]), "id": np.int64(3)}, "shape"),
        ({"x": np.array([1.0, 2.0]), "id": np.int64(-1)}, "sentinel"),
        ({"x": np.array([1.0, 2.0]), "id": np.int64(2)}, "Duplicate"),
        ({"x": np.array([1, 2], dtype=np.int64), "id": np.int64(3)}, "dtype"),
    ],
)
def test_replace_validates_before_delete(kwargs, match):
    archive = _archive()
    before = {name: np.array(value, copy=True) for name, value in archive._data.items()}
    versions = (archive._size, archive._structure_version, archive._value_version)
    archive.get_knn(np.array([0.0, 0.0]), 1)
    with pytest.raises(ValidationError, match=match):
        archive.add(f=np.array([9.0]), g=np.array([10.0]), cv=np.array(0.0), **kwargs)
    for name, value in before.items():
        np.testing.assert_array_equal(archive._data[name], value)
    assert (
        archive._size,
        archive._structure_version,
        archive._value_version,
    ) == versions
    assert archive._kdtree is not None


@pytest.mark.parametrize("kind", ["dict", "kwargs", "individual"])
def test_same_candidate_retry_updates_all_values(kind):
    archive = _archive()
    values = {
        "id": np.int64(1),
        "x": np.array([1.0, 2.0]),
        "f": np.array([30.0]),
        "g": np.array([40.0]),
        "cv": np.array(5.0),
    }
    if kind == "dict":
        archive.add(values)
    elif kind == "kwargs":
        archive.add(**values)
    else:
        source = Population(_attrs(), 1)
        source._append_internal(values, preserve_ids=True)
        archive.add(source[0])
    np.testing.assert_array_equal(archive.id, [1, 2])
    np.testing.assert_array_equal(archive.f[0], [30.0])
    np.testing.assert_array_equal(archive.g[0], [40.0])
    assert archive.cv[0] == 5.0


def test_prediction_x_row_alignment_is_strict():
    channel = PredictionChannel(np.ones((2, 1), dtype=np.float64))
    with pytest.raises(ValidationError, match="shape"):
        SurrogatePrediction({"objective": channel}, x=np.ones((1, 3)))
    prediction = SurrogatePrediction(
        {"objective": channel}, x=np.ones((2, 3), dtype=np.float32)
    )
    assert prediction.x is not None
    assert prediction.x.dtype == np.float64
    assert prediction.x.flags.owndata
    assert prediction.x.flags.c_contiguous


@pytest.mark.parametrize(
    "id_dtype, request_dtype, shape",
    [
        (np.int32, np.int64, ()),
        (np.int64, np.uint64, ()),
        (np.int64, object, ()),
        (np.int64, np.int64, (1,)),
    ],
)
def test_append_schema_requires_int64_scalar_ids(id_dtype, request_dtype, shape):
    attrs = _attrs()
    attrs[-1] = PopulationAttribute("id", id_dtype, shape, default=-1)
    attrs.append(PopulationAttribute("request_id", request_dtype, shape, default=-1))
    with pytest.raises(ValidationError, match="int64 scalar"):
        Archive(attrs, duplicate_policy="append")


def test_pareto_append_is_rejected():
    with pytest.raises(ValidationError, match="does not support append"):
        from saealib.population import ParetoArchive

        ParetoArchive(_attrs(), duplicate_policy="append")
