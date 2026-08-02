import json
import pickle
import subprocess
from typing import Any

import numpy as np
import pytest

from saealib.context import CURRENT_CHECKPOINT_SCHEMA_VERSION, OptimizationState
from saealib.exceptions import CheckpointError, ValidationError
from saealib.identity import IDAllocator
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem


class _PendingGadget:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return subprocess.call, (["touch", str(self.path)],)


def _attrs():
    return [
        PopulationAttribute("x", np.float64, (2,), default=np.nan),
        PopulationAttribute("f", np.float64, (1,), default=np.nan),
        PopulationAttribute("g", np.float64, (0,), default=0.0),
        PopulationAttribute("cv", np.float64, (), default=0.0),
        PopulationAttribute("id", np.int64, (), default=-1),
        PopulationAttribute("request_id", np.int64, (), default=-1),
    ]


def _problem():
    return Problem(
        func=lambda x: np.array([np.sum(x**2)]),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )


def _state():
    attrs = _attrs()
    population = Population(attrs, 4)
    population._extend_internal(
        {
            "x": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "f": np.array([[1.0], [2.0]]),
            "g": np.empty((2, 0)),
            "cv": np.zeros(2),
            "id": np.array([7, 8], dtype=np.int64),
            "request_id": np.array([17, 18], dtype=np.int64),
        },
        preserve_ids=True,
    )
    archive = Archive(
        attrs,
        4,
        key_attr="x",
        atol=1e-6,
        rtol=2e-6,
        duplicate_policy="replace",
    )
    archive._extend_internal(population, preserve_ids=True)
    pareto = ParetoArchive(attrs, 4, direction=np.array([-1.0]), eps_cv=0.25)
    pareto._extend_internal(population, preserve_ids=True)
    ctx = OptimizationState(
        problem=_problem(),
        populations={"main": population},
        archives={"main": archive, "pareto": pareto},
        rng=np.random.default_rng(3),
        candidate_id_allocator=IDAllocator(20),
        request_id_allocator=IDAllocator(30),
        fe=2,
        gen=4,
    )
    history = Archive(attrs, 2, duplicate_policy="append")
    history._extend_internal(population, preserve_ids=True)
    ctx.add_archive("history", history)
    ctx.add_population("backup", population.extract([0]))
    return ctx


def test_named_aliases_and_collection_validation():
    ctx = _state()
    assert ctx.population is ctx.populations["main"]
    assert ctx.archive is ctx.archives["main"]
    assert ctx.pareto_archive is ctx.archives["pareto"]
    with pytest.raises(ValidationError):
        ctx.add_archive("bad//name", Archive(_attrs()))
    with pytest.raises(ValidationError):
        ctx.add_population("other", object())


def test_named_collections_checkpoint_roundtrip(tmp_path):
    ctx = _state()
    path = tmp_path / "named.npz"
    ctx.save(path)
    loaded = OptimizationState.load(path, ctx.problem)
    assert loaded.population is loaded.populations["main"]
    assert loaded.archive is loaded.archives["main"]
    assert loaded.pareto_archive is loaded.archives["pareto"]
    assert loaded.archives["history"].duplicate_policy == "append"
    assert loaded.archive.key_attr == "x"
    assert loaded.archive.atol == 1e-6
    assert loaded.archive.rtol == 2e-6
    assert loaded.pareto_archive.eps_cv == 0.25
    np.testing.assert_array_equal(loaded.archives["history"].id, [7, 8])
    assert loaded.candidate_id_allocator.next_value == 20
    assert loaded.request_id_allocator.next_value == 30
    assert loaded.data["resumed"] is True


def test_current_schema_and_allocator_continuity(tmp_path):
    ctx = _state()
    path = tmp_path / "current.npz"
    ctx.save(path)
    raw = np.load(path, allow_pickle=False)
    assert int(raw["_checkpoint_schema_version"]) == CURRENT_CHECKPOINT_SCHEMA_VERSION
    loaded = OptimizationState.load(path, ctx.problem)
    assert loaded.candidate_id_allocator.allocate(1).tolist() == [20]
    assert loaded.request_id_allocator.allocate(1).tolist() == [30]


def test_pareto_none_direction_roundtrips(tmp_path):
    ctx = _state()
    ctx.pareto_archive.direction = None
    path = tmp_path / "none-direction.npz"
    ctx.save(path)
    loaded = OptimizationState.load(path, ctx.problem)
    assert loaded.pareto_archive.direction is None


def test_v1_checkpoint_is_migrated_without_mutating_payload(tmp_path):
    ctx = _state()
    schema = [
        {
            "name": attr.name,
            "dtype": np.dtype(attr.dtype).str,
            "shape": list(attr.shape),
            "default": "__nan__"
            if isinstance(attr.default, float) and np.isnan(attr.default)
            else attr.default,
        }
        for attr in ctx.archive.schema.values()
    ]
    payload: dict[str, Any] = {
        "_checkpoint_schema_version": np.array(1),
        "_schema": np.frombuffer(json.dumps(schema).encode(), dtype=np.uint8),
        "_rng_state": np.frombuffer(
            json.dumps(
                ctx.rng.bit_generator.state, default=lambda x: x.tolist()
            ).encode(),
            dtype=np.uint8,
        ),
        "_fe": np.array(ctx.fe),
        "_gen": np.array(ctx.gen),
        "_next_candidate_id": np.array(20),
        "_next_request_id": np.array(30),
        "_pending_evaluations": np.frombuffer(
            pickle.dumps({}, protocol=4), dtype=np.uint8
        ),
        "_archive_size": np.array(len(ctx.archive)),
        "_pop_size": np.array(len(ctx.population)),
        "_pareto_size": np.array(len(ctx.pareto_archive)),
    }
    for prefix, collection in (
        ("archive", ctx.archive),
        ("pop", ctx.population),
        ("pareto", ctx.pareto_archive),
    ):
        for name, array in collection._data.items():
            payload[f"{prefix}__{name}"] = array[: len(collection)]
    path = tmp_path / "legacy.npz"
    np.savez(path, **payload)
    before = {key: np.array(value, copy=True) for key, value in payload.items()}
    loaded = OptimizationState.load(path, ctx.problem)
    assert loaded.archive is loaded.archives["main"]
    bad_payload = dict(payload)
    bad_payload["_next_candidate_id"] = np.array(-1)
    bad_path = tmp_path / "legacy-bad-allocator.npz"
    np.savez(bad_path, **bad_payload)
    with pytest.raises(CheckpointError, match="allocator"):
        OptimizationState.load(bad_path, ctx.problem)
    for key, value in before.items():
        np.testing.assert_array_equal(np.asarray(payload[key]), value)


def test_invalid_manifest_fails_atomically(tmp_path):
    ctx = _state()
    path = tmp_path / "valid.npz"
    ctx.save(path)
    raw = np.load(path, allow_pickle=False)
    payload = dict(raw.items())
    manifest = json.loads(bytes(payload["_manifest"]).decode())
    manifest["archives"][0]["subtype"] = "UnknownArchive"
    payload["_manifest"] = np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8)
    bad = tmp_path / "bad.npz"
    np.savez(bad, **payload)
    with pytest.raises(CheckpointError, match="unknown archive subtype"):
        OptimizationState.load(bad, ctx.problem)


def test_append_history_preserves_repeated_candidate_observations(tmp_path):
    ctx = _state()
    history = ctx.archives["history"]
    history.add(
        id=np.int64(7),
        request_id=np.int64(19),
        x=np.array([0.1, 0.2], dtype=np.float64),
        f=np.array([3.0], dtype=np.float64),
        g=np.empty(0, dtype=np.float64),
        cv=np.array(0.0, dtype=np.float64),
    )
    with pytest.raises(ValidationError, match="request_id"):
        Archive(_attrs()[:-1], 2, duplicate_policy="append")
    with pytest.raises(ValidationError, match="pair"):
        history.add(
            id=np.int64(7),
            request_id=np.int64(19),
            x=np.array([0.1, 0.2], dtype=np.float64),
            f=np.array([4.0], dtype=np.float64),
            g=np.empty(0, dtype=np.float64),
            cv=np.array(0.0, dtype=np.float64),
        )
    path = tmp_path / "history.npz"
    ctx.save(path)
    loaded = OptimizationState.load(path, ctx.problem)
    np.testing.assert_array_equal(loaded.archives["history"].id, [7, 8, 7])
    np.testing.assert_array_equal(loaded.archives["history"].request_id, [17, 18, 19])


def test_named_collection_mutations_and_legacy_replace():
    ctx = _state()
    with pytest.raises(ValidationError):
        ctx.archives.pop("main")
    ctx.archives.clear()
    assert ctx.archives["main"] is ctx.archive
    assert ctx.archives["pareto"] is ctx.pareto_archive
    with pytest.raises(ValidationError):
        ctx.archives.update({"main": ctx.pareto_archive})
    with pytest.raises(ValidationError):
        ctx.archives |= {"pareto": ctx.archive}
    replacement = ctx.population.extract([0])
    updated = ctx.replace(population=replacement)
    assert updated.populations["main"] is replacement
    assert updated.populations["backup"] is ctx.populations["backup"]
    with pytest.raises(ValidationError):
        ctx.replace(population=replacement, populations=ctx.populations)
    ctx.add_population("extra", ctx.population.extract([0]))
    key, value = ctx.populations.popitem()
    assert key == "extra"
    assert value is not None
    ctx.populations.pop("backup")
    with pytest.raises(ValidationError):
        ctx.populations.popitem()


@pytest.mark.parametrize(
    "value",
    [
        np.array(-1),
        np.array(True),
        np.array(1.5),
        np.array(2**63, dtype=np.uint64),
    ],
)
def test_checkpoint_rejects_corrupt_allocator_values(tmp_path, value):
    ctx = _state()
    path = tmp_path / "valid.npz"
    ctx.save(path)
    raw = dict(np.load(path, allow_pickle=False).items())
    raw["_next_candidate_id"] = value
    bad = tmp_path / "bad.npz"
    np.savez(bad, **raw)
    with pytest.raises(CheckpointError, match=r"allocator|scalar|int64"):
        OptimizationState.load(bad, ctx.problem)


def test_data_keys_and_resumed_marker_are_safe(tmp_path):
    ctx = _state().replace(
        data={
            "nested": {"items": np.array([1, 2], dtype=np.int64)},
            "resumed": False,
        }
    )
    path = tmp_path / "data.npz"
    ctx.save(path)
    loaded = OptimizationState.load(path, ctx.problem)
    assert loaded.data["resumed"] is True
    assert loaded.data["nested"]["items"] == [1, 2]
    with pytest.raises(CheckpointError, match="keys must be strings"):
        _state().replace(data={1: "collision", "1": "value"}).save(
            tmp_path / "bad-data.npz"
        )


def test_v1_pending_pickle_is_not_deserialized(tmp_path):
    ctx = _state()
    schema = [
        {
            "name": attr.name,
            "dtype": np.dtype(attr.dtype).str,
            "shape": list(attr.shape),
            "default": "__nan__"
            if isinstance(attr.default, float) and np.isnan(attr.default)
            else attr.default,
        }
        for attr in ctx.archive.schema.values()
    ]
    payload: dict[str, Any] = {
        "_checkpoint_schema_version": np.array(1),
        "_schema": np.frombuffer(json.dumps(schema).encode(), dtype=np.uint8),
        "_rng_state": np.frombuffer(
            json.dumps(
                ctx.rng.bit_generator.state, default=lambda x: x.tolist()
            ).encode(),
            dtype=np.uint8,
        ),
        "_fe": np.array(ctx.fe),
        "_gen": np.array(ctx.gen),
        "_next_candidate_id": np.array(20),
        "_next_request_id": np.array(30),
        "_archive_size": np.array(len(ctx.archive)),
        "_pop_size": np.array(len(ctx.population)),
        "_pareto_size": np.array(len(ctx.pareto_archive)),
        "_pending_evaluations": np.frombuffer(
            pickle.dumps(_PendingGadget(tmp_path / "executed")), dtype=np.uint8
        ),
    }
    for prefix, collection in (
        ("archive", ctx.archive),
        ("pop", ctx.population),
        ("pareto", ctx.pareto_archive),
    ):
        for name, array in collection._data.items():
            payload[f"{prefix}__{name}"] = array[: len(collection)]
    path = tmp_path / "unsafe.npz"
    np.savez(path, **payload)
    with pytest.raises(CheckpointError, match="safe empty"):
        OptimizationState.load(path, ctx.problem)
    assert not (tmp_path / "executed").exists()
