import json
import pickle
import subprocess
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from saealib.context import CURRENT_CHECKPOINT_SCHEMA_VERSION, OptimizationState
from saealib.core.contracts.data import Fixed
from saealib.core.contracts.representation import ParameterSpec, RepresentationSpec
from saealib.core.state import STATE_MIGRATORS, StateKey
from saealib.exceptions import CheckpointError, ValidationError
from saealib.identity import IDAllocator
from saealib.population import (
    Archive,
    DenseVectorBatch,
    ObjectBatch,
    ParetoArchive,
    Population,
    PopulationAttribute,
)
from saealib.problem import Problem
from saealib.space import ObjectSpace
from saealib.stages import PendingEvaluationContextStage


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


def test_reservation_properties_are_derived_from_pending_evaluations():
    ctx = _state().replace(
        pending_evaluations={
            1: SimpleNamespace(
                request=SimpleNamespace(candidate_ids=np.array([8, 3], dtype=np.int64)),
                reserved_cost=1.25,
            ),
            2: SimpleNamespace(
                request=SimpleNamespace(candidate_ids=np.array([3, 9], dtype=np.int64)),
                reserved_cost=2.75,
            ),
        }
    )
    np.testing.assert_array_equal(ctx.pending_candidate_ids, [3, 8, 9])
    assert ctx.reserved_fe == 4
    assert ctx.reserved_cost == 4.0


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
    assert CURRENT_CHECKPOINT_SCHEMA_VERSION == 3
    ctx = _state()
    path = tmp_path / "current.npz"
    ctx.save(path)
    raw = np.load(path, allow_pickle=False)
    assert int(raw["_checkpoint_schema_version"]) == CURRENT_CHECKPOINT_SCHEMA_VERSION
    loaded = OptimizationState.load(path, ctx.problem)
    assert loaded.candidate_id_allocator.allocate(1).tolist() == [20]
    assert loaded.request_id_allocator.allocate(1).tolist() == [30]

    entries = json.loads(bytes(raw["_state_entries"]).decode())
    versions = {
        (item["key"]["namespace"], item["key"]["name"]): item["key"]["schema_version"]
        for item in entries
    }
    assert versions[("populations", "main")] == 2
    assert versions[("populations", "backup")] == 2
    assert versions[("archives", "main")] == 1


def test_population_key_construction_does_not_register_migrators():
    """Named key construction must not mutate the migration registry."""
    before = STATE_MIGRATORS.registered()
    _state()
    assert STATE_MIGRATORS.registered() == before


def _rewrite_population_entry_as_v1(path, population_name="main"):
    """Rewrite one population entry to the pre-codec v1 shape."""
    raw = dict(np.load(path, allow_pickle=False).items())
    entries = json.loads(bytes(raw["_state_entries"]).decode())
    for index, item in enumerate(entries):
        key = item["key"]
        if key["namespace"] != "populations" or key["name"] != population_name:
            continue
        genomes = item["value"].pop("genomes")
        raw[f"_entry_{index}__x"] = raw.pop(genomes["array"])
        key["schema_version"] = 1
        item["target_schema_version"] = 2
        break
    else:
        raise AssertionError(f"populations/{population_name} entry was not found")
    raw["_state_entries"] = np.frombuffer(json.dumps(entries).encode(), dtype=np.uint8)
    np.savez(path, **raw)


def test_v3_population_genome_roundtrip_uses_codec(tmp_path):
    ctx = _state()
    space = ctx.problem.space
    original = space.services.require("GenomeCodec")
    calls = {"encode": 0, "decode": 0}

    class SpyCodec:
        def encode(self, genomes):
            calls["encode"] += 1
            return original.encode(genomes)

        def decode(self, payload):
            calls["decode"] += 1
            return original.decode(payload)

    space.services.register("GenomeCodec", SpyCodec())
    path = tmp_path / "population-codec.npz"
    ctx.save(path)
    raw = np.load(path, allow_pickle=False)
    entries = json.loads(bytes(raw["_state_entries"]).decode())
    main_index, main = next(
        (index, item)
        for index, item in enumerate(entries)
        if item["key"]
        == {"namespace": "populations", "name": "main", "schema_version": 2}
    )
    assert "genomes" in main["value"]
    assert f"_entry_{main_index}__x" not in raw.files

    loaded = OptimizationState.load(path, ctx.problem)
    assert calls == {"encode": 2, "decode": 2}
    assert isinstance(loaded.population.genomes, DenseVectorBatch)
    assert isinstance(ctx.population.genomes, DenseVectorBatch)
    np.testing.assert_array_equal(
        loaded.population.genomes.array, ctx.population.genomes.array
    )


def test_v3_population_v1_entry_uses_registered_real_key_migrator(tmp_path):
    ctx = _state()
    path = tmp_path / "population-v1-entry.npz"
    ctx.save(path)
    _rewrite_population_entry_as_v1(path)

    loaded = OptimizationState.load(path, ctx.problem)
    migrated_key = StateKey(namespace="populations", name="main", schema_version=2)
    assert loaded.get_state(migrated_key) is loaded.population
    np.testing.assert_array_equal(loaded.population.x, ctx.population.x)


def test_v3_population_v1_entry_without_migrator_names_key_and_versions(tmp_path):
    ctx = _state()
    path = tmp_path / "population-missing-migrator.npz"
    ctx.save(path)
    _rewrite_population_entry_as_v1(path)

    migration_id = ("populations", "main", 1)
    original_migrators = STATE_MIGRATORS._migrators
    STATE_MIGRATORS._migrators = {
        key: value for key, value in original_migrators.items() if key != migration_id
    }
    try:
        with pytest.raises(CheckpointError) as error:
            OptimizationState.load(path, ctx.problem)
    finally:
        STATE_MIGRATORS._migrators = original_migrators

    message = str(error.value)
    assert "populations/main" in message
    assert "v1" in message and "v2" in message
    assert "v1 -> v2" in message


def test_v3_population_v1_arbitrary_name_resolves_at_load(tmp_path):
    """A legacy named population loads without prior StateKey construction."""
    ctx = _state()
    path = tmp_path / "population-arbitrary-v1-entry.npz"
    ctx.save(path)
    _rewrite_population_entry_as_v1(path, "backup")

    migration_id = ("populations", "backup", 1)
    original_migrators = STATE_MIGRATORS._migrators
    STATE_MIGRATORS._migrators = {
        key: value for key, value in original_migrators.items() if key != migration_id
    }
    try:
        loaded = OptimizationState.load(path, ctx.problem)
    finally:
        STATE_MIGRATORS._migrators = original_migrators

    migrated_key = StateKey(namespace="populations", name="backup", schema_version=2)
    assert loaded.get_state(migrated_key) is loaded.populations["backup"]
    np.testing.assert_array_equal(
        loaded.populations["backup"].x, ctx.populations["backup"].x
    )


def test_object_space_population_save_requires_genome_codec(tmp_path):
    ctx = _state()
    object_population = Population(
        [PopulationAttribute("f", np.float64, (1,), default=np.nan)],
        genomes=ObjectBatch(["genome"]),
    )
    ctx.problem._space = _object_space()
    ctx = ctx.replace(population=object_population)

    with pytest.raises(
        CheckpointError,
        match=r"GenomeCodec is required to save population populations/main",
    ):
        ctx.save(tmp_path / "object-space.npz")


def _object_space():
    """Build an ObjectSpace with a registered representation kind."""
    representation = RepresentationSpec(
        kind="sequence",
        parameters=(
            ParameterSpec(name="alphabet", value=Fixed(value=frozenset({"a"}))),
            ParameterSpec(name="min_length", value=Fixed(value=1)),
            ParameterSpec(name="max_length", value=Fixed(value=10)),
        ),
    )
    return ObjectSpace(representation)


def test_object_space_population_load_requires_genome_codec(tmp_path):
    ctx = _state()
    path = tmp_path / "vector-space.npz"
    ctx.save(path)
    ctx.problem._space = _object_space()

    with pytest.raises(CheckpointError, match=r"GenomeCodec.*populations/main"):
        OptimizationState.load(path, ctx.problem)


def test_v2_helper_writes_a_self_consistent_v2_checkpoint(tmp_path):
    ctx = _state()
    path = tmp_path / "helper-v2.npz"
    ctx._save_v2(path)
    raw = np.load(path, allow_pickle=False)
    assert int(raw["_checkpoint_schema_version"]) == 2
    assert json.loads(bytes(raw["_manifest"]).decode())["schema_version"] == 2
    loaded = OptimizationState.load(path, ctx.problem)
    assert loaded.fe == ctx.fe
    assert loaded.gen == ctx.gen
    assert loaded.candidate_id_allocator.next_value == 20
    np.testing.assert_array_equal(loaded.population.x, ctx.population.x)
    assert loaded.data == {"resumed": True}


def test_v3_round_trips_every_store_entry_and_custom_state(tmp_path):
    ctx = _state()
    ctx.offspring = ctx.population
    custom_key = StateKey(namespace="user", name="g5b_custom", schema_version=1)
    ctx.set_state(custom_key, {"answer": 42})
    path = tmp_path / "v3-store.npz"
    ctx.save(path)

    raw = np.load(path, allow_pickle=False)
    entries = json.loads(bytes(raw["_state_entries"]).decode())
    saved_keys = {(item["key"]["namespace"], item["key"]["name"]) for item in entries}
    assert saved_keys == {(key.namespace, key.name) for key in ctx._store._values}
    loaded = OptimizationState.load(path, ctx.problem)
    assert loaded.offspring is not None
    np.testing.assert_array_equal(loaded.offspring.x, loaded.population.x)
    assert loaded.get_state(custom_key) == {"answer": 42}
    assert loaded._store._values.keys() == ctx._store._values.keys()


def test_store_fields_have_no_class_attribute_residue():
    for name in (
        "predictions",
        "feedback_result",
        "offspring",
    ):
        assert name not in vars(OptimizationState)
    ctx = _state()
    marker = object()
    ctx.predictions = marker
    assert ctx.predictions is marker


def test_getstate_excludes_derived_evaluation_ids():
    ctx = _state().replace(
        evaluation_new_ids=np.array([1, 2], dtype=np.int64),
        evaluation_update_new_ids=[np.array([3], dtype=np.int64)],
    )
    serialized = ctx.__getstate__()
    assert "evaluation_new_ids" not in serialized
    assert "evaluation_update_new_ids" not in serialized


def test_v3_registered_entry_migrator_is_used_at_load_time(tmp_path):
    key = StateKey(namespace="user", name="g5b_migrated", schema_version=1)
    STATE_MIGRATORS.register(
        key.namespace,
        key.name,
        1,
        lambda value: {**cast(dict[str, Any], value), "migrated": True},
    )
    ctx = _state()
    ctx.set_state(key, {"answer": 42})
    path = tmp_path / "migrated.npz"
    ctx.save(path)
    raw = dict(np.load(path, allow_pickle=False).items())
    entries = json.loads(bytes(raw["_state_entries"]).decode())
    for item in entries:
        if item["key"]["name"] == key.name:
            item["key"]["schema_version"] = 1
            item["target_schema_version"] = 2
    raw["_state_entries"] = np.frombuffer(json.dumps(entries).encode(), dtype=np.uint8)
    np.savez(path, **raw)
    loaded = OptimizationState.load(path, ctx.problem)
    migrated_key = StateKey(namespace=key.namespace, name=key.name, schema_version=2)
    assert loaded.get_state(migrated_key) == {"answer": 42, "migrated": True}


def test_v3_missing_migrator_is_a_load_time_checkpoint_error(tmp_path):
    key = StateKey(namespace="user", name="g5b_no_migrator", schema_version=1)
    ctx = _state()
    ctx.set_state(key, "payload")
    path = tmp_path / "missing-migrator.npz"
    ctx.save(path)
    raw = dict(np.load(path, allow_pickle=False).items())
    entries = json.loads(bytes(raw["_state_entries"]).decode())
    for item in entries:
        if item["key"]["name"] == key.name:
            item["target_schema_version"] = 2
    raw["_state_entries"] = np.frombuffer(json.dumps(entries).encode(), dtype=np.uint8)
    np.savez(path, **raw)
    with pytest.raises(CheckpointError) as error:
        OptimizationState.load(path, ctx.problem)
    message = str(error.value)
    assert "user/g5b_no_migrator" in message
    assert "v1" in message and "v2" in message
    assert "Registered migrators" in message


def test_v3_future_entry_schema_version_is_rejected(tmp_path):
    key = StateKey(namespace="user", name="g5b_future", schema_version=1)
    ctx = _state()
    ctx.set_state(key, "payload")
    path = tmp_path / "future-entry.npz"
    ctx.save(path)
    raw = dict(np.load(path, allow_pickle=False).items())
    entries = json.loads(bytes(raw["_state_entries"]).decode())
    for item in entries:
        if item["key"]["name"] == key.name:
            item["key"]["schema_version"] = 4
            item["target_schema_version"] = 3
    raw["_state_entries"] = np.frombuffer(json.dumps(entries).encode(), dtype=np.uint8)
    np.savez(path, **raw)
    with pytest.raises(CheckpointError, match=r"g5b_future.*v4.*v3"):
        OptimizationState.load(path, ctx.problem)


def test_checkpoint_separates_async_fatal_and_derived_reservations(tmp_path):
    ctx = _state().replace(
        async_fatal={"request_id": 4, "reason": "backend stopped"},
        data={
            "async_fatal": {"user": True},
            "pending_candidate_ids": [99],
            "reserved_fe": 1,
            "reserved_cost": 2.5,
        },
    )
    path = tmp_path / "async-fatal.npz"
    ctx.save(path)
    loaded = OptimizationState.load(path, ctx.problem)
    assert loaded.async_fatal == {"request_id": 4, "reason": "backend stopped"}
    assert loaded.data["async_fatal"] == {"user": True}
    assert loaded.data["pending_candidate_ids"] == [99]
    assert loaded.data["reserved_fe"] == 1
    assert loaded.data["reserved_cost"] == 2.5


def test_pending_context_stage_does_not_checkpoint_derived_reservations(tmp_path):
    ctx = _state().replace(data={"custom": "kept"})
    ctx = PendingEvaluationContextStage(None).execute(ctx)
    path = tmp_path / "derived-reservations.npz"
    ctx.save(path)
    loaded = OptimizationState.load(path, ctx.problem)
    assert loaded.data == {"custom": "kept", "resumed": True}
    assert (
        not {
            "pending_candidate_ids",
            "reserved_fe",
            "reserved_cost",
        }
        & loaded.data.keys()
    )


def test_v2_legacy_builtin_reservations_are_not_restored(tmp_path):
    ctx = _state()
    source = tmp_path / "source.npz"
    ctx._save_v2(source)
    raw = dict(np.load(source, allow_pickle=False).items())
    raw.pop("_async_fatal")
    raw["_data"] = np.frombuffer(
        json.dumps(
            {
                "pending_candidate_ids": [1],
                "reserved_fe": 1,
                "reserved_cost": 1.0,
                "custom": "kept",
            }
        ).encode(),
        dtype=np.uint8,
    )
    legacy = tmp_path / "legacy-v2-reservations.npz"
    np.savez(legacy, **raw)
    loaded = OptimizationState.load(legacy, ctx.problem)
    assert loaded.data["custom"] == "kept"
    assert (
        not {
            "pending_candidate_ids",
            "reserved_fe",
            "reserved_cost",
        }
        & loaded.data.keys()
    )


def test_v2_legacy_async_fatal_is_migrated_and_derived_keys_ignored(tmp_path):
    ctx = _state()
    source = tmp_path / "source.npz"
    ctx._save_v2(source)
    raw = dict(np.load(source, allow_pickle=False).items())
    raw.pop("_async_fatal")
    raw["_data"] = np.frombuffer(
        json.dumps(
            {
                "async_fatal": {"request_id": 7, "reason": "legacy"},
                "custom": "kept",
            }
        ).encode(),
        dtype=np.uint8,
    )
    legacy = tmp_path / "legacy-v2.npz"
    np.savez(legacy, **raw)
    loaded = OptimizationState.load(legacy, ctx.problem)
    assert loaded.async_fatal == {"request_id": 7, "reason": "legacy"}
    assert loaded.data["custom"] == "kept"
    assert "async_fatal" not in loaded.data


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
        "_data": np.frombuffer(
            json.dumps(
                {
                    "pending_candidate_ids": [7],
                    "reserved_fe": 1,
                    "reserved_cost": 0.5,
                }
            ).encode(),
            dtype=np.uint8,
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
    assert loaded.data == {"resumed": True}
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
    ctx._save_v2(path)
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
    ctx._save_v2(path)
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
