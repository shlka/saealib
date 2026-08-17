"""Tests for per-entry checkpoint state migration."""

from typing import cast

import pytest

import saealib.core.state.migration as migration_module
from saealib.core.state import STATE_MIGRATORS as PACKAGE_STATE_MIGRATORS
from saealib.core.state.keys import StateKey
from saealib.core.state.migration import (
    STATE_MIGRATORS,
    StateMigrationRegistry,
)
from saealib.exceptions import CheckpointError, ConfigurationError


def key(name: str = "value", version: int = 1) -> StateKey[object]:
    """Build a user state key for migration tests."""
    return StateKey(namespace="user", name=name, schema_version=version)


def test_migrators_compose_in_version_order() -> None:
    registry = StateMigrationRegistry()
    registry.register("user", "value", 1, lambda value: cast(int, value) + 1)
    registry.register("user", "value", 2, lambda value: cast(int, value) * 10)

    migrated_key, migrated_value = registry.migrate(
        key(),
        3,
        target_version=3,
    )

    assert migrated_key == key(version=3)
    assert migrated_value == 40


def test_missing_migrator_names_key_versions_and_registered_migrators() -> None:
    registry = StateMigrationRegistry()
    registry.register("user", "other", 1, lambda value: value)

    with pytest.raises(CheckpointError) as error:
        registry.migrate(key(), "payload", target_version=2)

    message = str(error.value)
    assert "user/value" in message
    assert "v1" in message
    assert "v2" in message
    assert "user/other@v1->v2" in message


def test_missing_intermediate_migrator_identifies_the_cut() -> None:
    registry = StateMigrationRegistry()
    registry.register("user", "value", 1, lambda value: value)

    with pytest.raises(CheckpointError, match=r"v2 -> v3") as error:
        registry.migrate(key(), "payload", target_version=3)

    message = str(error.value)
    assert "user/value" in message
    assert "from v1 to v3" in message
    assert "user/value@v1->v2" in message


def test_newer_entry_cannot_be_read_by_an_older_target() -> None:
    registry = StateMigrationRegistry()

    with pytest.raises(CheckpointError) as error:
        registry.migrate(key(version=4), "payload", target_version=3)

    message = str(error.value)
    assert "user/value" in message
    assert "v4" in message
    assert "v3" in message
    assert "backward migration is unsupported" in message


def test_duplicate_migrator_registration_is_rejected() -> None:
    registry = StateMigrationRegistry()
    registry.register("user", "value", 1, lambda value: value)

    with pytest.raises(ConfigurationError, match="already registered"):
        registry.register("user", "value", 1, lambda value: value)


def test_same_version_needs_no_migrator() -> None:
    registry = StateMigrationRegistry()
    state_key = key(version=2)
    value = {"ready": True}

    migrated_key, migrated_value = registry.migrate(
        state_key,
        value,
        target_version=2,
    )

    assert migrated_key == state_key
    assert migrated_value is value


def test_registered_returns_identifiers_in_registration_order() -> None:
    registry = StateMigrationRegistry()
    registry.register("user", "second", 2, lambda value: value)
    registry.register("user", "first", 1, lambda value: value)

    assert registry.registered() == (
        ("user", "second", 2),
        ("user", "first", 1),
    )


def test_default_registry_is_shared_by_module_and_package_api() -> None:
    assert STATE_MIGRATORS is PACKAGE_STATE_MIGRATORS
    assert isinstance(STATE_MIGRATORS, StateMigrationRegistry)


def test_migration_public_names_have_no_compatibility_aliases() -> None:
    assert set(migration_module.__all__) == {
        "STATE_MIGRATORS",
        "Migrator",
        "StateMigrationRegistry",
    }
    assert not hasattr(migration_module, "MigrationRegistry")
    assert not hasattr(migration_module, "StateMigrator")
