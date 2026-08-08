"""Per-entry schema migration for checkpoint state values."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

from saealib.core.contracts.vocabulary import validate_name
from saealib.core.state.keys import StateKey
from saealib.exceptions import CheckpointError, ConfigurationError, ValidationError

__all__ = [
    "STATE_MIGRATORS",
    "Migrator",
    "StateMigrationRegistry",
]


ValueT = TypeVar("ValueT")
MigratedValueT = TypeVar("MigratedValueT")

Migrator = Callable[[object], object]
_MigrationId = tuple[str, str, int]


def _population_entry_v1_to_v2(value: object) -> object:
    """Keep the restored population value while advancing its entry version.

    At the value level, population entries are identical in v1 and v2.  The
    decoder absorbs the encoding difference between the legacy ``x`` column
    and the v2 ``GenomeCodec`` payload before this migrator is called.
    """
    return value


def _validate_version(version: int, *, field_name: str) -> int:
    """Validate and return a positive schema version."""
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValidationError(f"{field_name} must be a positive integer")
    return version


class StateMigrationRegistry:
    """Register and apply one-step, per-entry state migrations.

    A registered migrator converts one entry from ``from_version`` to
    ``from_version + 1``.  The registry deliberately has no operation that
    converts a whole checkpoint: entries can have independent migration paths.
    """

    def __init__(self) -> None:
        """Create an empty state migration registry."""
        self._migrators: dict[_MigrationId, Migrator] = {}

    def register(
        self,
        namespace: str,
        name: str,
        from_version: int,
        migrator: Migrator,
    ) -> None:
        """Register a one-step migrator for a state entry.

        The registered callable is the only migration from ``from_version``
        to ``from_version + 1`` for this ``(namespace, name)``.  Registering a
        second callable for the same transition is a configuration error.
        """
        migration_id = self._migration_id(namespace, name, from_version)
        if not callable(migrator):
            raise ValidationError("A state migrator must be callable")
        if migration_id in self._migrators:
            raise ConfigurationError(
                "State migrator is already registered for "
                f"{self._format_migration_id(migration_id)}"
            )
        self._migrators[migration_id] = migrator

    def registered(self) -> tuple[tuple[str, str, int], ...]:
        """Return registered migration identifiers in registration order."""
        return tuple(self._migrators)

    def migrate(
        self,
        key: StateKey[ValueT],
        value: ValueT,
        *,
        target_version: int,
    ) -> tuple[StateKey[MigratedValueT], MigratedValueT]:
        """Migrate one state entry to ``target_version``.

        Migrations are composed in ascending version order.  Loading an entry
        without a complete path is a checkpoint error; it is not a compiler
        diagnostic.  A target older than the entry is rejected because this
        registry only supports forward, one-step migrations.
        """
        if not isinstance(key, StateKey):
            raise ValidationError("migrate() expects a StateKey")
        target_version = _validate_version(target_version, field_name="target_version")
        if target_version < key.schema_version:
            raise CheckpointError(
                "Cannot migrate state key "
                f"{self._format_key(key)} from version v{key.schema_version} "
                f"to target version v{target_version}: backward migration is "
                "unsupported. "
                f"Registered migrators: {self._format_registered()}"
            )

        current_value: object = value
        for from_version in range(key.schema_version, target_version):
            migration_id = (key.namespace, key.name, from_version)
            migrator = self._migrators.get(migration_id)
            if migrator is None:
                raise CheckpointError(
                    "Missing state migrator for key "
                    f"{self._format_key(key)} at version transition "
                    f"v{from_version} -> v{from_version + 1} while migrating "
                    f"from v{key.schema_version} to v{target_version}. "
                    f"Registered migrators: {self._format_registered()}"
                )
            current_value = migrator(current_value)

        migrated_key = StateKey[MigratedValueT](
            namespace=key.namespace,
            name=key.name,
            schema_version=target_version,
        )
        return migrated_key, cast(MigratedValueT, current_value)

    def migrate_value(
        self,
        key: StateKey[ValueT],
        value: ValueT,
        *,
        target_version: int,
    ) -> object:
        """Migrate one value when the caller does not need the new key."""
        _, migrated_value = self.migrate(
            key,
            value,
            target_version=target_version,
        )
        return migrated_value

    @staticmethod
    def _migration_id(
        namespace: str,
        name: str,
        from_version: int,
    ) -> _MigrationId:
        """Validate a migration identifier and return its canonical tuple."""
        validate_name(namespace)
        validate_name(name)
        _validate_version(from_version, field_name="from_version")
        return namespace, name, from_version

    @staticmethod
    def _format_key(key: StateKey[ValueT]) -> str:
        """Format a state key without exposing its repr implementation."""
        return f"{key.namespace}/{key.name}"

    @staticmethod
    def _format_migration_id(migration_id: _MigrationId) -> str:
        """Format a registered one-step migration identifier."""
        namespace, name, from_version = migration_id
        return f"{namespace}/{name}@v{from_version}->v{from_version + 1}"

    def _format_registered(self) -> str:
        """Format all registered migrators for an actionable error."""
        if not self._migrators:
            return "(none)"
        return ", ".join(
            self._format_migration_id(migration_id) for migration_id in self._migrators
        )


STATE_MIGRATORS = StateMigrationRegistry()
