import pytest

from saealib.core.state.keys import STATE_NAMESPACES, StateKey
from saealib.exceptions import ValidationError


def test_core_state_namespaces_are_registered() -> None:
    assert STATE_NAMESPACES.names() == (
        "populations",
        "archives",
        "proposals",
        "feedback",
        "evaluations",
        "surrogates",
        "algorithms",
        "runtime",
        "user",
    )


def test_state_key_equality_includes_schema_version() -> None:
    first = StateKey[int](
        namespace="populations",
        name="main",
        schema_version=1,
    )
    same = StateKey[int](
        namespace="populations",
        name="main",
        schema_version=1,
    )
    newer = StateKey[int](
        namespace="populations",
        name="main",
        schema_version=2,
    )

    assert first == same
    assert first != newer


def test_state_key_rejects_invalid_schema_versions() -> None:
    with pytest.raises(ValidationError):
        StateKey[object](
            namespace="runtime",
            name="status",
            schema_version=0,
        )
