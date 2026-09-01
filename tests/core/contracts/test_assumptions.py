from typing import Any, cast

import pytest

from saealib.core.contracts.assumptions import (
    ASSUMPTION_KEYS,
    AssumptionSet,
    validate_assumption_name,
)
from saealib.core.contracts.vocabulary import validate_name
from saealib.exceptions import ValidationError


def test_assumptions_use_restrictive_unaware_defaults() -> None:
    assert ASSUMPTION_KEYS.names() == (
        "observation_schema.fixed",
        "evaluation.deterministic",
        "population.fixed_size",
        "state.checkpointable",
    )
    assumptions = AssumptionSet()

    assert assumptions["observation_schema.fixed"] is True
    assert assumptions["evaluation.deterministic"] is True
    assert assumptions["population.fixed_size"] is True
    assert assumptions["state.checkpointable"] is True
    assert "feedback.complete_batch" not in ASSUMPTION_KEYS


def test_assumption_names_use_a_dotted_rule_only_here() -> None:
    with pytest.raises(ValidationError):
        validate_name("observation_schema.fixed")

    assert validate_assumption_name("observation_schema.fixed") == (
        "observation_schema.fixed"
    )


def test_assumption_values_override_defaults_and_allow_extensions() -> None:
    assumptions = AssumptionSet(
        {
            "population.fixed_size": False,
            "extension.future_flag": True,
        }
    )

    assert assumptions["population.fixed_size"] is False
    assert assumptions["extension.future_flag"] is True
    assert ASSUMPTION_KEYS.get("extension.future_flag") is None


def test_assumption_set_rejects_bad_values_and_missing_keys() -> None:
    with pytest.raises(ValidationError):
        AssumptionSet(cast(Any, {"extension.future_flag": 1}))
    with pytest.raises(ValidationError):
        AssumptionSet({"invalid key": True})

    with pytest.raises(KeyError):
        AssumptionSet()["extension.future_flag"]
