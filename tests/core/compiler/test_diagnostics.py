from typing import Any

import pytest

from saealib.core.compiler.diagnostics import (
    ContractPath,
    Diagnostic,
    DiagnosticBag,
    Severity,
)
from saealib.exceptions import ValidationError


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            ContractPath(components=("algorithm", "integer_crossover")),
            "algorithm.integer_crossover",
        ),
        (
            ContractPath(components=("algorithm",), role="proposer"),
            "algorithm[proposer]",
        ),
        (
            ContractPath(components=("algorithm", "crossover"), port="offspring"),
            "algorithm.crossover.offspring",
        ),
        (
            ContractPath(components=("algorithm",), role="proposer", port="genomes"),
            "algorithm[proposer].genomes",
        ),
    ],
)
def test_contract_path_renders_components_role_and_port(
    path: ContractPath, expected: str
) -> None:
    assert str(path) == expected


def test_contract_path_rejects_an_empty_role() -> None:
    with pytest.raises(ValidationError, match="role must be a non-empty string"):
        ContractPath(components=("algorithm",), role="")


def test_diagnostic_requires_a_contract_path() -> None:
    diagnostic_constructor: Any = Diagnostic
    with pytest.raises(TypeError):
        diagnostic_constructor(
            severity=Severity.ERROR,
            code="unknown_data_spec",
            message="Unknown kind",
        )


def test_diagnostic_preserves_an_unregistered_code() -> None:
    diagnostic = Diagnostic(
        severity=Severity.ERROR,
        code="typo_in_rule_code",
        message="Unknown code",
        path=ContractPath(components=("algorithm",)),
    )

    assert diagnostic.code == "typo_in_rule_code"


def test_diagnostic_bag_reports_error_severity() -> None:
    bag = DiagnosticBag()
    path = ContractPath(components=("algorithm",), port="input")
    bag.append(
        Diagnostic(
            severity=Severity.WARNING,
            code="unknown_data_spec",
            message="Warning",
            path=path,
        )
    )

    assert not bag.has_errors
    bag.append(
        Diagnostic(
            severity=Severity.ERROR,
            code="schema_variable_unbound",
            message="Error",
            path=path,
        )
    )

    assert bag.has_errors
    assert len(list(bag)) == 2
