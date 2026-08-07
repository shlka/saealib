from typing import Any

import pytest

from saealib.core.compiler.diagnostics import (
    ContractPath,
    Diagnostic,
    DiagnosticBag,
    Severity,
)


def test_contract_path_renders_components_and_port() -> None:
    path = ContractPath(components=("algorithm", "surrogate"), port="predictions")

    assert str(path) == "algorithm.surrogate.predictions"


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
