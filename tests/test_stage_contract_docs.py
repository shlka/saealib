"""Focused checks for generated operational Stage contract documentation."""

from __future__ import annotations

from pathlib import Path

import saealib.stages as stages_module
from saealib._stage_docs import (
    BEGIN_MARKER,
    END_MARKER,
    render_stage_contracts,
    update_stage_docs,
)
from saealib.pipeline import Stage
from saealib.stages import discover_builtin_stages

DOCS_PATH = Path(__file__).parents[1] / "docs/components/stage.md"
EXTENSION_DOCS_PATH = DOCS_PATH.parent / "extension_guidelines.md"


def test_production_discovery_lists_all_operational_stages() -> None:
    stages = discover_builtin_stages()
    module_classes = tuple(
        cls
        for cls in vars(stages_module).values()
        if isinstance(cls, type)
        and cls is not Stage
        and issubclass(cls, Stage)
        and cls.__module__ == stages_module.__name__
        and "contract" in cls.__dict__
    )

    # Independent floor: dropping one current built-in from production
    # discovery must fail even if the module scan below is changed with it.
    assert len(stages) >= 20
    assert len(set(stages)) == len(stages)
    assert all(issubclass(stage, Stage) for stage in stages)
    assert stages == module_classes


def test_generated_table_contains_every_discovered_class_and_name() -> None:
    docs = DOCS_PATH.read_text(encoding="utf-8")
    generated = docs.split(BEGIN_MARKER, 1)[1].split(END_MARKER, 1)[0]
    expected = {(stage.__name__, stage.name) for stage in discover_builtin_stages()}
    actual = {
        (cells[0], cells[1])
        for line in generated.splitlines()
        if line.startswith("|") and not line.startswith("|---")
        for cells in [[cell.strip() for cell in line.strip("|").split(" | ")]]
        if cells[0] != "Class"
    }

    assert actual == expected
    assert len(actual) == len(expected)
    assert "11 built-in" not in docs
    assert "built-in 11" not in docs
    extension_docs = EXTENSION_DOCS_PATH.read_text(encoding="utf-8")
    assert "11 built-in" not in extension_docs
    assert "built-in 11" not in extension_docs


def test_docs_generation_is_reproducible_and_contract_driven() -> None:
    original = DOCS_PATH.read_text(encoding="utf-8")
    assert update_stage_docs(DOCS_PATH) is False
    assert DOCS_PATH.read_text(encoding="utf-8") == original

    table = render_stage_contracts()
    assert "| TellStage | tell |" in table
    assert "`proposals.offspring`" in table
    assert "| InitializationStage | initialization |" in table
