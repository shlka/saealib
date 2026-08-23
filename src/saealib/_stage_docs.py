"""Generate the built-in Stage contract section of the documentation."""

from __future__ import annotations

from pathlib import Path

from saealib.core.state import StateKey
from saealib.stages import _builtin_stage_instances_for_contracts

BEGIN_MARKER = "<!-- BEGIN GENERATED STAGE CONTRACTS: do not edit -->"
END_MARKER = "<!-- END GENERATED STAGE CONTRACTS -->"


def _keys(keys: tuple[StateKey[object], ...]) -> str:
    return ", ".join(f"`{key.namespace}.{key.name}`" for key in keys) or "—"


def render_stage_contracts() -> str:
    """Render a deterministic Markdown table from real Stage contracts."""
    rows = [
        "| Class | Name | Label | Notation | Reads | Writes | Exports |",
        "|---|---|---|---|---|---|---|",
    ]
    for stage in _builtin_stage_instances_for_contracts():
        contract = stage.contract()
        state = contract.state
        cells = (
            type(stage).__name__,
            stage.name,
            stage.label,
            stage.notation,
            _keys(state.reads),
            _keys(state.writes),
            _keys(state.exports),
        )
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def render_stage_docs(text: str) -> str:
    """Replace the fixed marker section in the Stage documentation."""
    start = text.index(BEGIN_MARKER) + len(BEGIN_MARKER)
    end = text.index(END_MARKER, start)
    generated = f"\n\n{render_stage_contracts()}\n"
    return text[:start] + generated + text[end:]


def update_stage_docs(path: Path) -> bool:
    """Update *path* and return whether its contents changed."""
    original = path.read_text(encoding="utf-8")
    updated = render_stage_docs(original)
    if updated != original:
        path.write_text(updated, encoding="utf-8")
        return True
    return False
