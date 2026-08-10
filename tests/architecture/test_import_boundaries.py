"""AST-enforced dependency boundaries for the low-level core package."""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_TOP_LEVEL_PACKAGES = {
    "algorithms",
    "operators",
    "surrogate",
    "acquisition",
    "strategies",
    "problem",
    "population",
    "pipeline",
}

# Keep exceptions exact and intentional.  These are implementation leaves that
# still need population runtime types; all shared descriptors use saealib.identity.
IMPORT_ALLOWLIST: dict[tuple[str, str], str] = {
    (
        "src/saealib/core/state/patch.py",
        "saealib.population.genome",
    ): "StatePatch stores the Population genome batch representation.",
    (
        "src/saealib/core/state/store.py",
        "saealib.population",
    ): "StateStore owns the runtime Population state value.",
    (
        "src/saealib/core/graph_builder.py",
        "saealib.pipeline",
    ): "The graph builder retains the legacy Stage/Pipeline compatibility bridge.",
}


class _Imports(ast.NodeVisitor):
    """Collect every import statement, including local compatibility imports."""

    def __init__(self) -> None:
        self.imports: list[tuple[ast.Import | ast.ImportFrom, str]] = []

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.extend((node, alias.name) for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is not None:
            self.imports.append((node, node.module))


def _core_files() -> list[Path]:
    root = Path(__file__).parents[2]
    return sorted((root / "src/saealib/core").glob("**/*.py"))


def test_core_has_no_forbidden_module_level_imports() -> None:
    violations: list[str] = []
    seen_allowlist: set[tuple[str, str]] = set()
    root = Path(__file__).parents[2]

    for path in _core_files():
        relative = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        collector = _Imports()
        collector.visit(tree)
        for node, module in collector.imports:
            parts = module.split(".")
            if len(parts) < 2 or parts[0] != "saealib":
                continue
            package = parts[1]
            if package not in FORBIDDEN_TOP_LEVEL_PACKAGES:
                continue
            key = (relative, module)
            if key in IMPORT_ALLOWLIST:
                seen_allowlist.add(key)
            else:
                violations.append(f"{relative}:{node.lineno}: from {module}")

    assert not violations, "Forbidden core imports:\n" + "\n".join(violations)
    assert seen_allowlist == set(IMPORT_ALLOWLIST), (
        "Allowlist contains stale or unobserved entries: "
        + repr(set(IMPORT_ALLOWLIST) - seen_allowlist)
    )


def test_candidate_ids_use_one_identity_across_compatibility_exports() -> None:
    from saealib.core.contracts import CandidateIds as CoreCandidateIds
    from saealib.identity import CandidateIds as IdentityCandidateIds
    from saealib.population import CandidateIds as PopulationPackageCandidateIds
    from saealib.population.population import CandidateIds as PopulationCandidateIds

    assert CoreCandidateIds is IdentityCandidateIds
    assert PopulationCandidateIds is IdentityCandidateIds
    assert PopulationPackageCandidateIds is IdentityCandidateIds
