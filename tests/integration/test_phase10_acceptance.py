"""Cross-phase acceptance guards for the Phase 10 plugin boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, cast

import numpy as np

from saealib.core.compiler import CompileContext, Compiler
from saealib.core.compiler.diagnostics import Severity
from saealib.core.state import OPTIMIZATION_STATE_INITIAL_KEYS
from saealib.optimizer import Optimizer
from saealib.space import PermutationSpace, SequenceSpace
from saealib.strategies.direct import DirectStrategy
from saealib.strategies.ib import IndividualBasedStrategy

_PHASE9_PATH = Path(__file__).parents[1] / "test_phase9_representations.py"
_PHASE9_SPEC = importlib.util.spec_from_file_location(
    "phase9_representations", _PHASE9_PATH
)
assert _PHASE9_SPEC is not None and _PHASE9_SPEC.loader is not None
phase9 = importlib.util.module_from_spec(_PHASE9_SPEC)
_PHASE9_SPEC.loader.exec_module(phase9)
phase10_plugin = importlib.import_module("test_phase10_plugin")


# Keep these names explicit: this is the inventory guard for the six Phase 10
# boundary areas represented by the eight committed focused tests.
PHASE10_FOCUSED_TESTS = (
    "test_u10_0_external_plugin_proposes_evaluates_and_delivers_feedback",
    "test_u10_0_plugin_component_compiles_without_error_diagnostics",
    "test_u10_1_differential_evolution_preserves_targets_and_replaces_improvements",
    "test_u10_1_multifidelity_accepts_sparse_low_and_selects_high_promotion",
    "test_u10_2_cmaes_updates_only_on_complete_generation_feedback",
    "test_u10_2_moead_updates_only_the_child_subproblem_neighborhood",
    "test_u10_3_map_elites_uses_behavior_cells_and_quality_replacement",
    "test_u10_3_coevolution_joins_named_blocks_and_updates_one_block",
)


def _phase9_profiles() -> tuple[tuple[str, Any, Any, Any], ...]:
    """Reuse Phase 9's real problem, algorithm, and external-space helpers."""
    permutation = PermutationSpace(3)
    sequence = SequenceSpace((0, 1), 1, 3)
    phase9._register_custom_kind()
    custom = phase9._CustomSpace()
    genome_ga = phase9.GenomeGA(
        phase9.OrderCrossover(),
        phase9.SwapMutation(),
        phase9.SequentialSelection(),
        phase9.TruncationSelection(),
    )
    sequence_ga = phase9.GenomeGA(
        phase9._ExternalCrossover(),
        phase9.SequenceMutation(alphabet=(0, 1), min_length=1, max_length=3),
        phase9.SequentialSelection(),
        phase9.TruncationSelection(),
    )
    return (
        (
            "vector-direct",
            phase9._vector_problem(),
            phase9._vector_ga(),
            DirectStrategy(),
        ),
        (
            "vector-individual-based",
            phase9._vector_problem(),
            phase9._vector_ga(),
            IndividualBasedStrategy(0.5),
        ),
        (
            "permutation",
            phase9._genome_problem(permutation, lambda x: np.asarray([sum(x)])),
            genome_ga,
            DirectStrategy(),
        ),
        (
            "variable-length-sequence",
            phase9._genome_problem(sequence, lambda x: np.asarray([len(x)])),
            sequence_ga,
            DirectStrategy(),
        ),
        (
            "external-custom-object-graph",
            phase9._genome_problem(custom, lambda x: np.asarray([float(x)])),
            phase9._custom_ga(),
            DirectStrategy(),
        ),
    )


def test_u10_4_cross_phase_focused_inventory_guard() -> None:
    assert len(PHASE10_FOCUSED_TESTS) == 8
    assert all(
        callable(getattr(phase10_plugin, name, None)) for name in PHASE10_FOCUSED_TESTS
    )
    assert len(_phase9_profiles()) == 5


def test_u10_4_phase9_graph_profiles_compile_through_current_public_path() -> None:
    profile_names = tuple(profile[0] for profile in _phase9_profiles())
    assert profile_names == (
        "vector-direct",
        "vector-individual-based",
        "permutation",
        "variable-length-sequence",
        "external-custom-object-graph",
    )

    for name, problem, algorithm, strategy in _phase9_profiles():
        optimizer = (
            Optimizer(problem, seed=1).set_algorithm(algorithm).set_strategy(strategy)
        )
        optimizer._resolve_defaults()
        graph = cast(Any, optimizer.strategy).build_graph(optimizer)
        plan = Compiler().compile(
            graph,
            CompileContext(
                space=optimizer.problem.space,
                problem=optimizer.problem,
                initial_state_keys=OPTIMIZATION_STATE_INITIAL_KEYS,
            ),
        )
        assert [
            diagnostic
            for diagnostic in plan.diagnostics
            if diagnostic.severity is Severity.ERROR
        ] == [], name


def test_u10_4_plugin_uses_core_compiler_without_representation_branch() -> None:
    compiler_root = Path(__file__).parents[2] / "src" / "saealib" / "core" / "compiler"
    compiler_source = "\n".join(
        path.read_text(encoding="utf-8") for path in compiler_root.glob("*.py")
    )
    assert "phase10_plugin" not in compiler_source
    assert "Phase10Plugin" not in compiler_source
