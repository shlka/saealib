"""Drift-guard tests for saealib's top-level export tiers.

See the "Export tiers" comment at the top of src/saealib/__init__.py for the
Tier 1 / Tier 2 / namespace-only policy these tests enforce.
"""

import importlib
import importlib.util
from pathlib import Path
from typing import Any, cast

import pytest

import saealib

# Subpackages whose __all__ must be fully covered by (Tier 1 + Tier 2 + the
# allowlist below). saealib.registry is deliberately excluded: it defines no
# __all__, and its get/build/to_spec namespace-only status is covered by
# tests/test_registry.py::TestTopLevelExport instead.
SCANNED_SUBPACKAGES = [
    "saealib.acquisition",
    "saealib.algorithms",
    "saealib.benchmarks",
    "saealib.callback",
    "saealib.comparators",
    "saealib.defaults",
    "saealib.exceptions",
    "saealib.execution",
    "saealib.operators",
    "saealib.population",
    "saealib.problem",
    "saealib.space",
    "saealib.strategies",
    "saealib.surrogate",
    "saealib.utils",
    "saealib.variables",
]

NAMESPACE_ONLY: dict[str, set[str]] = {
    "saealib.algorithms": {
        "FeedbackConsumer",
        "LegacyPopulationAlgorithmAdapter",
        "ProposalRequest",
        "Proposer",
    },
    "saealib.benchmarks": {
        "ackley",
        "dtlz1",
        "dtlz2",
        "dtlz3",
        "dtlz4",
        "dtlz5",
        "dtlz6",
        "dtlz7",
        "rastrigin",
        "rosenbrock",
        "sphere",
        "zdt1",
        "zdt2",
        "zdt3",
        "zdt4",
        "zdt5",
        "zdt6",
    },
    "saealib.defaults": {"dump_preset", "load_defaults", "load_preset"},
    "saealib.population": {
        "DenseVectorBatch",
        "GenomeBatch",
        "ObjectBatch",
        "PermutationBatch",
        "VariableLengthBatch",
    },
    "saealib.space": {
        "BoundsService",
        "CloneService",
        "ComparisonService",
        "DenseNumericView",
        "DerivedSamplingService",
        "DerivedValidationService",
        "DistanceService",
        "EquivalenceService",
        "FeatureEncoder",
        "FingerprintService",
        "GenomeCodec",
        "ObjectSpace",
        "PermutationSpace",
        "SamplingService",
        "SearchSpace",
        "ServiceRegistry",
        "ValidationResult",
        "ValidationService",
        "VectorSpace",
        "SequenceSpace",
    },
}

REMOVED_EXPERIMENTAL_EXPORTS = {
    "ArchiveSnapshot",
    "CooperativeCoevolution",
    "CorrelatedQuadraticSurrogate",
    "DynamicArchiveSelector",
    "EvaluationWorkflowResult",
    "FidelityEvaluator",
    "FidelityPromotionRunner",
    "FidelityWorkflowResult",
    "MigrationPolicy",
    "RepeatedEvaluationRunner",
    "SeededNoiseEvaluator",
    "reference_problem",
}

REMOVED_CANONICAL_ALIASES = {
    "EvaluationPolicy",
    "FeedbackPolicy",
    "DensityAcquisition",
    "NichingAcquisition",
    "RouletteWheelSelection",
    "MigrationPolicy",
    "AsyncScheduler",
}


def test_tier1_and_tier2_do_not_overlap():
    assert set(saealib.__all__).isdisjoint(saealib._TIER2_MAP)


def test_tier1_and_tier2_names_resolve():
    for name in list(saealib.__all__) + list(saealib._TIER2_MAP):
        assert hasattr(saealib, name), f"{name} listed but not resolvable"


def test_experimental_generality_code_is_not_a_library_export():
    assert REMOVED_EXPERIMENTAL_EXPORTS.isdisjoint(vars(saealib))
    assert importlib.util.find_spec("saealib.generality") is None
    assert not (Path(__file__).parents[1] / "src/saealib/generality.py").exists()


def test_removed_aliases_are_not_importable():
    for name in REMOVED_CANONICAL_ALIASES:
        assert not hasattr(saealib, name)
    for module_name in (
        "saealib.acquisition",
        "saealib.execution",
        "saealib.operators",
        "saealib.policies",
    ):
        module = importlib.import_module(module_name)
        assert all(not hasattr(module, name) for name in REMOVED_CANONICAL_ALIASES)


@pytest.mark.parametrize(
    ("module_name", "name"),
    [("saealib", name) for name in sorted(REMOVED_CANONICAL_ALIASES)],
)
def test_removed_root_names_fail_import(module_name, name):
    with pytest.raises(ImportError):
        exec(f"from {module_name} import {name}", {})


def test_removed_parameter_names_fail_construction():
    from saealib.execution import AsyncEvaluationScheduler, SerialEvaluator
    from saealib.stages import FeedbackStage

    with pytest.raises(TypeError):
        cast(Any, AsyncEvaluationScheduler)(SerialEvaluator(), feedback_policy=None)
    with pytest.raises(TypeError):
        cast(Any, FeedbackStage)(policy=None)


def test_subpackage_exports_are_covered():
    covered = set(saealib.__all__) | set(saealib._TIER2_MAP)
    for modname in SCANNED_SUBPACKAGES:
        mod = importlib.import_module(modname)
        allowed_extra = NAMESPACE_ONLY.get(modname, set())
        for name in getattr(mod, "__all__", []):
            assert name in covered or name in allowed_extra, (
                f"{modname}.{name} is public but neither listed at the "
                "saealib top level nor in the NAMESPACE_ONLY allowlist"
            )
