"""Tests for the bundled defaults/presets file (saealib/defaults/presets.yaml).

Also guards against staleness: every static component spec in the file must
still be buildable via the Registry, so a renamed/removed constructor
parameter is caught here instead of silently breaking Optimizer defaults.
"""

from saealib.defaults import load_defaults
from saealib.registry import build


class TestLoadDefaults:
    def test_returns_dict_with_schema_version(self):
        defaults = load_defaults()
        assert defaults["schema_version"] == 1

    def test_has_presets_by_algorithm_by_problem_shape_and_fallback(self):
        defaults = load_defaults()
        assert "presets" in defaults
        assert "by_algorithm" in defaults
        assert "by_problem_shape" in defaults
        assert "fallback" in defaults

    def test_fallback_references_an_existing_preset(self):
        defaults = load_defaults()
        assert defaults["fallback"] in defaults["presets"]

    def test_by_algorithm_entries_reference_existing_presets(self):
        defaults = load_defaults()
        for preset_name in defaults["by_algorithm"].values():
            assert preset_name in defaults["presets"]

    def test_by_problem_shape_entries_reference_existing_presets(self):
        defaults = load_defaults()
        for rule in defaults["by_problem_shape"]:
            assert rule["preset"] in defaults["presets"]

    def test_cached_across_calls(self):
        assert load_defaults() is load_defaults()


class TestPresetsAreBuildable:
    """Staleness guard: static specs must still construct via Registry.build."""

    def test_ga_rbf_ib_algorithm_spec_builds(self):
        from saealib.algorithms.ga import GA

        preset = load_defaults()["presets"]["ga_rbf_ib"]
        alg = build(preset["algorithm"])
        assert isinstance(alg, GA)

    def test_ga_rbf_ib_strategy_spec_builds(self):
        from saealib.strategies.ib import IndividualBasedStrategy

        preset = load_defaults()["presets"]["ga_rbf_ib"]
        strategy = build(preset["strategy"])
        assert isinstance(strategy, IndividualBasedStrategy)
        assert strategy.evaluation_ratio == 0.1

    def test_ga_rbf_ib_surrogate_manager_training_set_spec_builds(self):
        from saealib.surrogate.training_set import KNNObjectiveSet

        preset = load_defaults()["presets"]["ga_rbf_ib"]
        training_set_spec = preset["surrogate_manager"]["params"]["training_set"]
        training_set = build(training_set_spec)
        assert isinstance(training_set, KNNObjectiveSet)
        assert training_set.n_neighbors == 50

    def test_ga_rbf_ib_surrogate_manager_requires_injected_surrogate(self):
        """surrogate/acquisition are problem-dependent and intentionally
        omitted from the file; building the bare spec must fail clearly
        rather than silently constructing an incomplete manager."""
        import pytest

        from saealib.exceptions import ValidationError

        preset = load_defaults()["presets"]["ga_rbf_ib"]
        with pytest.raises(ValidationError):
            build(preset["surrogate_manager"])
