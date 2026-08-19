"""Unit tests for saealib.api: minimize/maximize preset support and _UnsetType."""

import numpy as np

from saealib import maximize, minimize
from saealib.api import _UNSET, _UnsetType
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ps import PreSelectionStrategy


class TestUnsetType:
    def test_repr_is_unset(self):
        assert repr(_UNSET) == "UNSET"

    def test_instance_of_dedicated_type(self):
        assert isinstance(_UNSET, _UnsetType)


class TestMinimizePreset:
    def test_completes_with_preset_dict(self):
        preset = {
            "strategy": {
                "type": "GenerationBasedStrategy",
                "params": {"gen_ctrl": 2},
            },
        }
        result = minimize(
            lambda x: np.sum(x**2),
            dim=2,
            lb=[-1, -1],
            ub=[1, 1],
            algorithm="GA",
            surrogate="rbf",
            preset=preset,
            max_fe=30,
            pop_size=6,
            seed=0,
            verbose=False,
        )
        assert isinstance(result.x, np.ndarray)
        assert result.fe > 0

    def test_preset_strategy_is_actually_used(self, monkeypatch):
        calls = {"gb": 0, "ps": 0}
        orig_gb_graph = GenerationBasedStrategy.build_graph
        orig_ps_graph = PreSelectionStrategy.build_graph

        def gb_graph(self, provider):
            calls["gb"] += 1
            return orig_gb_graph(self, provider)

        def ps_graph(self, provider):
            calls["ps"] += 1
            return orig_ps_graph(self, provider)

        # Observe the selected strategy at the graph/plan boundary, rather
        # than strategy.step(), which sync runtime execution intentionally
        # does not call.
        monkeypatch.setattr(GenerationBasedStrategy, "build_graph", gb_graph)
        monkeypatch.setattr(PreSelectionStrategy, "build_graph", ps_graph)

        preset = {
            "strategy": {
                "type": "GenerationBasedStrategy",
                "params": {"gen_ctrl": 2},
            },
        }
        minimize(
            lambda x: np.sum(x**2),
            dim=2,
            lb=[-1, -1],
            ub=[1, 1],
            algorithm="GA",
            surrogate="rbf",
            preset=preset,
            max_fe=30,
            pop_size=6,
            seed=0,
            verbose=False,
        )
        assert calls["gb"] > 0
        assert calls["ps"] == 0

    def test_explicit_strategy_overrides_preset(self, monkeypatch):
        calls = {"gb": 0, "ps": 0}
        orig_gb_graph = GenerationBasedStrategy.build_graph
        orig_ps_graph = PreSelectionStrategy.build_graph

        def gb_graph(self, provider):
            calls["gb"] += 1
            return orig_gb_graph(self, provider)

        def ps_graph(self, provider):
            calls["ps"] += 1
            return orig_ps_graph(self, provider)

        monkeypatch.setattr(GenerationBasedStrategy, "build_graph", gb_graph)
        monkeypatch.setattr(PreSelectionStrategy, "build_graph", ps_graph)

        # preset requests PreSelectionStrategy, but the explicit strategy="gb"
        # argument must take precedence.
        preset = {
            "strategy": {
                "type": "PreSelectionStrategy",
                "params": {"n_candidates": 6, "n_select": 2},
            },
        }
        minimize(
            lambda x: np.sum(x**2),
            dim=2,
            lb=[-1, -1],
            ub=[1, 1],
            algorithm="GA",
            surrogate="rbf",
            strategy="gb",
            preset=preset,
            max_fe=30,
            pop_size=6,
            seed=0,
            verbose=False,
        )
        assert calls["gb"] > 0
        assert calls["ps"] == 0


class TestMaximizePreset:
    def test_completes_with_preset_dict(self):
        preset = {
            "strategy": {
                "type": "GenerationBasedStrategy",
                "params": {"gen_ctrl": 2},
            },
        }
        result = maximize(
            lambda x: -np.sum(x**2),
            dim=2,
            lb=[-1, -1],
            ub=[1, 1],
            algorithm="GA",
            surrogate="rbf",
            preset=preset,
            max_fe=30,
            pop_size=6,
            seed=0,
            verbose=False,
        )
        assert isinstance(result.x, np.ndarray)
        assert result.fe > 0


class TestPopSizeExceedsDefaultArchive:
    def test_minimize_accepts_pop_size_larger_than_default_archive(self):
        dim = 2
        result = minimize(
            lambda x: np.sum(x**2),
            dim=dim,
            lb=[-1] * dim,
            ub=[1] * dim,
            pop_size=5 * dim + 1,
            max_fe=30,
            seed=0,
            verbose=False,
        )
        assert isinstance(result.x, np.ndarray)
        assert result.fe > 0
