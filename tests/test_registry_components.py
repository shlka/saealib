"""Integration tests: built-in components self-register and Registry.build
reproduces objects equivalent to the hardcoded resolvers in saealib.api."""

import numpy as np
import pytest

from saealib.algorithms.ga import GA
from saealib.algorithms.pso import PSO
from saealib.operators.crossover import CrossoverBLXAlpha
from saealib.operators.mutation import MutationUniform
from saealib.operators.selection import SequentialSelection, TruncationSelection
from saealib.registry import build, get
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.strategies.ps import PreSelectionStrategy
from saealib.surrogate.manager import LocalSurrogateManager
from saealib.surrogate.rbf import RBFSurrogate, gaussian_kernel
from saealib.surrogate.training_set import KNNObjectiveSet


class TestBuiltinComponentsRegistered:
    @pytest.mark.parametrize(
        ("name", "cls"),
        [
            ("GA", GA),
            ("PSO", PSO),
            ("RBFSurrogate", RBFSurrogate),
            ("LocalSurrogateManager", LocalSurrogateManager),
            ("KNNObjectiveSet", KNNObjectiveSet),
            ("IndividualBasedStrategy", IndividualBasedStrategy),
            ("GenerationBasedStrategy", GenerationBasedStrategy),
            ("PreSelectionStrategy", PreSelectionStrategy),
            ("CrossoverBLXAlpha", CrossoverBLXAlpha),
            ("MutationUniform", MutationUniform),
            ("SequentialSelection", SequentialSelection),
            ("TruncationSelection", TruncationSelection),
        ],
    )
    def test_registered_under_class_name(self, name, cls):
        assert get(name) is cls


class TestBuildMatchesApiResolvers:
    """Each spec below mirrors a hardcoded resolver in saealib/api.py."""

    def test_ga_default_spec_matches_resolve_algorithm(self):
        spec = {
            "type": "GA",
            "params": {
                "crossover": {
                    "type": "CrossoverBLXAlpha",
                    "params": {"prob": 0.7, "alpha": 0.4},
                },
                "mutation": {
                    "type": "MutationUniform",
                    "params": {"prob_var": 0.3},
                },
                "parent_selection": {"type": "SequentialSelection"},
                "survivor_selection": {"type": "TruncationSelection"},
            },
        }
        ga = build(spec)

        assert isinstance(ga, GA)
        assert isinstance(ga.crossover, CrossoverBLXAlpha)
        assert ga.crossover.prob == 0.7
        assert ga.crossover.alpha == 0.4
        assert isinstance(ga.mutation, MutationUniform)
        assert ga.mutation.prob_var == 0.3
        assert isinstance(ga.parent_selection, SequentialSelection)
        assert isinstance(ga.survivor_selection, TruncationSelection)

    def test_rbf_surrogate_manager_spec_matches_resolve_surrogate(self):
        direction = np.array([-1.0])
        spec = {
            "type": "LocalSurrogateManager",
            "params": {
                "surrogate": {
                    "type": "RBFSurrogate",
                    "params": {"kernel": gaussian_kernel, "dim": 5},
                },
                "acquisition": {
                    "type": "MeanPrediction",
                    "params": {"direction": direction},
                },
                "training_set": {
                    "type": "KNNObjectiveSet",
                    "params": {"n_neighbors": 50},
                },
            },
        }
        manager = build(spec)

        assert isinstance(manager, LocalSurrogateManager)
        assert isinstance(manager.surrogate, RBFSurrogate)
        assert manager.acquisition.direction is direction
        assert isinstance(manager.training_set, KNNObjectiveSet)
        assert manager.training_set.n_neighbors == 50

    @pytest.mark.parametrize(
        ("spec", "cls", "attrs"),
        [
            (
                {
                    "type": "IndividualBasedStrategy",
                    "params": {"evaluation_ratio": 0.1},
                },
                IndividualBasedStrategy,
                {"evaluation_ratio": 0.1},
            ),
            (
                {"type": "GenerationBasedStrategy", "params": {"gen_ctrl": 5}},
                GenerationBasedStrategy,
                {"gen_ctrl": 5},
            ),
            (
                {
                    "type": "PreSelectionStrategy",
                    "params": {"n_candidates": 40, "n_select": 4},
                },
                PreSelectionStrategy,
                {"n_candidates": 40, "n_select": 4},
            ),
        ],
    )
    def test_strategy_spec_matches_resolve_strategy(self, spec, cls, attrs):
        strategy = build(spec)
        assert isinstance(strategy, cls)
        for attr, expected in attrs.items():
            assert getattr(strategy, attr) == expected
