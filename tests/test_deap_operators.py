"""Tests for the optional DEAP operator adapters."""

from __future__ import annotations

import builtins
import random
from importlib import import_module
from typing import Any

import numpy as np
import pytest

deap = pytest.importorskip("deap")
base = import_module("deap.base")
tools = import_module("deap.tools")

from saealib import GA, TournamentSelection, TruncationSelection, minimize  # noqa: E402
from saealib.operators import DeapCrossover, DeapMutation  # noqa: E402
from saealib.operators._deap_rng import seeded_global_random  # noqa: E402


def _operators(dim: int, calls: dict[str, int] | None = None):
    lower = [-1.0] * dim
    upper = [1.0] * dim
    toolbox = base.Toolbox()
    toolbox.register(
        "mate",
        tools.cxSimulatedBinaryBounded,
        eta=10.0,
        low=lower,
        up=upper,
    )
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        eta=15.0,
        low=lower,
        up=upper,
        indpb=1.0 / dim,
    )
    if calls is None:
        return DeapCrossover(getattr(toolbox, "mate")), DeapMutation(
            getattr(toolbox, "mutate")
        )

    def mate(ind1, ind2):
        calls["mate"] += 1
        return getattr(toolbox, "mate")(ind1, ind2)

    def mutate(individual):
        calls["mutate"] += 1
        return getattr(toolbox, "mutate")(individual)

    return DeapCrossover(mate), DeapMutation(mutate)


def test_adapters_run_and_preserve_bounds():
    crossover, mutation = _operators(5)
    parents = np.random.default_rng(1).uniform(-1, 1, size=(4, 2, 5))
    bounds = (np.full(5, -1.0), np.full(5, 1.0))
    children = crossover.crossover_batch(parents, bounds, np.random.default_rng(2))
    mutants = mutation.mutate_batch(parents[:, 0], bounds, np.random.default_rng(3))
    assert children.shape == parents.shape
    assert mutants.shape == parents[:, 0].shape
    assert np.all((children >= -1) & (children <= 1))
    assert np.all((mutants >= -1) & (mutants <= 1))


def test_adapters_do_not_require_runtime_bounds_service():
    crossover, mutation = _operators(3)
    assert crossover.contract().ports["crossover"].inputs[0].required_services == ()
    assert mutation.contract().ports["mutation"].inputs[0].required_services == ()


def test_same_seed_reproduces_and_different_seed_changes_result():
    crossover, mutation = _operators(6)
    parents = np.random.default_rng(4).uniform(-1, 1, size=(3, 2, 6))
    candidates = parents[:, 0]
    bounds = (np.full(6, -1.0), np.full(6, 1.0))

    first = crossover.crossover_batch(parents, bounds, np.random.default_rng(8))
    second = crossover.crossover_batch(parents, bounds, np.random.default_rng(8))
    third = crossover.crossover_batch(parents, bounds, np.random.default_rng(9))
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)

    first = mutation.mutate_batch(candidates, bounds, np.random.default_rng(8))
    second = mutation.mutate_batch(candidates, bounds, np.random.default_rng(8))
    third = mutation.mutate_batch(candidates, bounds, np.random.default_rng(9))
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)


def test_global_random_state_restored_on_success_and_exception():
    random.seed(123)
    before = random.getstate()
    with seeded_global_random(np.random.default_rng(1)):
        assert random.getstate() != before
    assert random.getstate() == before

    with pytest.raises(RuntimeError), seeded_global_random(np.random.default_rng(2)):
        raise RuntimeError("boom")
    assert random.getstate() == before


def test_adapters_restore_global_random_state_when_operator_raises():
    def fail_crossover(ind1, ind2):
        raise RuntimeError("crossover failure")

    def fail_mutation(individual):
        raise RuntimeError("mutation failure")

    parents = np.zeros((1, 2, 3))
    random.seed(456)
    before = random.getstate()
    with pytest.raises(RuntimeError, match="crossover failure"):
        DeapCrossover(fail_crossover).crossover_batch(
            parents, rng=np.random.default_rng(1)
        )
    assert random.getstate() == before

    random.seed(789)
    before = random.getstate()
    with pytest.raises(RuntimeError, match="mutation failure"):
        DeapMutation(fail_mutation).mutate_batch(
            parents[:, 0],
            (np.full(3, -1.0), np.full(3, 1.0)),
            rng=np.random.default_rng(2),
        )
    assert random.getstate() == before


def test_missing_deap_has_install_command(monkeypatch: pytest.MonkeyPatch):
    real_import = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "deap" or name.startswith("deap."):
            raise ImportError("blocked")
        del args, kwargs
        return real_import(name)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for factory in (
        lambda: DeapCrossover(lambda ind1, ind2: (ind1, ind2)),
        lambda: DeapMutation(lambda individual: (individual,)),
    ):
        with pytest.raises(ImportError, match=r"pip install.*saealib\[deap\]"):
            factory()


def test_ga_uses_deap_adapters():
    calls = {"mate": 0, "mutate": 0}
    crossover, mutation = _operators(4, calls)
    result = minimize(
        lambda x: np.sum(x**2),
        dim=4,
        lb=[-1.0] * 4,
        ub=[1.0] * 4,
        algorithm=GA(
            crossover=crossover,
            mutation=mutation,
            parent_selection=TournamentSelection(2),
            survivor_selection=TruncationSelection(),
        ),
        surrogate="rbf",
        max_fe=80,
        pop_size=8,
        seed=11,
        verbose=False,
    )
    assert result.fe > 0
    assert np.isfinite(result.f).all()
    assert calls["mate"] > 0
    assert calls["mutate"] > 0
