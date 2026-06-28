"""Tests for RandomInitializer and SobolInitializer."""

from __future__ import annotations

import numpy as np
import pytest

from saealib import (
    GA,
    CrossoverBLXAlpha,
    LHSInitializer,
    MutationUniform,
    RandomInitializer,
    SequentialSelection,
    SobolInitializer,
    TruncationSelection,
)
from saealib.comparators import SingleObjectiveComparator
from saealib.execution.evaluator import SerialEvaluator
from saealib.execution.initializer import Initializer
from saealib.problem import Problem

DIM = 3
N_ARCHIVE = 8
N_POP = 4

LB = [-2.0] * DIM
UB = [3.0] * DIM


def _make_problem() -> Problem:
    return Problem(
        func=lambda x: np.array([np.sum(x**2)]),
        dim=DIM,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=LB,
        ub=UB,
        comparator=SingleObjectiveComparator(),
    )


class _MockProvider:
    seed: int | None = None

    def __init__(self):
        from saealib.callback import CallbackManager

        self.algorithm = GA(
            crossover=CrossoverBLXAlpha(crossover_rate=0.9, alpha=0.4),
            mutation=MutationUniform(mutation_rate=0.1),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        )
        self.evaluator = SerialEvaluator()
        self.cbmanager = CallbackManager()

    def dispatch(self, event):
        self.cbmanager.dispatch(event)


@pytest.fixture
def problem():
    return _make_problem()


@pytest.fixture
def provider():
    return _MockProvider()


# ---------------------------------------------------------------------------
# Shared behaviour: all initializers must satisfy these properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "initializer",
    [
        RandomInitializer(N_ARCHIVE, N_POP, seed=0),
        SobolInitializer(N_ARCHIVE, N_POP, seed=0),
        LHSInitializer(N_ARCHIVE, N_POP, seed=0),
    ],
    ids=["random", "sobol", "lhs"],
)
class TestInitializerContract:
    def test_archive_size(self, initializer, problem, provider):
        ctx = initializer.initialize(provider, problem)
        assert len(ctx.archive) == N_ARCHIVE

    def test_population_size(self, initializer, problem, provider):
        ctx = initializer.initialize(provider, problem)
        assert len(ctx.population) == N_POP

    def test_fe_equals_n_archive(self, initializer, problem, provider):
        ctx = initializer.initialize(provider, problem)
        assert ctx.fe == N_ARCHIVE

    def test_archive_x_within_bounds(self, initializer, problem, provider):
        ctx = initializer.initialize(provider, problem)
        x = ctx.archive.get_array("x")
        lb = np.array(LB)
        ub = np.array(UB)
        assert np.all(x >= lb - 1e-9)
        assert np.all(x <= ub + 1e-9)

    def test_is_initializer_subclass(self, initializer, problem, provider):
        assert isinstance(initializer, Initializer)


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


def test_random_initializer_importable_from_top_level():
    import saealib

    assert saealib.RandomInitializer is RandomInitializer


def test_sobol_initializer_importable_from_top_level():
    import saealib

    assert saealib.SobolInitializer is SobolInitializer


def test_random_initializer_importable_from_execution():
    import saealib.execution.initializer as m

    assert m.RandomInitializer is RandomInitializer


def test_sobol_initializer_importable_from_execution():
    import saealib.execution.initializer as m

    assert m.SobolInitializer is SobolInitializer


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_random_initializer_same_seed_reproducible(problem, provider):
    ctx1 = RandomInitializer(N_ARCHIVE, N_POP, seed=42).initialize(provider, problem)
    provider2 = _MockProvider()
    ctx2 = RandomInitializer(N_ARCHIVE, N_POP, seed=42).initialize(provider2, problem)
    np.testing.assert_array_equal(
        ctx1.archive.get_array("x"), ctx2.archive.get_array("x")
    )


def test_sobol_initializer_same_seed_reproducible(problem, provider):
    ctx1 = SobolInitializer(N_ARCHIVE, N_POP, seed=42).initialize(provider, problem)
    provider2 = _MockProvider()
    ctx2 = SobolInitializer(N_ARCHIVE, N_POP, seed=42).initialize(provider2, problem)
    np.testing.assert_array_equal(
        ctx1.archive.get_array("x"), ctx2.archive.get_array("x")
    )
