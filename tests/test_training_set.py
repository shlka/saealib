"""Tests for surrogate training set strategy objects."""

from __future__ import annotations

import numpy as np
import pytest

from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationContext
from saealib.population import Archive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.surrogate.training_set import (
    ArchiveObjectiveSet,
    FeasibilityClassificationSet,
    KNNObjectiveSet,
    TrainingSet,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DIM = 2
N_OBJ = 1

_ATTRS = [
    PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
    PopulationAttribute(name="f", dtype=np.float64, shape=(N_OBJ,)),
    PopulationAttribute(name="g", dtype=np.float64, shape=(0,)),
    PopulationAttribute(name="cv", dtype=np.float64, shape=()),
]


def _make_problem(eps: float = 1e-6) -> Problem:
    return Problem(
        func=lambda x: np.array([np.sum(x**2)]),
        dim=DIM,
        n_obj=N_OBJ,
        weight=np.array([-1.0]),
        lb=[-5.0] * DIM,
        ub=[5.0] * DIM,
        eps=eps,
        comparator=SingleObjectiveComparator(),
    )


def _make_archive(n: int = 10) -> Archive:
    arc = Archive(_ATTRS, init_capacity=n + 5)
    rng = np.random.default_rng(42)
    for _ in range(n):
        x = rng.uniform(-2.0, 2.0, size=DIM)
        f = np.array([np.sum(x**2)])
        arc.add(x=x, f=f, cv=0.0)
    return arc


def _make_archive_with_cv(cvs: list[float]) -> Archive:
    """Archive where each point has a prescribed cv value."""
    arc = Archive(_ATTRS, init_capacity=len(cvs) + 5)
    rng = np.random.default_rng(0)
    for cv in cvs:
        x = rng.uniform(-2.0, 2.0, size=DIM)
        f = np.array([np.sum(x**2)])
        arc.add(x=x, f=f, cv=float(cv))
    return arc


def _make_population_with_cv(cvs: list[float]) -> Population:
    """Population where each individual has a prescribed cv value."""
    pop = Population(_ATTRS, init_capacity=len(cvs) + 5)
    rng = np.random.default_rng(1)
    xs = rng.uniform(-2.0, 2.0, size=(len(cvs), DIM))
    fs = np.array([[np.sum(x**2)] for x in xs])
    cv_arr = np.array(cvs, dtype=float)
    pop.extend({"x": xs, "f": fs, "cv": cv_arr})
    return pop


def _make_ctx(
    archive: Archive | None = None,
    population: Population | None = None,
    eps: float = 1e-6,
) -> OptimizationContext:
    problem = _make_problem(eps=eps)
    arc = archive if archive is not None else _make_archive()
    pop = population if population is not None else _make_population_with_cv([0.0] * 5)
    return OptimizationContext(
        problem=problem,
        population=pop,
        archive=arc,
        rng=np.random.default_rng(0),
        fe=10,
        gen=1,
    )


# ===========================================================================
# TrainingSet ABC
# ===========================================================================
class TestTrainingSetABC:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            TrainingSet()  # type: ignore[abstract]


# ===========================================================================
# ArchiveObjectiveSet
# ===========================================================================
class TestArchiveObjectiveSet:
    def test_returns_all_archive_points(self) -> None:
        arc = _make_archive(10)
        ts = ArchiveObjectiveSet()
        data = ts.build(arc, None, None)
        assert data.train_x.shape == (10, DIM)
        assert data.train_y.shape == (10, N_OBJ)

    def test_values_match_archive(self) -> None:
        arc = _make_archive(8)
        ts = ArchiveObjectiveSet()
        data = ts.build(arc, None, None)
        np.testing.assert_array_equal(data.train_x, arc.x)
        np.testing.assert_array_equal(data.train_y, arc.f)

    def test_candidate_x_ignored(self) -> None:
        arc = _make_archive(5)
        ts = ArchiveObjectiveSet()
        cand = np.zeros(DIM)
        data = ts.build(arc, None, None, candidate_x=cand)
        assert data.train_x.shape == (5, DIM)


# ===========================================================================
# KNNObjectiveSet
# ===========================================================================
class TestKNNObjectiveSet:
    def test_returns_k_neighbours(self) -> None:
        arc = _make_archive(20)
        ts = KNNObjectiveSet(n_neighbors=5)
        cand = np.zeros(DIM)
        data = ts.build(arc, None, None, candidate_x=cand)
        assert data.train_x.shape == (5, DIM)
        assert data.train_y.shape == (5, N_OBJ)

    def test_requires_candidate_x(self) -> None:
        arc = _make_archive(10)
        ts = KNNObjectiveSet(n_neighbors=5)
        with pytest.raises(ValueError, match="requires candidate_x"):
            ts.build(arc, None, None, candidate_x=None)

    def test_default_n_neighbors(self) -> None:
        ts = KNNObjectiveSet()
        assert ts.n_neighbors == 50

    def test_clamped_to_archive_size(self) -> None:
        arc = _make_archive(8)
        ts = KNNObjectiveSet(n_neighbors=50)
        cand = np.zeros(DIM)
        data = ts.build(arc, None, None, candidate_x=cand)
        assert data.train_x.shape[0] <= 8


# ===========================================================================
# FeasibilityClassificationSet
# ===========================================================================
class TestFeasibilityClassificationSet:
    def test_all_feasible(self) -> None:
        arc = _make_archive_with_cv([0.0, 0.0, 0.0, 0.0])
        ts = FeasibilityClassificationSet(source="archive")
        data = ts.build(arc, None, None)
        np.testing.assert_array_equal(data.train_y, np.ones(4))

    def test_all_infeasible(self) -> None:
        arc = _make_archive_with_cv([1.0, 2.0, 0.5])
        ts = FeasibilityClassificationSet(source="archive")
        data = ts.build(arc, None, None)
        np.testing.assert_array_equal(data.train_y, np.zeros(3))

    def test_mixed_labels(self) -> None:
        arc = _make_archive_with_cv([0.0, 1.0, 0.0, 2.0])
        ts = FeasibilityClassificationSet(source="archive")
        data = ts.build(arc, None, None)
        np.testing.assert_array_equal(data.train_y, [1.0, 0.0, 1.0, 0.0])

    def test_eps_from_ctx(self) -> None:
        """Boundary value: cv == eps is feasible."""
        arc = _make_archive_with_cv([0.1, 0.2, 0.3])
        ctx = _make_ctx(archive=arc, eps=0.15)
        ts = FeasibilityClassificationSet(source="archive")
        data = ts.build(arc, None, ctx)
        # cv=0.1 <= 0.15 → 1, cv=0.2 > 0.15 → 0, cv=0.3 > 0.15 → 0
        np.testing.assert_array_equal(data.train_y, [1.0, 0.0, 0.0])

    def test_eps_default_when_ctx_none(self) -> None:
        """Without ctx, eps defaults to 1e-6."""
        arc = _make_archive_with_cv([0.0, 1e-7, 1e-5])
        ts = FeasibilityClassificationSet(source="archive")
        data = ts.build(arc, None, None)
        # 0.0 <= 1e-6 → 1, 1e-7 <= 1e-6 → 1, 1e-5 > 1e-6 → 0
        np.testing.assert_array_equal(data.train_y, [1.0, 1.0, 0.0])

    def test_source_population(self) -> None:
        pop = _make_population_with_cv([0.0, 0.5, 0.0])
        arc = _make_archive()
        ts = FeasibilityClassificationSet(source="population")
        data = ts.build(arc, pop, None)
        np.testing.assert_array_equal(data.train_y, [1.0, 0.0, 1.0])

    def test_source_population_none_raises(self) -> None:
        arc = _make_archive()
        ts = FeasibilityClassificationSet(source="population")
        with pytest.raises(ValueError, match="population"):
            ts.build(arc, None, None)

    def test_train_x_shape(self) -> None:
        arc = _make_archive_with_cv([0.0, 1.0, 0.0])
        ts = FeasibilityClassificationSet(source="archive")
        data = ts.build(arc, None, None)
        assert data.train_x.shape == (3, DIM)
        assert data.train_y.shape == (3,)

    def test_labels_are_float(self) -> None:
        arc = _make_archive_with_cv([0.0, 1.0])
        ts = FeasibilityClassificationSet(source="archive")
        data = ts.build(arc, None, None)
        assert data.train_y.dtype == float
