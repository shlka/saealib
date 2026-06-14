"""Tests for surrogate training set strategy objects."""

from __future__ import annotations

import numpy as np
import pytest

from saealib.comparators import SingleObjectiveComparator
from saealib.context import OptimizationContext
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.surrogate.training_set import (
    ArchiveObjectiveSet,
    ConstraintObjectiveSet,
    FeasibilityClassificationSet,
    KNNConstraintObjectiveSet,
    KNNObjectiveSet,
    LevelBasedSet,
    PairwiseComparisonSet,
    TopKBipartitionSet,
    TrainingSet,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DIM = 2
N_OBJ = 1
N_CONSTRAINTS = 2

_ATTRS = [
    PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
    PopulationAttribute(name="f", dtype=np.float64, shape=(N_OBJ,)),
    PopulationAttribute(name="g", dtype=np.float64, shape=(0,)),
    PopulationAttribute(name="cv", dtype=np.float64, shape=()),
]

_ATTRS_G = [
    PopulationAttribute(name="x", dtype=np.float64, shape=(DIM,)),
    PopulationAttribute(name="f", dtype=np.float64, shape=(N_OBJ,)),
    PopulationAttribute(name="g", dtype=np.float64, shape=(N_CONSTRAINTS,)),
    PopulationAttribute(name="cv", dtype=np.float64, shape=()),
]


def _make_problem(eps_cv: float = 1e-6) -> Problem:
    return Problem(
        func=lambda x: np.array([np.sum(x**2)]),
        dim=DIM,
        n_obj=N_OBJ,
        direction=np.array([-1.0]),
        lb=[-5.0] * DIM,
        ub=[5.0] * DIM,
        eps_cv=eps_cv,
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


def _make_archive_with_g(n: int = 10) -> Archive:
    """Archive with 2 constraint columns filled with known values."""
    arc = Archive(_ATTRS_G, init_capacity=n + 5)
    rng = np.random.default_rng(7)
    for i in range(n):
        x = rng.uniform(-2.0, 2.0, size=DIM)
        f = np.array([np.sum(x**2)])
        g = np.array([float(i), float(-i)])  # known values for assertion
        arc.add(x=x, f=f, g=g, cv=0.0)
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
    eps_cv: float = 1e-6,
) -> OptimizationContext:
    problem = _make_problem(eps_cv=eps_cv)
    arc = archive if archive is not None else _make_archive()
    pop = population if population is not None else _make_population_with_cv([0.0] * 5)
    pareto_arc = ParetoArchive(_ATTRS, init_capacity=20, direction=np.array([-1.0]))
    return OptimizationContext(
        problem=problem,
        population=pop,
        archive=arc,
        pareto_archive=pareto_arc,
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
# ConstraintObjectiveSet
# ===========================================================================
class TestConstraintObjectiveSet:
    def test_returns_all_archive_points(self) -> None:
        arc = _make_archive_with_g(8)
        ts = ConstraintObjectiveSet()
        data = ts.build(arc, None, None)
        assert data.train_x.shape == (8, DIM)
        assert data.train_y.shape == (8, N_CONSTRAINTS)
        np.testing.assert_array_equal(data.train_y, arc.g)

    def test_train_y_is_g_not_f(self) -> None:
        arc = _make_archive_with_g(5)
        ts = ConstraintObjectiveSet()
        data = ts.build(arc, None, None)
        # train_y must differ from f (different columns)
        assert data.train_y.shape[1] == N_CONSTRAINTS
        assert data.train_y.shape[1] != arc.f.shape[1] or not np.allclose(
            data.train_y, arc.f
        )

    def test_raises_on_zero_constraints(self) -> None:
        arc = _make_archive(5)  # uses _ATTRS with shape=(0,) for g
        ts = ConstraintObjectiveSet()
        with pytest.raises(ValueError, match="0 columns"):
            ts.build(arc, None, None)

    def test_candidate_x_is_ignored(self) -> None:
        arc = _make_archive_with_g(6)
        ts = ConstraintObjectiveSet()
        candidate = np.zeros(DIM)
        data = ts.build(arc, None, None, candidate_x=candidate)
        assert data.train_x.shape[0] == 6


# ===========================================================================
# KNNConstraintObjectiveSet
# ===========================================================================
class TestKNNConstraintObjectiveSet:
    def test_returns_k_neighbours(self) -> None:
        arc = _make_archive_with_g(10)
        ts = KNNConstraintObjectiveSet(n_neighbors=5)
        candidate = np.zeros(DIM)
        data = ts.build(arc, None, None, candidate_x=candidate)
        assert data.train_x.shape == (5, DIM)
        assert data.train_y.shape == (5, N_CONSTRAINTS)

    def test_requires_candidate_x(self) -> None:
        arc = _make_archive_with_g(5)
        ts = KNNConstraintObjectiveSet(n_neighbors=3)
        with pytest.raises(ValueError, match="candidate_x"):
            ts.build(arc, None, None, candidate_x=None)

    def test_raises_on_zero_constraints(self) -> None:
        arc = _make_archive(5)
        ts = KNNConstraintObjectiveSet(n_neighbors=3)
        with pytest.raises(ValueError, match="0 columns"):
            ts.build(arc, None, None, candidate_x=np.zeros(DIM))

    def test_clamped_to_archive_size(self) -> None:
        arc = _make_archive_with_g(4)
        ts = KNNConstraintObjectiveSet(n_neighbors=50)
        data = ts.build(arc, None, None, candidate_x=np.zeros(DIM))
        assert data.train_x.shape[0] == 4


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
        ctx = _make_ctx(archive=arc, eps_cv=0.15)
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


# ===========================================================================
# TopKBipartitionSet
# ===========================================================================


def _make_archive_with_f(
    f_values: list[float], cvs: list[float] | None = None
) -> Archive:
    """Archive where each point has a prescribed scalar objective."""
    if cvs is None:
        cvs = [0.0] * len(f_values)
    arc = Archive(_ATTRS, init_capacity=len(f_values) + 5)
    rng = np.random.default_rng(2)
    for fv, cv in zip(f_values, cvs):
        x = rng.uniform(-2.0, 2.0, size=DIM)
        arc.add(x=x, f=np.array([fv]), cv=float(cv))
    return arc


class TestTopKBipartitionSet:
    def test_label_count(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0, 3.0, 4.0])
        ctx = _make_ctx(archive=arc)
        ts = TopKBipartitionSet(source="archive", top_ratio=0.5)
        data = ts.build(arc, None, ctx)
        assert data.train_y.shape == (4,)
        assert int(data.train_y.sum()) == 2  # top 2 = label 1

    def test_best_get_label_one(self) -> None:
        """After sorting best-first, the first k entries should be label 1."""
        # f=1.0 is best (minimization), f=4.0 is worst
        arc = _make_archive_with_f([4.0, 1.0, 3.0, 2.0])
        ctx = _make_ctx(archive=arc)
        ts = TopKBipartitionSet(source="archive", top_ratio=0.5)
        data = ts.build(arc, None, ctx)
        # sorted order: f=1.0, f=2.0 → label 1; f=3.0, f=4.0 → label 0
        np.testing.assert_array_equal(data.train_y[:2], [1.0, 1.0])
        np.testing.assert_array_equal(data.train_y[2:], [0.0, 0.0])

    def test_at_least_one_label_one(self) -> None:
        """top_ratio=0 should still yield at least one label 1."""
        arc = _make_archive_with_f([1.0, 2.0, 3.0])
        ctx = _make_ctx(archive=arc)
        ts = TopKBipartitionSet(source="archive", top_ratio=0.0)
        data = ts.build(arc, None, ctx)
        assert data.train_y.sum() >= 1.0

    def test_ctx_none_raises(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0])
        ts = TopKBipartitionSet(source="archive")
        with pytest.raises(ValueError, match="ctx"):
            ts.build(arc, None, None)

    def test_source_population_none_raises(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0])
        ctx = _make_ctx(archive=arc)
        ts = TopKBipartitionSet(source="population")
        with pytest.raises(ValueError, match="population"):
            ts.build(arc, None, ctx)

    def test_source_population(self) -> None:
        pop = _make_population_with_cv([0.0, 0.0, 0.0, 0.0])
        arc = _make_archive()
        ctx = _make_ctx(archive=arc, population=pop)
        ts = TopKBipartitionSet(source="population", top_ratio=0.5)
        data = ts.build(arc, pop, ctx)
        assert data.train_x.shape == (4, DIM)
        assert data.train_y.shape == (4,)

    def test_train_x_shape(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0, 3.0])
        ctx = _make_ctx(archive=arc)
        ts = TopKBipartitionSet(source="archive")
        data = ts.build(arc, None, ctx)
        assert data.train_x.shape == (3, DIM)

    def test_labels_are_float(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0])
        ctx = _make_ctx(archive=arc)
        ts = TopKBipartitionSet(source="archive")
        data = ts.build(arc, None, ctx)
        assert data.train_y.dtype == float


# ===========================================================================
# LevelBasedSet
# ===========================================================================


class TestLevelBasedSet:
    def test_three_levels_equal_split(self) -> None:
        """6 individuals, 3 levels → 2 per level, labels [0,0,1,1,2,2]."""
        arc = _make_archive_with_f([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        ctx = _make_ctx(archive=arc)
        ts = LevelBasedSet(source="archive", n_levels=3)
        data = ts.build(arc, None, ctx)
        np.testing.assert_array_equal(data.train_y, [0.0, 0.0, 1.0, 1.0, 2.0, 2.0])

    def test_remainder_goes_to_last_level(self) -> None:
        """7 individuals, 3 levels → per=2; labels [0,0,1,1,2,2,2]."""
        arc = _make_archive_with_f([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        ctx = _make_ctx(archive=arc)
        ts = LevelBasedSet(source="archive", n_levels=3)
        data = ts.build(arc, None, ctx)
        np.testing.assert_array_equal(data.train_y, [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0])

    def test_best_individuals_get_level_zero(self) -> None:
        """Level 0 corresponds to best-sorted (smallest f for minimization)."""
        arc = _make_archive_with_f([5.0, 1.0, 3.0, 2.0])
        ctx = _make_ctx(archive=arc)
        ts = LevelBasedSet(source="archive", n_levels=2)
        data = ts.build(arc, None, ctx)
        # sorted order: f=1.0, f=2.0 → level 0; f=3.0, f=5.0 → level 1
        np.testing.assert_array_equal(data.train_y[:2], [0.0, 0.0])
        np.testing.assert_array_equal(data.train_y[2:], [1.0, 1.0])

    def test_default_n_levels(self) -> None:
        ts = LevelBasedSet()
        assert ts.n_levels == 5

    def test_ctx_none_raises(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0])
        ts = LevelBasedSet(source="archive")
        with pytest.raises(ValueError, match="ctx"):
            ts.build(arc, None, None)

    def test_source_population_none_raises(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0])
        ctx = _make_ctx(archive=arc)
        ts = LevelBasedSet(source="population")
        with pytest.raises(ValueError, match="population"):
            ts.build(arc, None, ctx)

    def test_train_x_shape(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0, 3.0, 4.0, 5.0])
        ctx = _make_ctx(archive=arc)
        ts = LevelBasedSet(source="archive", n_levels=5)
        data = ts.build(arc, None, ctx)
        assert data.train_x.shape == (5, DIM)
        assert data.train_y.shape == (5,)

    def test_labels_are_float(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0, 3.0])
        ctx = _make_ctx(archive=arc)
        ts = LevelBasedSet(source="archive", n_levels=3)
        data = ts.build(arc, None, ctx)
        assert data.train_y.dtype == float


# ===========================================================================
# PairwiseComparisonSet
# ===========================================================================


class TestPairwiseComparisonSet:
    def test_all_pairs_by_default(self) -> None:
        """n_pairs=None → all n*(n-1)/2 pairs generated."""
        arc = _make_archive_with_f([1.0, 2.0, 3.0, 4.0])
        ctx = _make_ctx(archive=arc)
        ts = PairwiseComparisonSet(source="archive")
        data = ts.build(arc, None, ctx)
        n = 4
        assert data.train_x.shape == (n * (n - 1) // 2, DIM * 2)
        assert data.train_y.shape == (n * (n - 1) // 2,)

    def test_n_pairs_limits_count(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0, 3.0, 4.0, 5.0])
        ctx = _make_ctx(archive=arc)
        ts = PairwiseComparisonSet(source="archive", n_pairs=3)
        data = ts.build(arc, None, ctx)
        assert data.train_x.shape == (3, DIM * 2)
        assert data.train_y.shape == (3,)

    def test_n_pairs_exceeds_total_uses_all(self) -> None:
        """n_pairs >= n*(n-1)/2 → all pairs used."""
        arc = _make_archive_with_f([1.0, 2.0, 3.0])
        ctx = _make_ctx(archive=arc)
        ts = PairwiseComparisonSet(source="archive", n_pairs=999)
        data = ts.build(arc, None, ctx)
        assert data.train_x.shape == (3, DIM * 2)  # 3*(3-1)/2 = 3

    def test_train_x_shape_is_2_dim(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0, 3.0])
        ctx = _make_ctx(archive=arc)
        ts = PairwiseComparisonSet(source="archive")
        data = ts.build(arc, None, ctx)
        assert data.train_x.shape[1] == DIM * 2

    def test_label_ordering_best_vs_worst(self) -> None:
        """For a pair (best, worst), label should be 1 (best wins)."""
        arc = _make_archive_with_f([1.0, 10.0])  # idx 0 = best, idx 1 = worst
        ctx = _make_ctx(archive=arc)
        ts = PairwiseComparisonSet(source="archive")
        data = ts.build(arc, None, ctx)
        # Only one pair (0, 1): f=1.0 beats f=10.0 → label 1
        assert data.train_y[0] == pytest.approx(1.0)

    def test_labels_are_binary(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0, 3.0, 4.0])
        ctx = _make_ctx(archive=arc)
        ts = PairwiseComparisonSet(source="archive")
        data = ts.build(arc, None, ctx)
        assert set(data.train_y.tolist()).issubset({0.0, 1.0})

    def test_ctx_none_raises(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0])
        ts = PairwiseComparisonSet(source="archive")
        with pytest.raises(ValueError, match="ctx"):
            ts.build(arc, None, None)

    def test_source_population_none_raises(self) -> None:
        arc = _make_archive_with_f([1.0, 2.0])
        ctx = _make_ctx(archive=arc)
        ts = PairwiseComparisonSet(source="population")
        with pytest.raises(ValueError, match="population"):
            ts.build(arc, None, ctx)

    def test_custom_rng(self) -> None:
        """Custom rng produces reproducible sampling."""
        arc = _make_archive_with_f([1.0, 2.0, 3.0, 4.0, 5.0])
        ctx = _make_ctx(archive=arc)
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        ts1 = PairwiseComparisonSet(source="archive", n_pairs=4, rng=rng1)
        ts2 = PairwiseComparisonSet(source="archive", n_pairs=4, rng=rng2)
        d1 = ts1.build(arc, None, ctx)
        d2 = ts2.build(arc, None, ctx)
        np.testing.assert_array_equal(d1.train_x, d2.train_x)
        np.testing.assert_array_equal(d1.train_y, d2.train_y)

    def test_no_duplicate_pairs(self) -> None:
        """Each (i, j) pair appears at most once."""
        arc = _make_archive_with_f([1.0, 2.0, 3.0, 4.0, 5.0])
        ctx = _make_ctx(archive=arc)
        ts = PairwiseComparisonSet(source="archive", n_pairs=8)
        data = ts.build(arc, None, ctx)
        # Each row in train_x is unique (half-vector x_a or full [x_a, x_b])
        assert len(data.train_x) == len(np.unique(data.train_x, axis=0))
