"""
Tests for decomposition strategies and DecompositionComparator.

Coverage:
- uniform_weight_vectors: count, non-negativity, sum-to-one, edge cases
- WeightedSumDecomposition: formula, ideal_point ignored
- TchebycheffDecomposition: formula, zero-weight handling
- PBIDecomposition: d1/d2 geometry, theta parameter
- DecompositionComparator: sort, compare_population, compare, feasibility rule,
  caching, direction transform, ideal_point auto-compute
- Non-convex Pareto front: WS vs Tchebycheff behavior on ZDT2-like problem
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from saealib.decomposition import (
    DecompositionComparator,
    PBIDecomposition,
    TchebycheffDecomposition,
    WeightedSumDecomposition,
)
from saealib.population import Population, PopulationAttribute
from saealib.utils.weight_vectors import uniform_weight_vectors

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pop(f: np.ndarray, cv: np.ndarray | None = None) -> Population:
    n, n_obj = f.shape
    if cv is None:
        cv = np.zeros(n)
    attrs = [
        PopulationAttribute(name="f", dtype=np.float64, shape=(n_obj,)),
        PopulationAttribute(name="cv", dtype=np.float64, shape=(), default=0.0),
    ]
    pop = Population(attrs, init_capacity=n + 1)
    for i in range(n):
        pop.append(f=f[i], cv=float(cv[i]))
    return pop


# ===========================================================================
# uniform_weight_vectors
# ===========================================================================


class TestUniformWeightVectors:
    def test_n_obj2_h4_count(self) -> None:
        """n_obj=2, H=4: C(2+4-1, 4) = C(5,4) = 5 vectors."""
        W = uniform_weight_vectors(2, 4)
        assert W.shape == (5, 2)

    def test_n_obj3_h4_count(self) -> None:
        """n_obj=3, H=4: C(3+4-1, 4) = C(6,4) = 15 vectors."""
        W = uniform_weight_vectors(3, 4)
        assert W.shape == (15, 3)

    def test_n_obj4_h3_count(self) -> None:
        """n_obj=4, H=3: C(4+3-1, 3) = C(6,3) = 20 vectors."""
        W = uniform_weight_vectors(4, 3)
        expected = math.comb(4 + 3 - 1, 3)
        assert W.shape[0] == expected

    def test_sum_to_one(self) -> None:
        W = uniform_weight_vectors(3, 5)
        assert np.allclose(W.sum(axis=1), 1.0)

    def test_non_negative(self) -> None:
        W = uniform_weight_vectors(3, 5)
        assert np.all(W >= 0.0)

    def test_h1_yields_identity_like_rows(self) -> None:
        """H=1: n_obj vectors, each is a standard basis vector."""
        W = uniform_weight_vectors(3, 1)
        assert W.shape == (3, 3)
        assert np.allclose(W.sum(axis=1), 1.0)

    def test_n_obj2_h1_values(self) -> None:
        W = uniform_weight_vectors(2, 1)
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        assert np.allclose(W, expected)

    def test_invalid_n_obj(self) -> None:
        with pytest.raises(ValueError, match="n_obj"):
            uniform_weight_vectors(1, 4)

    def test_invalid_n_divisions(self) -> None:
        with pytest.raises(ValueError, match="n_divisions"):
            uniform_weight_vectors(2, 0)


# ===========================================================================
# WeightedSumDecomposition
# ===========================================================================


class TestWeightedSumDecomposition:
    def setup_method(self) -> None:
        self.ws = WeightedSumDecomposition()
        self.f = np.array([[1.0, 2.0], [3.0, 0.5]])
        self.w = np.array([0.5, 0.5])
        self.z = np.array([0.0, 0.0])  # not used

    def test_basic_formula(self) -> None:
        scores = self.ws.aggregate(self.f, self.w, self.z)
        expected = np.array([1.5, 1.75])
        np.testing.assert_allclose(scores, expected)

    def test_ideal_point_ignored(self) -> None:
        s1 = self.ws.aggregate(self.f, self.w, np.array([0.0, 0.0]))
        s2 = self.ws.aggregate(self.f, self.w, np.array([100.0, 100.0]))
        np.testing.assert_array_equal(s1, s2)

    def test_output_shape(self) -> None:
        rng = np.random.default_rng(0)
        f = rng.random((10, 3))
        w = np.array([0.3, 0.3, 0.4])
        z = np.zeros(3)
        scores = self.ws.aggregate(f, w, z)
        assert scores.shape == (10,)

    def test_unequal_weights(self) -> None:
        f = np.array([[2.0, 1.0]])
        w = np.array([0.8, 0.2])
        z = np.zeros(2)
        scores = self.ws.aggregate(f, w, z)
        np.testing.assert_allclose(scores, [1.8])


# ===========================================================================
# TchebycheffDecomposition
# ===========================================================================


class TestTchebycheffDecomposition:
    def setup_method(self) -> None:
        self.tch = TchebycheffDecomposition()
        self.w = np.array([0.5, 0.5])
        self.z = np.array([0.0, 0.0])

    def test_basic_formula(self) -> None:
        f = np.array([[2.0, 1.0], [1.0, 2.0]])
        scores = self.tch.aggregate(f, self.w, self.z)
        # max(0.5*2, 0.5*1) = 1.0; max(0.5*1, 0.5*2) = 1.0
        np.testing.assert_allclose(scores, [1.0, 1.0])

    def test_asymmetric_weights(self) -> None:
        f = np.array([[1.0, 2.0]])
        w = np.array([0.3, 0.7])
        z = np.zeros(2)
        scores = self.tch.aggregate(f, w, z)
        # max(0.3*1, 0.7*2) = max(0.3, 1.4) = 1.4
        np.testing.assert_allclose(scores, [1.4])

    def test_ideal_point_shift(self) -> None:
        f = np.array([[3.0, 5.0]])
        z = np.array([1.0, 2.0])
        scores = self.tch.aggregate(f, self.w, z)
        # max(0.5*|3-1|, 0.5*|5-2|) = max(1.0, 1.5) = 1.5
        np.testing.assert_allclose(scores, [1.5])

    def test_zero_weight_replaced(self) -> None:
        """Zero weight components do not produce zero score (1e-6 floor)."""
        f = np.array([[10.0, 1.0]])
        w = np.array([0.0, 1.0])  # first weight is zero
        z = np.zeros(2)
        scores = self.tch.aggregate(f, w, z)
        # max(1e-6*10, 1.0*1) = max(1e-5, 1.0) = 1.0
        np.testing.assert_allclose(scores, [1.0])

    def test_output_shape(self) -> None:
        rng = np.random.default_rng(0)
        f = rng.random((7, 3))
        w = np.array([0.4, 0.3, 0.3])
        z = np.zeros(3)
        assert self.tch.aggregate(f, w, z).shape == (7,)


# ===========================================================================
# PBIDecomposition
# ===========================================================================


class TestPBIDecomposition:
    def test_theta_property(self) -> None:
        pbi = PBIDecomposition(theta=3.0)
        assert pbi.theta == 3.0

    def test_parallel_point_d2_is_zero(self) -> None:
        """A point exactly on the weight-vector ray has d2 = 0."""
        pbi = PBIDecomposition(theta=5.0)
        z = np.array([0.0, 0.0])
        w = np.array([1.0, 1.0])
        # Point on ray: (2, 2) — exactly aligned with w
        f = np.array([[2.0, 2.0]])
        scores = pbi.aggregate(f, w, z)
        # d1 = |(2,2)·(1/√2, 1/√2)| = 4/√2 = 2√2
        # d2 = 0
        d1 = np.abs(np.array([2.0, 2.0]) @ (w / np.linalg.norm(w)))
        np.testing.assert_allclose(scores, [d1], rtol=1e-6)

    def test_perpendicular_point_d1_is_zero(self) -> None:
        """A point orthogonal to w from z* has d1 ≈ 0."""
        pbi = PBIDecomposition(theta=5.0)
        w = np.array([1.0, 0.0])  # weight vector along f1 axis
        z = np.array([0.0, 0.0])
        # Point (0, 3) is perpendicular to w from z*
        f = np.array([[0.0, 3.0]])
        scores = pbi.aggregate(f, w, z)
        # d1 ≈ 0, d2 = 3, score = 0 + 5*3 = 15
        np.testing.assert_allclose(scores, [15.0], atol=1e-6)

    def test_theta_effect(self) -> None:
        """Higher theta penalizes off-axis points more."""
        z = np.array([0.0, 0.0])
        w = np.array([1.0, 0.0])
        f = np.array([[1.0, 2.0]])  # some off-axis point
        s_low = PBIDecomposition(theta=1.0).aggregate(f, w, z)[0]
        s_high = PBIDecomposition(theta=10.0).aggregate(f, w, z)[0]
        assert s_high > s_low

    def test_output_shape(self) -> None:
        rng = np.random.default_rng(0)
        pbi = PBIDecomposition()
        f = rng.random((8, 2))
        w = np.array([0.5, 0.5])
        z = np.zeros(2)
        assert pbi.aggregate(f, w, z).shape == (8,)


# ===========================================================================
# DecompositionComparator
# ===========================================================================


class TestDecompositionComparatorSort:
    def test_sort_by_score_ascending(self) -> None:
        f = np.array([[2.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
        # WeightedSum scores with w=[0.5,0.5]: [1.0, 1.0, 1.0] — all equal
        # Tchebycheff with w=[0.5,0.5], z=[0,0]: [1.0, 0.5, 1.0]
        # So [1, 0, 2] or [1, 2, 0]
        pop = _make_pop(f)
        cmp = DecompositionComparator(
            TchebycheffDecomposition(),
            weights=np.array([0.5, 0.5]),
            ideal_point=np.array([0.0, 0.0]),
        )
        order = cmp.sort_population(pop)
        assert order[0] == 1  # balanced point ranks first

    def test_feasibility_first(self) -> None:
        f = np.array([[1.0, 1.0], [0.0, 0.0]])
        cv = np.array([1.0, 0.0])  # first is infeasible
        pop = _make_pop(f, cv)
        cmp = DecompositionComparator(
            WeightedSumDecomposition(),
            weights=np.array([1.0, 1.0]),
            ideal_point=np.array([0.0, 0.0]),
        )
        order = cmp.sort_population(pop)
        assert order[0] == 1  # feasible ranks first despite worse f

    def test_infeasible_sorted_by_cv(self) -> None:
        f = np.zeros((3, 2))
        cv = np.array([2.0, 0.5, 1.0])
        pop = _make_pop(f, cv)
        cmp = DecompositionComparator(
            WeightedSumDecomposition(),
            weights=np.array([0.5, 0.5]),
            ideal_point=np.zeros(2),
        )
        order = cmp.sort_population(pop)
        # All infeasible; sorted by ascending cv: 0.5, 1.0, 2.0 → idx 1, 2, 0
        np.testing.assert_array_equal(order, [1, 2, 0])

    def test_sort_cache_consistency(self) -> None:
        f = np.array([[2.0, 0.0], [1.0, 1.0]])
        pop = _make_pop(f)
        cmp = DecompositionComparator(
            TchebycheffDecomposition(),
            weights=np.array([0.5, 0.5]),
            ideal_point=np.zeros(2),
        )
        order1 = cmp.sort_population(pop)
        order2 = cmp.sort_population(pop)
        np.testing.assert_array_equal(order1, order2)

    def test_ideal_point_auto_compute(self) -> None:
        """ideal_point=None: computed as per-objective min of feasible set."""
        # With z* auto-computed from [1,3] and [3,1], z* = [1,1]
        # Tchebycheff scores with w=[0.5,0.5]:
        #   (1,3): max(0.5*0, 0.5*2) = 1.0
        #   (3,1): max(0.5*2, 0.5*0) = 1.0  → tied, stable sort preserves order
        f = np.array([[1.0, 3.0], [3.0, 1.0]])
        pop = _make_pop(f)
        cmp = DecompositionComparator(
            TchebycheffDecomposition(),
            weights=np.array([0.5, 0.5]),
            ideal_point=None,
        )
        order = cmp.sort_population(pop)
        assert len(order) == 2

    def test_direction_transform(self) -> None:
        """Maximize objectives: direction=+1 flips to minimize before aggregation."""
        # Two solutions in maximize frame: [5,1] and [1,5]
        # With direction=[+1,+1], f_min = f * (-1) = [-5,-1] and [-1,-5]
        # z* (min of f_min) = [-5,-5]
        # Tchebycheff with w=[0.5,0.5]:
        #   [-5,-1] - [-5,-5] = [0, 4] -> max(0, 2) = 2
        #   [-1,-5] - [-5,-5] = [4, 0] -> max(2, 0) = 2  (tied)
        f = np.array([[5.0, 1.0], [1.0, 5.0]])
        pop = _make_pop(f)
        cmp = DecompositionComparator(
            TchebycheffDecomposition(),
            weights=np.array([0.5, 0.5]),
            direction=np.array([1.0, 1.0]),
        )
        order = cmp.sort_population(pop)
        assert len(order) == 2


class TestDecompositionComparatorCompare:
    def setup_method(self) -> None:
        self.cmp = DecompositionComparator(
            TchebycheffDecomposition(),
            weights=np.array([0.5, 0.5]),
            ideal_point=np.array([0.0, 0.0]),
        )

    def test_better_is_minus_one(self) -> None:
        # (1,1): score = max(0.5, 0.5) = 0.5
        # (2,2): score = max(1.0, 1.0) = 1.0  → a better
        result = self.cmp.compare(np.array([1.0, 1.0]), 0.0, np.array([2.0, 2.0]), 0.0)
        assert result == -1

    def test_worse_is_plus_one(self) -> None:
        result = self.cmp.compare(np.array([2.0, 2.0]), 0.0, np.array([1.0, 1.0]), 0.0)
        assert result == 1

    def test_equal_within_eps(self) -> None:
        cmp = DecompositionComparator(
            WeightedSumDecomposition(),
            weights=np.array([0.5, 0.5]),
            ideal_point=np.zeros(2),
            eps_obj=0.5,
        )
        # (1,1): 1.0, (1.5, 0.5): 1.0 — exactly equal (within eps)
        result = cmp.compare(np.array([1.0, 1.0]), 0.0, np.array([1.5, 0.5]), 0.0)
        assert result == 0

    def test_both_infeasible_lower_cv_wins(self) -> None:
        result = self.cmp.compare(np.array([0.0, 0.0]), 0.5, np.array([0.0, 0.0]), 1.0)
        assert result == -1

    def test_feasible_beats_infeasible(self) -> None:
        result = self.cmp.compare(
            np.array([10.0, 10.0]), 0.0, np.array([0.0, 0.0]), 1.0
        )
        assert result == -1

    def test_infeasible_loses_to_feasible(self) -> None:
        result = self.cmp.compare(
            np.array([0.0, 0.0]), 1.0, np.array([10.0, 10.0]), 0.0
        )
        assert result == 1

    def test_compare_population_consistent(self) -> None:
        f = np.array([[2.0, 0.0], [1.0, 1.0]])
        pop = _make_pop(f)
        # Tchebycheff w=[0.5,0.5] z=[0,0]: (2,0)->1.0, (1,1)->0.5
        result = self.cmp.compare_population(pop, 0, 1)
        assert result == 1  # idx 0 is worse

    def test_no_ideal_point_pairwise_min(self) -> None:
        """When ideal_point=None, compare() uses pair-wise min as approximation."""
        cmp = DecompositionComparator(
            TchebycheffDecomposition(),
            weights=np.array([0.5, 0.5]),
            ideal_point=None,
        )
        fa = np.array([1.0, 3.0])
        fb = np.array([3.0, 1.0])
        # z* = min(fa, fb) = [1, 1]
        # score(fa) = max(0.5*0, 0.5*2) = 1.0
        # score(fb) = max(0.5*2, 0.5*0) = 1.0 → equal
        result = cmp.compare(fa, 0.0, fb, 0.0)
        assert result == 0


# ===========================================================================
# Non-convex Pareto front: WS vs Tchebycheff
# ===========================================================================


class TestNonConvexParetoBehavior:
    """
    Verify decomposition behavior on a ZDT2-like non-convex Pareto front.

    Pareto front: f2 = 1 - f1^2, f1 in [0, 1].

    With w = [0.5, 0.5] and z* = [0, 0]:
    - WeightedSum score = 0.5*f1 + 0.5*f2 = 0.5*f1 + 0.5*(1-f1^2)
      minimized at f1=0 or f1=1 (score=0.5); maximized at f1=0.5 (score=0.625).
      → WS CANNOT reach the balanced interior point.
    - Tchebycheff score = max(0.5*f1, 0.5*f2)
      minimized at f1 ≈ 0.618 (score ≈ 0.309); extremes score 0.5.
      → Tchebycheff CAN reach the balanced interior.
    """

    # f1=0 → f2=1; f1≈0.618 → f2≈0.618; f1=1 → f2=0
    F_EXTREME_A = np.array([0.0, 1.0])
    F_BALANCED = np.array([0.618, 0.618])
    F_EXTREME_B = np.array([1.0, 0.0])
    W = np.array([0.5, 0.5])
    Z = np.array([0.0, 0.0])

    def _scores(self, decomp, points):
        return decomp.aggregate(np.array(points), self.W, self.Z)

    def test_ws_balanced_point_is_worst(self) -> None:
        """WeightedSum assigns the highest (worst) score to the balanced point."""
        ws = WeightedSumDecomposition()
        s = self._scores(ws, [self.F_EXTREME_A, self.F_BALANCED, self.F_EXTREME_B])
        # balanced (index 1) must have the highest score (worst)
        assert s[1] > s[0] and s[1] > s[2]

    def test_tch_balanced_point_is_best(self) -> None:
        """Tchebycheff assigns the lowest (best) score to the balanced point."""
        tch = TchebycheffDecomposition()
        s = self._scores(tch, [self.F_EXTREME_A, self.F_BALANCED, self.F_EXTREME_B])
        # balanced (index 1) must have the lowest score (best)
        assert s[1] < s[0] and s[1] < s[2]

    def test_pbi_balanced_point_is_best(self) -> None:
        """PBI also assigns the best score to the balanced point."""
        pbi = PBIDecomposition(theta=5.0)
        s = self._scores(pbi, [self.F_EXTREME_A, self.F_BALANCED, self.F_EXTREME_B])
        assert s[1] < s[0] and s[1] < s[2]

    def test_comparator_tch_sort_prefers_balanced(self) -> None:
        """DecompositionComparator with Tchebycheff ranks balanced point first."""
        f = np.stack([self.F_EXTREME_A, self.F_BALANCED, self.F_EXTREME_B])
        pop = _make_pop(f)
        cmp = DecompositionComparator(
            TchebycheffDecomposition(),
            weights=self.W,
            ideal_point=self.Z,
        )
        order = cmp.sort_population(pop)
        assert order[0] == 1  # index 1 = balanced point

    def test_comparator_ws_sort_prefers_extreme(self) -> None:
        """DecompositionComparator with WeightedSum does NOT prefer balanced point."""
        f = np.stack([self.F_EXTREME_A, self.F_BALANCED, self.F_EXTREME_B])
        pop = _make_pop(f)
        cmp = DecompositionComparator(
            WeightedSumDecomposition(),
            weights=self.W,
            ideal_point=self.Z,
        )
        order = cmp.sort_population(pop)
        # Balanced point must NOT be first
        assert order[0] != 1
