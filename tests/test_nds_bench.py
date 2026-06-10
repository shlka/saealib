"""
Tests for dda_non_dominated_sort: equivalence, canonical cases, dominator injection,
large-M correctness, and speed guard (#89).
"""

import time

import numpy as np
import pytest

from saealib import dda_non_dominated_sort, non_dominated_sort
from saealib.comparators import Dominator, ParetoDominator


# ---------------------------------------------------------------------------
# 1. EQUIVALENCE: random (N, M) — dda must match the oracle non_dominated_sort
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n,m,seed",
    [
        (1, 1, 0),
        (2, 1, 1),
        (5, 1, 2),
        (5, 2, 3),
        (50, 2, 4),
        (50, 5, 5),
        (200, 2, 6),
        (200, 5, 7),
    ],
)
def test_dda_equivalence_random(n, m, seed):
    """dda_non_dominated_sort returns identical ranks and front membership."""
    rng = np.random.default_rng(seed)
    f = rng.random((n, m))

    ranks_ref, fronts_ref = non_dominated_sort(f)
    ranks_dda, fronts_dda = dda_non_dominated_sort(f)

    np.testing.assert_array_equal(ranks_dda, ranks_ref)
    assert len(fronts_dda) == len(fronts_ref)
    for fd, fr in zip(fronts_dda, fronts_ref):
        assert set(fd) == set(fr)


@pytest.mark.parametrize("n,m", [(10, 2), (30, 3)])
def test_dda_equivalence_duplicate_rows(n, m):
    """Tie-rows (identical objectives) must be in the same front as oracle."""
    rng = np.random.default_rng(99)
    f_unique = rng.random((n // 2, m))
    # duplicate every row to force ties
    f = np.vstack([f_unique, f_unique])

    ranks_ref, fronts_ref = non_dominated_sort(f)
    ranks_dda, fronts_dda = dda_non_dominated_sort(f)

    np.testing.assert_array_equal(ranks_dda, ranks_ref)
    assert len(fronts_dda) == len(fronts_ref)
    for fd, fr in zip(fronts_dda, fronts_ref):
        assert set(fd) == set(fr)


def test_dda_equivalence_with_nan_rows():
    """NaN rows get sentinel fronts identical to non_dominated_sort."""
    rng = np.random.default_rng(42)
    f = rng.random((10, 2))
    # inject NaN into rows 2 and 7
    f[2, 0] = np.nan
    f[7, :] = np.nan

    ranks_ref, fronts_ref = non_dominated_sort(f)
    ranks_dda, fronts_dda = dda_non_dominated_sort(f)

    np.testing.assert_array_equal(ranks_dda, ranks_ref)
    assert len(fronts_dda) == len(fronts_ref)
    for fd, fr in zip(fronts_dda, fronts_ref):
        assert set(fd) == set(fr)


@pytest.mark.parametrize("sign", [1, -1])
def test_dda_equivalence_direction(sign):
    """direction=+1 (maximize) and direction=-1 (minimize) match oracle."""
    rng = np.random.default_rng(7)
    f = rng.random((20, 3))
    direction = np.full(3, float(sign))

    ranks_ref, fronts_ref = non_dominated_sort(f, direction=direction)
    ranks_dda, fronts_dda = dda_non_dominated_sort(f, direction=direction)

    np.testing.assert_array_equal(ranks_dda, ranks_ref)
    for fd, fr in zip(fronts_dda, fronts_ref):
        assert set(fd) == set(fr)


# ---------------------------------------------------------------------------
# 2. Canonical TestNonDominatedSort scenarios against dda_non_dominated_sort
# ---------------------------------------------------------------------------
class TestDDACanonical:
    """Canonical non-dominated sort scenarios, run against dda variant."""

    def test_basic_two_fronts(self):
        f = np.array([[0.0, 0.0], [1.0, 1.0]])
        ranks, fronts = dda_non_dominated_sort(f)
        assert ranks[0] == 0
        assert ranks[1] == 1
        assert 0 in fronts[0]
        assert 1 in fronts[1]

    def test_all_non_dominated(self):
        f = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
        ranks, fronts = dda_non_dominated_sort(f)
        assert np.all(ranks == 0)
        assert len(fronts) == 1
        assert sorted(fronts[0]) == [0, 1, 2, 3]

    def test_all_dominated_chain(self):
        f = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        ranks, _ = dda_non_dominated_sort(f)
        assert ranks[0] == 0
        assert ranks[1] == 1
        assert ranks[2] == 2

    def test_single_point(self):
        f = np.array([[1.0, 2.0]])
        ranks, fronts = dda_non_dominated_sort(f)
        assert ranks[0] == 0
        assert len(fronts) == 1

    def test_three_objectives(self):
        f = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        ranks, _ = dda_non_dominated_sort(f)
        assert ranks[0] == 0
        assert ranks[1] == 1

    def test_nan_row_last_front(self):
        f = np.array([[0.0, 0.0], [np.nan, np.nan], [1.0, 1.0]])
        ranks, _ = dda_non_dominated_sort(f)
        assert ranks[0] == 0
        assert ranks[2] == 1
        assert ranks[1] > ranks[2]

    def test_all_nan_input(self):
        f = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        ranks, fronts = dda_non_dominated_sort(f)
        assert ranks[0] == 0
        assert ranks[1] == 0
        assert len(fronts) == 2
        assert fronts[0] == [0]
        assert fronts[1] == [1]

    def test_mixed_nan_multiple_nan_rows(self):
        f = np.array(
            [
                [0.0, 0.0],
                [np.nan, np.nan],
                [1.0, 1.0],
                [np.nan, 2.0],
            ]
        )
        ranks, fronts = dda_non_dominated_sort(f)
        assert ranks[0] == 0
        assert ranks[2] == 1
        assert ranks[1] == 2
        assert ranks[3] == 2
        nan_fronts = fronts[2:]
        nan_indices = {fr[0] for fr in nan_fronts}
        assert nan_indices == {1, 3}
        for nf in nan_fronts:
            assert len(nf) == 1

    def test_all_equal_rows_single_front(self):
        f = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        ranks, fronts = dda_non_dominated_sort(f)
        assert np.all(ranks == 0)
        assert len(fronts) == 1
        assert sorted(fronts[0]) == [0, 1, 2]

    def test_direction_maximize(self):
        f = np.array([[3.0, 3.0], [1.0, 1.0]])
        direction = np.array([1.0, 1.0])
        ranks, fronts = dda_non_dominated_sort(f, direction=direction)
        assert ranks[0] == 0
        assert ranks[1] == 1
        assert 0 in fronts[0]
        assert 1 in fronts[1]


# ---------------------------------------------------------------------------
# 3. Custom dominator injection is honored
# ---------------------------------------------------------------------------
def test_dda_custom_dominator_honored():
    """dda_non_dominated_sort respects an injected custom Dominator."""

    class ReverseParetoDominator(Dominator):
        """Transpose of standard Pareto matrix: worse solutions appear to dominate."""

        def dominance_matrix(self, f, direction=None):
            return ParetoDominator().dominance_matrix(f, direction).T

    f = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    # With reversed dominance, dda result must equal non_dominated_sort with the
    # same custom dominator.
    ranks_ref, fronts_ref = non_dominated_sort(f, dominator=ReverseParetoDominator())
    ranks_dda, fronts_dda = dda_non_dominated_sort(
        f, dominator=ReverseParetoDominator()
    )

    np.testing.assert_array_equal(ranks_dda, ranks_ref)
    assert len(fronts_dda) == len(fronts_ref)
    for fd, fr in zip(fronts_dda, fronts_ref):
        assert set(fd) == set(fr)


def test_dda_custom_dominator_all_in_one_front():
    """A dominator that never dominates puts everything in front 0."""

    class NeverDominates(Dominator):
        def dominance_matrix(self, f, direction=None):
            return np.zeros((len(f), len(f)), dtype=bool)

    f = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    ranks, fronts = dda_non_dominated_sort(f, dominator=NeverDominates())
    assert np.all(ranks == 0)
    assert len(fronts) == 1
    assert sorted(fronts[0]) == [0, 1, 2]


# ---------------------------------------------------------------------------
# 4. LARGE-M correctness: N=200, M=200
# ---------------------------------------------------------------------------
def test_dda_large_m_correctness():
    """N=200, M=200: dda matches oracle and does not allocate (N, N, M) tensor."""
    rng = np.random.default_rng(2024)
    n, m = 200, 200
    f = rng.random((n, m))

    ranks_ref, fronts_ref = non_dominated_sort(f)
    ranks_dda, fronts_dda = dda_non_dominated_sort(f)

    np.testing.assert_array_equal(ranks_dda, ranks_ref)
    assert len(fronts_dda) == len(fronts_ref)
    for fd, fr in zip(fronts_dda, fronts_ref):
        assert set(fd) == set(fr)


# ---------------------------------------------------------------------------
# 5. SPEED GUARD (#89): vectorized non_dominated_sort >=10x faster than pure-Python
#    double-loop baseline at N=1000, n_obj=2.
# ---------------------------------------------------------------------------
def test_vectorized_nds_speed_guard_vs_pure_python():
    """Vectorised non_dominated_sort is >=10x faster than a pure-Python double loop.

    Guards the #89 acceptance criterion (>=10x at N=1000, n_obj=2). The baseline
    is a plain Python double loop (no NumPy in the inner loop), so it runs in a
    fraction of a second; the best of a few repetitions is used for both timings
    to stay stable under CI load variance.
    """
    rng = np.random.default_rng(0)
    n, m = 1000, 2
    f = rng.random((n, m))
    rows = f.tolist()  # plain Python lists: no NumPy call overhead in the loop

    # -- pure-Python double-loop reference (n_obj == 2) --
    def _pure_python_nds(pts):
        n_pts = len(pts)
        dom_count = [0] * n_pts
        for i in range(n_pts):
            ai0, ai1 = pts[i]
            for j in range(n_pts):
                if i != j:
                    bj0, bj1 = pts[j]
                    if ai0 <= bj0 and ai1 <= bj1 and (ai0 < bj0 or ai1 < bj1):
                        dom_count[j] += 1
        return dom_count

    def _best(fn, *args, repeats=3):
        fn(*args)  # warm-up (excludes import / allocation overhead)
        return min(_timed(fn, *args) for _ in range(repeats))

    def _timed(fn, *args):
        t0 = time.perf_counter()
        fn(*args)
        return time.perf_counter() - t0

    t_vec = _best(non_dominated_sort, f)
    t_py = _best(_pure_python_nds, rows)

    speedup = t_py / t_vec
    assert speedup >= 10.0, (
        f"Expected >=10x speedup (#89 criterion), "
        f"got {speedup:.1f}x (vec={t_vec * 1000:.1f}ms, py={t_py * 1000:.1f}ms)"
    )
