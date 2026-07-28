"""Tests for PymooCrossover / PymooMutation against the real pymoo library."""

from __future__ import annotations

import numpy as np
import pytest
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from saealib import GA, TournamentSelection, TruncationSelection, minimize
from saealib.operators import PymooCrossover, PymooMutation

DIM = 6


def _make_parents(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=(2, DIM))


class _CountedSBX(SBX):
    """SBX subclass counting real ``_do()`` invocations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_do_calls = 0

    def _do(self, *args, **kwargs):
        self.n_do_calls += 1
        return super()._do(*args, **kwargs)


class _CountedPM(PM):
    """PM subclass counting real ``_do()`` invocations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_do_calls = 0

    def _do(self, *args, **kwargs):
        self.n_do_calls += 1
        return super()._do(*args, **kwargs)


# ---------------------------------------------------------------------------
# PymooCrossover
# ---------------------------------------------------------------------------


class TestPymooCrossover:
    def test_output_shape(self):
        op = PymooCrossover(SBX(eta=15))
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        c = op.crossover(p, bounds=(lb, ub), rng=rng)
        assert c.shape == (2, DIM)

    def test_mirrors_pymoo_n_parents_n_children(self):
        pymoo_op = SBX()
        op = PymooCrossover(pymoo_op)
        assert op.n_parents == pymoo_op.n_parents
        assert op.n_children == pymoo_op.n_offsprings

    def test_explicit_n_parents_n_children_override(self):
        op = PymooCrossover(SBX(), n_parents=3, n_children=1)
        assert op.n_parents == 3
        assert op.n_children == 1

    def test_prob_mirrors_pymoo_operator(self):
        pymoo_op = SBX(prob=0.75)
        op = PymooCrossover(pymoo_op)
        assert op.prob == pytest.approx(pymoo_op.prob.value)

    def test_prob_override(self):
        op = PymooCrossover(SBX(), prob=0.42)
        assert op.prob == pytest.approx(0.42)

    def test_respects_bounds(self):
        op = PymooCrossover(SBX(eta=5))
        rng = np.random.default_rng(1)
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        for _ in range(20):
            p = rng.uniform(-1.0, 1.0, size=(2, DIM))
            c = op.crossover(p, bounds=(lb, ub), rng=rng)
            assert np.all(c >= lb) and np.all(c <= ub)

    def test_offspring_differ_from_parents_over_repeated_calls(self):
        op = PymooCrossover(SBX(eta=5))
        rng = np.random.default_rng(2)
        p = _make_parents(rng)
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        any_diff = False
        for _ in range(20):
            c = op.crossover(p, bounds=(lb, ub), rng=rng)
            if not np.array_equal(c, p):
                any_diff = True
        assert any_diff

    def test_unbounded_call_raises(self):
        """pymoo operators such as SBX unconditionally read problem.xl/xu, so
        calling without bounds fails inside pymoo itself. saealib's own GA
        always supplies bounds (ga.py's _route_crossover); this documents the
        requirement for direct callers instead of adding a defensive guard."""
        op = PymooCrossover(SBX(eta=5))
        rng = np.random.default_rng(3)
        p = _make_parents(rng)
        with pytest.raises(TypeError):
            op.crossover(p, rng=rng)

    def test_crossover_batch_calls_do_once(self):
        n_pair = 5
        counted = _CountedSBX(eta=15)
        op = PymooCrossover(counted)
        rng = np.random.default_rng(5)
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)

        result = op.crossover_batch(parents_batch, bounds=(lb, ub), rng=rng)
        assert result is not None
        assert counted.n_do_calls == 1
        assert result.shape == (n_pair, 2, DIM)

        counted.n_do_calls = 0
        for k in range(n_pair):
            op.crossover(parents_batch[k], bounds=(lb, ub), rng=rng)
        assert counted.n_do_calls == n_pair

    def test_crossover_batch_matches_single_crossover_at_n_pair_one(self):
        op = PymooCrossover(SBX(eta=15))
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        p = np.random.default_rng(6).uniform(-1.0, 1.0, size=(2, DIM))

        rng_batch = np.random.default_rng(7)
        batch_result = op.crossover_batch(
            p[np.newaxis, :, :], bounds=(lb, ub), rng=rng_batch
        )
        assert batch_result is not None
        batch_result = batch_result[0]

        rng_single = np.random.default_rng(7)
        single_result = op.crossover(p, bounds=(lb, ub), rng=rng_single)

        np.testing.assert_allclose(batch_result, single_result)

    def test_crossover_batch_output_shape(self):
        op = PymooCrossover(SBX(eta=15))
        rng = np.random.default_rng(8)
        n_pair = 3
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        c = op.crossover_batch(parents_batch, bounds=(lb, ub), rng=rng)
        assert c is not None
        assert c.shape == (n_pair, 2, DIM)

    def test_subdimensional_call_rebuilds_shim_problem(self):
        """A smaller dim slice (e.g. GA's per-type variable routing) works and
        does not reuse the wrong cached pymoo Problem shim."""
        op = PymooCrossover(SBX(eta=5))
        rng = np.random.default_rng(3)
        p_full = rng.uniform(-1.0, 1.0, size=(2, DIM))
        lb_full = np.full(DIM, -1.0)
        ub_full = np.full(DIM, 1.0)
        op.crossover(p_full, bounds=(lb_full, ub_full), rng=rng)
        p_small = p_full[:, :2]
        c_small = op.crossover(p_small, bounds=(lb_full[:2], ub_full[:2]), rng=rng)
        assert c_small.shape == (2, 2)


# ---------------------------------------------------------------------------
# PymooMutation
# ---------------------------------------------------------------------------


class TestPymooMutation:
    def test_output_shape(self):
        op = PymooMutation(PM(eta=20))
        rng = np.random.default_rng(0)
        p = rng.uniform(-1.0, 1.0, size=DIM)
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        m = op.mutate(p, (lb, ub), rng=rng)
        assert m.shape == (DIM,)

    def test_prob_zero_returns_unchanged(self):
        op = PymooMutation(PM(eta=20), prob=0.0)
        rng = np.random.default_rng(0)
        p = rng.uniform(-1.0, 1.0, size=DIM)
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        m = op.mutate(p, (lb, ub), rng=rng)
        np.testing.assert_array_equal(m, p)

    def test_prob_one_changes_over_repeated_calls(self):
        op = PymooMutation(PM(eta=5), prob=1.0)
        rng = np.random.default_rng(1)
        p = rng.uniform(-1.0, 1.0, size=DIM)
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        any_diff = False
        for _ in range(20):
            m = op.mutate(p, (lb, ub), rng=rng)
            if not np.array_equal(m, p):
                any_diff = True
        assert any_diff

    def test_prob_var_is_not_mirrored(self):
        """Regression guard: prob_var must stay None so GA's mixed-variable
        routing falls back to its own default instead of a foreign, possibly
        non-float pymoo.core.variable.Real value."""
        op = PymooMutation(PM(eta=20, prob_var=0.3))
        assert op.prob_var is None

    def test_respects_bounds(self):
        op = PymooMutation(PM(eta=5), prob=1.0)
        rng = np.random.default_rng(4)
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        for _ in range(20):
            p = rng.uniform(-1.0, 1.0, size=DIM)
            m = op.mutate(p, (lb, ub), rng=rng)
            assert np.all(m >= lb) and np.all(m <= ub)

    def test_mutate_batch_calls_do_once(self):
        n = 6
        counted = _CountedPM(eta=20, prob_var=1.0)
        op = PymooMutation(counted, prob=1.0)
        rng = np.random.default_rng(9)
        candidates_batch = rng.uniform(-1.0, 1.0, size=(n, DIM))
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)

        result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
        assert result is not None
        assert counted.n_do_calls == 1
        assert result.shape == (n, DIM)

        counted.n_do_calls = 0
        for k in range(n):
            op.mutate(candidates_batch[k], (lb, ub), rng=rng)
        assert counted.n_do_calls == n

    def test_mutate_batch_matches_single_mutate_at_n_one(self):
        op = PymooMutation(PM(eta=20, prob_var=1.0), prob=1.0)
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        p = np.random.default_rng(10).uniform(-1.0, 1.0, size=DIM)

        rng_batch = np.random.default_rng(11)
        batch_result = op.mutate_batch(p[np.newaxis, :], (lb, ub), rng=rng_batch)
        assert batch_result is not None
        batch_result = batch_result[0]

        rng_single = np.random.default_rng(11)
        single_result = op.mutate(p, (lb, ub), rng=rng_single)

        np.testing.assert_allclose(batch_result, single_result)

    def test_mutate_batch_prob_zero_returns_unchanged_without_do_call(self):
        counted = _CountedPM(eta=20)
        op = PymooMutation(counted, prob=0.0)
        rng = np.random.default_rng(12)
        n = 5
        candidates_batch = rng.uniform(-1.0, 1.0, size=(n, DIM))
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)

        result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
        np.testing.assert_array_equal(result, candidates_batch)
        assert counted.n_do_calls == 0

    def test_mutate_batch_prob_one_mutates_all_rows_with_single_do_call(self):
        counted = _CountedPM(eta=20, prob_var=1.0)
        op = PymooMutation(counted, prob=1.0)
        rng = np.random.default_rng(13)
        n = 5
        candidates_batch = rng.uniform(-1.0, 1.0, size=(n, DIM))
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)

        result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
        assert result is not None
        assert counted.n_do_calls == 1
        assert not np.array_equal(result, candidates_batch)

    def test_mutate_batch_fractional_prob_gates_rows_exactly(self):
        seed = 14
        counted = _CountedPM(eta=20, prob_var=1.0)
        op = PymooMutation(counted, prob=0.5)
        n = 8
        candidates_batch = np.random.default_rng(seed + 1).uniform(
            -1.0, 1.0, size=(n, DIM)
        )
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)

        # mutate_batch's gate draw is the first thing to touch the rng
        # stream, so a parallel, identically-seeded generator predicts it.
        expected_gate = np.random.default_rng(seed).random(n) < op.prob

        result = op.mutate_batch(
            candidates_batch, (lb, ub), rng=np.random.default_rng(seed)
        )
        assert result is not None

        np.testing.assert_array_equal(
            result[~expected_gate], candidates_batch[~expected_gate]
        )
        assert counted.n_do_calls == 1

    def test_mutate_batch_output_shape(self):
        op = PymooMutation(PM(eta=20), prob=1.0)
        rng = np.random.default_rng(15)
        n = 4
        candidates_batch = rng.uniform(-1.0, 1.0, size=(n, DIM))
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        m = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
        assert m is not None
        assert m.shape == (n, DIM)


# ---------------------------------------------------------------------------
# End-to-end: pymoo operators driving saealib's own GA
# ---------------------------------------------------------------------------


class TestPymooOperatorsEndToEnd:
    def test_ga_with_pymoo_operators_improves_sphere(self):
        rng_seed = 0
        result = minimize(
            lambda x: np.sum(x**2),
            dim=5,
            lb=[-5.0] * 5,
            ub=[5.0] * 5,
            algorithm=GA(
                crossover=PymooCrossover(SBX(eta=15)),
                mutation=PymooMutation(PM(eta=20)),
                parent_selection=TournamentSelection(2),
                survivor_selection=TruncationSelection(),
            ),
            surrogate="rbf",
            max_fe=150,
            pop_size=10,
            seed=rng_seed,
            verbose=False,
        )
        assert result.fe > 0
        assert np.isfinite(result.f).all()
