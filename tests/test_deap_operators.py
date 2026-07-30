"""Tests for DeapCrossover / DeapMutation against the real DEAP library."""

from __future__ import annotations

import random

import numpy as np
import pytest
from deap import base, tools

from saealib import GA, TournamentSelection, TruncationSelection, minimize
from saealib.exceptions import ValidationError
from saealib.operators import DeapCrossover, DeapMutation

DIM = 6


def _make_parents(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=(2, DIM))


class _CountedOperator:
    """Wraps a DEAP callable, counting how many times it is actually invoked."""

    def __init__(self, func):
        self._func = func
        self.n_calls = 0

    def __call__(self, *args, **kwargs):
        self.n_calls += 1
        return self._func(*args, **kwargs)


def _sbx_operator(lb: np.ndarray, ub: np.ndarray, eta: float = 15.0):
    toolbox = base.Toolbox()
    toolbox.register(
        "mate",
        tools.cxSimulatedBinaryBounded,
        eta=eta,
        low=lb.tolist(),
        up=ub.tolist(),
    )
    return toolbox.mate


def _pm_operator(lb: np.ndarray, ub: np.ndarray, eta: float = 20.0, indpb: float = 1.0):
    toolbox = base.Toolbox()
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        eta=eta,
        low=lb.tolist(),
        up=ub.tolist(),
        indpb=indpb,
    )
    return toolbox.mutate


# ---------------------------------------------------------------------------
# DeapCrossover
# ---------------------------------------------------------------------------


class TestDeapCrossover:
    def test_output_shape(self):
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        op = DeapCrossover(_sbx_operator(lb, ub))
        rng = np.random.default_rng(0)
        p = _make_parents(rng)
        c = op.crossover(p, bounds=(lb, ub), rng=rng)
        assert c.shape == (2, DIM)

    def test_default_prob_is_one(self):
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        op = DeapCrossover(_sbx_operator(lb, ub))
        assert op.prob == pytest.approx(1.0)

    def test_prob_override(self):
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        op = DeapCrossover(_sbx_operator(lb, ub), prob=0.42)
        assert op.prob == pytest.approx(0.42)

    def test_respects_bounds(self):
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        op = DeapCrossover(_sbx_operator(lb, ub, eta=5.0))
        rng = np.random.default_rng(1)
        for _ in range(20):
            p = rng.uniform(-1.0, 1.0, size=(2, DIM))
            c = op.crossover(p, bounds=(lb, ub), rng=rng)
            assert np.all(c >= lb) and np.all(c <= ub)

    def test_offspring_differ_from_parents_over_repeated_calls(self):
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        op = DeapCrossover(_sbx_operator(lb, ub, eta=5.0))
        rng = np.random.default_rng(2)
        p = _make_parents(rng)
        any_diff = False
        for _ in range(20):
            c = op.crossover(p, bounds=(lb, ub), rng=rng)
            if not np.array_equal(c, p):
                any_diff = True
        assert any_diff

    def test_crossover_batch_output_shape(self):
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        op = DeapCrossover(_sbx_operator(lb, ub))
        rng = np.random.default_rng(8)
        n_pair = 3
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        c = op.crossover_batch(parents_batch, bounds=(lb, ub), rng=rng)
        assert c.shape == (n_pair, 2, DIM)

    def test_crossover_batch_calls_operator_once_per_pair(self):
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        counted = _CountedOperator(_sbx_operator(lb, ub, eta=15.0))
        op = DeapCrossover(counted)
        rng = np.random.default_rng(5)
        n_pair = 5
        parents_batch = rng.uniform(-1.0, 1.0, size=(n_pair, 2, DIM))
        result = op.crossover_batch(parents_batch, bounds=(lb, ub), rng=rng)
        assert counted.n_calls == n_pair
        assert result.shape == (n_pair, 2, DIM)

    def test_crossover_batch_empty_returns_empty_without_calling_operator(self):
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        counted = _CountedOperator(_sbx_operator(lb, ub))
        op = DeapCrossover(counted)
        rng = np.random.default_rng(6)
        parents_batch = np.empty((0, 2, DIM))
        result = op.crossover_batch(parents_batch, bounds=(lb, ub), rng=rng)
        assert result.shape == (0, 2, DIM)
        assert counted.n_calls == 0

    def test_crossover_batch_matches_single_crossover_at_n_pair_one(self):
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        op = DeapCrossover(_sbx_operator(lb, ub, eta=15.0))
        p = np.random.default_rng(6).uniform(-1.0, 1.0, size=(2, DIM))

        rng_batch = np.random.default_rng(7)
        batch_result = op.crossover_batch(
            p[np.newaxis, :, :], bounds=(lb, ub), rng=rng_batch
        )[0]

        rng_single = np.random.default_rng(7)
        single_result = op.crossover(p, bounds=(lb, ub), rng=rng_single)

        np.testing.assert_allclose(batch_result, single_result)

    def test_batch_rows_get_distinct_random_streams(self):
        """One seeded ``random`` state spans the whole batch, so identical
        parent pairs must not all come out mutated identically -- a
        regression guard against reseeding once per row instead of once per
        batch call."""
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        op = DeapCrossover(_sbx_operator(lb, ub, eta=5.0))
        rng = np.random.default_rng(21)
        pair = rng.uniform(-1.0, 1.0, size=(2, DIM))
        parents_batch = np.tile(pair, (4, 1, 1))
        result = op.crossover_batch(parents_batch, bounds=(lb, ub), rng=rng)
        first = result[0]
        assert any(not np.allclose(first, result[k]) for k in range(1, 4))

    def test_copy_safety_original_parents_untouched(self):
        """DEAP's cxSimulatedBinaryBounded mutates its ind1/ind2 arguments in
        place; the adapter must not let that leak into the caller's array."""
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        op = DeapCrossover(_sbx_operator(lb, ub, eta=5.0))
        rng = np.random.default_rng(9)
        parents_batch = rng.uniform(-1.0, 1.0, size=(3, 2, DIM))
        snapshot = parents_batch.copy()
        op.crossover_batch(parents_batch, bounds=(lb, ub), rng=rng)
        np.testing.assert_array_equal(parents_batch, snapshot)

    @pytest.mark.parametrize(
        "bad_operator",
        [
            lambda ind1, ind2: (ind1,),  # wrong arity: 1-tuple instead of 2
            lambda ind1, ind2: [ind1, ind2],  # not a tuple at all
            lambda ind1, ind2: (ind1[:-1], ind2),  # wrong shape
            lambda ind1, ind2: (["a"] * len(ind1), ind2),  # not float-coercible
        ],
    )
    def test_malformed_return_raises_validation_error(self, bad_operator):
        op = DeapCrossover(bad_operator)
        rng = np.random.default_rng(10)
        p = _make_parents(rng)
        with pytest.raises(ValidationError):
            op.crossover(p, rng=rng)

    def test_rng_state_restored_after_normal_call(self):
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        op = DeapCrossover(_sbx_operator(lb, ub))
        random.seed(1234)
        pre_state = random.getstate()
        rng = np.random.default_rng(11)
        p = _make_parents(rng)
        op.crossover(p, bounds=(lb, ub), rng=rng)
        assert random.getstate() == pre_state

    def test_rng_state_restored_after_operator_raises(self):
        def _raising_operator(ind1, ind2):
            raise RuntimeError("boom")

        op = DeapCrossover(_raising_operator)
        random.seed(5678)
        pre_state = random.getstate()
        rng = np.random.default_rng(12)
        p = _make_parents(rng)
        with pytest.raises(RuntimeError, match="boom"):
            op.crossover(p, rng=rng)
        assert random.getstate() == pre_state


# ---------------------------------------------------------------------------
# DeapMutation
# ---------------------------------------------------------------------------


class TestDeapMutation:
    def test_output_shape(self):
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        op = DeapMutation(_pm_operator(lb, ub))
        rng = np.random.default_rng(0)
        p = rng.uniform(-1.0, 1.0, size=DIM)
        m = op.mutate(p, (lb, ub), rng=rng)
        assert m.shape == (DIM,)

    def test_prob_zero_returns_unchanged_without_operator_call(self):
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        counted = _CountedOperator(_pm_operator(lb, ub))
        op = DeapMutation(counted, prob=0.0)
        rng = np.random.default_rng(0)
        p = rng.uniform(-1.0, 1.0, size=DIM)
        m = op.mutate(p, (lb, ub), rng=rng)
        np.testing.assert_array_equal(m, p)
        assert counted.n_calls == 0

    def test_prob_one_changes_over_repeated_calls(self):
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        op = DeapMutation(_pm_operator(lb, ub, eta=5.0), prob=1.0)
        rng = np.random.default_rng(1)
        p = rng.uniform(-1.0, 1.0, size=DIM)
        any_diff = False
        for _ in range(20):
            m = op.mutate(p, (lb, ub), rng=rng)
            if not np.array_equal(m, p):
                any_diff = True
        assert any_diff

    def test_prob_var_is_not_mirrored(self):
        """Regression guard: prob_var must stay None so GA's mixed-variable
        routing falls back to its own default instead of a foreign indpb."""
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        op = DeapMutation(_pm_operator(lb, ub, indpb=0.3))
        assert op.prob_var is None

    def test_respects_bounds(self):
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        op = DeapMutation(_pm_operator(lb, ub, eta=5.0), prob=1.0)
        rng = np.random.default_rng(4)
        for _ in range(20):
            p = rng.uniform(-1.0, 1.0, size=DIM)
            m = op.mutate(p, (lb, ub), rng=rng)
            assert np.all(m >= lb) and np.all(m <= ub)

    def test_mutate_batch_calls_operator_once_per_gated_row(self):
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        counted = _CountedOperator(_pm_operator(lb, ub, eta=20.0))
        op = DeapMutation(counted, prob=1.0)
        rng = np.random.default_rng(9)
        n = 6
        candidates_batch = rng.uniform(-1.0, 1.0, size=(n, DIM))
        result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
        assert counted.n_calls == n
        assert result.shape == (n, DIM)

    def test_batch_rows_get_distinct_random_streams(self):
        """One seeded ``random`` state spans the whole gated subset, so
        identical candidate rows must not all come out mutated identically
        -- a regression guard against reseeding once per row instead of once
        per batch call."""
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        op = DeapMutation(_pm_operator(lb, ub, eta=5.0, indpb=1.0), prob=1.0)
        rng = np.random.default_rng(22)
        row = rng.uniform(-1.0, 1.0, size=DIM)
        candidates_batch = np.tile(row, (4, 1))
        result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
        first = result[0]
        assert any(not np.allclose(first, result[k]) for k in range(1, 4))

    def test_mutate_derives_single_row_batch(self):
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        op = DeapMutation(_pm_operator(lb, ub, eta=20.0), prob=1.0)
        assert "mutate" not in vars(DeapMutation)

        p = np.random.default_rng(10).uniform(-1.0, 1.0, size=DIM)

        rng_batch = np.random.default_rng(11)
        batch_result = op.mutate_batch(p[np.newaxis, :], (lb, ub), rng=rng_batch)[0]

        rng_single = np.random.default_rng(11)
        single_result = op.mutate(p, (lb, ub), rng=rng_single)

        np.testing.assert_allclose(batch_result, single_result)

    def test_mutate_batch_prob_zero_returns_unchanged_without_operator_call(self):
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        counted = _CountedOperator(_pm_operator(lb, ub))
        op = DeapMutation(counted, prob=0.0)
        rng = np.random.default_rng(12)
        n = 5
        candidates_batch = rng.uniform(-1.0, 1.0, size=(n, DIM))
        result = op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
        np.testing.assert_array_equal(result, candidates_batch)
        assert counted.n_calls == 0

    def test_mutate_batch_fractional_prob_gates_rows_exactly(self):
        seed = 14
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        counted = _CountedOperator(_pm_operator(lb, ub, eta=20.0))
        op = DeapMutation(counted, prob=0.5)
        n = 8
        candidates_batch = np.random.default_rng(seed + 1).uniform(
            -1.0, 1.0, size=(n, DIM)
        )

        # mutate_batch's gate draw is the first thing to touch the rng
        # stream, so a parallel, identically-seeded generator predicts it.
        expected_gate = np.random.default_rng(seed).random(n) < op.prob

        result = op.mutate_batch(
            candidates_batch, (lb, ub), rng=np.random.default_rng(seed)
        )

        np.testing.assert_array_equal(
            result[~expected_gate], candidates_batch[~expected_gate]
        )
        assert counted.n_calls == int(expected_gate.sum())

    def test_copy_safety_original_candidates_untouched(self):
        """DEAP's mutPolynomialBounded mutates its individual argument in
        place; the adapter must not let that leak into the caller's array."""
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        op = DeapMutation(_pm_operator(lb, ub, eta=5.0), prob=1.0)
        rng = np.random.default_rng(9)
        candidates_batch = rng.uniform(-1.0, 1.0, size=(4, DIM))
        snapshot = candidates_batch.copy()
        op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
        np.testing.assert_array_equal(candidates_batch, snapshot)

    @pytest.mark.parametrize(
        "bad_operator",
        [
            lambda individual: (individual, individual),  # wrong arity
            lambda individual: [individual],  # not a tuple at all
            lambda individual: (individual[:-1],),  # wrong shape
            lambda individual: (["a"] * len(individual),),  # not float-coercible
        ],
    )
    def test_malformed_return_raises_validation_error(self, bad_operator):
        op = DeapMutation(bad_operator, prob=1.0)
        rng = np.random.default_rng(10)
        p = rng.uniform(-1.0, 1.0, size=DIM)
        with pytest.raises(ValidationError):
            op.mutate(p, (np.full(DIM, -1.0), np.full(DIM, 1.0)), rng=rng)

    def test_rng_state_restored_after_normal_call(self):
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        op = DeapMutation(_pm_operator(lb, ub), prob=1.0)
        random.seed(1234)
        pre_state = random.getstate()
        rng = np.random.default_rng(11)
        p = rng.uniform(-1.0, 1.0, size=DIM)
        op.mutate(p, (lb, ub), rng=rng)
        assert random.getstate() == pre_state

    def test_rng_state_restored_after_operator_raises(self):
        def _raising_operator(individual):
            raise RuntimeError("boom")

        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        op = DeapMutation(_raising_operator, prob=1.0)
        random.seed(5678)
        pre_state = random.getstate()
        rng = np.random.default_rng(12)
        p = rng.uniform(-1.0, 1.0, size=DIM)
        with pytest.raises(RuntimeError, match="boom"):
            op.mutate(p, (lb, ub), rng=rng)
        assert random.getstate() == pre_state

    def test_rng_state_untouched_when_no_row_gated(self):
        lb = np.full(DIM, -1.0)
        ub = np.full(DIM, 1.0)
        op = DeapMutation(_pm_operator(lb, ub), prob=0.0)
        random.seed(999)
        pre_state = random.getstate()
        rng = np.random.default_rng(13)
        candidates_batch = rng.uniform(-1.0, 1.0, size=(5, DIM))
        op.mutate_batch(candidates_batch, (lb, ub), rng=rng)
        assert random.getstate() == pre_state


# ---------------------------------------------------------------------------
# End-to-end: DEAP operators driving saealib's own GA
# ---------------------------------------------------------------------------


class TestDeapOperatorsEndToEnd:
    def test_ga_with_deap_operators_improves_sphere(self):
        dim = 5
        lb = np.full(dim, -5.0)
        ub = np.full(dim, 5.0)
        rng_seed = 0
        result = minimize(
            lambda x: np.sum(x**2),
            dim=dim,
            lb=lb.tolist(),
            ub=ub.tolist(),
            algorithm=GA(
                crossover=DeapCrossover(_sbx_operator(lb, ub, eta=15.0)),
                mutation=DeapMutation(_pm_operator(lb, ub, eta=20.0, indpb=1.0 / dim)),
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
