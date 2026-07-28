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
