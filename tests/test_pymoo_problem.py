"""Tests for PymooProblem against the real pymoo library."""

from __future__ import annotations

import numpy as np
import pytest
from pymoo.core.problem import Problem as PmProb
from pymoo.problems import get_problem

from saealib import minimize
from saealib.exceptions import ValidationError
from saealib.execution.evaluator import SerialEvaluator
from saealib.problem.constraint import EqualityConstraint, InequalityConstraint
from saealib.problem.pymoo_problem import PymooProblem
from saealib.variables import ContinuousVariable


class _CountedTinyEq(PmProb):
    """Custom pymoo problem with both G and H, counting real _evaluate calls."""

    def __init__(self) -> None:
        super().__init__(
            n_var=2,
            n_obj=1,
            n_ieq_constr=1,
            n_eq_constr=1,
            xl=np.array([-1.0, -1.0]),
            xu=np.array([1.0, 1.0]),
        )
        self.n_evaluate_calls = 0

    def _evaluate(self, x, out, *args, **kwargs):
        self.n_evaluate_calls += 1
        out["F"] = np.sum(x**2, axis=1, keepdims=True)
        out["G"] = (x[:, 0] - 0.5).reshape(-1, 1)
        out["H"] = (x[:, 1] - 0.2).reshape(-1, 1)


class TestPymooProblem:
    def test_dim_bounds_and_objectives_from_zdt1(self):
        zdt1 = get_problem("zdt1")
        problem = PymooProblem(zdt1)
        assert problem.dim == 30
        assert problem.n_obj == 2
        np.testing.assert_array_equal(problem.direction, np.array([-1.0, -1.0]))
        np.testing.assert_allclose(problem.lb, zdt1.xl)
        np.testing.assert_allclose(problem.ub, zdt1.xu)
        assert all(isinstance(v, ContinuousVariable) for v in problem.variables)

    def test_evaluate_matches_pymoo_numerically(self):
        zdt1 = get_problem("zdt1")
        problem = PymooProblem(zdt1)
        rng = np.random.default_rng(0)
        for _ in range(5):
            x = rng.uniform(zdt1.xl, zdt1.xu)
            out = zdt1.evaluate(x[np.newaxis, :], return_as_dictionary=True)
            np.testing.assert_allclose(problem.evaluate(x), out["F"][0])

    def test_constrained_g_matches_verbatim(self):
        bnh = get_problem("bnh")
        problem = PymooProblem(bnh)
        assert problem.n_constraints == bnh.n_ieq_constr
        assert all(isinstance(c, InequalityConstraint) for c in problem.constraints)
        rng = np.random.default_rng(1)
        x = rng.uniform(bnh.xl, bnh.xu)
        g_expected = bnh.evaluate(x[np.newaxis, :], return_as_dictionary=True)["G"][0]
        g_actual, _ = problem.evaluate_constraints(x)
        np.testing.assert_allclose(g_actual, g_expected)

    def test_equality_constraint_is_mapped(self):
        pymoo_problem = _CountedTinyEq()
        problem = PymooProblem(pymoo_problem)
        assert problem.n_constraints == 2
        assert isinstance(problem.constraints[0], InequalityConstraint)
        assert isinstance(problem.constraints[1], EqualityConstraint)

        x = np.array([0.3, 0.1])
        g, _ = problem.evaluate_constraints(x)
        np.testing.assert_allclose(g, [x[0] - 0.5, x[1] - 0.2])

    def test_eval_cache_avoids_redundant_pymoo_calls_row_by_row(self):
        pymoo_problem = _CountedTinyEq()
        problem = PymooProblem(pymoo_problem)
        x = np.random.default_rng(2).uniform(-1.0, 1.0, size=(5, 2))
        # Bypass evaluate_batch entirely, exercising the row-loop path
        # (evaluate_constraints() + evaluate() called back-to-back per row,
        # as SerialEvaluator's row-loop fallback does when evaluate_batch is
        # unavailable).
        for xi in x:
            g_i, _ = problem.evaluate_constraints(xi)
            problem.evaluate(xi, g_i)
        # one evaluate_constraints() (2 constraints) + one evaluate() per row;
        # the cache must collapse this to exactly one real pymoo call per row.
        assert pymoo_problem.n_evaluate_calls == 5

    def test_serial_evaluator_uses_evaluate_batch_single_pymoo_call(self):
        pymoo_problem = _CountedTinyEq()
        problem = PymooProblem(pymoo_problem)
        evaluator = SerialEvaluator()
        x = np.random.default_rng(2).uniform(-1.0, 1.0, size=(5, 2))
        evaluator.evaluate_batch(x, problem)
        # SerialEvaluator prefers PymooProblem.evaluate_batch, which calls
        # the wrapped pymoo problem's own vectorized evaluate() once for the
        # whole batch, collapsing all 5 rows into a single real pymoo call.
        assert pymoo_problem.n_evaluate_calls == 1

    def test_evaluate_batch_single_pymoo_call(self):
        pymoo_problem = _CountedTinyEq()
        problem = PymooProblem(pymoo_problem)
        x = np.random.default_rng(2).uniform(-1.0, 1.0, size=(5, 2))
        problem.evaluate_batch(x)
        assert pymoo_problem.n_evaluate_calls == 1

    def test_evaluate_batch_g_column_order_ieq_then_eq(self):
        pymoo_problem = _CountedTinyEq()
        problem = PymooProblem(pymoo_problem)
        x = np.array([[0.3, 0.1], [-0.2, 0.5]])

        result = problem.evaluate_batch(x)
        assert result is not None
        _, g_batch = result

        assert g_batch.shape == (2, 2)
        np.testing.assert_allclose(g_batch[:, 0], x[:, 0] - 0.5)
        np.testing.assert_allclose(g_batch[:, 1], x[:, 1] - 0.2)

    def test_evaluate_batch_empty_batch(self):
        pymoo_problem = _CountedTinyEq()
        problem = PymooProblem(pymoo_problem)

        result = problem.evaluate_batch(np.empty((0, 2)))
        assert result is not None
        f_batch, g_batch = result

        assert f_batch.shape == (0, 1)
        assert g_batch.shape == (0, 2)

    def test_evaluate_batch_matches_zdt1(self):
        zdt1 = get_problem("zdt1")
        problem = PymooProblem(zdt1)
        rng = np.random.default_rng(3)
        x = rng.uniform(zdt1.xl, zdt1.xu, size=(5, zdt1.n_var))

        result = problem.evaluate_batch(x)
        assert result is not None
        f_batch, g_batch = result

        out = zdt1.evaluate(x, return_as_dictionary=True)
        np.testing.assert_allclose(f_batch, out["F"])
        assert g_batch.shape == (5, 0)

        for i, xi in enumerate(x):
            g_i, _ = problem.evaluate_constraints(xi)
            f_i = problem.evaluate(xi, g_i)
            np.testing.assert_allclose(f_batch[i], f_i)
            np.testing.assert_allclose(g_batch[i], g_i)

    def test_evaluate_batch_matches_bnh(self):
        bnh = get_problem("bnh")
        problem = PymooProblem(bnh)
        rng = np.random.default_rng(4)
        x = rng.uniform(bnh.xl, bnh.xu, size=(5, bnh.n_var))

        result = problem.evaluate_batch(x)
        assert result is not None
        f_batch, g_batch = result

        out = bnh.evaluate(x, return_as_dictionary=True)
        np.testing.assert_allclose(f_batch, out["F"])
        np.testing.assert_allclose(g_batch, out["G"])

        for i, xi in enumerate(x):
            g_i, _ = problem.evaluate_constraints(xi)
            f_i = problem.evaluate(xi, g_i)
            np.testing.assert_allclose(f_batch[i], f_i)
            np.testing.assert_allclose(g_batch[i], g_i)

    def test_missing_bounds_raises_validation_error(self):
        class Unbounded(PmProb):
            def __init__(self):
                super().__init__(n_var=2, n_obj=1)

            def _evaluate(self, x, out, *args, **kwargs):
                out["F"] = np.sum(x**2, axis=1, keepdims=True)

        with pytest.raises(ValidationError):
            PymooProblem(Unbounded())

    def test_end_to_end_minimize(self):
        sphere = get_problem("sphere")
        problem = PymooProblem(sphere)
        result = minimize(
            problem,
            surrogate="rbf",
            max_fe=100,
            pop_size=10,
            seed=0,
            verbose=False,
        )
        assert result.fe > 0
        assert np.isfinite(result.f).all()
