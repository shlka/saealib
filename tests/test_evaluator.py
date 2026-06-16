"""
Tests for the Evaluator abstraction (Issue #85, Commit 1).

Tests cover:
- SerialEvaluator: batch shapes for f / g / cv
- Equivalence with per-candidate Problem.evaluate / evaluate_constraints
- No-constraint problems (g shape (n, 0), cv all zeros)
- Single-row input handling
- Optimizer wiring: default evaluator and set_evaluator chaining
"""

import numpy as np
import pytest

from saealib import EvaluationResult, Evaluator, Optimizer, SerialEvaluator
from saealib.execution.evaluator import JoblibEvaluator
from saealib.problem import InequalityConstraint, Problem


def _sphere_problem(constraints=None):
    return Problem(
        func=lambda x: np.sum(x**2),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0, -5.0],
        ub=[5.0, 5.0],
        constraints=constraints,
    )


class TestSerialEvaluator:
    def test_is_evaluator_subclass(self):
        assert issubclass(SerialEvaluator, Evaluator)

    def test_batch_shapes_with_constraints(self):
        p = _sphere_problem(constraints=[InequalityConstraint(lambda x: x[0] - 1.0)])
        x = np.array([[0.0, 0.0], [2.0, 0.0]])
        result = SerialEvaluator().evaluate_batch(x, p)
        assert isinstance(result, EvaluationResult)
        assert result.f.shape == (2, 1)
        assert result.g.shape == (2, 1)
        assert result.cv.shape == (2,)

    def test_values_match_per_candidate(self):
        p = _sphere_problem(constraints=[InequalityConstraint(lambda x: x[0] - 1.0)])
        x = np.array([[0.0, 0.0], [2.0, 0.0], [-3.0, 1.0]])
        result = SerialEvaluator().evaluate_batch(x, p)
        for i, xi in enumerate(x):
            assert result.f[i] == pytest.approx(p.evaluate(xi))
            g_i, cv_i = p.evaluate_constraints(xi)
            assert result.g[i] == pytest.approx(g_i)
            assert result.cv[i] == pytest.approx(cv_i)

    def test_cv_aggregation(self):
        p = _sphere_problem(constraints=[InequalityConstraint(lambda x: x[0] - 1.0)])
        x = np.array([[0.0, 0.0], [2.0, 0.0]])
        result = SerialEvaluator().evaluate_batch(x, p)
        assert result.cv == pytest.approx([0.0, 1.0])

    def test_no_constraints(self):
        p = _sphere_problem()
        x = np.array([[0.0, 0.0], [2.0, 0.0]])
        result = SerialEvaluator().evaluate_batch(x, p)
        assert result.g.shape == (2, 0)
        assert np.all(result.cv == 0.0)

    def test_single_row_input(self):
        p = _sphere_problem()
        result = SerialEvaluator().evaluate_batch(np.array([[3.0, 4.0]]), p)
        assert result.f.shape == (1, 1)
        assert result.f[0, 0] == pytest.approx(25.0)


joblib = pytest.importorskip("joblib", reason="joblib not installed")


class TestJoblibEvaluator:
    def test_import_error_raised_at_init(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "joblib":
                raise ImportError("joblib not found")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="saealib\\[parallel\\]"):
            JoblibEvaluator()

    def test_is_evaluator_subclass(self):
        assert issubclass(JoblibEvaluator, Evaluator)

    def test_default_properties(self):
        ev = JoblibEvaluator()
        assert ev.n_jobs == -1
        assert ev.backend == "loky"

    def test_custom_properties(self):
        ev = JoblibEvaluator(n_jobs=2, backend="threading")
        assert ev.n_jobs == 2
        assert ev.backend == "threading"

    def test_results_match_serial(self):
        p = _sphere_problem(constraints=[InequalityConstraint(lambda x: x[0] - 1.0)])
        x = np.array([[0.0, 0.0], [2.0, 0.0], [-3.0, 1.0]])
        serial = SerialEvaluator().evaluate_batch(x, p)
        parallel = JoblibEvaluator(n_jobs=2, backend="loky").evaluate_batch(x, p)
        assert parallel.f == pytest.approx(serial.f)
        assert parallel.g == pytest.approx(serial.g)
        assert parallel.cv == pytest.approx(serial.cv)

    def test_no_constraints_shape(self):
        p = _sphere_problem()
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = JoblibEvaluator(n_jobs=2, backend="loky").evaluate_batch(x, p)
        assert result.f.shape == (2, 1)
        assert result.g.shape == (2, 0)
        assert result.cv.shape == (2,)
        assert np.all(result.cv == 0.0)

    def test_threading_backend(self):
        p = _sphere_problem()
        x = np.array([[1.0, 0.0], [0.0, 1.0]])
        serial = SerialEvaluator().evaluate_batch(x, p)
        parallel = JoblibEvaluator(n_jobs=2, backend="threading").evaluate_batch(x, p)
        assert parallel.f == pytest.approx(serial.f)

    def test_result_is_evaluation_result(self):
        p = _sphere_problem()
        x = np.array([[1.0, 1.0]])
        result = JoblibEvaluator(n_jobs=1, backend="loky").evaluate_batch(x, p)
        assert isinstance(result, EvaluationResult)

    def test_lazy_import_accessible_via_saealib(self):
        import saealib

        assert saealib.JoblibEvaluator is JoblibEvaluator


class TestOptimizerWiring:
    def test_default_evaluator_is_serial(self):
        opt = Optimizer(_sphere_problem())
        assert isinstance(opt.evaluator, SerialEvaluator)

    def test_set_evaluator_chains(self):
        opt = Optimizer(_sphere_problem())
        ev = SerialEvaluator()
        assert opt.set_evaluator(ev) is opt
        assert opt.evaluator is ev
