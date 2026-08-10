"""
Tests for the Evaluator abstraction.

Tests cover:
- SerialEvaluator: batch shapes for f / g / cv
- Equivalence with per-candidate Problem.evaluate / evaluate_constraints
- No-constraint problems (g shape (n, 0), cv all zeros)
- Single-row input handling
- Optimizer wiring: default evaluator and set_evaluator chaining
- SerialEvaluator's Problem.evaluate_batch fast path: call
  counting, numerical equivalence with the row-loop fallback, and the
  empty-batch edge case
"""

from typing import Any, cast

import numpy as np
import pytest

from saealib import EvaluationResult, Evaluator, Optimizer, SerialEvaluator
from saealib.exceptions import EvaluationProtocolError, ValidationError
from saealib.execution.evaluator import (
    EvaluationRequest,
    EvaluationStatus,
    EvaluationUpdate,
    JoblibEvaluator,
    PendingEvaluation,
)
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


class _CountedBatchSphereProblem(Problem):
    """
    Sphere ``Problem`` overriding ``evaluate_batch`` to score the whole batch
    in one call, counting calls to ``evaluate_batch``, ``evaluate``, and
    ``evaluate_constraints`` so tests can assert the fast path avoids the
    row-by-row ones.
    """

    def __init__(self, constraints=None):
        super().__init__(
            func=lambda x: np.sum(x**2),
            dim=2,
            n_obj=1,
            direction=np.array([-1.0]),
            lb=[-5.0, -5.0],
            ub=[5.0, 5.0],
            constraints=constraints,
        )
        self.n_evaluate_batch_calls = 0
        self.n_evaluate_calls = 0
        self.n_evaluate_constraints_calls = 0

    def evaluate_batch(self, x):
        self.n_evaluate_batch_calls += 1
        n = len(x)
        n_c = len(self.constraints)
        f_raw = np.sum(x**2, axis=1, keepdims=True)
        if n_c and n:
            g_raw = np.array(
                [[c.func(xi) for c in self.constraints] for xi in x], dtype=float
            )
        else:
            g_raw = np.empty((n, n_c), dtype=float)
        return f_raw, g_raw

    def evaluate(self, x, g=None):
        self.n_evaluate_calls += 1
        return super().evaluate(x, g)

    def evaluate_constraints(self, x):
        self.n_evaluate_constraints_calls += 1
        return super().evaluate_constraints(x)


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


class TestEvaluatorBoundaryValidation:
    def test_evaluation_result_rejects_misaligned_optional_channels(self):
        with pytest.raises(ValidationError, match="channel lengths"):
            EvaluationResult(np.ones((2, 1)), np.ones((1, 0)), np.zeros(2))
        with pytest.raises(ValidationError, match="candidate_ids length"):
            EvaluationResult(
                np.ones((2, 1)),
                np.ones((2, 0)),
                np.zeros(2),
                candidate_ids=np.array([1], dtype=np.int64),
            )
        with pytest.raises(ValidationError, match="candidate_ids must be unique"):
            EvaluationResult(
                np.ones((2, 1)),
                np.ones((2, 0)),
                np.zeros(2),
                candidate_ids=np.array([1, 1], dtype=np.int64),
            )
        with pytest.raises(ValidationError, match="cost length"):
            EvaluationResult(
                np.ones((2, 1)),
                np.ones((2, 0)),
                np.zeros(2),
                cost=np.ones(1),
            )
        with pytest.raises(ValidationError, match="noise must have"):
            EvaluationResult(
                np.ones((2, 1)),
                np.ones((2, 0)),
                np.zeros(2),
                noise=np.ones((2, 2)),
            )
        with pytest.raises(ValidationError, match="outputs"):
            EvaluationResult(
                np.ones((2, 1)),
                np.ones((2, 0)),
                np.zeros(2),
                outputs={"bad": np.ones((1, 1), dtype=np.float64)},
            )

    def test_request_update_and_pending_records_reject_invalid_identity(self):
        with pytest.raises(ValidationError, match="request_id"):
            EvaluationRequest(
                cast(Any, np.array([1, 2])),
                np.array([1, 2]),
                np.ones((2, 1)),
            )
        with pytest.raises(ValidationError, match="unique and match"):
            EvaluationRequest(
                np.int64(1), np.array([1, 1], dtype=np.int64), np.ones((2, 1))
            )
        request = EvaluationRequest(
            np.int64(1), np.array([1, 2], dtype=np.int64), np.ones((2, 1))
        )
        with pytest.raises(EvaluationProtocolError, match="non-negative"):
            EvaluationUpdate(
                request.request_id,
                EvaluationStatus.COMPLETED,
                np.array([], dtype=np.int64),
                sequence=-1,
            )
        with pytest.raises(EvaluationProtocolError, match="must be unique"):
            EvaluationUpdate(
                request.request_id,
                EvaluationStatus.PARTIAL,
                np.array([1, 1], dtype=np.int64),
            )
        with pytest.raises(EvaluationProtocolError, match="candidate_ids"):
            EvaluationUpdate(
                request.request_id,
                EvaluationStatus.COMPLETED,
                np.array([1], dtype=np.int64),
                EvaluationResult(np.ones((1, 1)), np.ones((1, 0)), np.zeros(1)),
            )
        with pytest.raises(ValidationError, match="original_candidate_ids"):
            PendingEvaluation(
                request,
                EvaluationStatus.PENDING,
                np.array([], dtype=np.int64),
                original_candidate_ids=np.array([1, 1], dtype=np.int64),
            )
        with pytest.raises(ValidationError, match="reserved_cost"):
            PendingEvaluation(
                request,
                EvaluationStatus.PENDING,
                np.array([], dtype=np.int64),
                reserved_cost=-1.0,
            )
        with pytest.raises(ValidationError, match="retry_count"):
            PendingEvaluation(
                request,
                EvaluationStatus.PENDING,
                np.array([], dtype=np.int64),
                retry_count=-1,
            )


class TestSerialEvaluatorBatchHook:
    """SerialEvaluator's fast path via Problem.evaluate_batch."""

    def test_evaluate_batch_called_once_no_row_calls(self):
        p = _CountedBatchSphereProblem(
            constraints=[InequalityConstraint(lambda x: x[0] - 1.0)]
        )
        x = np.array([[0.0, 0.0], [2.0, 0.0], [-3.0, 1.0]])
        SerialEvaluator().evaluate_batch(x, p)
        assert p.n_evaluate_batch_calls == 1
        assert p.n_evaluate_calls == 0
        assert p.n_evaluate_constraints_calls == 0

    def test_values_match_row_loop_fallback_no_constraints(self):
        x = np.array([[0.0, 0.0], [2.0, 0.0], [-3.0, 1.0]])
        hooked = SerialEvaluator().evaluate_batch(x, _CountedBatchSphereProblem())
        fallback = SerialEvaluator().evaluate_batch(x, _sphere_problem())
        np.testing.assert_allclose(hooked.f, fallback.f)
        np.testing.assert_allclose(hooked.g, fallback.g)
        np.testing.assert_allclose(hooked.cv, fallback.cv)

    def test_values_match_row_loop_fallback_with_constraints(self):
        x = np.array([[0.0, 0.0], [2.0, 0.0], [-3.0, 1.0]])
        hooked = SerialEvaluator().evaluate_batch(
            x,
            _CountedBatchSphereProblem(
                constraints=[InequalityConstraint(lambda x: x[0] - 1.0)]
            ),
        )
        fallback = SerialEvaluator().evaluate_batch(
            x, _sphere_problem(constraints=[InequalityConstraint(lambda x: x[0] - 1.0)])
        )
        np.testing.assert_allclose(hooked.f, fallback.f)
        np.testing.assert_allclose(hooked.g, fallback.g)
        np.testing.assert_allclose(hooked.cv, fallback.cv)

    def test_empty_batch_no_constraints(self):
        p = _CountedBatchSphereProblem()
        result = SerialEvaluator().evaluate_batch(np.empty((0, 2)), p)
        assert result.f.shape == (0, 1)
        assert result.g.shape == (0, 0)
        assert result.cv.shape == (0,)

    def test_empty_batch_with_constraints(self):
        p = _CountedBatchSphereProblem(
            constraints=[InequalityConstraint(lambda x: x[0] - 1.0)]
        )
        result = SerialEvaluator().evaluate_batch(np.empty((0, 2)), p)
        assert result.f.shape == (0, 1)
        assert result.g.shape == (0, 1)
        assert result.cv.shape == (0,)


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
