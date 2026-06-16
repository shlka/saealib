"""Tests for GA with mixed-variable routing."""

from __future__ import annotations

import numpy as np
import pytest

from saealib import (
    GA,
    CategoricalVariable,
    ContinuousVariable,
    IntegerVariable,
    Problem,
)
from saealib.algorithms.ga import _route_crossover, _route_mutation
from saealib.operators import (
    CrossoverCategorical,
    CrossoverIntegerSBX,
    CrossoverSBX,
    MutationCategorical,
    MutationIntegerUniform,
    MutationPolynomial,
    TournamentSelection,
    TruncationSelection,
)


def _make_ga(**kwargs):
    return GA(
        crossover=CrossoverSBX(1.0, eta=20.0),
        mutation=MutationPolynomial(0.1, eta=20.0),
        parent_selection=TournamentSelection(2),
        survivor_selection=TruncationSelection(),
        **kwargs,
    )


def _make_problem_mixed():
    variables = [
        ContinuousVariable(0.0, 1.0),
        IntegerVariable(0, 9),
        CategoricalVariable(["a", "b", "c"]),
    ]
    return Problem(
        func=lambda x: np.array([x[0]]),
        dim=3,
        n_obj=1,
        direction=np.array([-1.0]),
        variables=variables,
    )


def _make_problem_continuous():
    return Problem(
        func=lambda x: np.array([x[0]]),
        dim=3,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[0.0, 0.0, 0.0],
        ub=[1.0, 1.0, 1.0],
    )


# ---------------------------------------------------------------------------
# GA constructor defaults
# ---------------------------------------------------------------------------


class TestGADefaults:
    def test_default_integer_crossover_type(self):
        ga = _make_ga()
        assert isinstance(ga.integer_crossover, CrossoverIntegerSBX)

    def test_default_integer_mutation_type(self):
        ga = _make_ga()
        assert isinstance(ga.integer_mutation, MutationIntegerUniform)

    def test_default_categorical_crossover_type(self):
        ga = _make_ga()
        assert isinstance(ga.categorical_crossover, CrossoverCategorical)

    def test_default_categorical_mutation_type(self):
        ga = _make_ga()
        assert isinstance(ga.categorical_mutation, MutationCategorical)

    def test_default_integer_crossover_rate_inherits(self):
        ga = _make_ga()
        assert ga.integer_crossover.crossover_rate == pytest.approx(1.0)

    def test_default_categorical_crossover_rate_inherits(self):
        ga = _make_ga()
        assert ga.categorical_crossover.crossover_rate == pytest.approx(1.0)

    def test_default_integer_mutation_rate_inherits(self):
        ga = _make_ga()
        assert ga.integer_mutation.mutation_rate == pytest.approx(0.1)

    def test_default_categorical_mutation_rate_inherits(self):
        ga = _make_ga()
        assert ga.categorical_mutation.mutation_rate == pytest.approx(0.1)

    def test_custom_integer_crossover(self):
        custom = CrossoverIntegerSBX(0.5, eta=5.0)
        ga = _make_ga(integer_crossover=custom)
        assert ga.integer_crossover is custom

    def test_custom_categorical_mutation(self):
        custom = MutationCategorical(0.3)
        ga = _make_ga(categorical_mutation=custom)
        assert ga.categorical_mutation is custom


# ---------------------------------------------------------------------------
# _route_crossover
# ---------------------------------------------------------------------------


class TestRouteCrossover:
    def setup_method(self):
        self.rng = np.random.default_rng(0)
        self.cont_op = CrossoverSBX(1.0, eta=20.0)
        self.int_op = CrossoverIntegerSBX(1.0, eta=20.0)
        self.cat_op = CrossoverCategorical(1.0)

    def test_fast_path_all_continuous(self):
        problem = _make_problem_continuous()
        parent = np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        result = _route_crossover(
            parent, rng1, problem, self.cont_op, self.int_op, self.cat_op
        )
        expected = self.cont_op.crossover(parent, rng=rng2)
        np.testing.assert_array_equal(result, expected)

    def test_mixed_output_shape(self):
        problem = _make_problem_mixed()
        parent = np.array([[0.5, 3.0, 1.0], [0.8, 7.0, 2.0]])
        result = _route_crossover(
            parent, self.rng, problem, self.cont_op, self.int_op, self.cat_op
        )
        assert result.shape == (2, 3)

    def test_integer_dims_are_rounded(self):
        problem = _make_problem_mixed()
        rng = np.random.default_rng(0)
        for _ in range(20):
            parent = np.array([[0.5, 3.0, 1.0], [0.8, 7.0, 2.0]])
            result = _route_crossover(
                parent, rng, problem, self.cont_op, self.int_op, self.cat_op
            )
            assert result[0, 1] == round(result[0, 1])
            assert result[1, 1] == round(result[1, 1])

    def test_categorical_dims_take_parent_value(self):
        problem = _make_problem_mixed()
        rng = np.random.default_rng(0)
        for _ in range(20):
            parent = np.array([[0.5, 3.0, 0.0], [0.8, 7.0, 2.0]])
            result = _route_crossover(
                parent, rng, problem, self.cont_op, self.int_op, self.cat_op
            )
            assert result[0, 2] in {0.0, 2.0}
            assert result[1, 2] in {0.0, 2.0}

    def test_continuous_dims_not_integer(self):
        problem = _make_problem_mixed()
        rng = np.random.default_rng(1)
        parent = np.array([[0.1, 3.0, 1.0], [0.9, 7.0, 0.0]])
        results = [
            _route_crossover(
                parent, rng, problem, self.cont_op, self.int_op, self.cat_op
            )
            for _ in range(30)
        ]
        cont_vals = np.array([r[:, 0] for r in results]).ravel()
        assert not np.all(cont_vals == np.round(cont_vals))


# ---------------------------------------------------------------------------
# _route_mutation
# ---------------------------------------------------------------------------


class TestRouteMutation:
    def setup_method(self):
        self.rng = np.random.default_rng(0)
        self.cont_op = MutationPolynomial(1.0, eta=20.0)
        self.int_op = MutationIntegerUniform(1.0)
        self.cat_op = MutationCategorical(1.0)

    def test_fast_path_all_continuous(self):
        problem = _make_problem_continuous()
        p = np.array([0.1, 0.5, 0.9])
        lb = problem.lb
        ub = problem.ub
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        result = _route_mutation(
            p, lb, ub, rng1, problem, self.cont_op, self.int_op, self.cat_op
        )
        expected = self.cont_op.mutate(p, (lb, ub), rng=rng2)
        np.testing.assert_array_equal(result, expected)

    def test_mixed_output_shape(self):
        problem = _make_problem_mixed()
        p = np.array([0.5, 3.0, 1.0])
        result = _route_mutation(
            p,
            problem.lb,
            problem.ub,
            self.rng,
            problem,
            self.cont_op,
            self.int_op,
            self.cat_op,
        )
        assert result.shape == (3,)

    def test_integer_dim_rounded(self):
        problem = _make_problem_mixed()
        rng = np.random.default_rng(0)
        for _ in range(20):
            p = np.array([0.5, 3.0, 1.0])
            result = _route_mutation(
                p,
                problem.lb,
                problem.ub,
                rng,
                problem,
                self.cont_op,
                self.int_op,
                self.cat_op,
            )
            assert result[1] == round(result[1])

    def test_categorical_dim_valid_index(self):
        problem = _make_problem_mixed()
        rng = np.random.default_rng(0)
        for _ in range(20):
            p = np.array([0.5, 3.0, 1.0])
            result = _route_mutation(
                p,
                problem.lb,
                problem.ub,
                rng,
                problem,
                self.cont_op,
                self.int_op,
                self.cat_op,
            )
            assert result[2] in {0.0, 1.0, 2.0}


# ---------------------------------------------------------------------------
# GA.ask() end-to-end with mixed problem
# ---------------------------------------------------------------------------


class _NoopProvider:
    """Minimal provider that silently discards dispatched events."""

    def dispatch(self, event):
        pass


def _make_ctx_for(problem, n_pop=4, seed=42):
    from saealib.context import OptimizationContext
    from saealib.population import (
        Archive,
        ParetoArchive,
        Population,
        PopulationAttribute,
    )

    dim = problem.dim
    n_obj = problem.n_obj
    attrs = [
        PopulationAttribute(name="x", dtype=np.float64, shape=(dim,)),
        PopulationAttribute(name="f", dtype=np.float64, shape=(n_obj,)),
        PopulationAttribute(name="g", dtype=np.float64, shape=(0,)),
        PopulationAttribute(name="cv", dtype=np.float64, shape=()),
    ]
    rng = np.random.default_rng(seed)
    pop = Population(attrs, init_capacity=n_pop + 2)
    arc = Archive(attrs, init_capacity=n_pop + 2)
    pareto_arc = ParetoArchive(
        attrs, init_capacity=n_pop + 2, direction=problem.direction
    )

    xs = problem.repair(rng.uniform(problem.lb, problem.ub, size=(n_pop, dim)))
    fs = np.zeros((n_pop, n_obj))
    pop.extend({"x": xs, "f": fs, "g": np.zeros((n_pop, 0)), "cv": np.zeros(n_pop)})
    arc.extend({"x": xs, "f": fs, "g": np.zeros((n_pop, 0)), "cv": np.zeros(n_pop)})

    return OptimizationContext(
        problem=problem,
        population=pop,
        archive=arc,
        pareto_archive=pareto_arc,
        rng=np.random.default_rng(seed + 1),
    )


class TestGAMixedAsk:
    def test_ask_returns_correct_shape(self):
        problem = _make_problem_mixed()
        ga = _make_ga()
        ctx = _make_ctx_for(problem)
        offspring = ga.ask(ctx, _NoopProvider(), n_offspring=4)
        assert offspring.get_array("x").shape == (4, 3)

    def test_ask_integer_dims_are_rounded(self):
        problem = _make_problem_mixed()
        ga = _make_ga()
        ctx = _make_ctx_for(problem, n_pop=8)
        offspring = ga.ask(ctx, _NoopProvider(), n_offspring=10)
        x = offspring.get_array("x")
        np.testing.assert_array_equal(x[:, 1], np.round(x[:, 1]))

    def test_ask_categorical_dims_valid(self):
        problem = _make_problem_mixed()
        ga = _make_ga()
        ctx = _make_ctx_for(problem, n_pop=8)
        offspring = ga.ask(ctx, _NoopProvider(), n_offspring=10)
        x = offspring.get_array("x")
        assert np.all((x[:, 2] >= 0) & (x[:, 2] <= 2))
        np.testing.assert_array_equal(x[:, 2], np.round(x[:, 2]))

    def test_ask_bounds_respected(self):
        problem = _make_problem_mixed()
        ga = _make_ga()
        ctx = _make_ctx_for(problem, n_pop=8)
        offspring = ga.ask(ctx, _NoopProvider(), n_offspring=20)
        x = offspring.get_array("x")
        assert np.all(x >= problem.lb)
        assert np.all(x <= problem.ub)

    def test_ask_continuous_problem_shape(self):
        """All-continuous problem fast path returns correct shape."""
        problem = _make_problem_continuous()
        ga = _make_ga()
        ctx = _make_ctx_for(problem)
        offspring = ga.ask(ctx, _NoopProvider(), n_offspring=4)
        assert offspring.get_array("x").shape == (4, 3)
