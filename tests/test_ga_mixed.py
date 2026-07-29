"""Tests for GA with mixed-variable routing."""

from __future__ import annotations

import numpy as np
import pytest

# tests/ has no __init__.py, so pytest's default (prepend) import mode puts
# tests/ itself on sys.path -- this makes test_operators.py importable as a
# top-level module from any other file under tests/, the same way it is
# from within test_operators.py.
from test_operators import _MutationUnbatched

from saealib import (
    GA,
    CategoricalVariable,
    ConfigurationError,
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
from saealib.operators.crossover import Crossover
from saealib.operators.mutation import Mutation


def _make_ga(**kwargs):
    return GA(
        crossover=CrossoverSBX(1.0, eta=20.0),
        mutation=MutationPolynomial(eta=20.0, prob_var=0.1),
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
# Stub crossovers with non-default n_children / n_parents
# ---------------------------------------------------------------------------


class _CrossoverN1C(Crossover):
    """n_children=1 stub."""

    n_children = 1

    def __init__(self):
        self.prob = 1.0

    def crossover(self, parent, bounds=None, rng=np.random.default_rng()):
        return parent[:1].copy()


class _CrossoverP3(Crossover):
    """n_parents=3 stub."""

    n_parents = 3

    def __init__(self):
        self.prob = 1.0

    def crossover(self, parent, bounds=None, rng=np.random.default_rng()):
        return parent[:2].copy()


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
        assert ga.integer_crossover.prob == pytest.approx(1.0)

    def test_default_categorical_crossover_rate_inherits(self):
        ga = _make_ga()
        assert ga.categorical_crossover.prob == pytest.approx(1.0)

    def test_default_integer_mutation_rate_inherits(self):
        ga = _make_ga()
        assert ga.integer_mutation.prob_var == pytest.approx(0.1)

    def test_default_categorical_mutation_rate_inherits(self):
        ga = _make_ga()
        assert ga.categorical_mutation.prob_var == pytest.approx(0.1)

    def test_custom_integer_crossover(self):
        custom = CrossoverIntegerSBX(0.5, eta=5.0)
        ga = _make_ga(integer_crossover=custom)
        assert ga.integer_crossover is custom

    def test_custom_categorical_mutation(self):
        custom = MutationCategorical(0.3)
        ga = _make_ga(categorical_mutation=custom)
        assert ga.categorical_mutation is custom

    def test_integer_crossover_n_children_mismatch_raises(self):
        with pytest.raises(ConfigurationError, match=r"integer_crossover\.n_children"):
            _make_ga(integer_crossover=_CrossoverN1C())

    def test_categorical_crossover_n_children_mismatch_raises(self):
        with pytest.raises(
            ConfigurationError, match=r"categorical_crossover\.n_children"
        ):
            _make_ga(categorical_crossover=_CrossoverN1C())

    def test_integer_crossover_n_parents_mismatch_raises(self):
        with pytest.raises(ConfigurationError, match=r"integer_crossover\.n_parents"):
            _make_ga(integer_crossover=_CrossoverP3())

    def test_categorical_crossover_n_parents_mismatch_raises(self):
        with pytest.raises(
            ConfigurationError, match=r"categorical_crossover\.n_parents"
        ):
            _make_ga(categorical_crossover=_CrossoverP3())


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
            parent,
            problem.lb,
            problem.ub,
            rng1,
            problem,
            self.cont_op,
            self.int_op,
            self.cat_op,
        )
        expected = self.cont_op.crossover(parent, (problem.lb, problem.ub), rng=rng2)
        np.testing.assert_array_equal(result, expected)

    def test_mixed_output_shape(self):
        problem = _make_problem_mixed()
        parent = np.array([[0.5, 3.0, 1.0], [0.8, 7.0, 2.0]])
        result = _route_crossover(
            parent,
            problem.lb,
            problem.ub,
            self.rng,
            problem,
            self.cont_op,
            self.int_op,
            self.cat_op,
        )
        assert result.shape == (2, 3)

    def test_integer_dims_are_rounded(self):
        problem = _make_problem_mixed()
        rng = np.random.default_rng(0)
        for _ in range(20):
            parent = np.array([[0.5, 3.0, 1.0], [0.8, 7.0, 2.0]])
            result = _route_crossover(
                parent,
                problem.lb,
                problem.ub,
                rng,
                problem,
                self.cont_op,
                self.int_op,
                self.cat_op,
            )
            assert result[0, 1] == round(result[0, 1])
            assert result[1, 1] == round(result[1, 1])

    def test_categorical_dims_take_parent_value(self):
        problem = _make_problem_mixed()
        rng = np.random.default_rng(0)
        for _ in range(20):
            parent = np.array([[0.5, 3.0, 0.0], [0.8, 7.0, 2.0]])
            result = _route_crossover(
                parent,
                problem.lb,
                problem.ub,
                rng,
                problem,
                self.cont_op,
                self.int_op,
                self.cat_op,
            )
            assert result[0, 2] in {0.0, 2.0}
            assert result[1, 2] in {0.0, 2.0}

    def test_continuous_dims_not_integer(self):
        problem = _make_problem_mixed()
        rng = np.random.default_rng(1)
        parent = np.array([[0.1, 3.0, 1.0], [0.9, 7.0, 0.0]])
        results = [
            _route_crossover(
                parent,
                problem.lb,
                problem.ub,
                rng,
                problem,
                self.cont_op,
                self.int_op,
                self.cat_op,
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
    from saealib.context import OptimizationState
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

    return OptimizationState(
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

    def test_ask_raises_when_primary_n_children_mutated(self):
        """Mutating crossover.n_children after init must raise ConfigurationError."""
        problem = _make_problem_mixed()
        ga = _make_ga()
        # Bypass __init__ validation by directly changing the attribute.
        ga.crossover.n_children = 1
        ctx = _make_ctx_for(problem)
        with pytest.raises(ConfigurationError, match=r"integer_crossover\.n_children"):
            ga.ask(ctx, _NoopProvider())

    def test_ask_raises_when_integer_crossover_replaced_with_mismatched(self):
        """Replacing integer_crossover with a different n_children must raise."""
        problem = _make_problem_mixed()
        ga = _make_ga()
        # _CrossoverN1C has n_children=1; crossover has n_children=2.
        ga.integer_crossover = _CrossoverN1C()
        ctx = _make_ctx_for(problem)
        with pytest.raises(ConfigurationError, match=r"integer_crossover\.n_children"):
            ga.ask(ctx, _NoopProvider())


# ---------------------------------------------------------------------------
# _mutate_candidates fallback-loop interleaving order (Issue #224, commit 6
# review fix)
# ---------------------------------------------------------------------------


class TestMutateCandidatesInterleaveOrder:
    """``_mutate_candidates``'s per-individual fallback loop (taken whenever
    ``self.mutation`` does not override ``mutate_batch``) must call
    ``_route_mutation`` and ``post_mutation`` interleaved per individual —
    i.e. ``mutate(0), post(0), mutate(1), post(1), ...`` — matching the
    pre-batching behaviour byte-for-byte. A prior version of this method
    ran the full ``_route_mutation`` loop first and only then a separate
    ``post_mutation`` loop, which is invisible for the default (identity,
    RNG-free) ``post_mutation`` hook but silently reorders RNG consumption
    for a ``with_post`` hook that draws from ``rng``. This test uses such a
    hook and cross-checks the library's output against an independently
    hand-written interleaved reference loop fed an identically-seeded RNG.

    Uses ``_MutationUnbatched`` (imported from ``tests/test_operators.py``)
    as the test vehicle rather than any real, evolving built-in
    ``Mutation`` class: MutationUniform (commit 9), MutationPolynomial
    (commit 10), and MutationGaussian (commit 11) all now override
    ``mutate_batch``, and each one gaining that override in turn broke this
    test's premise the moment it did. A dedicated test-local dummy that will
    never grow a ``mutate_batch`` override closes that hole permanently.
    ``_MutationUnbatched.mutate()`` still draws exactly one ``rng.random()``
    value per call (see its docstring) -- without that draw, this test would
    pass regardless of whether ``_mutate_candidates`` actually interleaves
    correctly, since a zero-draw ``mutate()`` produces byte-identical RNG
    consumption under both the correct interleaved order and the buggy
    two-pass order this test guards against.
    """

    def test_fallback_loop_matches_hand_written_interleaved_reference(self):
        def rng_consuming_hook(offspring, mutate_range, rng, ctx):
            # Visibly perturbs the output using a value drawn from rng, so
            # that a reordering of RNG draws changes the returned array.
            return offspring + rng.random() * 1e-3

        mutation = _MutationUnbatched(prob=0.5).with_post(rng_consuming_hook)
        # Sanity check: mirrors the production predicate in
        # _mutate_candidates that decides the fallback path is taken.
        # (Checking `type(mutation).mutate_batch is _MutationUnbatched
        # .mutate_batch` would be vacuous here, since with_post always
        # returns a shallow copy of the same type regardless of whether
        # that type overrides mutate_batch.)
        assert type(mutation).mutate_batch is Mutation.mutate_batch

        ga = GA(
            crossover=CrossoverSBX(1.0, eta=20.0),
            mutation=mutation,
            parent_selection=TournamentSelection(2),
            survivor_selection=TruncationSelection(),
        )
        problem = _make_problem_continuous()
        lb, ub = problem.lb, problem.ub
        cand = np.random.default_rng(1).uniform(lb, ub, size=(6, problem.dim))

        ctx = _make_ctx_for(problem)
        ctx.rng = np.random.default_rng(2024)
        actual = ga._mutate_candidates(ctx, cand.copy(), lb, ub, mixed=False)

        ref_rng = np.random.default_rng(2024)
        expected = cand.copy()
        for i in range(len(expected)):
            expected[i] = _route_mutation(
                expected[i],
                lb,
                ub,
                ref_rng,
                problem,
                mutation,
                ga.integer_mutation,
                ga.categorical_mutation,
            )
            expected[i] = mutation.post_mutation(expected[i], (lb, ub), ref_rng, ctx)

        np.testing.assert_allclose(actual, expected)

        # Diagnostic-and-guard: recompute the *buggy* two-pass order (all
        # _route_mutation calls first, then all post_mutation calls) from
        # the same seed. If _mutate_candidates ever regressed to that
        # ordering, `actual` would match `wrong_order` instead of
        # `expected` -- this assertion is what would catch it, and is also
        # the empirical proof that _MutationUnbatched's one-draw mutate()
        # keeps this test capable of discriminating between the two orders
        # (a zero-draw mutate() would make wrong_order == expected too).
        wrong_rng = np.random.default_rng(2024)
        wrong_order = cand.copy()
        for i in range(len(wrong_order)):
            wrong_order[i] = _route_mutation(
                wrong_order[i],
                lb,
                ub,
                wrong_rng,
                problem,
                mutation,
                ga.integer_mutation,
                ga.categorical_mutation,
            )
        for i in range(len(wrong_order)):
            wrong_order[i] = mutation.post_mutation(
                wrong_order[i], (lb, ub), wrong_rng, ctx
            )
        assert not np.allclose(actual, wrong_order)


# ---------------------------------------------------------------------------
# GA batch-dispatch consistency regression (Issue #224 follow-up fix):
# subclassing a batch-capable built-in operator and overriding only the
# scalar method must not fall through to the inherited (stale) batch method.
# ---------------------------------------------------------------------------


class _CustomSBX(CrossoverSBX):
    """Overrides only scalar crossover() with a no-op (returns the first
    n_children parents unchanged); crossover_batch is inherited from
    CrossoverSBX and would silently produce real SBX offspring instead if
    GA dispatched to it directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def crossover(self, parent, bounds=None, rng=np.random.default_rng()):
        self.calls += 1
        return parent[: self.n_children].copy()


class _CustomMutation(MutationPolynomial):
    """Overrides only scalar mutate() with a no-op; mutate_batch is
    inherited from MutationPolynomial and would silently apply real
    polynomial mutation instead if GA dispatched to it directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def mutate(self, p, mutate_range, rng=np.random.default_rng()):
        self.calls += 1
        return p.copy()


class TestGABatchDispatchConsistency:
    def test_custom_crossover_scalar_override_is_actually_invoked(self):
        custom_crossover = _CustomSBX(1.0, eta=20.0)
        ga = GA(
            crossover=custom_crossover,
            mutation=MutationPolynomial(prob=0.0, eta=20.0, prob_var=0.0),
            parent_selection=TournamentSelection(2),
            survivor_selection=TruncationSelection(),
        )
        problem = _make_problem_continuous()
        ctx = _make_ctx_for(problem, n_pop=6)
        n_offspring = 8
        offspring = ga.ask(ctx, _NoopProvider(), n_offspring=n_offspring)

        # The scalar override must actually have been invoked -- if GA
        # instead dispatched to the inherited (stale) crossover_batch, this
        # counter would stay at 0.
        assert custom_crossover.calls > 0

        # crossover() is a no-op copy of the parents, so with mutation
        # disabled (prob=0.0) every offspring row must exactly equal some
        # row drawn from the initial population -- real SBX offspring
        # (produced by the inherited crossover_batch) would essentially
        # never coincide exactly with a parent, for continuous-valued data.
        x = offspring.get_array("x")
        pop_x = ctx.population.get_array("x")
        for row in x:
            assert np.any(np.all(np.isclose(pop_x, row), axis=1))

    def test_custom_mutation_scalar_override_is_actually_invoked(self):
        custom_mutation = _CustomMutation(prob=1.0, eta=20.0, prob_var=1.0)
        ga = GA(
            # crossover disabled (prob=0.0) so offspring going into mutation
            # are exact copies of parents, isolating the mutation dispatch.
            crossover=CrossoverSBX(0.0, eta=20.0),
            mutation=custom_mutation,
            parent_selection=TournamentSelection(2),
            survivor_selection=TruncationSelection(),
        )
        problem = _make_problem_continuous()
        ctx = _make_ctx_for(problem, n_pop=6)
        n_offspring = 8
        offspring = ga.ask(ctx, _NoopProvider(), n_offspring=n_offspring)

        # The scalar override must actually have been invoked -- if GA
        # instead dispatched to the inherited (stale) mutate_batch, this
        # counter would stay at 0.
        assert custom_mutation.calls > 0

        # mutate() is a no-op copy, and crossover is disabled, so every
        # offspring row must exactly equal some row from the initial
        # population -- real polynomial mutation (produced by the inherited
        # mutate_batch, with prob_var=1.0 perturbing every dimension) would
        # essentially never coincide exactly with a parent.
        x = offspring.get_array("x")
        pop_x = ctx.population.get_array("x")
        for row in x:
            assert np.any(np.all(np.isclose(pop_x, row), axis=1))
