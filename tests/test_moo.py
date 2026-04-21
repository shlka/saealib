"""
Tests for multi-objective optimization support.

Tests cover:
- non_dominated_sort: basic ranking, NaN handling, all-non-dominated, all-dominated
- crowding_distance: boundary=inf, interior calculation, 2-point and 1-point edge cases
- crowding_distance_all_fronts: correct per-front assignment
- WeightedSumComparator: sort_population, compare_population, constraint handling
- NSGA2Comparator: sort_population, compare_population, cache, infeasible handling
- Problem: auto-selection of comparator (n_obj=1 / n_obj>1), custom injection
- MOO integration: 2-objective optimization with IndividualBasedStrategy
"""

import logging

import numpy as np

from saealib import (
    GA,
    CrossoverBLXAlpha,
    IndividualBasedStrategy,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    Problem,
    RBFsurrogate,
    SequentialSelection,
    Termination,
    TruncationSelection,
    WeightedSumComparator,
    crowding_distance,
    crowding_distance_all_fronts,
    gaussian_kernel,
    max_fe,
    non_dominated_sort,
)
from saealib.population import Population, PopulationAttribute
from saealib.problem import NSGA2Comparator, ParetoComparator, SingleObjectiveComparator

logging.getLogger("saealib.surrogate.rbf").setLevel(logging.CRITICAL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_pop(f_values: np.ndarray, cv_values: np.ndarray | None = None) -> Population:
    """
    Create a Population with 'f' and 'cv' attributes from arrays.

    Parameters
    ----------
    f_values : np.ndarray
        Objective matrix, shape (n, n_obj).
    cv_values : np.ndarray or None
        Constraint violations, shape (n,). Defaults to zeros.
    """
    n, n_obj = f_values.shape
    if cv_values is None:
        cv_values = np.zeros(n)
    attrs = [
        PopulationAttribute(name="f", dtype=np.float64, shape=(n_obj,)),
        PopulationAttribute(name="cv", dtype=np.float64, shape=(), default=0.0),
    ]
    pop = Population(attrs, init_capacity=n + 1)
    for i in range(n):
        pop.append(f=f_values[i], cv=float(cv_values[i]))
    return pop


# ===========================================================================
# non_dominated_sort Tests
# ===========================================================================
class TestNonDominatedSort:
    """Tests for the non_dominated_sort function."""

    def test_basic_two_fronts(self) -> None:
        """[0,0] dominates [1,1]; they form two separate fronts."""
        f = np.array([[0.0, 0.0], [1.0, 1.0]])
        ranks, fronts = non_dominated_sort(f)
        assert ranks[0] == 0
        assert ranks[1] == 1
        assert 0 in fronts[0]
        assert 1 in fronts[1]

    def test_all_non_dominated(self) -> None:
        """Points on the Pareto front are all in front 0."""
        # f1 trade-off with f2: each point differs on one objective
        f = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
        ranks, fronts = non_dominated_sort(f)
        assert np.all(ranks == 0)
        assert len(fronts) == 1
        assert sorted(fronts[0]) == [0, 1, 2, 3]

    def test_all_dominated_chain(self) -> None:
        """0 dominates 1 dominates 2: three separate fronts."""
        f = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        ranks, _fronts = non_dominated_sort(f)
        assert ranks[0] == 0
        assert ranks[1] == 1
        assert ranks[2] == 2

    def test_ranks_shape(self) -> None:
        rng = np.random.default_rng(0)
        f = rng.random((10, 2))
        ranks, _fronts = non_dominated_sort(f)
        assert ranks.shape == (10,)

    def test_nan_row_last_front(self) -> None:
        """NaN rows are placed in a sentinel front after all valid individuals."""
        f = np.array([[0.0, 0.0], [np.nan, np.nan], [1.0, 1.0]])
        ranks, _fronts = non_dominated_sort(f)
        assert ranks[0] == 0  # best valid point
        assert ranks[2] == 1  # dominated valid point
        # NaN point rank > all valid ranks
        assert ranks[1] > ranks[2]

    def test_single_point(self) -> None:
        f = np.array([[1.0, 2.0]])
        ranks, fronts = non_dominated_sort(f)
        assert ranks[0] == 0
        assert len(fronts) == 1

    def test_three_objectives(self) -> None:
        """Works for n_obj > 2."""
        f = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        ranks, _fronts = non_dominated_sort(f)
        assert ranks[0] == 0
        assert ranks[1] == 1


# ===========================================================================
# crowding_distance Tests
# ===========================================================================
class TestCrowdingDistance:
    """Tests for the crowding_distance function."""

    def test_boundary_points_are_infinite(self) -> None:
        """The two boundary solutions on each objective receive infinite distance."""
        f = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
        cd = crowding_distance(f)
        assert np.isinf(cd[0])
        assert np.isinf(cd[3])

    def test_two_points_both_infinite(self) -> None:
        """With only 2 points, both are boundary points → infinite distance."""
        f = np.array([[0.0, 1.0], [1.0, 0.0]])
        cd = crowding_distance(f)
        assert np.all(np.isinf(cd))

    def test_one_point_infinite(self) -> None:
        """Single-point front gets infinite crowding distance."""
        f = np.array([[1.0, 2.0]])
        cd = crowding_distance(f)
        assert np.isinf(cd[0])

    def test_interior_finite(self) -> None:
        """Interior points (not boundary) have finite crowding distance."""
        f = np.array([[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]])
        cd = crowding_distance(f)
        # interior points: indices 1, 2, 3
        assert np.all(np.isfinite(cd[1:4]))

    def test_output_shape(self) -> None:
        rng = np.random.default_rng(0)
        f = rng.random((6, 2))
        cd = crowding_distance(f)
        assert cd.shape == (6,)

    def test_uniform_spacing_symmetric(self) -> None:
        """Uniformly spaced points have equal interior crowding distances."""
        # 5 points uniformly spaced on the line f1+f2=4, step 1
        f = np.array([[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]])
        cd = crowding_distance(f)
        # all interior points should have the same distance
        interior = cd[1:4]
        assert np.allclose(interior, interior[0])


# ===========================================================================
# crowding_distance_all_fronts Tests
# ===========================================================================
class TestCrowdingDistanceAllFronts:
    """Tests for the crowding_distance_all_fronts function."""

    def test_output_shape(self) -> None:
        f = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        _, fronts = non_dominated_sort(f)
        cd = crowding_distance_all_fronts(f, fronts)
        assert cd.shape == (3,)

    def test_values_match_per_front(self) -> None:
        """Each front's CD should match crowding_distance computed on that front."""
        f = np.array([[0.0, 3.0], [1.5, 1.5], [3.0, 0.0], [2.0, 2.0]])
        _, fronts = non_dominated_sort(f)
        cd_all = crowding_distance_all_fronts(f, fronts)
        for front in fronts:
            if not front:
                continue
            idx = np.array(front)
            cd_front = crowding_distance(f[idx])
            np.testing.assert_array_almost_equal(cd_all[idx], cd_front)

    def test_empty_front_skipped(self) -> None:
        """An empty front list entry should not cause errors."""
        f = np.array([[0.0, 1.0], [1.0, 0.0]])
        fronts = [[0, 1], []]
        cd = crowding_distance_all_fronts(f, fronts)
        assert cd.shape == (2,)


# ===========================================================================
# WeightedSumComparator Tests
# ===========================================================================
class TestWeightedSumComparator:
    """Tests for WeightedSumComparator."""

    def test_sort_population_ascending_weighted_sum(self) -> None:
        """Sort by f @ weights descending (higher weighted sum = better)."""
        # weights = [-1, -1] → score = -(f1+f2); lower sum → higher score
        f = np.array([[1.0, 1.0], [0.5, 0.5], [2.0, 2.0]])
        pop = _make_pop(f)
        comp = WeightedSumComparator(weights=np.array([-1.0, -1.0]))
        order = comp.sort_population(pop)
        # sorted by weighted sum descending: idx=1 (sum=1), idx=0 (sum=2), idx=2 (sum=4)
        assert order[0] == 1
        assert order[2] == 2

    def test_sort_population_with_infeasible(self) -> None:
        """Infeasible (cv > eps) individuals come after all feasible ones."""
        f = np.array([[1.0, 1.0], [0.1, 0.1], [0.5, 0.5]])
        cv = np.array([0.0, 0.5, 0.0])  # idx=1 is infeasible
        pop = _make_pop(f, cv)
        comp = WeightedSumComparator(weights=np.array([-1.0, -1.0]))
        order = comp.sort_population(pop)
        # idx=1 (infeasible) should appear last
        assert order[-1] == 1

    def test_compare_population_a_better(self) -> None:
        f = np.array([[0.5, 0.5], [1.0, 1.0]])
        pop = _make_pop(f)
        comp = WeightedSumComparator(weights=np.array([-1.0, -1.0]))
        result = comp.compare_population(pop, 0, 1)
        assert result == -1  # a has lower sum → better

    def test_compare_population_b_better(self) -> None:
        f = np.array([[2.0, 2.0], [1.0, 1.0]])
        pop = _make_pop(f)
        comp = WeightedSumComparator(weights=np.array([-1.0, -1.0]))
        result = comp.compare_population(pop, 0, 1)
        assert result == 1  # b has lower sum → better

    def test_compare_population_equal(self) -> None:
        f = np.array([[1.0, 1.0], [1.0, 1.0]])
        pop = _make_pop(f)
        comp = WeightedSumComparator(weights=np.array([-1.0, -1.0]))
        result = comp.compare_population(pop, 0, 1)
        assert result == 0

    def test_compare_infeasible_vs_feasible(self) -> None:
        """Feasible solution is always preferred over infeasible."""
        f = np.array([[100.0, 100.0], [0.0, 0.0]])
        cv = np.array([0.5, 0.0])  # idx=0 infeasible, idx=1 feasible
        pop = _make_pop(f, cv)
        comp = WeightedSumComparator(weights=np.array([-1.0, -1.0]))
        result = comp.compare_population(pop, 0, 1)
        assert result == 1  # b (feasible) is better

    def test_single_objective_equivalent(self) -> None:
        """WeightedSumComparator with n_obj=1 behaves like SingleObjectiveComparator."""
        f = np.array([[2.0], [1.0], [3.0]])
        pop = _make_pop(f)
        comp = WeightedSumComparator(weights=np.array([-1.0]))
        order = comp.sort_population(pop)
        # ascending order of f: idx=1(1.0), idx=0(2.0), idx=2(3.0)
        assert order[0] == 1
        assert order[-1] == 2


# ===========================================================================
# ParetoComparator Tests
# ===========================================================================
class TestParetoComparator:
    """Tests for ParetoComparator (non-dominated sorting, no crowding distance)."""

    def test_is_subclass_of_comparator(self) -> None:
        from saealib import Comparator

        assert issubclass(ParetoComparator, Comparator)

    def test_nsga2_is_subclass_of_pareto(self) -> None:
        assert issubclass(NSGA2Comparator, ParetoComparator)

    def test_sort_population_first_front_first(self) -> None:
        """Non-dominated solutions (front 0) appear before dominated ones."""
        f = np.array([[0.0, 3.0], [3.0, 0.0], [2.0, 2.0]])
        pop = _make_pop(f)
        comp = ParetoComparator()
        order = comp.sort_population(pop)
        assert set(order[:2]) == {0, 1}
        assert order[2] == 2

    def test_sort_population_infeasible_last(self) -> None:
        """Infeasible individuals appear after all feasible ones."""
        f = np.array([[10.0, 10.0], [0.0, 0.0], [1.0, 1.0]])
        cv = np.array([1.0, 0.0, 0.0])
        pop = _make_pop(f, cv)
        comp = ParetoComparator()
        order = comp.sort_population(pop)
        assert order[-1] == 0

    def test_sort_population_infeasible_by_ascending_cv(self) -> None:
        """Multiple infeasible individuals are sorted by ascending cv."""
        f = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        cv = np.array([2.0, 0.5, 1.0])
        pop = _make_pop(f, cv)
        comp = ParetoComparator()
        order = comp.sort_population(pop)
        assert list(order) == [1, 2, 0]

    def test_sort_population_not_cached(self) -> None:
        """ParetoComparator does not write to the pareto_sort cache."""
        f = np.array([[0.0, 1.0], [1.0, 0.0]])
        pop = _make_pop(f)
        comp = ParetoComparator()
        comp.sort_population(pop)
        assert pop.get_cache("pareto_sort") is None

    def test_sort_population_nan_last_among_feasible(self) -> None:
        """NaN objective rows are placed last within the feasible block."""
        f = np.array([[0.0, 0.0], [np.nan, np.nan], [1.0, 1.0]])
        pop = _make_pop(f)
        comp = ParetoComparator()
        order = comp.sort_population(pop)
        assert order[0] == 0
        assert order[-1] == 1

    def test_sort_population_output_is_int_array(self) -> None:
        f = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        pop = _make_pop(f)
        comp = ParetoComparator()
        order = comp.sort_population(pop)
        assert np.issubdtype(order.dtype, np.integer)

    def test_compare_population_a_dominates_b(self) -> None:
        f = np.array([[0.0, 0.0], [1.0, 1.0]])
        pop = _make_pop(f)
        comp = ParetoComparator()
        assert comp.compare_population(pop, 0, 1) == -1

    def test_compare_population_b_dominates_a(self) -> None:
        f = np.array([[1.0, 1.0], [0.0, 0.0]])
        pop = _make_pop(f)
        comp = ParetoComparator()
        assert comp.compare_population(pop, 0, 1) == 1

    def test_compare_population_non_dominated(self) -> None:
        f = np.array([[0.0, 1.0], [1.0, 0.0]])
        pop = _make_pop(f)
        comp = ParetoComparator()
        assert comp.compare_population(pop, 0, 1) == 0

    def test_compare_infeasible_vs_feasible(self) -> None:
        """Feasible always beats infeasible regardless of objectives."""
        f = np.array([[0.0, 0.0], [100.0, 100.0]])
        cv = np.array([1.0, 0.0])
        pop = _make_pop(f, cv)
        comp = ParetoComparator()
        assert comp.compare_population(pop, 0, 1) == 1

    def test_compare_both_infeasible_lower_cv_wins(self) -> None:
        f = np.array([[0.0, 0.0], [0.0, 0.0]])
        cv = np.array([0.5, 2.0])
        pop = _make_pop(f, cv)
        comp = ParetoComparator()
        assert comp.compare_population(pop, 0, 1) == -1

    def test_weights_stored_but_not_used(self) -> None:
        """Weights are stored for interface compatibility but ignored."""
        f = np.array([[0.0, 1.0], [1.0, 0.0]])
        pop = _make_pop(f)
        comp_no_w = ParetoComparator()
        comp_with_w = ParetoComparator(weights=np.array([1.0, 2.0]))
        np.testing.assert_array_equal(
            comp_no_w.sort_population(pop),
            comp_with_w.sort_population(pop),
        )


# ===========================================================================
# NSGA2Comparator Tests
# ===========================================================================
class TestNSGA2Comparator:
    """Tests for NSGA2Comparator."""

    def test_sort_population_first_front_first(self) -> None:
        """Non-dominated solutions (front 0) appear before dominated ones."""
        # f[0] and f[1] are non-dominated; f[2] is dominated by both
        f = np.array([[0.0, 3.0], [3.0, 0.0], [2.0, 2.0]])
        pop = _make_pop(f)
        comp = NSGA2Comparator()
        order = comp.sort_population(pop)
        # idx 0 and 1 must appear before idx 2
        assert set(order[:2]) == {0, 1}
        assert order[2] == 2

    def test_sort_population_infeasible_last(self) -> None:
        """Infeasible individuals appear after all feasible ones."""
        f = np.array([[10.0, 10.0], [0.0, 0.0], [1.0, 1.0]])
        cv = np.array([1.0, 0.0, 0.0])  # idx=0 infeasible
        pop = _make_pop(f, cv)
        comp = NSGA2Comparator()
        order = comp.sort_population(pop)
        assert order[-1] == 0

    def test_sort_population_result_cached(self) -> None:
        """Second call returns the cached result without recomputing."""
        f = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        pop = _make_pop(f)
        comp = NSGA2Comparator()
        order1 = comp.sort_population(pop)
        order2 = comp.sort_population(pop)
        assert order1 is order2  # same object from cache

    def test_sort_population_cache_invalidated_on_mutation(self) -> None:
        """Cache is invalidated when the population is modified."""
        f = np.array([[0.0, 1.0], [1.0, 0.0]])
        pop = _make_pop(f)
        comp = NSGA2Comparator()
        order1 = comp.sort_population(pop)
        # mutate population
        pop.f = np.array([[0.5, 0.5], [0.5, 0.5]])
        order2 = comp.sort_population(pop)
        # objects should not be the same (cache was invalidated)
        assert order1 is not order2

    def test_compare_population_a_dominates_b(self) -> None:
        f = np.array([[0.0, 0.0], [1.0, 1.0]])
        pop = _make_pop(f)
        comp = NSGA2Comparator()
        result = comp.compare_population(pop, 0, 1)
        assert result == -1  # a dominates b

    def test_compare_population_b_dominates_a(self) -> None:
        f = np.array([[1.0, 1.0], [0.0, 0.0]])
        pop = _make_pop(f)
        comp = NSGA2Comparator()
        result = comp.compare_population(pop, 0, 1)
        assert result == 1  # b dominates a

    def test_compare_population_non_dominated(self) -> None:
        f = np.array([[0.0, 1.0], [1.0, 0.0]])
        pop = _make_pop(f)
        comp = NSGA2Comparator()
        result = comp.compare_population(pop, 0, 1)
        assert result == 0  # mutually non-dominated

    def test_compare_infeasible_vs_feasible(self) -> None:
        f = np.array([[0.0, 0.0], [0.0, 0.0]])
        cv = np.array([1.0, 0.0])  # a infeasible, b feasible
        pop = _make_pop(f, cv)
        comp = NSGA2Comparator()
        result = comp.compare_population(pop, 0, 1)
        assert result == 1  # b (feasible) is better

    def test_output_is_int_array(self) -> None:
        f = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        pop = _make_pop(f)
        comp = NSGA2Comparator()
        order = comp.sort_population(pop)
        assert order.dtype == np.intp or np.issubdtype(order.dtype, np.integer)

    def test_weights_stored_but_not_used_in_sorting(self) -> None:
        """weights is stored for interface compatibility but ignored in sorting."""
        f = np.array([[0.0, 1.0], [1.0, 0.0]])
        pop = _make_pop(f)
        comp_no_w = NSGA2Comparator()
        comp_with_w = NSGA2Comparator(weights=np.array([1.0, 2.0]))
        order_no_w = comp_no_w.sort_population(pop)
        order_with_w = comp_with_w.sort_population(pop)
        np.testing.assert_array_equal(order_no_w, order_with_w)


# ===========================================================================
# Problem Comparator Auto-selection Tests
# ===========================================================================
class TestProblemComparatorAutoSelection:
    """Tests for Problem's automatic comparator selection."""

    def test_single_objective_uses_single_objective_comparator(self) -> None:
        p = Problem(
            func=lambda x: np.sum(x**2),
            dim=2,
            n_obj=1,
            weight=np.array([-1.0]),
            lb=[-5.0, -5.0],
            ub=[5.0, 5.0],
        )
        assert isinstance(p.comparator, SingleObjectiveComparator)

    def test_multi_objective_uses_pareto_comparator(self) -> None:
        p = Problem(
            func=lambda x: np.array([np.sum(x**2), np.sum((x - 1) ** 2)]),
            dim=2,
            n_obj=2,
            weight=np.array([-1.0, -1.0]),
            lb=[-5.0, -5.0],
            ub=[5.0, 5.0],
        )
        assert isinstance(p.comparator, NSGA2Comparator)

    def test_custom_comparator_overrides_auto_selection(self) -> None:
        custom = WeightedSumComparator(weights=np.array([-1.0, -1.0]))
        p = Problem(
            func=lambda x: np.array([np.sum(x**2), np.sum((x - 1) ** 2)]),
            dim=2,
            n_obj=2,
            weight=np.array([-1.0, -1.0]),
            lb=[-5.0, -5.0],
            ub=[5.0, 5.0],
            comparator=custom,
        )
        assert p.comparator is custom

    def test_problem_evaluate_returns_n_obj_values(self) -> None:
        p = Problem(
            func=lambda x: np.array([np.sum(x**2), np.sum((x - 1) ** 2)]),
            dim=2,
            n_obj=2,
            weight=np.array([-1.0, -1.0]),
            lb=[-5.0, -5.0],
            ub=[5.0, 5.0],
        )
        result = p.evaluate(np.array([0.5, 0.5]))
        assert result.shape == (2,)


# ===========================================================================
# MOO Integration Test
# ===========================================================================
class TestMOOIntegration:
    """2-objective integration test using IndividualBasedStrategy."""

    def test_biobj_optimization_converges(self) -> None:
        """
        2-objective optimization on a simple bi-sphere problem.

        f1(x) = sum(x^2)      (minimize, weight=-1)
        f2(x) = sum((x-2)^2)  (minimize, weight=-1)

        Pareto front: x_i in [0, 2] for each dimension.
        """
        dim = 3
        seed = 42

        def bisphere(x: np.ndarray) -> np.ndarray:
            return np.array([np.sum(x**2), np.sum((x - 2.0) ** 2)])

        problem = Problem(
            func=bisphere,
            dim=dim,
            n_obj=2,
            weight=np.array([-1.0, -1.0]),
            lb=[0.0] * dim,
            ub=[2.0] * dim,
        )
        initializer = LHSInitializer(
            n_init_archive=5 * dim,
            n_init_population=4 * dim,
            seed=seed,
        )
        algorithm = GA(
            crossover=CrossoverBLXAlpha(crossover_rate=0.7, alpha=0.4),
            mutation=MutationUniform(mutation_rate=0.3),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        )
        termination = Termination(max_fe(80 * dim))
        surrogate = RBFsurrogate(gaussian_kernel, dim)
        strategy = IndividualBasedStrategy(evaluation_ratio=0.1)

        opt = (
            Optimizer(problem)
            .set_initializer(initializer)
            .set_algorithm(algorithm)
            .set_termination(termination)
            .set_surrogate(surrogate, n_neighbors=15)
            .set_strategy(strategy)
        )
        ctx = opt.run()

        assert ctx is not None

        # Archive has solutions
        archive_f = ctx.archive.get("f")
        assert archive_f is not None
        assert len(archive_f) > 0
        assert archive_f.shape[1] == 2

        # All objective values are finite
        assert np.all(np.isfinite(archive_f))

        # At least some non-dominated solutions exist (front 0 is non-empty)
        _ranks, fronts = non_dominated_sort(archive_f)
        assert len(fronts[0]) >= 1

        # Best f1 and f2 are reasonable (within the feasible region bounds)
        best_f1 = archive_f[:, 0].min()
        best_f2 = archive_f[:, 1].min()
        assert best_f1 >= 0.0
        assert best_f2 >= 0.0
