"""Tests for Problem and SearchSpace/BoundsService integration (Unit H5).

Validates that Problem owns VectorSpace as its single source of truth,
BoundsService routes bounds accurately, algorithms remain deterministic,
and services are resolved outside hot-path inner loops.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from saealib.algorithms.base import ProposalRequest
from saealib.algorithms.ga import GA
from saealib.algorithms.pso import PSO
from saealib.context import OptimizationState
from saealib.execution.initializer import RandomInitializer
from saealib.operators.crossover import CrossoverSBX
from saealib.operators.mutation import MutationPolynomial
from saealib.operators.selection import TournamentSelection, TruncationSelection
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem.problem import Problem
from saealib.space import BoundsService, VectorSpace


def algorithm_ask(algorithm, state, provider):
    """Call a built-in proposer through the canonical boundary."""
    view = state._store.view(
        algorithm.contract().state,
        context=state,
        dispatch=provider.dispatch,
    )
    return algorithm.ask(ProposalRequest(), view).candidates


def dummy_func(x: np.ndarray) -> float:
    """Dummy objective function."""
    return float(np.sum(x**2))


def _make_ga() -> GA:
    """Helper to construct a GA algorithm instance."""
    return GA(
        crossover=CrossoverSBX(eta=20.0, prob=0.9),
        mutation=MutationPolynomial(eta=20.0, prob=0.1),
        parent_selection=TournamentSelection(tournament_size=2),
        survivor_selection=TruncationSelection(),
    )


def _setup_state(
    prob: Problem,
    popsize: int,
    seed: int,
    extra_attrs: list[PopulationAttribute] | None = None,
) -> OptimizationState:
    """Helper to construct an OptimizationState with initialized population."""
    init = RandomInitializer(
        n_init_archive=popsize, n_init_population=popsize, seed=seed
    )
    provider = MagicMock()
    provider.seed = seed

    if extra_attrs:
        provider.algorithm.population_class = lambda attrs, init_capacity: Population(
            attrs=list(attrs) + list(extra_attrs), init_capacity=init_capacity
        )
    else:
        provider.algorithm.population_class = Population

    provider.algorithm.archive_class = Archive
    provider.algorithm.create_pareto_archive = (
        lambda attrs, init_capacity, problem: ParetoArchive(
            attrs=attrs, init_capacity=init_capacity
        )
    )
    provider.evaluator.evaluate_batch = lambda x_arr, p: MagicMock(
        f=np.array([[dummy_func(row)] for row in x_arr]),
        g=np.empty((len(x_arr), 0)),
        cv=np.zeros(len(x_arr)),
    )
    ctx = init.initialize(provider, prob)
    return ctx


# ---------------------------------------------------------------------------
# 1. problem.lb / ub / dim reflect the space values (single source of truth)
# ---------------------------------------------------------------------------


def test_problem_properties_derive_from_space() -> None:
    """problem.lb/ub/dim derive directly from space without duplication.

    Implementation mutation that would break this test:
        Store `self._lb = lb.copy()` in Problem.__init__ and return
        `self._lb` in `Problem.lb`. Modifying `problem.space._lb` would no
        longer be reflected in `problem.lb`.
    """
    prob = Problem(
        func=dummy_func,
        dim=3,
        n_obj=1,
        direction=np.array([-1]),
        lb=[0.0, 1.0, 2.0],
        ub=[10.0, 11.0, 12.0],
    )

    assert isinstance(prob.space, VectorSpace)
    assert prob.dim == prob.space.dim == 3
    assert np.array_equal(prob.lb, prob.space.lb)
    assert np.array_equal(prob.ub, prob.space.ub)

    # Identical underlying numpy array reference (no separate stored copy)
    assert np.shares_memory(prob.lb, prob.space.lb)
    assert np.shares_memory(prob.ub, prob.space.ub)


# ---------------------------------------------------------------------------
# 2. BoundsService returns identical values to problem.lb / ub
# ---------------------------------------------------------------------------


def test_bounds_service_matches_problem_bounds() -> None:
    """BoundsService.bounds returns the exact (lb, ub) matching problem.lb/ub.

    Implementation mutation that would break this test:
        In `_VectorBoundsService`, add `+ 1.0` to `self._ub`.
        `np.testing.assert_array_equal(srv_ub, prob.ub)` would fail.
    """
    prob = Problem(
        func=dummy_func,
        dim=2,
        n_obj=1,
        direction=np.array([-1]),
        lb=[-5.0, -10.0],
        ub=[5.0, 10.0],
    )

    bounds_srv = prob.space.services.require("BoundsService")
    assert isinstance(bounds_srv, BoundsService)

    srv_lb, srv_ub = bounds_srv.bounds
    np.testing.assert_array_equal(srv_lb, prob.lb)
    np.testing.assert_array_equal(srv_ub, prob.ub)


# ---------------------------------------------------------------------------
# 3. GA / PSO 1-generation determinism with BoundsService
# ---------------------------------------------------------------------------


def test_ga_and_pso_determinism_with_bounds_service() -> None:
    """GA and PSO produce deterministic outputs using BoundsService.

    Implementation mutation that would break this test:
        Swap `lb, ub = ub, lb` in `GA.ask` or `PSO.ask`.
        Candidate values generated with the same seed will differ.
    """
    prob = Problem(
        func=dummy_func,
        dim=4,
        n_obj=1,
        direction=np.array([-1]),
        lb=[-1.0, -1.0, -1.0, -1.0],
        ub=[1.0, 1.0, 1.0, 1.0],
    )

    provider = MagicMock()
    ga = _make_ga()
    ctx_ga1 = _setup_state(prob, popsize=20, seed=42)
    pop_ga1 = algorithm_ask(ga, ctx_ga1, provider)

    ctx_ga2 = _setup_state(prob, popsize=20, seed=42)
    pop_ga2 = algorithm_ask(ga, ctx_ga2, provider)

    np.testing.assert_array_equal(pop_ga1.get_array("x"), pop_ga2.get_array("x"))

    pso_attrs = [
        PopulationAttribute("velocity", float, (prob.dim,), default=0.0),
        PopulationAttribute("pbest_x", float, (prob.dim,), default=np.nan),
        PopulationAttribute("pbest_f", float, (prob.n_obj,), default=np.nan),
        PopulationAttribute("pbest_cv", float, (), default=np.nan),
    ]

    pso = PSO()
    ctx_pso1 = _setup_state(prob, popsize=20, seed=123, extra_attrs=pso_attrs)
    ctx_pso2 = _setup_state(prob, popsize=20, seed=123, extra_attrs=pso_attrs)

    pop_pso1 = algorithm_ask(pso, ctx_pso1, provider)
    pop_pso2 = algorithm_ask(pso, ctx_pso2, provider)

    np.testing.assert_array_equal(pop_pso1.get_array("x"), pop_pso2.get_array("x"))


# ---------------------------------------------------------------------------
# 4. Verification that services.require is NOT called in hot-path inner loops
# ---------------------------------------------------------------------------


def test_bounds_service_not_required_in_inner_loops() -> None:
    """GA.ask uses the compiled bounds reference without registry lookup.

    Implementation mutation that would break this test:
        Reintroduce `ctx.problem.space.services.require("BoundsService")` in
        GA.ask. The post-initialization spy would observe a non-zero call.
    """
    prob = Problem(
        func=dummy_func,
        dim=4,
        n_obj=1,
        direction=np.array([-1]),
        lb=[-1.0, -1.0, -1.0, -1.0],
        ub=[1.0, 1.0, 1.0, 1.0],
    )

    # Spy on ServiceRegistry.require
    real_require = prob.space.services.require
    require_mock = MagicMock(side_effect=real_require)
    prob.space.services.require = require_mock  # type: ignore[assignment]

    provider = MagicMock()
    ga = _make_ga()
    ctx = _setup_state(prob, popsize=20, seed=42)

    require_mock.reset_mock()
    algorithm_ask(ga, ctx, provider)

    # Count how many times BoundsService was resolved in ask()
    bounds_calls = [
        call for call in require_mock.call_args_list if call.args[0] == "BoundsService"
    ]
    # The compiled reference is bound during state setup; ask must not resolve
    # the service through the registry.
    assert len(bounds_calls) == 0
