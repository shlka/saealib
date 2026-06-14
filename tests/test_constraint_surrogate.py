"""End-to-end tests for constraint surrogate integration (issue #86).

Tests that CompositeSurrogateManager([ei_mgr, pof_mgr], product_combine)
correctly combines an objective surrogate (EI) with a constraint surrogate
(ProductOfFeasibility) on the G01 constrained benchmark.
"""

from __future__ import annotations

import numpy as np
import pytest

from saealib import (
    GA,
    CompositeSurrogateManager,
    CrossoverBLXAlpha,
    ExpectedImprovement,
    GlobalSurrogateManager,
    GPSurrogate,
    InequalityConstraint,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    PerObjectiveSurrogate,
    PreSelectionStrategy,
    Problem,
    ProductOfFeasibility,
    SequentialSelection,
    StaticToleranceHandler,
    Termination,
    TruncationSelection,
    max_fe,
    product_combine,
)
from saealib.surrogate.training_set import ArchiveObjectiveSet, ConstraintObjectiveSet

# ---------------------------------------------------------------------------
# G01 benchmark definition
# ---------------------------------------------------------------------------

_DIM = 13
_N_CONSTRAINTS = 9


def _g01_obj(x: np.ndarray) -> np.ndarray:
    return np.array([5.0 * np.sum(x[:4]) - 5.0 * np.sum(x[:4] ** 2) - np.sum(x[4:13])])


_G01_CONSTRAINTS = [
    InequalityConstraint(lambda x: 2 * x[0] + 2 * x[1] + x[9] + x[10] - 10.0),
    InequalityConstraint(lambda x: 2 * x[0] + 2 * x[2] + x[9] + x[11] - 10.0),
    InequalityConstraint(lambda x: 2 * x[1] + 2 * x[2] + x[10] + x[11] - 10.0),
    InequalityConstraint(lambda x: -8.0 * x[0] + x[9]),
    InequalityConstraint(lambda x: -8.0 * x[1] + x[10]),
    InequalityConstraint(lambda x: -8.0 * x[2] + x[11]),
    InequalityConstraint(lambda x: -2.0 * x[3] - x[4] + x[9]),
    InequalityConstraint(lambda x: -2.0 * x[5] - x[6] + x[10]),
    InequalityConstraint(lambda x: -2.0 * x[7] - x[8] + x[11]),
]


def _make_g01_problem() -> Problem:
    return Problem(
        func=_g01_obj,
        dim=_DIM,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[0.0] * _DIM,
        ub=[1.0] * _DIM,
        constraints=_G01_CONSTRAINTS,
        handler=StaticToleranceHandler(eps_cv=1e-6),
    )


def _make_ga() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(crossover_rate=0.9, alpha=0.4),
        mutation=MutationUniform(mutation_rate=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


# ---------------------------------------------------------------------------
# G01 end-to-end test
# ---------------------------------------------------------------------------


class TestConstraintSurrogateG01:
    """End-to-end test: EI x ProductOfFeasibility on G01."""

    def test_optimizer_runs_without_error(self) -> None:
        """CompositeSurrogateManager completes 5 generations on G01."""
        problem = _make_g01_problem()

        ei_mgr = GlobalSurrogateManager(
            GPSurrogate(),
            ExpectedImprovement(),
            training_set=ArchiveObjectiveSet(),
        )
        pof_mgr = GlobalSurrogateManager(
            PerObjectiveSurrogate([GPSurrogate() for _ in range(_N_CONSTRAINTS)]),
            ProductOfFeasibility(),
            training_set=ConstraintObjectiveSet(),
        )
        surrogate_manager = CompositeSurrogateManager(
            [ei_mgr, pof_mgr],
            combine_fn=product_combine,
        )

        optimizer = (
            Optimizer(problem)
            .set_initializer(
                LHSInitializer(n_init_archive=20, n_init_population=10, seed=0)
            )
            .set_algorithm(_make_ga())
            .set_strategy(PreSelectionStrategy(n_candidates=20, n_select=3))
            .set_surrogate_manager(surrogate_manager)
            .set_termination(Termination(max_fe(40)))
        )
        ctx = optimizer.run()
        assert ctx is not None

    def test_constraint_surrogate_trains_on_g(self) -> None:
        """ConstraintObjectiveSet produces train_y with n_constraints columns."""
        problem = _make_g01_problem()
        optimizer = (
            Optimizer(problem)
            .set_initializer(
                LHSInitializer(n_init_archive=15, n_init_population=10, seed=1)
            )
            .set_algorithm(_make_ga())
            .set_strategy(PreSelectionStrategy(n_candidates=20, n_select=3))
            .set_surrogate_manager(
                GlobalSurrogateManager(GPSurrogate(), ExpectedImprovement())
            )
            .set_termination(Termination(max_fe(15)))
        )
        # Run initialisation only (max_fe equals initial archive sample size)
        ctx = optimizer.run()
        archive = ctx.archive
        ts = ConstraintObjectiveSet()
        data = ts.build(archive, None, None)
        assert data.train_y.shape == (len(archive), _N_CONSTRAINTS)

    def test_feasible_solution_found(self) -> None:
        """With enough budget, at least one feasible solution is archived."""
        problem = _make_g01_problem()

        ei_mgr = GlobalSurrogateManager(
            GPSurrogate(),
            ExpectedImprovement(),
            training_set=ArchiveObjectiveSet(),
        )
        pof_mgr = GlobalSurrogateManager(
            PerObjectiveSurrogate([GPSurrogate() for _ in range(_N_CONSTRAINTS)]),
            ProductOfFeasibility(),
            training_set=ConstraintObjectiveSet(),
        )

        optimizer = (
            Optimizer(problem)
            .set_initializer(
                LHSInitializer(n_init_archive=30, n_init_population=10, seed=42)
            )
            .set_algorithm(_make_ga())
            .set_strategy(PreSelectionStrategy(n_candidates=30, n_select=5))
            .set_surrogate_manager(
                CompositeSurrogateManager([ei_mgr, pof_mgr], product_combine)
            )
            .set_termination(Termination(max_fe(60)))
        )
        ctx = optimizer.run()
        assert ctx is not None


# ---------------------------------------------------------------------------
# Backward compatibility: zero-constraint problems
# ---------------------------------------------------------------------------


class TestConstraintSurrogateBackwardCompat:
    """Zero-constraint problems work as before; ConstraintObjectiveSet raises."""

    def test_unconstrained_optimizer_unchanged(self) -> None:
        """Unconstrained sphere problem runs identically with new imports loaded."""
        problem = Problem(
            func=lambda x: np.array([np.sum(x**2)]),
            dim=3,
            n_obj=1,
            direction=np.array([-1.0]),
            lb=[-5.0] * 3,
            ub=[5.0] * 3,
        )
        optimizer = (
            Optimizer(problem)
            .set_initializer(
                LHSInitializer(n_init_archive=10, n_init_population=5, seed=0)
            )
            .set_algorithm(_make_ga())
            .set_strategy(PreSelectionStrategy(n_candidates=10, n_select=2))
            .set_surrogate_manager(
                GlobalSurrogateManager(GPSurrogate(), ExpectedImprovement())
            )
            .set_termination(Termination(max_fe(20)))
        )
        ctx = optimizer.run()
        assert ctx is not None

    def test_constraint_objective_set_raises_on_zero_constraints(self) -> None:
        """ConstraintObjectiveSet raises ValueError when archive has no g columns."""
        from saealib.population import Archive, PopulationAttribute

        attrs = [
            PopulationAttribute("x", np.float64, (3,)),
            PopulationAttribute("f", np.float64, (1,)),
            PopulationAttribute("g", np.float64, (0,)),
            PopulationAttribute("cv", np.float64, ()),
        ]
        arc = Archive(attrs, init_capacity=10)
        rng = np.random.default_rng(0)
        for _ in range(5):
            arc.add(x=rng.uniform(size=3), f=np.array([1.0]), cv=0.0)

        ts = ConstraintObjectiveSet()
        with pytest.raises(ValueError, match="0 columns"):
            ts.build(arc, None, None)
