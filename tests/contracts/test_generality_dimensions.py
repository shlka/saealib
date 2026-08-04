"""Contract tests for direction, objective count, and constraints."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from examples._support import (
    constrained_problem,
    maximize_problem,
    reference_problem,
    two_objective_problem,
)
from saealib import (
    PSO,
    DirectStrategy,
    IndividualBasedStrategy,
    LHSInitializer,
    Optimizer,
    SklearnGPRSurrogate,
    Termination,
    max_fe,
)
from saealib.acquisition import EHVIAcquisition


def _run_direct(problem, seed=5):
    return (
        Optimizer(problem, seed=seed)
        .set_initializer(LHSInitializer(4, problem.dim, seed))
        .set_algorithm(PSO())
        .set_strategy(DirectStrategy(n_offspring=4))
        .set_termination(Termination(max_fe(16)))
        .run()
    )


@pytest.fixture(params=("direct", "surrogate"))
def run_optimizer(request):
    """Return a runner for one of the measured Unit 4 configurations."""

    def run(problem, seed=5):
        if request.param == "direct":
            return _run_direct(problem, seed)

        optimizer = (
            Optimizer(problem, seed=seed)
            .set_initializer(LHSInitializer(6, problem.dim, seed))
            .set_algorithm(PSO())
            .set_surrogate(
                SklearnGPRSurrogate(alpha=1e-6, optimizer=None, random_state=seed)
            )
            .set_strategy(IndividualBasedStrategy(0.5))
            .set_termination(Termination(max_fe(12)))
        )
        if problem.n_obj > 1:
            optimizer = optimizer.set_acquisition(EHVIAcquisition(n_samples=32))
        return optimizer.run()

    return run


@pytest.mark.parametrize("maximize", (False, True), ids=("minimize", "maximize"))
@pytest.mark.parametrize("n_obj", (1, 2), ids=("1-objective", "2-objective"))
@pytest.mark.parametrize(
    "constrained", (False, True), ids=("unconstrained", "constrained")
)
def test_all_generality_dimensions_complete(
    maximize, n_obj, constrained, run_optimizer
):
    """All direction/objective/constraint cells complete with feasible results."""
    if n_obj == 1:
        if constrained:
            problem = constrained_problem(maximize=maximize)
        elif maximize:
            problem = maximize_problem()
        else:
            problem = reference_problem(shift=0.2)
    else:
        problem = two_objective_problem(maximize=maximize, constrained=constrained)

    state = run_optimizer(problem)
    archive = state.archive
    np.testing.assert_array_equal(
        state.problem.direction, np.full(n_obj, 1.0 if maximize else -1.0)
    )
    assert state.problem.n_obj == n_obj
    assert archive.get_array("f").shape[1] == n_obj
    cv = archive.get_array("cv")
    feasible = cv <= problem.eps_cv
    assert state.fe > 0
    assert feasible.any()

    best_idx = problem.comparator.sort_population(archive)[0]
    assert feasible[best_idx]

    if n_obj == 1:
        f = archive.get_array("f")[:, 0]
        feasible_f = f[feasible]
        expected = feasible_f.max() if maximize else feasible_f.min()
        assert np.isclose(f[best_idx], expected)
    else:
        f = archive.get_array("f")
        assert np.any(np.abs(f[:, 0] - f[:, 1]) > 1e-9)


@pytest.mark.parametrize("maximize", (False, True), ids=("minimize", "maximize"))
@pytest.mark.parametrize("n_obj", (1, 2), ids=("1-objective", "2-objective"))
def test_direct_constraints_bind_and_preserve_comparator_best(maximize, n_obj):
    """The measured direct runs expose the binding constraint and feasible best."""

    def make_problem(is_constrained):
        if n_obj == 1:
            if is_constrained:
                return constrained_problem(maximize=maximize)
            return maximize_problem() if maximize else reference_problem(shift=0.2)
        return two_objective_problem(maximize=maximize, constrained=is_constrained)

    unconstrained = _run_direct(make_problem(False), seed=5)
    constrained = _run_direct(make_problem(True), seed=5)

    unconstrained_cv = unconstrained.archive.get_array("cv")
    constrained_cv = constrained.archive.get_array("cv")
    assert np.count_nonzero(constrained_cv <= 1e-6) < np.count_nonzero(
        unconstrained_cv <= 1e-6
    )
    best_idx = constrained.comparator.sort_population(constrained.archive)[0]
    assert constrained_cv[best_idx] <= 1e-6
    unconstrained_best = unconstrained.comparator.sort_population(
        unconstrained.archive
    )[0]
    assert not np.allclose(
        constrained.archive.x[best_idx], unconstrained.archive.x[unconstrained_best]
    )
    if n_obj == 1:
        assert np.sum(constrained.archive.x[best_idx]) <= 0.2
