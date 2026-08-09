"""Microbenchmark the common state replacement paths.

Run from the repository root with ``python benchmarks/state_store_replace.py``.
The benchmark intentionally uses ordinary scalar state writes, which is the
path optimized in ``StateStore.apply_patch`` and ``OptimizationState.replace``.
"""

from __future__ import annotations

import timeit

import numpy as np

from saealib.context import OptimizationState
from saealib.core.state import RUNTIME_GENERATION, StatePatch, StateStore
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem


def _make_optimization_state() -> OptimizationState:
    attrs = [PopulationAttribute(name="x", dtype=np.float64)]
    population = Population(attrs)
    archive = Archive(attrs)
    pareto_archive = ParetoArchive(attrs, direction=np.array([-1.0]))
    problem = Problem(
        func=lambda x: np.array([np.sum(x**2)]),
        dim=1,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0],
        ub=[1.0],
    )
    return OptimizationState(
        problem,
        population,
        archive,
        pareto_archive,
    )


def _state_store_replace(iterations: int) -> None:
    key = RUNTIME_GENERATION
    store = StateStore({key: 0})
    patch = StatePatch(writes={key: 1})
    for _ in range(iterations):
        store = store.apply_patch(patch)


def _optimization_state_replace(iterations: int) -> None:
    state = _make_optimization_state()
    for generation in range(1, iterations + 1):
        state = state.replace(gen=generation)


def main() -> None:
    """Print the minimum per-call time for each replacement path."""
    iterations = 10_000
    for name, function in (
        ("StateStore.apply_patch", _state_store_replace),
        ("OptimizationState.replace", _optimization_state_replace),
    ):
        elapsed = min(timeit.repeat(lambda: function(iterations), repeat=5, number=1))
        print(f"{name}: {elapsed / iterations * 1e6:.2f} us/call")


if __name__ == "__main__":
    main()
