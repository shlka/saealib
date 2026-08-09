"""Microbenchmark the private GA merge-pool allocation path.

Run from the repository root with ``uv run python benchmarks/ga_merge_pool.py``.
The optimized path is intentionally private; public ``Population.empty_like``
remains the reference path for all other callers.
"""

from __future__ import annotations

import timeit

import numpy as np

from saealib.algorithms.ga import _canonical_merge_pool
from saealib.population import Population, PopulationAttribute


def _population(size: int, dim: int) -> Population:
    attrs = [
        PopulationAttribute("x", np.float64, (dim,)),
        PopulationAttribute("f", np.float64, (1,)),
        PopulationAttribute("g", np.float64, (0,)),
        PopulationAttribute("cv", np.float64),
    ]
    population = Population(attrs, init_capacity=size)
    population.extend(
        {
            "x": np.zeros((size, dim)),
            "f": np.zeros((size, 1)),
            "g": np.zeros((size, 0)),
            "cv": np.zeros(size),
        }
    )
    return population


def main() -> None:
    """Print minimum allocation and complete pool-construction timings."""
    size, dim, repeats = 100, 10, 10_000
    population = _population(size, dim)
    offspring = _population(size, dim)

    def fast_allocation() -> None:
        _canonical_merge_pool(population, offspring, size * 2)

    def reference_allocation() -> None:
        population.empty_like(capacity=size * 2)

    def fast_pool() -> None:
        pool = _canonical_merge_pool(population, offspring, size * 2)
        assert pool is not None
        pool._extend_internal(population, preserve_ids=True)
        pool._extend_internal(offspring, preserve_ids=True)

    def reference_pool() -> None:
        pool = population.empty_like(capacity=size * 2)
        pool._extend_internal(population, preserve_ids=True)
        pool._extend_internal(offspring, preserve_ids=True)

    for name, function in (
        ("canonical allocation", fast_allocation),
        ("empty_like allocation", reference_allocation),
        ("canonical pool", fast_pool),
        ("empty_like pool", reference_pool),
    ):
        elapsed = min(timeit.repeat(function, repeat=5, number=repeats))
        print(f"{name}: {elapsed / repeats * 1e6:.2f} us/call")


if __name__ == "__main__":
    main()
