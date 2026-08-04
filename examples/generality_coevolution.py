"""Demonstrate cooperative coevolution using only public saealib APIs.

The shared-context objective is not monotone: each block commits an incumbent
whose archive fitness was measured with a stale context.  That stale-fitness
effect is inherent to this cooperative coevolution mechanism, not a saealib
bug.
"""

import numpy as np

from saealib import (
    PSO,
    DirectStrategy,
    LHSInitializer,
    Optimizer,
    Problem,
    Termination,
    max_fe,
)

DIMENSION = 4
COORDINATE_BLOCKS = ((0, 1), (2, 3))
TARGET = 0.3
EVALUATION_BUDGET = 40


def sphere_value(vector: np.ndarray) -> float:
    """Evaluate the shifted sphere at a complete decision vector."""
    return float(np.sum((vector - TARGET) ** 2))


class SharedVector:
    """Mutable full-length vector shared by all block optimizers."""

    def __init__(self, dimension: int) -> None:
        self.values = np.zeros(dimension, dtype=np.float64)

    def candidate(self, coordinates: tuple[int, ...], values: np.ndarray) -> np.ndarray:
        """Assemble block values with the current values of other coordinates."""
        full = self.values.copy()
        full[list(coordinates)] = values
        return full

    def replace(self, coordinates: tuple[int, ...], values: np.ndarray) -> None:
        """Write a block incumbent into the shared full vector."""
        self.values[list(coordinates)] = values


def make_block_problem(shared: SharedVector, coordinates: tuple[int, ...]) -> Problem:
    """Build a block problem that evaluates candidates in shared context."""

    def evaluate(row):
        return np.array([sphere_value(shared.candidate(coordinates, row))], np.float64)

    return Problem(
        func=evaluate,
        dim=len(coordinates),
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0] * len(coordinates),
        ub=[1.0] * len(coordinates),
    )


def make_block_optimizer(
    shared: SharedVector, coordinates: tuple[int, ...], seed: int
) -> Optimizer:
    """Build one optimizer for a disjoint coordinate block."""
    return (
        Optimizer(make_block_problem(shared, coordinates), seed=seed)
        .set_initializer(LHSInitializer(4, len(coordinates), seed))
        .set_algorithm(PSO())
        .set_strategy(DirectStrategy(n_offspring=4))
        .set_termination(Termination(max_fe(EVALUATION_BUDGET)))
    )


def best_candidate(state):
    """Return the comparator-best row in a block's archive."""
    order = state.comparator.sort_population(state.archive)
    return state.archive.get_array("x")[order[0]].copy()


def main():
    """Run two block optimizers against one mutable shared vector."""
    shared = SharedVector(DIMENSION)
    optimizers = [
        make_block_optimizer(shared, block, 11 + i)
        for i, block in enumerate(COORDINATE_BLOCKS)
    ]
    generators = [optimizer.iterate() for optimizer in optimizers]
    states = [next(generator) for generator in generators]
    live = [True] * len(generators)
    trace = [sphere_value(shared.values)]

    while any(live):
        for i, generator in enumerate(generators):
            if not live[i]:
                continue
            try:
                states[i] = next(generator)
            except StopIteration:
                live[i] = False
                continue
            shared.replace(COORDINATE_BLOCKS[i], best_candidate(states[i]))
            trace.append(sphere_value(shared.values))

    return shared, states, trace


if __name__ == "__main__":
    context, states, trace = main()
    print("final context:", np.round(context.values, 4))
    print("objective start -> end:", trace[0], "->", trace[-1])
