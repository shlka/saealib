"""Representation-agnostic genetic algorithm profile."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from saealib.algorithms.base import Algorithm
from saealib.context import OptimizationState
from saealib.core.contracts import PartSpec
from saealib.population import Archive, Population, PopulationAttribute
from saealib.population.genome import GenomeBatch
from saealib.problem import Problem

if TYPE_CHECKING:
    from saealib.operators.selection import ParentSelection, SurvivorSelection
    from saealib.optimizer import Dispatchable

__all__ = ["GenomeGA"]


def _crossover_genomes(
    operator: object,
    parents: GenomeBatch,
    *,
    n_pair: int,
    n_parents: int,
    rng: np.random.Generator,
) -> GenomeBatch:
    method = getattr(operator, "crossover_genomes", None)
    if method is None:
        method = getattr(operator, "apply", None)
    if not callable(method):
        raise TypeError("crossover must provide crossover_genomes()")
    if getattr(operator, "crossover_genomes", None) is not None:
        return method(parents, n_pair=n_pair, n_parents=n_parents, rng=rng)
    return method(parents, rng=rng)


def _mutate_genomes(
    operator: object,
    candidates: GenomeBatch,
    *,
    rng: np.random.Generator,
    space: object,
) -> GenomeBatch:
    method = getattr(operator, "mutate_genomes", None)
    if method is None:
        method = getattr(operator, "apply", None)
    if not callable(method):
        raise TypeError("mutation must provide mutate_genomes()")
    if getattr(operator, "mutate_genomes", None) is not None:
        return method(candidates, rng=rng, space=space)
    return method(candidates, rng=rng)


class GenomeGA(Algorithm):
    """A small ``ask``/``tell`` GA that only uses the GenomeBatch protocol."""

    def __init__(
        self,
        crossover: object,
        mutation: object,
        parent_selection: ParentSelection,
        survivor_selection: SurvivorSelection,
    ) -> None:
        self.crossover = crossover
        self.mutation = mutation
        self.parent_selection = parent_selection
        self.survivor_selection = survivor_selection
        self.n_parents = int(getattr(crossover, "n_parents", 2))
        self.n_children = int(getattr(crossover, "n_children", 2))
        if self.n_parents < 1 or self.n_children < 1:
            raise ValueError("crossover parent and child counts must be positive")

    def contract(self):
        """Return the component contract for this genetic algorithm."""
        base = super().contract()
        proposer = base.ports["proposer"]
        output = proposer.outputs[0]
        output = replace(output, required_services=())
        return replace(
            base,
            ports={
                "proposer": replace(proposer, outputs=(output,)),
                **{
                    key: value for key, value in base.ports.items() if key != "proposer"
                },
            },
            parts=(
                PartSpec(
                    name="crossover", contract=cast(Any, self.crossover).contract()
                ),
                PartSpec(name="mutation", contract=cast(Any, self.mutation).contract()),
                PartSpec(
                    name="parent_selection", contract=self.parent_selection.contract()
                ),
                PartSpec(
                    name="survivor_selection",
                    contract=self.survivor_selection.contract(),
                ),
            ),
        )

    def get_required_attrs(self, problem: Problem) -> list[PopulationAttribute]:
        """Return population attributes required by this algorithm."""
        return []

    @property
    def population_class(self) -> type[Population]:
        """Return the population implementation used by the algorithm."""
        return Population

    @property
    def archive_class(self) -> type[Archive]:
        """Return the archive implementation used by the algorithm."""
        return Archive

    def ask(
        self,
        ctx: OptimizationState,
        provider: Dispatchable,
        n_offspring: int | None = None,
    ) -> Population:
        """Generate offspring from the current optimization state."""
        popsize = len(ctx.population)
        target = popsize if n_offspring is None else n_offspring
        if target < 0:
            raise ValueError("n_offspring must be non-negative")
        if target == 0:
            return ctx.population.empty_like(capacity=0)
        n_pair = math.ceil(target / self.n_children)
        parent_indices = self.parent_selection.select(
            ctx,
            ctx.population,
            n_pair=n_pair,
            n_parents=self.n_parents,
            rng=ctx.rng,
        )
        parents = ctx.population.genomes.take(np.asarray(parent_indices).reshape(-1))
        offspring = _crossover_genomes(
            self.crossover,
            parents,
            n_pair=n_pair,
            n_parents=self.n_parents,
            rng=ctx.rng,
        )
        offspring = _mutate_genomes(
            self.mutation, offspring, rng=ctx.rng, space=ctx.problem.space
        )
        offspring = offspring.take(np.arange(target, dtype=np.intp))

        candidates = ctx.population.empty_like(capacity=target)
        columns: dict[str, np.ndarray | GenomeBatch] = {
            "f": np.full((target, ctx.problem.n_obj), np.nan),
            "g": np.zeros((target, ctx.problem.n_constraints)),
            "cv": np.zeros(target),
            "id": np.full(target, -1, dtype=np.int64),
        }
        columns["genome"] = offspring
        candidates.extend(columns)
        return candidates

    def tell(
        self,
        ctx: OptimizationState,
        provider: Dispatchable,
        offspring: Population,
    ) -> None:
        """Update the population with evaluated offspring."""
        population = ctx.population
        pool = population.empty_like(capacity=len(population) + len(offspring))
        pool._extend_internal(population, preserve_ids=True)
        pool._extend_internal(offspring, preserve_ids=True)
        survivor_idx = self.survivor_selection.select(ctx, pool, len(population))
        if not population._replace_from_population(
            pool, survivor_idx, preserve_ids=True
        ):
            population.clear()
            population._extend_internal(pool.extract(survivor_idx), preserve_ids=True)
