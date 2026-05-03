"""Genetic Algorithm module."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from saealib.algorithms.base import Algorithm
from saealib.callback import PostAskEvent, PostCrossoverEvent, PostMutationEvent
from saealib.context import OptimizationContext
from saealib.operators.repair import repair_clipping
from saealib.population import Archive, Population, PopulationAttribute
from saealib.problem import Problem

if TYPE_CHECKING:
    from saealib.operators.crossover import Crossover
    from saealib.operators.mutation import Mutation
    from saealib.operators.selection import ParentSelection, SurvivorSelection
    from saealib.optimizer import ComponentProvider

RepairFunc = Callable[[np.ndarray, tuple[np.ndarray, np.ndarray]], np.ndarray]


class GA(Algorithm):
    """
    Genetic Algorithm class.

    Attributes
    ----------
    crossover : Crossover
        Crossover operator.
    mutation : Mutation
        Mutation operator.
    parent_selection : ParentSelection
        Parent selection operator.
    survivor_selection : SurvivorSelection
        Survivor selection operator.
    """

    def __init__(
        self,
        crossover: Crossover,
        mutation: Mutation,
        parent_selection: ParentSelection,
        survivor_selection: SurvivorSelection,
        repair: RepairFunc | None = repair_clipping,
    ):
        """
        Initialize GA (Genetic Algorithm) class.

        Parameters
        ----------
        crossover : Crossover
            Crossover operator.
        mutation : Mutation
            Mutation operator.
        parent_selection : ParentSelection
            Parent selection operator.
        survivor_selection : SurvivorSelection
            Survivor selection operator.
        repair : RepairFunc or None, optional
            Repair function applied after crossover and after mutation.
            Defaults to ``repair_clipping``. Pass ``None`` to disable repair.
        """
        super().__init__()
        self.crossover = crossover
        self.mutation = mutation
        self.parent_selection = parent_selection
        self.survivor_selection = survivor_selection
        self.repair = repair

    def get_required_attrs(self, problem: Problem) -> list[PopulationAttribute]:
        """Return algorithm-specific attributes (GA needs none beyond the defaults)."""
        return []

    @property
    def population_class(self):
        """Return the population class."""
        return Population

    @property
    def archive_class(self):
        """Return the archive class."""
        return Archive

    def ask(
        self,
        ctx: OptimizationContext,
        provider: ComponentProvider,
        n_offspring: int | None = None,
    ) -> Population:
        """
        Generate offspring via crossover and mutation.

        Parameters
        ----------
        ctx : OptimizationContext
            Current optimization context.
        provider : ComponentProvider
            Component provider.
        n_offspring : int or None, optional
            Number of offspring. Defaults to the current population size.

        Returns
        -------
        Population
        """
        cand = np.empty((0, ctx.dim))
        pop = ctx.population.get_array("x")
        popsize = len(pop)
        target = n_offspring if n_offspring is not None else popsize
        lb = ctx.problem.lb
        ub = ctx.problem.ub
        n_pair = math.ceil(target / 2)
        parent_idx_m = (
            self.parent_selection.select(
                ctx,
                ctx.population,
                n_pair=n_pair,
                n_parents=self.crossover.n_parents,
                rng=ctx.rng,
            )
            % popsize
        )
        for i in range(n_pair):
            parent = pop[parent_idx_m[i]]
            if ctx.rng.random() < self.crossover.crossover_rate:
                c = self.crossover.crossover(parent, rng=ctx.rng)
            else:
                c = parent.copy()
            cand = np.vstack((cand, c))
        if self.repair is not None:
            cand = self.repair(cand, (lb, ub))
        post_co = PostCrossoverEvent(ctx=ctx, provider=provider, candidates=cand)
        provider.dispatch(post_co)
        cand = post_co.candidates

        cand_len = len(cand)
        for i in range(cand_len):
            cand[i] = self.mutation.mutate(cand[i], (lb, ub), rng=ctx.rng)
        if self.repair is not None:
            cand = self.repair(cand, (lb, ub))
        post_mut = PostMutationEvent(ctx=ctx, provider=provider, candidates=cand)
        provider.dispatch(post_mut)
        cand = post_mut.candidates

        post_ask = PostAskEvent(ctx=ctx, provider=provider, candidates=cand)
        provider.dispatch(post_ask)
        cand = post_ask.candidates

        cand_pop = ctx.population.empty_like(capacity=target)
        cand_pop.extend({"x": cand[:target]})
        return cand_pop

    def tell(
        self,
        ctx: OptimizationContext,
        provider: ComponentProvider,
        offspring: Population,
    ):
        """
        Update the population using (μ+λ) survivor selection.

        Parameters
        ----------
        ctx : OptimizationContext
            Current optimization context.
        provider : ComponentProvider
            Component provider.
        offspring : Population
            Offspring population.
        """
        popsize = len(ctx.population)

        pool = ctx.population.empty_like(capacity=popsize + len(offspring))
        pool.extend(ctx.population)
        pool.extend(offspring)

        survivor_idx = self.survivor_selection.select(ctx, pool, popsize)

        ctx.population.clear()
        ctx.population.extend(pool.extract(survivor_idx))
