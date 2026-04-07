"""
Genetic Algorithm class.

This module contains the implementation of Genetic Algorithm.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np

from saealib.algorithms.base import Algorithm
from saealib.callback import CallbackEvent, PostAskArgs
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
        """
        Return the list of attributes required for Population based on a Problem object.

        GA requires only the default attributes ('x', 'f', 'g', 'cv').

        Parameters
        ----------
        problem : Problem
            The Problem object being referenced.

        Returns
        -------
        list[PopulationAttribute]
            List of attributes required in addition to the default attributes.
        """
        return []  # only default attrs

    @property
    def population_class(self):
        """Return the population class."""
        return Population

    @property
    def archive_class(self):
        """Return the archive class."""
        return Archive

    def ask(self, ctx: OptimizationContext, provider: ComponentProvider) -> Population:
        """
        Generate offspring solutions.

        Parameters
        ----------
        ctx : OptimizationContext
            A dataclass object that holds internal information about the Optimizer.
        provider : ComponentProvider
            Objects of the class in which the component is exposed (ex. Optimizer).

        Returns
        -------
        Population
            Generated offspring solutions. shape = (popsize, dim).
        """
        # TODO: Modified the ask method to return candidate solutions of arbitrary
        # size without depending on popsize.
        cand = np.empty((0, ctx.dim))
        pop = ctx.population.get_array("x")
        popsize = len(pop)
        lb = ctx.problem.lb
        ub = ctx.problem.ub
        n_pair = math.ceil(popsize / 2)
        parent_idx_m = self.parent_selection.select(
            ctx,
            ctx.population,
            n_pair=n_pair,
            n_parents=2,  # TODO: receive n_parents from Crossover
            rng=ctx.rng,
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
        provider.dispatch(CallbackEvent.POST_CROSSOVER, PostAskArgs(ctx=ctx, candidates=cand))
        cand_len = len(cand)
        for i in range(cand_len):
            cand[i] = self.mutation.mutate(cand[i], (lb, ub), rng=ctx.rng)
        if self.repair is not None:
            cand = self.repair(cand, (lb, ub))
        provider.dispatch(CallbackEvent.POST_MUTATION, PostAskArgs(ctx=ctx, candidates=cand))
        provider.dispatch(CallbackEvent.POST_ASK, PostAskArgs(ctx=ctx, candidates=cand))

        cand_pop = ctx.population.empty_like(capacity=popsize)
        cand_pop.extend({"x": cand[:popsize]})
        return cand_pop

    def tell(
        self,
        ctx: OptimizationContext,
        provider: ComponentProvider,
        offspring: Population,
    ):
        """
        Update the population with offspring solutions.

        Optimizer population is updated in-place.

        Parameters
        ----------
        ctx : OptimizationContext
            A dataclass object that holds internal information about the Optimizer.
        provider : ComponentProvider
            Objects of the class in which the component is exposed (ex. Optimizer).
        offspring : Population
            Population object representing offspring.

        Returns
        -------
        None

        Notes
        -----
        Uses (μ+λ) replacement strategy: merges parents and offspring into
        a pool, then selects the best popsize individuals via
        survivor_selection.
        """
        popsize = len(ctx.population)

        # Build μ+λ pool
        pool = ctx.population.empty_like(capacity=popsize + len(offspring))
        pool.extend(ctx.population)
        pool.extend(offspring)

        # Select survivors
        survivor_idx = self.survivor_selection.select(ctx, pool, popsize)

        # Update population
        ctx.population.clear()
        ctx.population.extend(pool.extract(survivor_idx))
