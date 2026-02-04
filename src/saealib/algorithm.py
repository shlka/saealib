"""
Evolutionary Algorithm Module.

This module contains the implementation of evolutionary algorithms.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from saealib.population import Population, PopulationAttribute, Archive
from saealib.callback import CallbackEvent
from saealib.context import OptimizationContext
from saealib.problem import Problem

if TYPE_CHECKING:
    from saealib.optimizer import ComponentProvider
    from saealib.operators.crossover import Crossover
    from saealib.operators.mutation import Mutation
    from saealib.operators.selection import ParentSelection, SurvivorSelection
    from saealib.optimizer import Optimizer


class Algorithm(ABC):
    """Base class for evolutionary algorithms."""

    @abstractmethod
    def get_required_attrs(self, problem: Problem) -> list[PopulationAttribute]:
        pass

    @property
    @abstractmethod
    def population_class(self) -> type[Population]:
        pass

    @property
    @abstractmethod
    def archive_class(self) -> type[Archive]:
        pass


    @abstractmethod
    def ask(self, ctx: OptimizationContext, provider: ComponentProvider) -> Population:
        """
        Generate offspring solutions.

        Parameters
        ----------
        optimizer : Optimizer
            Optimizer instance.

        Returns
        -------
        np.ndarray
            Generated offspring solutions. shape = (popsize, dim).
        """
        pass

    @abstractmethod
    def tell(
        self, ctx: OptimizationContext, provider: ComponentProvider, offspring: Population
    ) -> None:
        """
        Update the population with offspring solutions.

        Parameters
        ----------
        optimizer : Optimizer
            Optimizer instance.
        offspring : np.ndarray
            Offspring solutions. shape = (popsize, dim).
        offspring_fit : np.ndarray
            Offspring fitness values. shape = (popsize, ).
        """
        pass


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
        """
        super().__init__()
        self.crossover = crossover
        self.mutation = mutation
        self.parent_selection = parent_selection
        self.survivor_selection = survivor_selection

    def get_required_attrs(self, problem: Problem) -> list[PopulationAttribute]:
        """
        Returns the list of attributes required for Population based on a Problem object. 
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
        return [] # only default attrs

    @property
    def population_class(self):
        return Population

    @property
    def archive_class(self):
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
        np.ndarray
            Generated offspring solutions. shape = (popsize, dim).
        """
        # TODO: Modified the ask method to return candidate solutions of arbitrary size without depending on popsize.
        cand = np.empty((0, ctx.dim))
        pop = ctx.population.get_array("x")
        pop_f = ctx.population.get_array("f")
        popsize = len(pop)
        lb = ctx.problem.lb
        ub = ctx.problem.ub
        n_pair = math.ceil(popsize / 2)
        parent_idx_m = self.parent_selection.select(
            provider,
            pop,
            pop_f,
            np.zeros(popsize),# TODO: use cv if constraints are defined
            n_pair=n_pair,
            n_parents=2,# TODO: recieve n_parents from Crossover
            rng=ctx.rng)
        for i in range(n_pair):
            parent = pop[parent_idx_m[i]]
            if ctx.rng.random() < self.crossover.crossover_rate:
                c = self.crossover.crossover(parent, rng=ctx.rng)
            else:
                c = parent.copy()
            cand = np.vstack((cand, c))
        cand = provider.dispatch(CallbackEvent.POST_CROSSOVER, ctx=ctx, data=cand)
        cand_len = len(cand)
        for i in range(cand_len):
            cand[i] = self.mutation.mutate(cand[i], (lb, ub), rng=ctx.rng)
        cand = provider.dispatch(CallbackEvent.POST_MUTATION, ctx=ctx, data=cand)

        cand_pop = ctx.population.empty_like(capacity=popsize)
        cand_pop.extend({"x": cand[:popsize]})
        return cand_pop

    def tell(self, ctx: OptimizationContext, provider: ComponentProvider, offspring: Population):
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
        These optimizer attributes are used:
        - population (read/write)
        - popsize (readonly)
        - problem.comparator (readonly)
        """
        cmp = ctx.comparator
        # select a best solution in parent
        best_idx = cmp.sort(
            ctx.population.get_array("f"), ctx.population.get_array("cv")
        )
        parent_best = ctx.population.get_array("x")[best_idx]
        parent_best_fit = ctx.population.get_array("f")[best_idx]
        parent = np.delete(ctx.population.get_array("x"), best_idx, axis=0)
        parent_fit = np.delete(ctx.population.get_array("f"), best_idx, axis=0)
        # update population and fitness
        pop_cand = np.vstack((parent_best, parent, offspring.get_array("x")))
        fit_cand = np.hstack((parent_best_fit, parent_fit, offspring.get_array("f")))
        # TODO: use cv if constraints are defined
        cand_idx = cmp.sort(fit_cand, np.zeros_like(fit_cand))
        pop_cand = pop_cand[cand_idx]
        fit_cand = fit_cand[cand_idx]

        popsize = len(ctx.population)
        ctx.population.clear()
        ctx.population.extend(
            {"x": pop_cand[: popsize], "f": fit_cand[: popsize]}
        )
