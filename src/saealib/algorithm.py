"""
Evolutionary Algorithm Module

This module contains the implementation of evolutionary algorithms.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import math

import numpy as np

from saealib.callback import CallbackEvent

if TYPE_CHECKING:
    from .optimizer import Optimizer
    from .operators.crossover import Crossover
    from .operators.mutation import Mutation
    from .operators.selection import ParentSelection, SurvivorSelection


class Algorithm(ABC):
    """
    Base class for evolutionary algorithms.
    """
    @abstractmethod
    def ask(self, optimizer: Optimizer) -> np.ndarray:
        pass

    @abstractmethod
    def tell(self, optimizer: Optimizer, offspring: np.ndarray, offspring_fit: np.ndarray) -> None:
        pass



class GA(Algorithm):
    """
    Genetic Algorithm class.
    """
    def __init__(self, crossover: Crossover, mutation: Mutation, parent_selection: ParentSelection, survivor_selection: SurvivorSelection):
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
        super().__init__()
        self.crossover = crossover
        self.mutation = mutation
        self.parent_selection = parent_selection
        self.survivor_selection = survivor_selection

    def ask(self, optimizer: Optimizer):
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
        
        Notes
        -----
        These optimizer attributes are used:
        - population (readonly)
        - popsize (readonly)
        - problem.dim (readonly)
        - problem.lb (readonly)
        - problem.ub (readonly)
        - rng (readonly)
        - dispatch (method)
        """
        candidate = np.empty((0, optimizer.problem.dim))
        popsize = len(optimizer.population)
        pop = optimizer.population.get("x")
        lb = optimizer.problem.lb
        ub = optimizer.problem.ub
        n_pair = math.ceil(popsize / 2)
        parent_idx_m = self.parent_selection.select(
            optimizer,
            pop,
            optimizer.population.get("f"),
            np.zeros(popsize),# TODO: use cv if constraints are defined
            n_pair=n_pair,
            n_parents=2,# TODO: recieve n_parents from Crossover
            rng=optimizer.rng)
        for i in range(n_pair):
            parent = pop[parent_idx_m[i]]
            if optimizer.rng.random() < self.crossover.crossover_rate:
                c = self.crossover.crossover(parent, rng=optimizer.rng)
            else:
                c = parent.copy()
            candidate = np.vstack((candidate, c))
        candidate = optimizer.dispatch(CallbackEvent.POST_CROSSOVER, data=candidate)
        candidate_len = len(candidate)
        for i in range(candidate_len):
            candidate[i] = self.mutation.mutate(candidate[i], (lb, ub), rng=optimizer.rng)
        candidate = optimizer.dispatch(CallbackEvent.POST_MUTATION, data=candidate)
        return candidate[:optimizer.popsize]

    def tell(self, optimizer: Optimizer, offspring: np.ndarray, offspring_fit: np.ndarray):
        """
        Update the population with offspring solutions. Optimizer population is updated in-place.

        Parameters
        ----------
        optimizer : Optimizer
            Optimizer instance.
        offspring : np.ndarray
            Offspring solutions. shape = (popsize, dim).
        offspring_fit : np.ndarray
            Offspring fitness values. shape = (popsize, ).
        
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
        cmp = optimizer.problem.comparator
        # select a best solution in parent
        best_idx = np.argmin(optimizer.population.get("f"))
        parent_best = optimizer.population.get("x")[best_idx]
        parent_best_fit = optimizer.population.get("f")[best_idx]
        parent = np.delete(optimizer.population.get("x"), best_idx, axis=0)
        parent_fit = np.delete(optimizer.population.get("f"), best_idx, axis=0)
        # update population and fitness
        pop_cand = np.vstack((parent_best, parent, offspring))
        fit_cand = np.hstack((parent_best_fit, parent_fit, offspring_fit))  
        # TODO: use cv if constraints are defined
        cand_idx = cmp.sort(fit_cand, np.zeros_like(fit_cand))
        pop_cand = pop_cand[cand_idx]
        fit_cand = fit_cand[cand_idx]
        optimizer.population.set("x", pop_cand[:optimizer.popsize])
        optimizer.population.set("f", fit_cand[:optimizer.popsize])
