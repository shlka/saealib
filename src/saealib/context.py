"""
Context Module

This module contains the implementation of the optimization context.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from saealib.problem import Problem, Comparator
    from saealib.population import Population, Archive


@dataclass
class OptimizationContext:
    """
    Optimization Context class.
    Manage the state of the optimization process.

    Attributes
    ----------
    problem : Problem
        Problem instance.
    population : Population
        Population instance.
    archive : Archive
        Archive instance.
    rng : np.random.Generator
        Random number generator.
    fe : int
        Number of function evaluations.
    gen : int
        Number of generations.
    metadata : dict
        Metadata.
    """
    problem: Problem

    population: Population
    archive: Archive
    rng: np.random.Generator

    fe: int=0
    gen: int=0

    metadata: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return self.problem.dim

    @property
    def n_obj(self) -> int:
        return self.problem.n_obj

    @property
    def lb(self) -> np.ndarray:
        return self.problem.lb

    @property
    def ub(self) -> np.ndarray:
        return self.problem.ub

    @property
    def weight(self) -> np.ndarray:
        return self.problem.weight

    @property
    def comparator(self) -> Comparator:
        return self.problem.comparator

    def count_fe(self, count: int = 1) -> None:
        """
        Count function evaluations.

        Parameters
        ----------
        count : int
            Number of function evaluations to add.
        """
        self.fe += count

    def count_generation(self) -> None:
        """
        Count generations.
        """
        self.gen += 1
