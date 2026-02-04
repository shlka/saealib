"""
Context Module

This module contains the implementation of the optimization context.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from saealib.problem import Problem
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

        """
        Count function evaluations.

        Parameters
        ----------
        count : int
            Number of function evaluations to add.
        """
        self.fe += count

    def next_generation(self) -> None:
        """
        Count generations.
        """
        self.gen += 1
