from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from saealib.problem import Problem
    from saealib.population import Population, Archive


@dataclass
class OptimizationContext:
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

    def count_fe(self, count: int = 1):
        self.fe += count

    def next_generation(self):
        self.gen += 1
